/**
* This file is part of UC-SLAM, a fork of ORB-SLAM3.
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2026 The UC-SLAM authors.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "UcslamTypes.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

namespace ORB_SLAM3
{

const double EdgeLogDepthBase::kMinDepth = 1e-6;

// ------------------------------------------------------------------ EdgeLogDepth --

double EdgeLogDepth::depth() const
{
    const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    const ImuCamPose& P = VPose->estimate();
    return P.Rcw[cam_idx].row(2).dot(VPoint->estimate()) + P.tcw[cam_idx](2);
}

void EdgeLogDepth::computeError()
{
    const VertexLogScale* VA = static_cast<const VertexLogScale*>(_vertices[2]);
    const double z = depth();
    if(z > kMinDepth)
        _error(0,0) = std::log(z) - VA->estimate() - _measurement;
    else
        _error(0,0) = 0.0;
}

void EdgeLogDepth::linearizeOplus()
{
    const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
    const ImuCamPose& P = VPose->estimate();

    const Eigen::Matrix3d& Rcw = P.Rcw[cam_idx];
    const Eigen::Vector3d Xc = Rcw * VPoint->estimate() + P.tcw[cam_idx];
    const double z = Xc(2);

    if(!(z > kMinDepth))
    {
        _jacobianOplus[0].setZero();
        _jacobianOplus[1].setZero();
        _jacobianOplus[2].setZero();
        return;
    }

    const Eigen::Vector3d Xb = P.Rbc[cam_idx] * Xc + P.tbc[cam_idx];
    const Eigen::Matrix3d& Rcb = P.Rcb[cam_idx];
    const double invz = 1.0 / z;

    // d Xc / d [ur, ut] = Rcb [ hat(Xb) | -I ], the negative of EdgeMono's SE3deriv.
    Eigen::Matrix<double,3,6> dXc;
    dXc.block<3,3>(0,0) = Skew(Xb);
    dXc.block<3,3>(0,3) = -Eigen::Matrix3d::Identity();

    _jacobianOplus[0].block<1,3>(0,0) = invz * Rcw.row(2);
    _jacobianOplus[1].block<1,6>(0,0) = invz * (Rcb.row(2) * dXc);
    _jacobianOplus[2](0,0) = -1.0;
}

// --------------------------------------------------------------- EdgeLogDepthSE3 --

double EdgeLogDepthSE3::depth() const
{
    const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    const g2o::VertexSE3Expmap* VPose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    return VPose->estimate().map(VPoint->estimate())(2);
}

void EdgeLogDepthSE3::computeError()
{
    const VertexLogScale* VA = static_cast<const VertexLogScale*>(_vertices[2]);
    const double z = depth();
    if(z > kMinDepth)
        _error(0,0) = std::log(z) - VA->estimate() - _measurement;
    else
        _error(0,0) = 0.0;
}

void EdgeLogDepthSE3::linearizeOplus()
{
    const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    const g2o::VertexSE3Expmap* VPose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    const g2o::SE3Quat& T = VPose->estimate();
    const Eigen::Vector3d Xc = T.map(VPoint->estimate());
    const double z = Xc(2);

    if(!(z > kMinDepth))
    {
        _jacobianOplus[0].setZero();
        _jacobianOplus[1].setZero();
        _jacobianOplus[2].setZero();
        return;
    }

    const double invz = 1.0 / z;
    const Eigen::Matrix3d Rcw = T.rotation().toRotationMatrix();

    // Third row of [ -hat(Xc) | I ].
    Eigen::Matrix<double,1,6> dz;
    dz << Xc(1), -Xc(0), 0.0, 0.0, 0.0, 1.0;

    _jacobianOplus[0].block<1,3>(0,0) = invz * Rcw.row(2);
    _jacobianOplus[1].block<1,6>(0,0) = invz * dz;
    _jacobianOplus[2](0,0) = -1.0;
}

// ----------------------------------------------------------- dense pose marginal --

bool UcslamDensePoseSchur(const UcslamBlockSolverX& solver, Eigen::MatrixXd& S, bool lowerOnly)
{
    typedef g2o::BlockSolverX::PoseMatrixType PoseBlock;
    typedef g2o::BlockSolverX::LandmarkMatrixType LandmarkBlock;
    typedef g2o::BlockSolverX::PoseLandmarkMatrixType PoseLandmarkBlock;

    const g2o::SparseBlockMatrix<PoseBlock>* Hpp = solver.Hpp();
    if(!Hpp)
        return false;
    const int n = Hpp->cols();
    if(n <= 0)
        return false;

    S.setZero(n, n);

    // Hpp is stored as its upper triangle (row block <= column block).
    const std::vector<g2o::SparseBlockMatrix<PoseBlock>::IntBlockMap>& cols = Hpp->blockCols();
    for(size_t c = 0; c < cols.size(); c++)
    {
        for(g2o::SparseBlockMatrix<PoseBlock>::IntBlockMap::const_iterator it = cols[c].begin(); it != cols[c].end(); ++it)
        {
            const int r = it->first;
            const PoseBlock* B = it->second;
            if(!B)
                continue;
            const int rb = Hpp->rowBaseOfBlock(r), cb = Hpp->colBaseOfBlock((int)c);
            // UC-SLAM (attempt 08): lowerOnly writes an off-diagonal block only below the block diagonal.
            if(lowerOnly && r != (int)c)
            {
                S.block(cb, rb, B->cols(), B->rows()) = B->transpose();
                continue;
            }
            S.block(rb, cb, B->rows(), B->cols()) = *B;
            if(r != (int)c)
                S.block(cb, rb, B->cols(), B->rows()) = B->transpose();
        }
    }

    const g2o::SparseBlockMatrix<LandmarkBlock>* Hll = solver.Hll();
    const g2o::SparseBlockMatrix<PoseLandmarkBlock>* Hpl = solver.Hpl();
    if(!Hll || !Hpl || !const_cast<UcslamBlockSolverX&>(solver).schur())
        return true;

    const std::vector<g2o::SparseBlockMatrix<PoseLandmarkBlock>::IntBlockMap>& lcols = Hpl->blockCols();
    std::vector<std::pair<int, const PoseLandmarkBlock*> > blocks;
    for(size_t j = 0; j < lcols.size(); j++)
    {
        if(lcols[j].empty())
            continue;
        const LandmarkBlock* D = Hll->block((int)j, (int)j);
        if(!D)
            continue;
        Eigen::FullPivLU<Eigen::MatrixXd> lu(*D);
        if(!lu.isInvertible())
            return false;
        const Eigen::MatrixXd Dinv = lu.inverse();

        blocks.assign(lcols[j].begin(), lcols[j].end());
        for(size_t a = 0; a < blocks.size(); a++)
        {
            const PoseLandmarkBlock* Ba = blocks[a].second;
            if(!Ba)
                continue;
            const int baseA = Hpp->rowBaseOfBlock(blocks[a].first);
            const Eigen::MatrixXd BaDinv = (*Ba) * Dinv;
            // UC-SLAM (attempt 08): the blocks of lcols[j] come in increasing row order (a std::map), so
            // b <= a are the blocks on and below the block diagonal.
            const size_t bEnd = lowerOnly ? a + 1 : blocks.size();
            for(size_t b = 0; b < bEnd; b++)
            {
                const PoseLandmarkBlock* Bb = blocks[b].second;
                if(!Bb)
                    continue;
                const int baseB = Hpp->rowBaseOfBlock(blocks[b].first);
                if(lowerOnly)
                    S.block(baseA, baseB, Ba->rows(), Bb->rows()).noalias() -= BaDinv * Bb->transpose();
                else
                    S.block(baseA, baseB, Ba->rows(), Bb->rows()) -= BaDinv * Bb->transpose();
            }
        }
    }
    return true;
}

// ------------------------------------------------ attempt 08: diagonal-only marginals --

bool UcslamMarginalDiag(const Eigen::MatrixXd& S, const std::vector<int>& cols, Eigen::VectorXd& var,
                        double* rcond)
{
    var = Eigen::VectorXd::Constant(S.rows(), std::numeric_limits<double>::quiet_NaN());
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if(ldlt.info() != Eigen::Success || !ldlt.isPositive())
        return false;
    if(rcond)
        *rcond = ldlt.rcond();
    if(cols.empty())
        return true;
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(S.rows(), (Eigen::Index)cols.size());
    for(size_t j = 0; j < cols.size(); j++)
        E(cols[j], (Eigen::Index)j) = 1.0;
    const Eigen::MatrixXd X = ldlt.solve(E);
    for(size_t j = 0; j < cols.size(); j++)
        var(cols[j]) = X(cols[j], (Eigen::Index)j);
    return true;
}

// ------------------------------------------------- attempt 08: depth-edge subsample --

uint64_t UcslamSplitMix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t UcslamDepthCandKey(unsigned long pointId, unsigned long kfId)
{
    return UcslamSplitMix64((uint64_t)pointId * 0x9E3779B97F4A7C15ULL ^ (uint64_t)kfId);
}

bool UcslamSubsampleEligible(bool bFixedKF, int nObs)
{
    return !bFixedKF && nObs >= kUcslamSubsampleMinObs;
}

int UcslamSubsampleBudget(int n, double s, double eps)
{
    if(n <= 0)
        return 0;
    if(!(eps > 0.0) || !(s > 0.0))
        return n;
    const double q = s / (s + eps);
    const int k = (int)std::ceil(q * (double)n);
    return std::min(n, std::max(kUcslamSubsampleKMin, k));
}

namespace
{

// Cell index of x on [lo, hi] with kUcslamSubsampleGrid cells, clamped to the grid.
int UcslamSubsampleCell(double x, double lo, double hi)
{
    if(!(hi > lo))
        return 0;
    const double t = (x - lo) / (hi - lo) * kUcslamSubsampleGrid;
    if(!(t > 0.0))
        return 0;
    if(t >= (double)kUcslamSubsampleGrid)
        return kUcslamSubsampleGrid - 1;
    return (int)t;
}

} // namespace

double UcslamSubsampleKeyFrame(const std::vector<UcslamDepthCand>& cands, int K, double minX, double maxX,
                               double minY, double maxY, std::vector<char>& keep)
{
    const int n = (int)cands.size();
    if(n == 0 || K >= n)
    {
        keep.assign(n, 1);
        return 1.0;
    }
    keep.assign(n, 0);
    if(K <= 0)
        return 0.0;

    // The candidates in key order. The choice inside a cell and every sum below then do not depend on the
    // order in which the local BA met the observations.
    std::vector<int> order(n);
    for(int i = 0; i < n; i++)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&cands](int a, int b) {
        if(cands[a].key != cands[b].key)
            return cands[a].key < cands[b].key;
        return cands[a].pointId < cands[b].pointId;
    });

    const int G = kUcslamSubsampleGrid;
    std::vector<std::vector<int> > cell(G * G);
    for(int j = 0; j < n; j++)
    {
        const UcslamDepthCand& c = cands[order[j]];
        cell[UcslamSubsampleCell(c.v, minY, maxY) * G + UcslamSubsampleCell(c.u, minX, maxX)].push_back(order[j]);
    }

    // Largest remainder: cell g gets floor(K n_g / n) edges, and the K - sum of those left go one each to
    // the cells with the largest remainders, ties to the cell with the lowest key. Integer arithmetic, so
    // exact. A cell that gets one more has a positive remainder, hence an edge left.
    std::vector<int> quota(G * G, 0);
    std::vector<long long> rem(G * G, 0);
    int given = 0;
    std::vector<int> cells;
    for(int g = 0; g < G * G; g++)
    {
        if(cell[g].empty())
            continue;
        const long long p = (long long)K * (long long)cell[g].size();
        quota[g] = (int)(p / n);
        rem[g] = p % n;
        given += quota[g];
        cells.push_back(g);
    }
    std::sort(cells.begin(), cells.end(), [&](int a, int b) {
        if(rem[a] != rem[b])
            return rem[a] > rem[b];
        const uint64_t ka = cands[cell[a][0]].key, kb = cands[cell[b][0]].key;
        if(ka != kb)
            return ka < kb;
        return a < b;
    });
    for(int j = 0; j < K - given && j < (int)cells.size(); j++)
        quota[cells[j]]++;
    for(int g = 0; g < G * G; g++)
        for(int j = 0; j < quota[g] && j < (int)cell[g].size(); j++)
            keep[cell[g][j]] = 1;

    double wAll = 0.0, wKept = 0.0;
    for(int j = 0; j < n; j++)
    {
        const int i = order[j];
        wAll += cands[i].w;
        if(keep[i])
            wKept += cands[i].w;
    }
    return wKept > 0.0 ? wAll / wKept : 0.0;
}

double UcslamSubsampleShift(const std::vector<UcslamDepthCand>& cands, const std::vector<char>& keep)
{
    double wAll = 0.0, wrAll = 0.0, wKept = 0.0, wrKept = 0.0;
    for(size_t i = 0; i < cands.size(); i++)
    {
        wAll += cands[i].w;
        wrAll += cands[i].w * cands[i].r;
        if(i < keep.size() && keep[i])
        {
            wKept += cands[i].w;
            wrKept += cands[i].w * cands[i].r;
        }
    }
    if(!(wAll > 0.0) || !(wKept > 0.0))
        return 0.0;
    return (wrKept / wKept - wrAll / wAll) * std::sqrt(wAll);
}

UcslamDepthWeight UcslamWeightDepthEdge(double w, double s, double c)
{
    UcslamDepthWeight d;
    d.info = c * w;
    d.delta = kUcslamHuberDepth * std::sqrt(s * c);
    d.chi2Gate = kUcslamChi2Depth * s * c;
    return d;
}

} // namespace ORB_SLAM3
