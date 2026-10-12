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

#ifndef UCSLAM_TYPES_H
#define UCSLAM_TYPES_H

#include "G2oTypes.h"

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/types/types_sba.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace ORB_SLAM3
{

/*
 * UC-SLAM optimizer types, docs/fork_design.md Sec. 2.
 *
 * VertexLogScale holds a_k, the log of the ratio between the map depth and the
 * network depth of keyframe k. EdgeLogDepth ties one map point, one keyframe
 * pose and that keyframe's a_k to a network depth. EdgeLogScaleRW is the
 * random walk between the a_k of consecutive keyframes. EdgeLogScalePrior is
 * the absolute prior on the head of the chain.
 *
 * Every Jacobian below is analytic. test/test_ucslam_types.cc compares each
 * one with g2o's numeric linearizeOplus.
 */

// a_k. Plain additive update.
class VertexLogScale : public g2o::BaseVertex<1,double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexLogScale() { setEstimate(0.0); }
    explicit VertexLogScale(double a) { setEstimate(a); }

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() { setEstimate(0.0); }

    virtual void oplusImpl(const double* update_) { setEstimate(estimate() + update_[0]); }
};

// Shared interface of the two log-depth edges, so the local BA can gate them through one pointer.
class EdgeLogDepthBase : public g2o::BaseMultiEdge<1,double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeLogDepthBase() { resize(3); }
    virtual ~EdgeLogDepthBase() {}

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    // The workspace overload of the base, hidden by the analytic override below without this.
    using g2o::BaseMultiEdge<1,double>::linearizeOplus;

    // Depth of the point in the camera at the current estimate.
    virtual double depth() const = 0;
    bool isDepthPositive() const { return depth() > kMinDepth; }

    // Number of Jacobian columns: 3 (point), 6 (pose), 1 (a_k).
    static const int kCols = 10;

    // Analytic Jacobian as one row [d/dX (3) | d/dpose (6) | d/da (1)]. Needs a mapped workspace.
    Eigen::Matrix<double,1,kCols> GetJacobian()
    {
        linearizeOplus();
        Eigen::Matrix<double,1,kCols> J;
        J.block<1,3>(0,0) = _jacobianOplus[0];
        J.block<1,6>(0,3) = _jacobianOplus[1];
        J.block<1,1>(0,9) = _jacobianOplus[2];
        return J;
    }

    // g2o's central differences into the same workspace, for the test.
    Eigen::Matrix<double,1,kCols> GetJacobianNumeric()
    {
        g2o::BaseMultiEdge<1,double>::linearizeOplus();
        Eigen::Matrix<double,1,kCols> J;
        J.block<1,3>(0,0) = _jacobianOplus[0];
        J.block<1,6>(0,3) = _jacobianOplus[1];
        J.block<1,1>(0,9) = _jacobianOplus[2];
        return J;
    }

    // A point at or behind the camera plane has no log depth. Below this the edge contributes
    // nothing and the caller gates it through isDepthPositive.
    static const double kMinDepth;
};

/*
 * r = log z - a_k - log z_hat with z the depth of the point in camera cam_idx.
 * Vertices: [0] g2o::VertexSBAPointXYZ, [1] VertexPose (body frame tangent [ur, ut],
 * twb += Rwb ut, Rwb <- Rwb Exp(ur)), [2] VertexLogScale.
 * Measurement: log z_hat. Information: 1 / sigma_D^2.
 *
 *   d r / d X        = (1/z) e3^T Rcw
 *   d r / d [ur, ut] = (1/z) e3^T Rcb [ hat(Xb) | -I ]     with Xb = Rbc Xc + tbc
 *   d r / d a_k      = -1
 */
class EdgeLogDepth : public EdgeLogDepthBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit EdgeLogDepth(int cam_idx_ = 0) : cam_idx(cam_idx_) {}

    using EdgeLogDepthBase::linearizeOplus;
    void computeError();
    virtual void linearizeOplus();
    virtual double depth() const;

    const int cam_idx;
};

/*
 * The same residual for the visual local BA, whose pose vertex is g2o::VertexSE3Expmap holding
 * Tcw with the left update Tcw <- Exp([omega, upsilon]) Tcw. Vertices: [0] g2o::VertexSBAPointXYZ,
 * [1] g2o::VertexSE3Expmap, [2] VertexLogScale.
 *
 *   d r / d X                = (1/z) e3^T Rcw
 *   d r / d [omega, upsilon] = (1/z) e3^T [ -hat(Xc) | I ] = (1/z) [ y, -x, 0, 0, 0, 1 ]
 *   d r / d a_k              = -1
 */
class EdgeLogDepthSE3 : public EdgeLogDepthBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeLogDepthSE3() {}

    using EdgeLogDepthBase::linearizeOplus;
    void computeError();
    virtual void linearizeOplus();
    virtual double depth() const;
};

// r = a_next - a_prev - measurement (the measurement is 0 for the random walk).
// Information 1 / (sigma_rw^2 dt), plus the stored variance of a fixed end point.
class EdgeLogScaleRW : public g2o::BaseBinaryEdge<1,double,VertexLogScale,VertexLogScale>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeLogScaleRW() { setMeasurement(0.0); }

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    using g2o::BaseBinaryEdge<1,double,VertexLogScale,VertexLogScale>::linearizeOplus;

    void computeError()
    {
        const VertexLogScale* VPrev = static_cast<const VertexLogScale*>(_vertices[0]);
        const VertexLogScale* VNext = static_cast<const VertexLogScale*>(_vertices[1]);
        _error(0,0) = VNext->estimate() - VPrev->estimate() - _measurement;
    }

    virtual void linearizeOplus()
    {
        _jacobianOplusXi(0,0) = -1.0;
        _jacobianOplusXj(0,0) = 1.0;
    }

    Eigen::Matrix<double,1,2> GetJacobian()
    {
        linearizeOplus();
        Eigen::Matrix<double,1,2> J;
        J(0,0) = _jacobianOplusXi(0,0);
        J(0,1) = _jacobianOplusXj(0,0);
        return J;
    }

    Eigen::Matrix<double,1,2> GetJacobianNumeric()
    {
        g2o::BaseBinaryEdge<1,double,VertexLogScale,VertexLogScale>::linearizeOplus();
        Eigen::Matrix<double,1,2> J;
        J(0,0) = _jacobianOplusXi(0,0);
        J(0,1) = _jacobianOplusXj(0,0);
        return J;
    }
};

// r = a - mean. Information 1 / var. Placed on the keyframe that carries the absolute prior.
class EdgeLogScalePrior : public g2o::BaseUnaryEdge<1,double,VertexLogScale>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit EdgeLogScalePrior(double mean = 0.0) { setMeasurement(mean); }

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    using g2o::BaseUnaryEdge<1,double,VertexLogScale>::linearizeOplus;

    void computeError()
    {
        const VertexLogScale* VA = static_cast<const VertexLogScale*>(_vertices[0]);
        _error(0,0) = VA->estimate() - _measurement;
    }

    virtual void linearizeOplus() { _jacobianOplusXi(0,0) = 1.0; }

    double GetJacobian()
    {
        linearizeOplus();
        return _jacobianOplusXi(0,0);
    }

    double GetJacobianNumeric()
    {
        g2o::BaseUnaryEdge<1,double,VertexLogScale>::linearizeOplus();
        return _jacobianOplusXi(0,0);
    }
};

/*
 * A BlockSolverX that lets the local BA read the Hessian blocks it built. The local BA uses it
 * to compute the marginal variance of every a_k in the window after the last iteration:
 * the landmarks are eliminated, the dense pose Hessian is inverted, and the diagonal entry of
 * each VertexLogScale is read. The pose part of a local window is a few hundred rows at most.
 */
class UcslamBlockSolverX : public g2o::BlockSolverX
{
public:
    explicit UcslamBlockSolverX(LinearSolverType* linearSolver) : g2o::BlockSolverX(linearSolver) {}

    const g2o::SparseBlockMatrix<PoseMatrixType>* Hpp() const { return _Hpp; }
    const g2o::SparseBlockMatrix<LandmarkMatrixType>* Hll() const { return _Hll; }
    const g2o::SparseBlockMatrix<PoseLandmarkMatrixType>* Hpl() const { return _Hpl; }
};

// Dense Schur complement of the pose part, S = Hpp - Hpl Hll^-1 Hpl^T, rows and columns in the
// order of the vertices' hessianIndex. The caller has run buildStructure and buildSystem on the
// solver at the current linearization point. Returns false when there is no pose block or a
// landmark block is singular.
// UC-SLAM (UCSLAM.LbaMargDiagOnly, attempt 08): with lowerOnly only the blocks on and below the block
// diagonal are filled, the part Eigen::LDLT reads, and the products go into S without a temporary.
// The blocks above the block diagonal stay zero.
bool UcslamDensePoseSchur(const UcslamBlockSolverX& solver, Eigen::MatrixXd& S, bool lowerOnly = false);

// UC-SLAM (UCSLAM.LbaMargDiagOnly, attempt 08): the entries cols of the diagonal of S^-1, from the LDLT of
// S solved for the unit vectors of cols only, not for the identity. Reads the lower triangle of S. var has
// the size of S and is NaN outside cols. False, as for the full inverse, when the LDLT fails or S is not
// positive semidefinite. A non-null rcond receives the LDLT rcond estimate when the LDLT succeeds.
bool UcslamMarginalDiag(const Eigen::MatrixXd& S, const std::vector<int>& cols, Eigen::VectorXd& var,
                        double* rcond = NULL);

/*
 * UC-SLAM (UCSLAM.DepthEdgeSubsampleEps, attempt 08): subsample of the depth edges of the free keyframes of
 * the inertial local BA, docs/reports/2026-09-28-accuracy-improvement/logs/design8/judge/judge.txt Sec. 3.
 *
 * Under ESS deflation the n depth edges of one keyframe carry the information of s n independent
 * measurements, s = N_eff / N. The candidates of a free keyframe are its edges that pass the gate at the
 * initial estimate on points seen by at least kUcslamSubsampleMinObs keyframes of the window. K are kept,
 * K = min(n, max(kUcslamSubsampleKMin, ceil(q n))) with q = s / (s + eps), spread over a grid of the image
 * by largest remainder, lowest key first inside each cell. Each kept edge carries the weight
 * c_k = (sum of w over the candidates) / (sum of w over the kept), w = s / sigma^2, so that the information
 * on a_k of the keyframe is kept exactly. The Huber delta and the gate scale with c_k as they scale with s:
 * both still act on the whitened residual of one measurement. Points seen by fewer keyframes and every edge
 * of a fixed keyframe are kept with weight 1.
 */
const double kUcslamHuberDepth = 1.96;      // Huber delta on the whitened residual of one depth measurement
const double kUcslamChi2Depth = 3.841;      // gate on its square, chi-square 95 percent quantile, one dof
const int kUcslamSubsampleKMin = 20;        // smallest kept set of a keyframe
const int kUcslamSubsampleMinObs = 3;       // a point seen by fewer keyframes is exempt
const int kUcslamSubsampleGrid = 4;         // cells per image side

// One candidate depth edge of a keyframe.
struct UcslamDepthCand
{
    unsigned long pointId;
    double u, v;        // keypoint of the observation [px]
    double w;           // information of the edge, s / sigma^2
    double r;           // residual log z - a_k - log z_hat at the initial estimate, only for UcslamSubsampleShift
    uint64_t key;       // UcslamDepthCandKey(pointId, keyframe id)
};

// splitmix64, the mixer of the Sec. 9 cap (docs/fork_design.md Sec. 9, UCSLAM.MaxDepthEdgesPerKF).
uint64_t UcslamSplitMix64(uint64_t x);
// The order of a candidate inside its cell: splitmix64 of (point id times the golden ratio constant) xor keyframe id.
uint64_t UcslamDepthCandKey(unsigned long pointId, unsigned long kfId);
// True for an edge that may be left out: free keyframe and a point seen by at least kUcslamSubsampleMinObs keyframes of the window.
bool UcslamSubsampleEligible(bool bFixedKF, int nObs);
// K of a keyframe with n candidates. eps <= 0 or s <= 0 keeps all.
int UcslamSubsampleBudget(int n, double s, double eps);
// Keeps K of the candidates of one keyframe (keep[i] = 1) on the grid of the image bounds [minX, maxX] x
// [minY, maxY] and returns c_k. The result does not depend on the order of cands. K >= n keeps all and
// returns 1.
double UcslamSubsampleKeyFrame(const std::vector<UcslamDepthCand>& cands, int K, double minX, double maxX,
                               double minY, double maxY, std::vector<char>& keep);
// UCSLAM.DiagSubsampleCheck: d_sd = (m_kept - m_all) sqrt(sum of w over the candidates), with m the
// w-weighted mean residual. The shift of the one-dimensional a_k estimate of the keyframe at fixed geometry,
// in units of its model sd (design8 b_design/subsample_bench_check.py).
double UcslamSubsampleShift(const std::vector<UcslamDepthCand>& cands, const std::vector<char>& keep);

// Information, Huber delta and chi2 gate of a depth edge of information w under ESS ratio s that carries the
// weight c. c = 1 gives w, kUcslamHuberDepth sqrt(s) and kUcslamChi2Depth s, the values without a subsample.
struct UcslamDepthWeight
{
    double info, delta, chi2Gate;
};
UcslamDepthWeight UcslamWeightDepthEdge(double w, double s, double c);

} // namespace ORB_SLAM3

#endif // UCSLAM_TYPES_H
