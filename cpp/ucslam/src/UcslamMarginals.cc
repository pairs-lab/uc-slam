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

// Posterior covariance dump, docs/fork_design.md Sec. 4. See include/UcslamMarginals.h.
// This file is identical in the fork and in the stock tree (cpp/orbslam3_patches/0003).
// The fork compiles it with UCSLAM_FORK defined.

#include "UcslamMarginals.h"

#include "Atlas.h"
#include "G2oTypes.h"
#include "ImuTypes.h"
#include "KeyFrame.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapPoint.h"
#ifdef UCSLAM_FORK
#include "Optimizer.h"
#include "UcslamTypes.h"
#endif

#include "Thirdparty/g2o/g2o/core/jacobian_workspace.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/types/types_sba.h"

#include <Eigen/Dense>
#include <opencv2/core/persistence.hpp>

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace ORB_SLAM3
{

// configs/default.yaml, block marginals.
const double kMarginalsGaugeSigma = 1e-10;
const double kMarginalsPointRankRtol = 1e-12;
const double kMarginalsThreadWaitS = 60.0;

// ---------------------------------------------------------------------------
// The linear algebra
// ---------------------------------------------------------------------------

MarginalsProblem::Result::Result():
    ok(false), spdDense(false), spdInfo(false), spdSandwich(false),
    minPivotDense(NAN), minPivotInfo(NAN), asymInfo(NAN), asymRaw(NAN), asymSandwich(NAN),
    eigMinInfo(NAN), eigMaxInfo(NAN), condInfo(NAN),
    nLocal(0), nLocalRankDeficient(0), nLocalEmpty(0), nDenseUntouched(0),
    msLocal(0.0), msDense(0.0), msInfo(0.0), msSandwich(0.0), msEig(0.0) {}

MarginalsProblem::MarginalsProblem(): mbFrozen(false), mnKept(0), mnDense(0) {}

int MarginalsProblem::AddBlock(Kind kind, int dim)
{
    if(dim <= 0)
        throw std::invalid_argument("MarginalsProblem::AddBlock: dimension must be positive");
    if(kind != LOCAL && mbFrozen)
        throw std::logic_error("MarginalsProblem::AddBlock: KEPT and DENSE blocks go before Freeze");
    Block b;
    b.kind = kind;
    b.dim = dim;
    b.offset = -1;
    b.local = -1;
    b.touched = false;
    if(kind == KEPT)
    {
        b.offset = mnKept;
        mnKept += dim;
    }
    else if(kind == DENSE)
    {
        b.offset = mnDense;   // shifted behind the KEPT rows in Freeze
        mnDense += dim;
    }
    else
    {
        b.local = (int)mvLocal.size();
        Local l;
        l.dim = dim;
        l.H = Eigen::MatrixXd::Zero(dim, dim);
        l.B = Eigen::MatrixXd::Zero(dim, dim);
        mvLocal.push_back(l);
    }
    mvBlocks.push_back(b);
    return (int)mvBlocks.size() - 1;
}

void MarginalsProblem::Freeze()
{
    if(mbFrozen)
        return;
    for(size_t i = 0; i < mvBlocks.size(); i++)
        if(mvBlocks[i].kind == DENSE)
            mvBlocks[i].offset += mnKept;
    const int M = mnKept + mnDense;
    mH = Eigen::MatrixXd::Zero(M, M);
    mB = Eigen::MatrixXd::Zero(M, M);
    mbFrozen = true;
}

MarginalsProblem::Cross& MarginalsProblem::GetCross(Local& l, int zblock)
{
    for(size_t c = 0; c < l.cross.size(); c++)
        if(l.cross[c].zblock == zblock)
            return l.cross[c];
    Cross c;
    c.zblock = zblock;
    c.H = Eigen::MatrixXd::Zero(mvBlocks[zblock].dim, l.dim);
    c.B = Eigen::MatrixXd::Zero(mvBlocks[zblock].dim, l.dim);
    l.cross.push_back(c);
    return l.cross.back();
}

void MarginalsProblem::AddFactor(int n, const int* blocks, const double* const* J, int D, const double* Wp,
                                 const double* rp)
{
    if(!mbFrozen)
        throw std::logic_error("MarginalsProblem::AddFactor: call Freeze first");
    const Eigen::Map<const Eigen::MatrixXd> W(Wp, D, D);
    const Eigen::Map<const Eigen::VectorXd> r(rp, D);
    const Eigen::VectorXd Wr = W * r;

    std::vector<Eigen::MatrixXd> WJ(n);
    std::vector<Eigen::VectorXd> g(n);
    int nLocal = 0;
    for(int a = 0; a < n; a++)
    {
        Block& b = mvBlocks[blocks[a]];
        const Eigen::Map<const Eigen::MatrixXd> Ja(J[a], D, b.dim);
        WJ[a] = W * Ja;
        g[a] = Ja.transpose() * Wr;
        b.touched = true;
        if(b.kind == LOCAL)
            nLocal++;
    }
    if(nLocal > 1)
        throw std::logic_error("MarginalsProblem::AddFactor: a factor touches at most one LOCAL block");

    for(int a = 0; a < n; a++)
    {
        const Block& A = mvBlocks[blocks[a]];
        const Eigen::Map<const Eigen::MatrixXd> Ja(J[a], D, A.dim);
        for(int c = 0; c < n; c++)
        {
            const Block& C = mvBlocks[blocks[c]];
            if(A.kind != LOCAL && C.kind != LOCAL)
            {
                mH.block(A.offset, C.offset, A.dim, C.dim).noalias() += Ja.transpose() * WJ[c];
                mB.block(A.offset, C.offset, A.dim, C.dim).noalias() += g[a] * g[c].transpose();
            }
            else if(A.kind != LOCAL && C.kind == LOCAL)
            {
                Cross& x = GetCross(mvLocal[C.local], blocks[a]);
                x.H.noalias() += Ja.transpose() * WJ[c];
                x.B.noalias() += g[a] * g[c].transpose();
            }
            else if(A.kind == LOCAL && C.kind == LOCAL)
            {
                Local& l = mvLocal[A.local];
                l.H.noalias() += Ja.transpose() * WJ[c];
                l.B.noalias() += g[a] * g[c].transpose();
            }
            // A LOCAL and C not LOCAL is the transpose of the second case, which is stored.
        }
    }
}

void MarginalsProblem::AddInformation(int block, const Eigen::MatrixXd& info)
{
    if(!mbFrozen)
        throw std::logic_error("MarginalsProblem::AddInformation: call Freeze first");
    Block& b = mvBlocks[block];
    if(b.kind == LOCAL || info.rows() != b.dim || info.cols() != b.dim)
        throw std::invalid_argument("MarginalsProblem::AddInformation: a KEPT or DENSE block of matching size");
    mH.block(b.offset, b.offset, b.dim, b.dim) += info;
    b.touched = true;
}

namespace
{

typedef std::chrono::steady_clock Clock;

double MsSince(const Clock::time_point& t)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// max |A - A^T| / max |A|, 0 for an empty or zero matrix.
double RelAsym(const Eigen::MatrixXd& A)
{
    if(A.size() == 0)
        return 0.0;
    const double m = A.cwiseAbs().maxCoeff();
    if(!(m > 0.0))
        return 0.0;
    return (A - A.transpose()).cwiseAbs().maxCoeff() / m;
}

void Symmetrize(Eigen::MatrixXd& A)
{
    A = 0.5 * (A + A.transpose());
}

} // namespace

bool MarginalsProblem::Solve(double rankRtol, int gaugeDims, Result& res)
{
    res = Result();
    Freeze();
    const int nx = mnKept, ny = mnDense;
    if(nx == 0)
    {
        res.error = "no KEPT block";
        return false;
    }
    gaugeDims = std::max(0, std::min(gaugeDims, nx - 1));

    // 1. LOCAL blocks, one at a time, with a pseudo-inverse of the block.
    Clock::time_point t = Clock::now();
    std::vector<bool> vLocalTouched(mvLocal.size(), false);
    for(size_t i = 0; i < mvBlocks.size(); i++)
        if(mvBlocks[i].kind == LOCAL && mvBlocks[i].touched)
            vLocalTouched[mvBlocks[i].local] = true;
    for(size_t li = 0; li < mvLocal.size(); li++)
    {
        Local& l = mvLocal[li];
        if(!vLocalTouched[li])
            continue;
        res.nLocal++;
        Eigen::MatrixXd Hll = l.H;
        Symmetrize(Hll);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hll);
        const Eigen::VectorXd lam = es.eigenvalues();
        const double lmax = lam.maxCoeff();
        if(!(lmax > 0.0) || !std::isfinite(lmax))
        {
            res.nLocalEmpty++;
            l.cross.clear();
            continue;
        }
        Eigen::VectorXd inv(l.dim);
        bool bDropped = false;
        for(int k = 0; k < l.dim; k++)
        {
            if(lam(k) > rankRtol * lmax)
                inv(k) = 1.0 / lam(k);
            else
            {
                inv(k) = 0.0;
                bDropped = true;
            }
        }
        if(bDropped)
            res.nLocalRankDeficient++;
        const Eigen::MatrixXd P = es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
        Eigen::MatrixXd Bll = l.B;
        Symmetrize(Bll);

        const size_t nc = l.cross.size();
        std::vector<Eigen::MatrixXd> K(nc), KB(nc);
        for(size_t c = 0; c < nc; c++)
        {
            K[c] = l.cross[c].H * P;       // H_ul H_ll^+
            KB[c] = K[c] * Bll;
        }
        for(size_t c1 = 0; c1 < nc; c1++)
        {
            const Block& b1 = mvBlocks[l.cross[c1].zblock];
            for(size_t c2 = 0; c2 < nc; c2++)
            {
                const Block& b2 = mvBlocks[l.cross[c2].zblock];
                // H'_uv = H_uv - K_u H_lv
                mH.block(b1.offset, b2.offset, b1.dim, b2.dim).noalias() -= K[c1] * l.cross[c2].H.transpose();
                // B'_uv = B_uv - K_u B_lv - B_ul K_v^T + K_u B_ll K_v^T
                Eigen::MatrixXd dB = KB[c1] * K[c2].transpose();
                dB.noalias() -= K[c1] * l.cross[c2].B.transpose();
                dB.noalias() -= l.cross[c1].B * K[c2].transpose();
                mB.block(b1.offset, b2.offset, b1.dim, b2.dim) += dB;
            }
        }
        std::vector<Cross>().swap(l.cross);
    }
    res.msLocal = MsSince(t);

    // 2. DENSE blocks. A block without any factor is decoupled from everything and is left out
    // by giving it the identity as information.
    t = Clock::now();
    for(size_t i = 0; i < mvBlocks.size(); i++)
    {
        const Block& b = mvBlocks[i];
        if(b.kind == DENSE && !b.touched)
        {
            mH.block(b.offset, b.offset, b.dim, b.dim).setIdentity();
            res.nDenseUntouched++;
        }
    }
    Eigen::MatrixXd S = mH.topLeftCorner(nx, nx);
    Eigen::MatrixXd X;   // H_yy^-1 H_yx
    if(ny > 0)
    {
        Eigen::MatrixXd Hyy = mH.bottomRightCorner(ny, ny);
        Symmetrize(Hyy);
        Eigen::LDLT<Eigen::MatrixXd> ldy(Hyy);
        res.minPivotDense = ldy.vectorD().minCoeff();
        res.spdDense = ldy.info() == Eigen::Success && ldy.isPositive() && res.minPivotDense > 0.0;
        if(!res.spdDense)
        {
            res.msDense = MsSince(t);
            res.error = "the information of velocities, biases and log-scales is not positive definite";
            return false;
        }
        X = ldy.solve(mH.block(nx, 0, ny, nx));
        S.noalias() -= mH.block(0, nx, nx, ny) * X;
    }
    res.asymInfo = RelAsym(S.bottomRightCorner(nx - gaugeDims, nx - gaugeDims));
    Symmetrize(S);
    res.msDense = MsSince(t);

    // 3. The pose block.
    t = Clock::now();
    Eigen::LDLT<Eigen::MatrixXd> lds(S);
    res.minPivotInfo = lds.vectorD().minCoeff();
    res.spdInfo = lds.info() == Eigen::Success && lds.isPositive() && res.minPivotInfo > 0.0;
    if(!res.spdInfo)
    {
        res.msInfo = MsSince(t);
        res.error = "the pose information after elimination is not positive definite";
        return false;
    }
    res.covRaw = lds.solve(Eigen::MatrixXd::Identity(nx, nx));
    res.asymRaw = RelAsym(res.covRaw.bottomRightCorner(nx - gaugeDims, nx - gaugeDims));
    Symmetrize(res.covRaw);
    res.msInfo = MsSince(t);

    // 4. The sandwich, C = A B A^T for the KEPT rows, then S^-1 C S^-1.
    t = Clock::now();
    Eigen::MatrixXd C = mB.topLeftCorner(nx, nx);
    if(ny > 0)
    {
        const Eigen::MatrixXd T1 = X.transpose() * mB.block(nx, 0, ny, nx);
        C -= T1;
        C -= T1.transpose();
        const Eigen::MatrixXd ByyX = mB.bottomRightCorner(ny, ny) * X;
        C.noalias() += X.transpose() * ByyX;
    }
    Symmetrize(C);
    const Eigen::MatrixXd CR = C * res.covRaw;
    res.covSandwich = res.covRaw * CR;
    res.asymSandwich = RelAsym(res.covSandwich.bottomRightCorner(nx - gaugeDims, nx - gaugeDims));
    Symmetrize(res.covSandwich);
    res.msSandwich = MsSince(t);

    // 5. Checks without the gauge rows: eigenvalues of the information, Cholesky of the sandwich.
    t = Clock::now();
    const int nr = nx - gaugeDims;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esS(S.bottomRightCorner(nr, nr), Eigen::EigenvaluesOnly);
    if(esS.info() == Eigen::Success)
    {
        res.eigMinInfo = esS.eigenvalues()(0);
        res.eigMaxInfo = esS.eigenvalues()(nr - 1);
        res.condInfo = res.eigMaxInfo / res.eigMinInfo;
    }
    Eigen::LLT<Eigen::MatrixXd> llt(res.covSandwich.bottomRightCorner(nr, nr));
    res.spdSandwich = llt.info() == Eigen::Success;
    res.msEig = MsSince(t);

    res.ok = res.covRaw.allFinite() && res.covSandwich.allFinite();
    if(!res.ok)
        res.error = "a covariance has a non-finite entry";
    return res.ok;
}

// ---------------------------------------------------------------------------
// Settings and files
// ---------------------------------------------------------------------------

int ReadDumpMarginalsFlag(const std::string& settingsFile)
{
    if(settingsFile.empty())
        return 0;
    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if(!fs.isOpened())
        return 0;
    cv::FileNode node = fs["UCSLAM.DumpMarginals"];
    if(node.empty())
        return 0;
    if(node.isInt())
        return (int)node;
    if(node.isReal())
        return (int)std::lround((double)node);
    return 0;
}

namespace
{

// The depth edge constants of the fork's local BA (UcslamChainBuilder in src/Optimizer.cc):
// the Huber delta and the gate on the squared whitened residual, chi-square with one degree
// of freedom at 95 percent, both scaled with the ESS information factor there.
const double kHuberDepth = 1.96;
const double kChi2Depth = 3.841;

// Huber deltas and the inertial one of Optimizer::FullInertialBA.
const double kChi2Mono = 5.991;
const double kChi2Stereo = 7.815;
const double kChi2Inertial = 16.92;

struct DumpSettings
{
    std::string tree;
    bool depthFactor;
    double infoScale;   // N_eff / N under ESS deflation, else 1
    double sigma0;
    double sigmaRW;
};

DumpSettings CurrentSettings()
{
    DumpSettings s;
#ifdef UCSLAM_FORK
    const Settings::Ucslam& u = Optimizer::UcslamParams();
    s.tree = "fork";
    s.depthFactor = u.useDepthFactor != 0;
    s.infoScale = u.essDeflation ? (double)u.neffOverN : 1.0;
    s.sigma0 = u.sigma0;
    s.sigmaRW = u.sigmaRW;
#else
    s.tree = "stock";
    s.depthFactor = false;
    s.infoScale = 1.0;
    s.sigma0 = 0.0;
    s.sigmaRW = 0.0;
#endif
    return s;
}

bool MakeDirs(const std::string& path)
{
    if(path.empty())
        return true;
    std::string cur;
    std::stringstream ss(path);
    std::string part;
    if(path[0] == '/')
        cur = "/";
    while(std::getline(ss, part, '/'))
    {
        if(part.empty())
            continue;
        cur += part;
        if(mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
            return false;
        cur += "/";
    }
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string JsonNum(double v)
{
    if(!std::isfinite(v))
        return "null";
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

std::string JsonStr(const std::string& s)
{
    std::string o = "\"";
    for(size_t i = 0; i < s.size(); i++)
    {
        const char c = s[i];
        if(c == '"' || c == '\\')
        {
            o += '\\';
            o += c;
        }
        else if((unsigned char)c < 0x20)
        {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)(unsigned char)c);
            o += buf;
        }
        else
            o += c;
    }
    return o + "\"";
}

const char* JsonBool(bool b) { return b ? "true" : "false"; }

// NPY format version 1.0, float64, C order.
bool WriteNpy(const std::string& path, const Eigen::MatrixXd& M)
{
    const uint16_t one = 1;
    const bool bLittle = *reinterpret_cast<const unsigned char*>(&one) == 1;
    std::ostringstream h;
    h << "{'descr': '" << (bLittle ? "<" : ">") << "f8', 'fortran_order': False, 'shape': ("
      << M.rows() << ", " << M.cols() << "), }";
    std::string header = h.str();
    const size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    std::ofstream f(path.c_str(), std::ios::binary);
    if(!f)
        return false;
    const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
    f.write(magic, 8);
    const uint16_t len = (uint16_t)header.size();
    const char lenb[2] = {(char)(len & 0xff), (char)(len >> 8)};
    f.write(lenb, 2);
    f.write(header.data(), header.size());
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> R = M;
    f.write(reinterpret_cast<const char*>(R.data()), sizeof(double) * R.size());
    return (bool)f;
}

// Owns the g2o vertices of the dump. The edges are linearized one at a time and never enter a graph.
struct VertexOwner
{
    std::vector<g2o::OptimizableGraph::Vertex*> v;
    template<class T> T* Add(T* p) { v.push_back(p); return p; }
    ~VertexOwner() { for(size_t i = 0; i < v.size(); i++) delete v[i]; }
};

struct EdgeCount
{
    long n, nDown, nSkipped;
    EdgeCount(): n(0), nDown(0), nSkipped(0) {}
};

// Linearizes one edge at the current estimate and adds it to the problem with the Huber weight
// rho'(chi2) folded into its information. huberDelta <= 0 means the edge has no kernel.
class EdgeAdder
{
public:
    explicit EdgeAdder(MarginalsProblem& prob): mProb(prob)
    {
        mWs.updateSize(6, 9 * 6);   // EdgeInertial: six vertices, 9 x 6 Jacobian at most
        mWs.allocate();
    }

    bool Add(g2o::OptimizableGraph::Edge* e, const int* blocks, double huberDelta, EdgeCount& count)
    {
        e->computeError();
        const int D = e->dimension();
        const double chi2 = e->chi2();
        if(!std::isfinite(chi2))
        {
            count.nSkipped++;
            return false;
        }
        double w = 1.0;
        if(huberDelta > 0.0)
        {
            mHuber.setDelta(huberDelta);
            Eigen::Vector3d rho;
            mHuber.robustify(chi2, rho);
            w = rho[1];
        }
        e->linearizeOplus(mWs);
        const int nV = (int)e->vertices().size();
        const double* J[6];
        for(int k = 0; k < nV; k++)
        {
            const g2o::OptimizableGraph::Vertex* v = static_cast<const g2o::OptimizableGraph::Vertex*>(e->vertex(k));
            J[k] = mWs.workspaceForVertex(k);
            if(!Eigen::Map<const Eigen::MatrixXd>(J[k], D, v->dimension()).allFinite())
            {
                count.nSkipped++;
                return false;
            }
        }
        mW = w * Eigen::Map<const Eigen::MatrixXd>(e->informationData(), D, D);
        mProb.AddFactor(nV, blocks, J, D, mW.data(), e->errorData());
        count.n++;
        if(w < 1.0)
            count.nDown++;
        return true;
    }

private:
    MarginalsProblem& mProb;
    g2o::JacobianWorkspace mWs;
    g2o::RobustKernelHuber mHuber;
    Eigen::MatrixXd mW;
};

bool WaitForThreads(LocalMapping* pLocalMapper, LoopClosing* pLoopCloser, double waitS)
{
    const Clock::time_point t0 = Clock::now();
    while(true)
    {
        const bool bMapper = !pLocalMapper || pLocalMapper->isFinished();
        const bool bLoop = !pLoopCloser || (pLoopCloser->isFinished() && !pLoopCloser->isRunningGBA());
        if(bMapper && bLoop)
            return true;
        if(MsSince(t0) > 1000.0 * waitS)
            return false;
        usleep(5000);
    }
}

void RemoveIfPresent(const std::string& path)
{
    if(unlink(path.c_str()) != 0 && errno != ENOENT)
        std::cerr << "[UCSLAM] marginals: cannot remove " << path << std::endl;
}

// What the dump reads from the atlas. The caller holds the map's update mutex.
struct DumpInput
{
    Map* pMap;
    size_t nMaps;
    std::vector<KeyFrame*> vpKFs;   // good keyframes of pMap in order of timestamp
    std::map<KeyFrame*, int> kfIndex;
};

// Counts of one build, and the poses the covariance is computed at.
struct BuildOutput
{
    std::vector<Eigen::Matrix3d> Rwb;
    std::vector<Eigen::Vector3d> twb;
    EdgeCount cMono, cMonoRight, cStereo, cInertial, cGyroRW, cAccRW;
    EdgeCount cDepth, cChain, cMerge, cPrior, cUnary, cDetached;
    long nDepthGated, nPoints, nPointsUsed;
    int nImu, nDenseDims;
    double msBuild;
    BuildOutput(): nDepthGated(0), nPoints(0), nPointsUsed(0), nImu(0), nDenseDims(0), msBuild(0.0) {}
};

/*
 * Builds the full problem of the map into prob. gaugeDims 6 holds the whole pose of the first
 * keyframe, which is the contract. gaugeDims 4 holds its position and its yaw about the world
 * z axis only. That is the gauge freedom of an inertial problem, because gravity makes roll and
 * pitch observable. It exists for a diagnostic that measures what the six-dimensional gauge
 * changes.
 */
void BuildProblem(const DumpInput& in, const DumpSettings& s, int gaugeDims, MarginalsProblem& prob,
                  BuildOutput& o)
{
    const Clock::time_point t0 = Clock::now();
    const std::vector<KeyFrame*>& vpKFs = in.vpKFs;
    const std::map<KeyFrame*, int>& kfIndex = in.kfIndex;
    Map* pMap = in.pMap;
    const int N = (int)vpKFs.size();

    // Vertices and blocks. The poses are the KEPT blocks in the order of keyframes.csv.
    VertexOwner own;
    std::vector<VertexPose*> vP(N, static_cast<VertexPose*>(NULL));
    std::vector<VertexVelocity*> vV(N, static_cast<VertexVelocity*>(NULL));
    std::vector<VertexGyroBias*> vG(N, static_cast<VertexGyroBias*>(NULL));
    std::vector<VertexAccBias*> vA(N, static_cast<VertexAccBias*>(NULL));
    std::vector<int> bP(N, -1), bV(N, -1), bG(N, -1), bA(N, -1), bS(N, -1);
    for(int i = 0; i < N; i++)
    {
        vP[i] = own.Add(new VertexPose(vpKFs[i]));
        bP[i] = prob.AddBlock(MarginalsProblem::KEPT, 6);
    }
    int& nImu = o.nImu;
    for(int i = 0; i < N; i++)
    {
        if(!vpKFs[i]->bImu)
            continue;
        nImu++;
        vV[i] = own.Add(new VertexVelocity(vpKFs[i]));
        vG[i] = own.Add(new VertexGyroBias(vpKFs[i]));
        vA[i] = own.Add(new VertexAccBias(vpKFs[i]));
        bV[i] = prob.AddBlock(MarginalsProblem::DENSE, 3);
        bG[i] = prob.AddBlock(MarginalsProblem::DENSE, 3);
        bA[i] = prob.AddBlock(MarginalsProblem::DENSE, 3);
    }
#ifdef UCSLAM_FORK
    std::vector<VertexLogScale*> vS(N, static_cast<VertexLogScale*>(NULL));
    if(s.depthFactor)
    {
        for(int i = 0; i < N; i++)
        {
            // The value UcslamChainBuilder::AddVertex starts a node at.
            KeyFrame* pKF = vpKFs[i];
            double a = pKF->mLogScale;
            if(!(pKF->mLogScaleVar > 0.0) && pKF->mPrevKF && pKF->mPrevKF->mLogScaleVar > 0.0)
                a = pKF->mPrevKF->mLogScale;
            vS[i] = own.Add(new VertexLogScale(a));
            bS[i] = prob.AddBlock(MarginalsProblem::DENSE, 1);
        }
    }
#endif
    prob.Freeze();

    EdgeAdder adder(prob);
    EdgeCount &cMono = o.cMono, &cMonoRight = o.cMonoRight, &cStereo = o.cStereo;
    EdgeCount &cInertial = o.cInertial, &cGyroRW = o.cGyroRW, &cAccRW = o.cAccRW;
#ifdef UCSLAM_FORK
    EdgeCount &cDepth = o.cDepth, &cChain = o.cChain, &cMerge = o.cMerge;
    EdgeCount &cPrior = o.cPrior, &cUnary = o.cUnary, &cDetached = o.cDetached;
    long& nDepthGated = o.nDepthGated;
#endif

    // Inertial and bias random walk edges, as FullInertialBA with bInit false.
    for(int i = 0; i < N; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(!pKFi->mPrevKF)
            continue;
        std::map<KeyFrame*, int>::const_iterator itp = kfIndex.find(pKFi->mPrevKF);
        if(itp == kfIndex.end())
            continue;
        const int j = itp->second;
        if(!pKFi->bImu || !pKFi->mPrevKF->bImu || !pKFi->mpImuPreintegrated || !vV[j] || !vV[i])
            continue;

        std::unique_ptr<EdgeInertial> ei(new EdgeInertial(pKFi->mpImuPreintegrated));
        ei->setVertex(0, vP[j]);
        ei->setVertex(1, vV[j]);
        ei->setVertex(2, vG[j]);
        ei->setVertex(3, vA[j]);
        ei->setVertex(4, vP[i]);
        ei->setVertex(5, vV[i]);
        const int bi[6] = {bP[j], bV[j], bG[j], bA[j], bP[i], bV[i]};
        adder.Add(ei.get(), bi, std::sqrt(kChi2Inertial), cInertial);

        std::unique_ptr<EdgeGyroRW> egr(new EdgeGyroRW());
        egr->setVertex(0, vG[j]);
        egr->setVertex(1, vG[i]);
        egr->setInformation(pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse());
        const int bg[2] = {bG[j], bG[i]};
        adder.Add(egr.get(), bg, -1.0, cGyroRW);

        std::unique_ptr<EdgeAccRW> ear(new EdgeAccRW());
        ear->setVertex(0, vA[j]);
        ear->setVertex(1, vA[i]);
        ear->setInformation(pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse());
        const int ba[2] = {bA[j], bA[i]};
        adder.Add(ear.get(), ba, -1.0, cAccRW);
    }

#ifdef UCSLAM_FORK
    // The log-scale chain, as UcslamChainBuilder::AddChainEdges with every node free.
    if(s.depthFactor)
    {
        const double sigmaRW2 = s.sigmaRW * s.sigmaRW;
        std::unique_ptr<EdgeLogScaleRW> eRW(new EdgeLogScaleRW());
        std::unique_ptr<EdgeLogScalePrior> ePr(new EdgeLogScalePrior(0.0));
        Eigen::Matrix<double,1,1> I1;
        for(int i = 0; i < N; i++)
        {
            KeyFrame* pKF = vpKFs[i];
            KeyFrame* pPrev = pKF->mPrevKF;
            if(!pPrev)
                continue;
            std::map<KeyFrame*, int>::const_iterator itp = kfIndex.find(pPrev);
            if(itp == kfIndex.end())
                continue;
            const int j = itp->second;
            Map::LogScaleChainTerms terms;
            Map::GetLogScaleChainTerms(pKF, terms);
            double var;
            if(terms.segInfo > 0.0)
                var = 1.0 / terms.segInfo;
            else
            {
                const double dt = std::max(std::fabs(pKF->mTimeStamp - pPrev->mTimeStamp), 1e-6);
                var = std::max(sigmaRW2 * dt, 1e-12);
            }
            eRW->setVertex(0, vS[j]);
            eRW->setVertex(1, vS[i]);
            eRW->setMeasurement(0.0);
            I1(0,0) = 1.0 / var;
            eRW->setInformation(I1);
            const int b2[2] = {bS[j], bS[i]};
            adder.Add(eRW.get(), b2, -1.0, cChain);
        }
        const std::vector<Map::LogScaleMergePrior> vMerge = Map::GetLogScaleMergePriors();
        for(size_t k = 0; k < vMerge.size(); k++)
        {
            const Map::LogScaleMergePrior& m = vMerge[k];
            std::map<KeyFrame*, int>::const_iterator ia = kfIndex.find(m.pAbsorbedHead);
            std::map<KeyFrame*, int>::const_iterator is = kfIndex.find(m.pSurvivingHead);
            if(ia == kfIndex.end() || is == kfIndex.end() || !(m.var > 0.0))
                continue;
            eRW->setVertex(0, vS[is->second]);
            eRW->setVertex(1, vS[ia->second]);
            eRW->setMeasurement(m.mean);
            I1(0,0) = 1.0 / m.var;
            eRW->setInformation(I1);
            const int b2[2] = {bS[is->second], bS[ia->second]};
            adder.Add(eRW.get(), b2, -1.0, cMerge);
        }
        for(int i = 0; i < N; i++)
        {
            KeyFrame* pKF = vpKFs[i];
            const int b1[1] = {bS[i]};
            bool bAnchored = false;
            if(pKF->mbLogScalePrior && pKF->mLogScalePriorVar > 0.0)
            {
                ePr->setVertex(0, vS[i]);
                ePr->setMeasurement(pKF->mLogScalePriorMean);
                I1(0,0) = 1.0 / pKF->mLogScalePriorVar;
                ePr->setInformation(I1);
                adder.Add(ePr.get(), b1, -1.0, cPrior);
                bAnchored = true;
            }
            Map::LogScaleChainTerms terms;
            if(Map::GetLogScaleChainTerms(pKF, terms) && terms.unaryInfo > 0.0)
            {
                ePr->setVertex(0, vS[i]);
                ePr->setMeasurement(terms.unaryB / terms.unaryInfo);
                I1(0,0) = terms.unaryInfo;
                ePr->setInformation(I1);
                adder.Add(ePr.get(), b1, -1.0, cUnary);
            }
            // A node without a predecessor in the map and without the absolute prior is held at its
            // stored value with its stored variance, as the local BA holds the oldest node of a window.
            if(!bAnchored && (!pKF->mPrevKF || kfIndex.find(pKF->mPrevKF) == kfIndex.end()))
            {
                double var = (double)s.sigma0 * (double)s.sigma0;
                if(pKF->mLogScaleVar > 0.0)
                    var = pKF->mLogScaleVar;
                else if(pKF->mbLogScalePrior && pKF->mLogScalePriorVar > 0.0)
                    var = pKF->mLogScalePriorVar;
                ePr->setVertex(0, vS[i]);
                ePr->setMeasurement(vS[i]->estimate());
                I1(0,0) = 1.0 / var;
                ePr->setInformation(I1);
                adder.Add(ePr.get(), b1, -1.0, cDetached);
            }
        }
    }
    std::vector<int> vDepthState(N, 0);   // 0 not read, 1 ready, -1 none
    std::vector<std::vector<float> > vDepthZ(N), vDepthSigma(N);
    std::unique_ptr<EdgeLogDepth> eDepth(new EdgeLogDepth(0));
#endif

    // Map points and their reprojection edges, as FullInertialBA. Every point is one LOCAL block.
    std::unique_ptr<g2o::VertexSBAPointXYZ> vPoint(new g2o::VertexSBAPointXYZ());
    std::unique_ptr<EdgeMono> eMono(new EdgeMono(0));
    std::unique_ptr<EdgeMono> eMonoRight(new EdgeMono(1));
    std::unique_ptr<EdgeStereo> eStereo(new EdgeStereo(0));
    const double thHuberMono = std::sqrt(kChi2Mono);
    const double thHuberStereo = std::sqrt(kChi2Stereo);
    long& nPoints = o.nPoints;
    long& nPointsUsed = o.nPointsUsed;
    const std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    for(size_t k = 0; k < vpMPs.size(); k++)
    {
        MapPoint* pMP = vpMPs[k];
        if(!pMP || pMP->isBad())
            continue;
        nPoints++;
        const std::map<KeyFrame*, std::tuple<int,int> > observations = pMP->GetObservations();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int bPoint = -1;
        for(std::map<KeyFrame*, std::tuple<int,int> >::const_iterator mit = observations.begin();
            mit != observations.end(); mit++)
        {
            KeyFrame* pKFi = mit->first;
            std::map<KeyFrame*, int>::const_iterator it = kfIndex.find(pKFi);
            if(it == kfIndex.end() || pKFi->isBad())
                continue;
            const int i = it->second;
            const int leftIndex = std::get<0>(mit->second);
            if(bPoint < 0)
                bPoint = prob.AddBlock(MarginalsProblem::LOCAL, 3);
            const int b2[2] = {bPoint, bP[i]};

            if(leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0)   // monocular observation
            {
                const cv::KeyPoint& kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;
                eMono->setVertex(0, vPoint.get());
                eMono->setVertex(1, vP[i]);
                eMono->setMeasurement(obs);
                eMono->setInformation(Eigen::Matrix2d::Identity() * pKFi->mvInvLevelSigma2[kpUn.octave]);
                adder.Add(eMono.get(), b2, thHuberMono, cMono);
            }
            else if(leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0)   // stereo observation
            {
                const cv::KeyPoint& kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,3,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight[leftIndex];
                eStereo->setVertex(0, vPoint.get());
                eStereo->setVertex(1, vP[i]);
                eStereo->setMeasurement(obs);
                eStereo->setInformation(Eigen::Matrix3d::Identity() * pKFi->mvInvLevelSigma2[kpUn.octave]);
                adder.Add(eStereo.get(), b2, thHuberStereo, cStereo);
            }

            if(pKFi->mpCamera2)   // right camera observation, with FullInertialBA's index test
            {
                int rightIndex = std::get<1>(mit->second);
                if(rightIndex != -1 && rightIndex < (int)pKFi->mvKeysRight.size())
                {
                    rightIndex -= pKFi->NLeft;
                    const cv::KeyPoint& kp = pKFi->mvKeysRight[rightIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kp.pt.x, kp.pt.y;
                    eMonoRight->setVertex(0, vPoint.get());
                    eMonoRight->setVertex(1, vP[i]);
                    eMonoRight->setMeasurement(obs);
                    eMonoRight->setInformation(Eigen::Matrix2d::Identity() * pKFi->mvInvLevelSigma2[kp.octave]);
                    adder.Add(eMonoRight.get(), b2, thHuberMono, cMonoRight);
                }
            }

#ifdef UCSLAM_FORK
            // The log-depth edge of this observation, as the local BA adds and gates it.
            if(s.depthFactor && leftIndex != -1)
            {
                if(vDepthState[i] == 0)
                    vDepthState[i] = vpKFs[i]->GetDepthPrior(vDepthZ[i], vDepthSigma[i]) ? 1 : -1;
                if(vDepthState[i] == 1 && leftIndex < (int)vDepthZ[i].size() && leftIndex < (int)vDepthSigma[i].size())
                {
                    const float zhat = vDepthZ[i][leftIndex], sigma = vDepthSigma[i][leftIndex];
                    if(zhat > 0.f && sigma > 0.f)
                    {
                        eDepth->setVertex(0, vPoint.get());
                        eDepth->setVertex(1, vP[i]);
                        eDepth->setVertex(2, vS[i]);
                        eDepth->setMeasurement(std::log((double)zhat));
                        Eigen::Matrix<double,1,1> I1;
                        I1(0,0) = s.infoScale / ((double)sigma * (double)sigma);
                        eDepth->setInformation(I1);
                        bool bGate = !eDepth->isDepthPositive();
                        if(!bGate)
                        {
                            eDepth->computeError();
                            bGate = !(eDepth->chi2() <= kChi2Depth * s.infoScale);
                        }
                        if(bGate)
                            nDepthGated++;
                        else
                        {
                            const int b3[3] = {bPoint, bP[i], bS[i]};
                            adder.Add(eDepth.get(), b3, kHuberDepth * std::sqrt(s.infoScale), cDepth);
                        }
                    }
                }
            }
#endif
        }
        if(bPoint >= 0)
            nPointsUsed++;
    }

    // Gauge on the first keyframe's pose.
    const double gaugeInfo = 1.0 / (kMarginalsGaugeSigma * kMarginalsGaugeSigma);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(6, 6);
    if(gaugeDims == 4)
    {
        // R Exp(ur) = Rz(d) R gives ur = R^T e_z d, the yaw about world z seen in the body tangent.
        const Eigen::Vector3d a = vP[0]->estimate().Rwb.transpose() * Eigen::Vector3d::UnitZ();
        G.topLeftCorner(3, 3) = gaugeInfo * a * a.transpose();
        G.bottomRightCorner(3, 3) = gaugeInfo * Eigen::Matrix3d::Identity();
    }
    else
        G = gaugeInfo * Eigen::MatrixXd::Identity(6, 6);
    prob.AddInformation(bP[0], G);

    o.Rwb.resize(N);
    o.twb.resize(N);
    for(int i = 0; i < N; i++)
    {
        o.Rwb[i] = vP[i]->estimate().Rwb;
        o.twb[i] = vP[i]->estimate().twb;
    }
    o.nDenseDims = prob.NumDenseDims();
    o.msBuild = MsSince(t0);
}

bool DumpImpl(Atlas* pAtlas, const std::string& outDir, bool bThreadsFinished, double waitMs)
{
    const Clock::time_point t0 = Clock::now();
    const DumpSettings s = CurrentSettings();
    if(!pAtlas)
    {
        std::cout << "[UCSLAM] marginals: no atlas, nothing written" << std::endl;
        return false;
    }

    // The largest map, chosen as System::SaveKeyFrameTrajectoryEuRoC chooses it.
    const std::vector<Map*> vpMaps = pAtlas->GetAllMaps();
    Map* pMap = NULL;
    size_t nMax = 0;
    for(size_t i = 0; i < vpMaps.size(); i++)
    {
        if(vpMaps[i] && vpMaps[i]->GetAllKeyFrames().size() > nMax)
        {
            nMax = vpMaps[i]->GetAllKeyFrames().size();
            pMap = vpMaps[i];
        }
    }
    if(!pMap)
    {
        std::cout << "[UCSLAM] marginals: no map with keyframes, nothing written" << std::endl;
        return false;
    }
    std::unique_lock<std::mutex> lockMap(pMap->mMutexMapUpdate);

    std::vector<KeyFrame*> vpKFs;
    {
        const std::vector<KeyFrame*> vAll = pMap->GetAllKeyFrames();
        for(size_t i = 0; i < vAll.size(); i++)
            if(vAll[i] && !vAll[i]->isBad())
                vpKFs.push_back(vAll[i]);
    }
    std::sort(vpKFs.begin(), vpKFs.end(), [](KeyFrame* a, KeyFrame* b) {
        return a->mTimeStamp < b->mTimeStamp || (a->mTimeStamp == b->mTimeStamp && a->mnId < b->mnId);
    });
    const int N = (int)vpKFs.size();
    if(N == 0)
    {
        std::cout << "[UCSLAM] marginals: the largest map has no good keyframe, nothing written" << std::endl;
        return false;
    }
    DumpInput in;
    in.pMap = pMap;
    in.nMaps = vpMaps.size();
    in.vpKFs = vpKFs;
    for(int i = 0; i < N; i++)
        in.kfIndex[vpKFs[i]] = i;

    MarginalsProblem::Result res;
    BuildOutput o;
    {
        MarginalsProblem prob;
        BuildProblem(in, s, 6, prob, o);
        prob.Solve(kMarginalsPointRankRtol, 6, res);
    }

    // Diagnostic, off unless UCSLAM_MARGINALS_GAUGE4=1 is in the environment: the same problem with the
    // four-dimensional gauge, written to cov_raw_gauge4.npy and cov_sandwich_gauge4.npy.
    const char* pGauge4 = getenv("UCSLAM_MARGINALS_GAUGE4");
    const bool bGauge4 = pGauge4 && std::string(pGauge4) == "1";
    MarginalsProblem::Result res4;
    double msGauge4 = 0.0;
    if(bGauge4)
    {
        const Clock::time_point t4 = Clock::now();
        MarginalsProblem prob4;
        BuildOutput o4;
        BuildProblem(in, s, 4, prob4, o4);
        prob4.Solve(kMarginalsPointRankRtol, 6, res4);
        msGauge4 = MsSince(t4);
    }

    const EdgeCount &cMono = o.cMono, &cMonoRight = o.cMonoRight, &cStereo = o.cStereo;
    const EdgeCount &cInertial = o.cInertial, &cGyroRW = o.cGyroRW, &cAccRW = o.cAccRW;
    const EdgeCount &cDepth = o.cDepth, &cChain = o.cChain, &cMerge = o.cMerge;
    const EdgeCount &cPrior = o.cPrior, &cUnary = o.cUnary, &cDetached = o.cDetached;
    const int nImu = o.nImu;
    const long nPoints = o.nPoints, nPointsUsed = o.nPointsUsed, nDepthGated = o.nDepthGated;
    const double msBuild = o.msBuild;

    // Files. Old files of an earlier run in the same directory go first, so a failed dump never
    // leaves a covariance next to keyframes it does not belong to.
    const Clock::time_point tWrite = Clock::now();
    if(!MakeDirs(outDir))
    {
        std::cerr << "[UCSLAM] marginals: cannot create " << outDir << std::endl;
        return false;
    }
    const std::string dir = outDir.empty() ? std::string(".") : outDir;
    RemoveIfPresent(dir + "/meta.json");
    RemoveIfPresent(dir + "/cov_raw.npy");
    RemoveIfPresent(dir + "/cov_sandwich.npy");
    RemoveIfPresent(dir + "/keyframes.csv");
    RemoveIfPresent(dir + "/cov_raw_gauge4.npy");
    RemoveIfPresent(dir + "/cov_sandwich_gauge4.npy");

    bool bWritten = true;
    {
        std::ofstream f((dir + "/keyframes.csv").c_str());
        f << "kf_id,timestamp_s,px,py,pz,qx,qy,qz,qw\n";
        for(int i = 0; i < N; i++)
        {
            Eigen::Quaterniond q(o.Rwb[i]);
            q.normalize();
            char buf[512];
            snprintf(buf, sizeof(buf), "%lu,%.9f,%.9f,%.9f,%.9f,%.12f,%.12f,%.12f,%.12f\n",
                     vpKFs[i]->mnId, vpKFs[i]->mTimeStamp, o.twb[i](0), o.twb[i](1), o.twb[i](2),
                     q.x(), q.y(), q.z(), q.w());
            f << buf;
        }
        bWritten = bWritten && (bool)f;
    }
    if(res.ok)
    {
        bWritten = WriteNpy(dir + "/cov_raw.npy", res.covRaw) && bWritten;
        bWritten = WriteNpy(dir + "/cov_sandwich.npy", res.covSandwich) && bWritten;
    }
    if(bGauge4 && res4.ok)
    {
        WriteNpy(dir + "/cov_raw_gauge4.npy", res4.covRaw);
        WriteNpy(dir + "/cov_sandwich_gauge4.npy", res4.covSandwich);
    }
    const double msWrite = MsSince(tWrite);
    const double wallMs = MsSince(t0) - msGauge4;   // the diagnostic is not part of the dump's cost

    const bool bDepthEdges = s.depthFactor && cDepth.n > 0;
    std::ostringstream gauge;
    gauge << "the pose of the first keyframe (row 0, kf_id " << vpKFs[0]->mnId
          << ") is held by a prior with standard deviation " << kMarginalsGaugeSigma
          << " on all six tangent components [ur, ut]";
    std::string notes =
        "Gauss-Newton information of the full problem of the largest map at the final estimate, "
        "with each Huber kernel weight rho'(chi2) at that estimate folded into the information. "
        "Nothing is optimized. Map points are eliminated one at a time with a pseudo-inverse of "
        "their 3x3 block. Velocities, biases and log-scales are eliminated by a dense Schur "
        "complement. The pose block is inverted with LDLT. Reprojection information is "
        "invSigma2 of the octave as in FullInertialBA.";
    if(s.depthFactor)
        notes += " Log-depth edges above the chi-square gate or with a point behind the camera are "
                 "left out, as the local BA gates them.";
    if(!bThreadsFinished)
        notes += " The mapping threads did not finish within the wait, the estimate may have moved during the dump.";
    if(nImu > 0 && nImu < N)
        notes += " Some keyframes have no inertial state.";

    std::ostringstream m;
    m << "{\n";
    m << "  \"format\": \"ucslam-marginals-1\",\n";
    m << "  \"n_kf\": " << N << ",\n";
    m << "  \"gauge\": " << JsonStr(gauge.str()) << ",\n";
    m << "  \"includes_depth_edges\": " << JsonBool(bDepthEdges) << ",\n";
    m << "  \"n_edges_by_type\": {\"EdgeMono\": " << cMono.n << ", \"EdgeMonoRight\": " << cMonoRight.n
      << ", \"EdgeStereo\": " << cStereo.n << ", \"EdgeInertial\": " << cInertial.n
      << ", \"EdgeGyroRW\": " << cGyroRW.n << ", \"EdgeAccRW\": " << cAccRW.n
      << ", \"EdgeLogDepth\": " << cDepth.n << ", \"EdgeLogScaleRW\": " << cChain.n
      << ", \"EdgeLogScaleMerge\": " << cMerge.n << ", \"EdgeLogScalePrior\": " << cPrior.n
      << ", \"EdgeLogScaleUnary\": " << cUnary.n << ", \"EdgeLogScaleDetached\": " << cDetached.n
      << ", \"GaugePrior\": 1},\n";
    m << "  \"wall_ms\": " << JsonNum(wallMs) << ",\n";
    m << "  \"tree\": " << JsonStr(s.tree) << ",\n";
    m << "  \"notes\": " << JsonStr(notes) << ",\n";
    m << "  \"ok\": " << JsonBool(res.ok && bWritten) << ",\n";
    m << "  \"error\": " << JsonStr(res.ok ? (bWritten ? "" : "a file could not be written") : res.error) << ",\n";
    m << "  \"tangent\": \"[ur (3), ut (3)] of VertexPose, R_true = R_est Exp(ur), t_true = t_est + R_est ut\",\n";
    m << "  \"gauge_sigma\": " << JsonNum(kMarginalsGaugeSigma) << ",\n";
    m << "  \"point_rank_rtol\": " << JsonNum(kMarginalsPointRankRtol) << ",\n";
    m << "  \"map_id\": " << pMap->GetId() << ",\n";
    m << "  \"n_maps\": " << vpMaps.size() << ",\n";
    m << "  \"n_kf_inertial\": " << nImu << ",\n";
    m << "  \"n_points\": " << nPoints << ",\n";
    m << "  \"n_points_used\": " << nPointsUsed << ",\n";
    m << "  \"n_points_rank_deficient\": " << res.nLocalRankDeficient << ",\n";
    m << "  \"n_points_empty\": " << res.nLocalEmpty << ",\n";
    m << "  \"n_dense_blocks_without_edges\": " << res.nDenseUntouched << ",\n";
    m << "  \"state_dims\": {\"pose\": " << 6 * N << ", \"dense_eliminated\": "
      << o.nDenseDims << ", \"points\": " << 3 * nPointsUsed << "},\n";
    m << "  \"n_edges_downweighted_by_type\": {\"EdgeMono\": " << cMono.nDown << ", \"EdgeMonoRight\": "
      << cMonoRight.nDown << ", \"EdgeStereo\": " << cStereo.nDown << ", \"EdgeInertial\": "
      << cInertial.nDown << ", \"EdgeLogDepth\": " << cDepth.nDown << "},\n";
    m << "  \"n_edges_skipped_nonfinite\": " << (cMono.nSkipped + cMonoRight.nSkipped + cStereo.nSkipped
      + cInertial.nSkipped + cGyroRW.nSkipped + cAccRW.nSkipped + cDepth.nSkipped + cChain.nSkipped
      + cMerge.nSkipped + cPrior.nSkipped + cUnary.nSkipped + cDetached.nSkipped) << ",\n";
    m << "  \"depth_factor_on\": " << JsonBool(s.depthFactor) << ",\n";
    m << "  \"depth_info_scale\": " << JsonNum(s.infoScale) << ",\n";
    m << "  \"n_depth_edges_gated\": " << nDepthGated << ",\n";
    m << "  \"threads_finished\": " << JsonBool(bThreadsFinished) << ",\n";
    if(bGauge4)
        m << "  \"diagnostic_gauge4\": {\"ok\": " << JsonBool(res4.ok) << ", \"files\": \"cov_raw_gauge4.npy cov_sandwich_gauge4.npy\", "
          << "\"gauge\": \"position and yaw about world z of the first keyframe\", \"cond_pose_info\": "
          << JsonNum(res4.condInfo) << ", \"ms\": " << JsonNum(msGauge4) << "},\n";
    m << "  \"checks\": {\"spd_dense\": " << JsonBool(res.spdDense) << ", \"spd_info\": " << JsonBool(res.spdInfo)
      << ", \"spd_sandwich\": " << JsonBool(res.spdSandwich)
      << ", \"min_pivot_dense\": " << JsonNum(res.minPivotDense)
      << ", \"min_pivot_info\": " << JsonNum(res.minPivotInfo)
      << ", \"asym_info\": " << JsonNum(res.asymInfo) << ", \"asym_raw\": " << JsonNum(res.asymRaw)
      << ", \"asym_sandwich\": " << JsonNum(res.asymSandwich)
      << ", \"eig_min_info\": " << JsonNum(res.eigMinInfo) << ", \"eig_max_info\": " << JsonNum(res.eigMaxInfo)
      << ", \"cond_pose_info\": " << JsonNum(res.condInfo)
      << ", \"note\": \"eigenvalues, condition number, asymmetry and the sandwich Cholesky are taken without "
         "the six gauge rows of the first keyframe, whose information is 1/gauge_sigma^2\"},\n";
    m << "  \"timing_ms\": {\"wait_threads\": " << JsonNum(waitMs) << ", \"build\": " << JsonNum(msBuild)
      << ", \"points\": " << JsonNum(res.msLocal) << ", \"dense\": " << JsonNum(res.msDense)
      << ", \"pose_inverse\": " << JsonNum(res.msInfo) << ", \"sandwich\": " << JsonNum(res.msSandwich)
      << ", \"checks\": " << JsonNum(res.msEig) << ", \"write\": " << JsonNum(msWrite) << "}\n";
    m << "}\n";
    {
        std::ofstream f((dir + "/meta.json").c_str());
        f << m.str();
        bWritten = bWritten && (bool)f;
    }

    std::cout << "[UCSLAM] marginals: tree=" << s.tree << " n_kf=" << N << " depth_edges=" << cDepth.n
              << " points=" << nPointsUsed << " ok=" << (res.ok && bWritten ? 1 : 0)
              << " spd_info=" << res.spdInfo << " spd_sandwich=" << res.spdSandwich
              << " cond=" << res.condInfo << " wall_ms=" << wallMs << " wait_ms=" << waitMs
              << " dir=" << dir << (res.ok ? "" : " error=" + res.error) << std::endl;
    return res.ok && bWritten;
}

} // namespace

bool DumpMarginalsAtShutdown(Atlas* pAtlas, LocalMapping* pLocalMapper, LoopClosing* pLoopCloser,
                             const std::string& outDir)
{
    const Clock::time_point t = Clock::now();
    const bool bThreads = WaitForThreads(pLocalMapper, pLoopCloser, kMarginalsThreadWaitS);
    const double waitMs = MsSince(t);
    try
    {
        return DumpImpl(pAtlas, outDir, bThreads, waitMs);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[UCSLAM] marginals: failed, " << ex.what() << std::endl;
        return false;
    }
}

} // namespace ORB_SLAM3
