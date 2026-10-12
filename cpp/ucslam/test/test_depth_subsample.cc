/**
* This file is part of UC-SLAM, a fork of ORB-SLAM3.
*
* Standalone test of attempt 08, a faster inertial local BA
* (docs/reports/2026-09-28-accuracy-improvement/logs/design8/judge/judge.txt Sec. 3).
*
*   U1. UcslamSubsampleBudget and UcslamSubsampleKeyFrame: K = min(n, max(20, ceil(q n))), exactly K kept,
*       the kept weighted information equals the candidate information of the keyframe (1e-12), the grid
*       quota by largest remainder with the lowest keys kept inside each cell, the exemptions of
*       UcslamSubsampleEligible, and a result that does not depend on the order of the candidates.
*   U2. UcslamWeightDepthEdge on a real EdgeLogDepth: over residuals of 0 to 5 sigma the Huber cost of a kept
*       edge is c_k times the cost of the edge without weight, the IRLS weight is the same, and the gate
*       decides the same.
*   U3. Known geometry (points and poses fixed): Var(a_k) with the subsample equals Var(a_k) with every
*       depth edge (1e-9).
*   U4. Free points (marginalized, seen through EdgeMono): the mean over 32 seeds of the mean over the
*       keyframes of Var_sub(a_k) / Var_full(a_k) is at least 1 - 1e-3. The subsample must not claim more
*       information than the full set.
*       Also K = 21 at n = 145 with the float settings 0.008f and 0.05f.
*   U5. UcslamMarginalDiag against the full inverse of the same LDLT, on random SPD matrices of size 400
*       and on the dense Schur complement of a rig with free poses (1e-12). The rig's lower fill against its
*       full fill gets 1e-12 / rcond, since last-bit differences of S scale with the conditioning.
*   U6. UcslamDensePoseSchur with lowerOnly: the same lower triangle (1e-12 of max|S|) and the same LDLT
*       diagonal (1e-12 of max|D|) as the full fill.
*
* Exit code 0 when every check passes, 1 otherwise. No gtest.
*
* Copyright (C) 2026 The UC-SLAM authors.
* Distributed under the GNU General Public License v3 or later, like ORB-SLAM3. See LICENSE.
*/

#include "UcslamTypes.h"
#include "G2oTypes.h"
#include "CameraModels/Pinhole.h"

#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>

using namespace ORB_SLAM3;

namespace
{

int gFailures = 0;

void Check(bool ok, const char* what)
{
    if(!ok)
    {
        std::cout << "FAIL: " << what << std::endl;
        gFailures++;
    }
}

// One PASS or FAIL line per part.
void Report(const char* part, int nFailuresBefore)
{
    std::cout << (gFailures == nFailuresBefore ? "PASS " : "FAIL ") << part << std::endl;
}

const double kS = 0.008;            // UCSLAM.NeffOverN of the EuRoC configs
const double kEps = 0.05;           // eps of the draft

// The EuRoC camera and IMU extrinsics of configs/orbslam3, as in test_ucslam_types.cc.
struct Rig
{
    GeometricCamera* cam;
    int width, height;
    Eigen::Matrix3d Rbc;
    Eigen::Vector3d tbc;
};

ImuCamPose MakePose(const Rig& rig, const Eigen::Matrix3d& Rwb, const Eigen::Vector3d& twb)
{
    const Eigen::Matrix3d Rcb = rig.Rbc.transpose();
    const Eigen::Vector3d tcb = -Rcb * rig.tbc;
    const Eigen::Matrix3d Rcw = Rcb * Rwb.transpose();
    const Eigen::Vector3d tcw = Rcb * (-Rwb.transpose() * twb) + tcb;
    ImuCamPose pose;
    pose.its = 0;
    std::vector<Eigen::Matrix3d> vRcw(1, Rcw), vRbc(1, rig.Rbc);
    std::vector<Eigen::Vector3d> vtcw(1, tcw), vtbc(1, rig.tbc);
    pose.SetParam(vRcw, vtcw, vRbc, vtbc, 0.0);
    pose.pCamera.assign(1, rig.cam);
    return pose;
}

// ------------------------------------------------------------------------------------------ U1 --

int CellOf(double x, double lo, double hi)
{
    const int k = (int)std::floor((x - lo) / (hi - lo) * kUcslamSubsampleGrid);
    return std::max(0, std::min(kUcslamSubsampleGrid - 1, k));
}

std::vector<UcslamDepthCand> RandomCands(std::mt19937& rng, int n, unsigned long kfId, double s, bool bClustered,
                                         double W, double H)
{
    std::uniform_real_distribution<double> uu(0.0, W), vv(0.0, H), sig(0.05, 0.3), rr(-1.0, 1.0);
    std::uniform_int_distribution<unsigned long> id(0, 10000000);
    std::set<unsigned long> used;
    std::vector<UcslamDepthCand> c(n);
    for(int i = 0; i < n; i++)
    {
        unsigned long p = id(rng);
        while(!used.insert(p).second)
            p = id(rng);
        c[i].pointId = p;
        // Clustered: most points in the upper left part of the image, some cells empty.
        c[i].u = bClustered ? 0.3 * uu(rng) : uu(rng);
        c[i].v = bClustered ? 0.4 * vv(rng) : vv(rng);
        const double sg = sig(rng);
        c[i].w = s / (sg * sg);
        c[i].r = 0.1 * rr(rng);
        c[i].key = UcslamDepthCandKey(p, kfId);
    }
    return c;
}

void TestU1()
{
    const int before = gFailures;
    const double W = 752.0, H = 480.0;
    std::mt19937 rng(20261001u);
    const int ns[] = {0, 1, 5, 19, 20, 21, 40, 144, 145, 146, 150, 187, 400, 1000};
    const double epss[] = {0.02, 0.05, 0.10, 0.20};
    const double ss[] = {0.008, 0.022};
    double maxInfoDev = 0.0;
    int nInst = 0, nKeptTotal = 0, nCandTotal = 0;
    for(size_t a = 0; a < sizeof(ns) / sizeof(ns[0]); a++)
    for(size_t b = 0; b < sizeof(epss) / sizeof(epss[0]); b++)
    for(size_t d = 0; d < sizeof(ss) / sizeof(ss[0]); d++)
    for(int clustered = 0; clustered < 2; clustered++)
    {
        const int n = ns[a];
        const double eps = epss[b], s = ss[d];
        const unsigned long kfId = 1 + (unsigned long)nInst;
        const std::vector<UcslamDepthCand> cands = RandomCands(rng, n, kfId, s, clustered != 0, W, H);
        nInst++;

        // K from the draft, computed here.
        const int K = UcslamSubsampleBudget(n, s, eps);
        const int Kref = n == 0 ? 0 : std::min(n, std::max(20, (int)std::ceil(s / (s + eps) * n)));
        Check(K == Kref, "U1 K = min(n, max(20, ceil(q n)))");
        Check(K <= n && K >= std::min(n, 20) && K >= s / (s + eps) * n - 1e-9, "U1 K within its bounds");

        std::vector<char> keep;
        const double c = UcslamSubsampleKeyFrame(cands, K, 0.0, W, 0.0, H, keep);
        Check((int)keep.size() == n, "U1 one keep flag per candidate");
        int nKept = 0;
        double wAll = 0.0, wKept = 0.0;
        for(int i = 0; i < n; i++)
        {
            wAll += cands[i].w;
            if(keep[i])
            {
                nKept++;
                wKept += cands[i].w;
            }
        }
        nKeptTotal += nKept;
        nCandTotal += n;
        Check(nKept == K, "U1 exactly K candidates kept");
        if(n > 0)
            maxInfoDev = std::max(maxInfoDev, std::fabs(c * wKept - wAll) / wAll);
        if(K >= n)
            Check(c == 1.0, "U1 c = 1 when every candidate is kept");
        else
            Check(c > 1.0, "U1 c > 1 when candidates are left out");

        // Grid: cell g keeps floor(K n_g / n) or one more, the cells with one more have the largest
        // remainders, and inside a cell every kept key is below every dropped key.
        if(n > 0 && K < n)
        {
            const int G = kUcslamSubsampleGrid;
            std::vector<int> ng(G * G, 0), kg(G * G, 0);
            std::vector<uint64_t> maxKept(G * G, 0), minDropped(G * G, ~(uint64_t)0);
            for(int i = 0; i < n; i++)
            {
                const int g = CellOf(cands[i].v, 0.0, H) * G + CellOf(cands[i].u, 0.0, W);
                ng[g]++;
                if(keep[i])
                {
                    kg[g]++;
                    maxKept[g] = std::max(maxKept[g], cands[i].key);
                }
                else
                    minDropped[g] = std::min(minDropped[g], cands[i].key);
            }
            int nPlus = 0, sumFloor = 0;
            long long minRemPlus = (long long)n, maxRemRest = -1;
            for(int g = 0; g < G * G; g++)
            {
                if(ng[g] == 0)
                    continue;
                const long long p = (long long)K * ng[g];
                const int fl = (int)(p / n);
                const long long rem = p % n;
                sumFloor += fl;
                Check(kg[g] == fl || kg[g] == fl + 1, "U1 cell quota is floor(K n_g / n) or one more");
                if(kg[g] == fl + 1)
                {
                    nPlus++;
                    minRemPlus = std::min(minRemPlus, rem);
                }
                else
                    maxRemRest = std::max(maxRemRest, rem);
                if(kg[g] > 0 && kg[g] < ng[g])
                    Check(maxKept[g] < minDropped[g], "U1 lowest keys kept inside a cell");
            }
            Check(nPlus == K - sumFloor, "U1 the remainder goes one edge per cell");
            Check(nPlus == 0 || minRemPlus >= maxRemRest, "U1 largest remainders get the extra edges");
        }

        // The same candidates in another order give the same kept set and the same c, bit for bit.
        std::vector<UcslamDepthCand> shuffled = cands;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        std::vector<char> keep2;
        const double c2 = UcslamSubsampleKeyFrame(shuffled, K, 0.0, W, 0.0, H, keep2);
        std::set<unsigned long> k1, k2;
        for(int i = 0; i < n; i++)
        {
            if(keep[i])
                k1.insert(cands[i].pointId);
            if(keep2[i])
                k2.insert(shuffled[i].pointId);
        }
        Check(k1 == k2 && c2 == c, "U1 the result does not depend on the order of the candidates");
    }
    Check(maxInfoDev <= 1e-12, "U1 kept weighted information equals the candidate information (1e-12)");

    // Exemptions: never a fixed keyframe, never a point seen by 1 or 2 keyframes.
    Check(!UcslamSubsampleEligible(true, 1) && !UcslamSubsampleEligible(true, 3) && !UcslamSubsampleEligible(true, 50),
          "U1 an edge of a fixed keyframe is never a candidate");
    Check(!UcslamSubsampleEligible(false, 0) && !UcslamSubsampleEligible(false, 1) && !UcslamSubsampleEligible(false, 2),
          "U1 points seen by fewer than 3 keyframes are exempt");
    Check(UcslamSubsampleEligible(false, 3) && UcslamSubsampleEligible(false, 12),
          "U1 points seen by 3 or more keyframes of a free keyframe are candidates");
    Check(UcslamSubsampleBudget(100, kS, 0.0) == 100 && UcslamSubsampleBudget(0, kS, kEps) == 0 &&
          UcslamSubsampleBudget(7, kS, kEps) == 7, "U1 budget corner cases (eps 0, n 0, n below 20)");
    // The local BA takes s and eps from float settings. (double)0.008f and (double)0.05f lie slightly above the
    // decimals, so q n passes 20 at n = 145 and K is 21 (review B-2, Python gives 21 too, 20 with the decimals).
    Check(UcslamSubsampleBudget(145, (double)0.008f, (double)0.05f) == 21,
          "U1 K = 21 at n = 145 with the float settings s = 0.008f, eps = 0.05f");

    // The mixer is splitmix64 (reference values from the Python code of design8 b_design).
    Check(UcslamSplitMix64(0) == 0xE220A8397B1DCDAFULL, "U1 splitmix64(0)");
    Check(UcslamDepthCandKey(12345, 678) == 0xD74FAAD6B1575071ULL, "U1 key of point 12345 in keyframe 678");

    std::printf("U1 instances=%d candidates=%d kept=%d max_rel_info_dev=%.3e\n", nInst, nCandTotal, nKeptTotal, maxInfoDev);
    Report("U1 selection, weight, K bounds, exemptions, determinism", before);
}

// ------------------------------------------------------------------------------------------ U2 --

void TestU2(const Rig& rig)
{
    const int before = gFailures;
    {
        const double w = kS / (0.1 * 0.1);
        const UcslamDepthWeight d1 = UcslamWeightDepthEdge(w, kS, 1.0);
        Check(d1.info == w && d1.delta == kUcslamHuberDepth * std::sqrt(kS) && d1.chi2Gate == kUcslamChi2Depth * kS,
              "U2 c = 1 gives the information, delta and gate without subsample");
    }

    // One point 3 m in front of an identity body pose, a_k = 0.
    const ImuCamPose pose = MakePose(rig, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    VertexPose VP;
    VP.setId(1);
    VP.setEstimate(pose);
    const Eigen::Vector3d Xc(0.2, -0.1, 3.0);
    g2o::VertexSBAPointXYZ VX;
    VX.setId(0);
    VX.setEstimate(pose.Rcw[0].transpose() * (Xc - pose.tcw[0]));
    VertexLogScale VA(0.0);
    VA.setId(2);
    const double logz = std::log(Xc(2));

    const double cs[] = {1.0, 2.5, 7.25, 20.0};
    const double sigmas[] = {0.08, 0.2};
    const double ss[] = {0.008, 0.022};
    double maxRho = 0.0, maxWeight = 0.0;
    int nSteps = 0, nGateBad = 0;
    for(size_t a = 0; a < 4; a++)
    for(size_t b = 0; b < 2; b++)
    for(size_t d = 0; d < 2; d++)
    {
        const double c = cs[a], sigma = sigmas[b], s = ss[d];
        const double w = s / (sigma * sigma);
        const UcslamDepthWeight d0 = UcslamWeightDepthEdge(w, s, 1.0), dc = UcslamWeightDepthEdge(w, s, c);
        g2o::RobustKernelHuber k0, kc;
        k0.setDelta(d0.delta);
        kc.setDelta(dc.delta);
        for(int i = -1; i < 1000; i++)
        {
            // t from 0 to 5 in steps of 0.005, off the switch points, both signs.
            const double t = i < 0 ? 0.0 : 0.0025 + 0.005 * i;
            const double sign = (i % 2 == 0) ? 1.0 : -1.0;
            EdgeLogDepth e0(0), ec(0);
            EdgeLogDepth* es[2] = {&e0, &ec};
            const double infos[2] = {d0.info, dc.info};
            for(int m = 0; m < 2; m++)
            {
                es[m]->setVertex(0, &VX);
                es[m]->setVertex(1, &VP);
                es[m]->setVertex(2, &VA);
                es[m]->setMeasurement(logz - sign * t * sigma);    // r = sign t sigma
                Eigen::Matrix<double,1,1> info;
                info(0,0) = infos[m];
                es[m]->setInformation(info);
                es[m]->computeError();
            }
            Eigen::Vector3d r0, rc;
            k0.robustify(e0.chi2(), r0);
            kc.robustify(ec.chi2(), rc);
            const double ref = c * r0[0];
            maxRho = std::max(maxRho, std::fabs(rc[0] - ref) / std::max(std::fabs(ref), 1e-300));
            if(ref == 0.0)
                maxRho = std::max(maxRho, std::fabs(rc[0]));
            maxWeight = std::max(maxWeight, std::fabs(rc[1] - r0[1]));
            const bool g0 = e0.chi2() > d0.chi2Gate, gc = ec.chi2() > dc.chi2Gate, gt = t > std::sqrt(kUcslamChi2Depth);
            if(g0 != gc || gc != gt)
                nGateBad++;
            nSteps++;
        }
    }
    std::printf("U2 steps=%d max_rel_err rho_kept vs c rho=%.3e max_abs_err IRLS weight=%.3e gate_mismatches=%d\n",
                nSteps, maxRho, maxWeight, nGateBad);
    // The vendored g2o stores the Huber delta^2 as float (Thirdparty/g2o/g2o/core/robust_kernel_impl.h:84, "float dsqr"),
    // so the outlier branch rho = 2 sqrt(e) delta - dsqr carries float rounding of delta^2 (about 6e-8 relative). The
    // scaling is exact in double, so this check uses float precision. The IRLS weight and gate checks stay exact.
    Check(maxRho <= 1e-6, "U2 Huber cost of a kept edge is c_k times the cost without weight (float delta^2 in g2o)");
    Check(maxWeight <= 1e-12, "U2 IRLS weight of a kept edge equals the weight without subsample");
    Check(nGateBad == 0, "U2 the gate of a kept edge acts at |r / sigma| = sqrt(3.841) like the edge without weight");
    Report("U2 kernel and gate scaling over 0 to 5 sigma", before);
}

// ----------------------------------------------------------------------------------- U3 to U6 --

// One observation of a window.
struct Obs
{
    int k, j;               // keyframe, point
    double u, v;            // measured keypoint [px]
    double logz;            // measured log z_hat (a_k = 0 in the scene)
    double sigma;           // sigma of the network depth
};

struct Scene
{
    std::vector<ImuCamPose> poses;
    std::vector<Eigen::Vector3d> points;
    std::vector<int> nObs;  // keyframes that see the point
    std::vector<Obs> obs;
};

// nKF keyframes 0.15 m apart looking along body z. Each point is seen by m consecutive keyframes,
// m = 1 + Geometric(0.35), until every keyframe has nPerKF observations. The depth residuals stay
// within 1.5 sigma, so every depth edge passes the gate and sits in the quadratic Huber region.
Scene MakeScene(std::mt19937& rng, const Rig& rig, int nKF, int nPerKF)
{
    Scene sc;
    std::uniform_real_distribution<double> unit(-1.0, 1.0), uu(10.0, rig.width - 10.0), vv(10.0, rig.height - 10.0),
        zz(1.5, 6.0), sg(0.08, 0.20);
    std::normal_distribution<double> npx(0.0, 0.5), nz(0.0, 0.3);
    std::geometric_distribution<int> geo(0.35);
    for(int k = 0; k < nKF; k++)
    {
        const Eigen::Matrix3d Rwb = ExpSO3(Eigen::Vector3d(0.02 * unit(rng), 0.02 * unit(rng), 0.02 * unit(rng)));
        const Eigen::Vector3d twb(0.15 * k, 0.02 * unit(rng), 0.02 * unit(rng));
        sc.poses.push_back(MakePose(rig, Rwb, twb));
    }
    std::vector<int> load(nKF, 0);
    for(int guard = 0; guard < 1000000 && *std::min_element(load.begin(), load.end()) < nPerKF; guard++)
    {
        const int k0 = (int)(std::min_element(load.begin(), load.end()) - load.begin());
        const int m = std::min(1 + geo(rng), nKF);
        const int first = std::max(0, std::min(k0, nKF - m));
        const cv::Point2f p((float)uu(rng), (float)vv(rng));
        const Eigen::Vector3f ray = rig.cam->unprojectEig(p);
        const Eigen::Vector3d Xc = ray.cast<double>() / (double)ray(2) * zz(rng);
        const ImuCamPose& P0 = sc.poses[k0];
        const Eigen::Vector3d Xw = P0.Rcw[0].transpose() * (Xc - P0.tcw[0]);
        bool ok = true;
        for(int k = first; k < first + m && ok; k++)
        {
            const ImuCamPose& P = sc.poses[k];
            const Eigen::Vector2d q = P.Project(Xw, 0);
            ok = (P.Rcw[0] * Xw + P.tcw[0])(2) > 0.5 && q(0) >= 1.0 && q(0) < rig.width - 1.0 &&
                 q(1) >= 1.0 && q(1) < rig.height - 1.0;
        }
        if(!ok)
            continue;
        const int j = (int)sc.points.size();
        sc.points.push_back(Xw);
        sc.nObs.push_back(m);
        for(int k = first; k < first + m; k++)
        {
            const ImuCamPose& P = sc.poses[k];
            const Eigen::Vector2d q = P.Project(Xw, 0);
            Obs o;
            o.k = k;
            o.j = j;
            o.u = q(0) + npx(rng);
            o.v = q(1) + npx(rng);
            o.sigma = sg(rng);
            o.logz = std::log((P.Rcw[0] * Xw + P.tcw[0])(2)) + o.sigma * std::max(-1.5, std::min(1.5, nz(rng)));
            sc.obs.push_back(o);
            load[k]++;
        }
    }
    return sc;
}

// The weight the inertial local BA gives each depth edge of a window whose keyframes are all free: c_k for
// a kept candidate, 0 for a candidate left out, 1 for an exempt edge.
std::vector<double> SubsampleWeights(const Scene& sc, const Rig& rig, double s, double eps, int& nDropped)
{
    std::vector<double> weight(sc.obs.size(), 1.0);
    nDropped = 0;
    for(int k = 0; k < (int)sc.poses.size(); k++)
    {
        std::vector<UcslamDepthCand> cands;
        std::vector<int> idx;
        for(size_t i = 0; i < sc.obs.size(); i++)
        {
            const Obs& o = sc.obs[i];
            if(o.k != k || !UcslamSubsampleEligible(false, sc.nObs[o.j]))
                continue;
            UcslamDepthCand c;
            c.pointId = 1000 + (unsigned long)o.j;
            c.u = o.u;
            c.v = o.v;
            c.w = s / (o.sigma * o.sigma);
            c.r = 0.0;
            c.key = UcslamDepthCandKey(c.pointId, 1 + (unsigned long)k);
            cands.push_back(c);
            idx.push_back((int)i);
        }
        std::vector<char> keep;
        const int K = UcslamSubsampleBudget((int)cands.size(), s, eps);
        const double c = UcslamSubsampleKeyFrame(cands, K, 0.0, rig.width, 0.0, rig.height, keep);
        for(size_t m = 0; m < idx.size(); m++)
        {
            weight[idx[m]] = keep[m] ? c : 0.0;
            if(!keep[m])
                nDropped++;
        }
    }
    return weight;
}

// The g2o graph of a window, built as the local BA builds it, and its Hessian at the scene's estimate.
struct Window
{
    g2o::SparseOptimizer opt;
    UcslamBlockSolverX* solver;
    std::vector<VertexLogScale*> as;
    Window(): solver(NULL) {}
};

// weight[i] scales the depth edge of observation i (0 leaves it out). bFreePoints: points marginalized
// and seen through EdgeMono, else fixed. bFreePoses: every pose but the first is free, else all fixed.
// Every a_k is free, with a prior on a_0 and a random walk between neighbours.
void BuildWindow(Window& w, const Scene& sc, const std::vector<double>& weight, double s, bool bFreePoints, bool bFreePoses)
{
    g2o::BlockSolverX::LinearSolverType* linear = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    w.solver = new UcslamBlockSolverX(linear);
    w.opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(w.solver));
    const int nKF = (int)sc.poses.size();
    std::vector<VertexPose*> vp;
    for(int k = 0; k < nKF; k++)
    {
        VertexPose* v = new VertexPose();
        v->setEstimate(sc.poses[k]);
        v->setId(k);
        v->setFixed(!bFreePoses || k == 0);
        w.opt.addVertex(v);
        vp.push_back(v);
    }
    for(int k = 0; k < nKF; k++)
    {
        VertexLogScale* v = new VertexLogScale(0.0);
        v->setId(nKF + k);
        w.opt.addVertex(v);
        w.as.push_back(v);
    }
    Eigen::Matrix<double,1,1> info;
    {
        EdgeLogScalePrior* p = new EdgeLogScalePrior(0.0);
        p->setVertex(0, w.as[0]);
        info(0,0) = 1.0 / (0.05 * 0.05);
        p->setInformation(info);
        w.opt.addEdge(p);
    }
    for(int k = 0; k + 1 < nKF; k++)
    {
        EdgeLogScaleRW* e = new EdgeLogScaleRW();
        e->setVertex(0, w.as[k]);
        e->setVertex(1, w.as[k + 1]);
        info(0,0) = 1.0 / (0.05 * 0.05);
        e->setInformation(info);
        w.opt.addEdge(e);
    }
    std::vector<g2o::VertexSBAPointXYZ*> pts;
    for(size_t j = 0; j < sc.points.size(); j++)
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId(2 * nKF + (int)j);
        v->setEstimate(sc.points[j]);
        if(bFreePoints)
            v->setMarginalized(true);
        else
            v->setFixed(true);
        w.opt.addVertex(v);
        pts.push_back(v);
    }
    for(size_t i = 0; i < sc.obs.size(); i++)
    {
        const Obs& o = sc.obs[i];
        if(bFreePoints)
        {
            EdgeMono* em = new EdgeMono(0);
            em->setVertex(0, pts[o.j]);
            em->setVertex(1, vp[o.k]);
            em->setMeasurement(Eigen::Vector2d(o.u, o.v));
            em->setInformation(Eigen::Matrix2d::Identity() * (1.0 / 1.44));
            w.opt.addEdge(em);
        }
        if(!(weight[i] > 0.0))
            continue;
        const UcslamDepthWeight dw = UcslamWeightDepthEdge(s / (o.sigma * o.sigma), s, weight[i]);
        EdgeLogDepth* ed = new EdgeLogDepth(0);
        ed->setVertex(0, pts[o.j]);
        ed->setVertex(1, vp[o.k]);
        ed->setVertex(2, w.as[o.k]);
        ed->setMeasurement(o.logz);
        info(0,0) = dw.info;
        ed->setInformation(info);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(dw.delta);
        ed->setRobustKernel(rk);
        w.opt.addEdge(ed);
    }
    // The Hessian at this estimate, as ComputeMarginals builds it after the last iteration.
    w.opt.initializeOptimization();
    w.opt.computeActiveErrors();
    w.opt.solver()->init();
    w.solver->buildStructure();
    w.solver->buildSystem();
}

std::vector<int> ScaleColumns(const Window& w)
{
    std::vector<int> cols;
    for(size_t k = 0; k < w.as.size(); k++)
        cols.push_back(w.as[k]->colInHessian());
    return cols;
}

// Var(a_k) of the window through the full path of ComputeMarginals: dense Schur, LDLT, inverse.
bool WindowVar(const Window& w, std::vector<double>& var)
{
    Eigen::MatrixXd S;
    if(!UcslamDensePoseSchur(*w.solver, S))
        return false;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if(ldlt.info() != Eigen::Success || !ldlt.isPositive())
        return false;
    const Eigen::MatrixXd C = ldlt.solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
    const std::vector<int> cols = ScaleColumns(w);
    var.resize(cols.size());
    for(size_t k = 0; k < cols.size(); k++)
        var[k] = C(cols[k], cols[k]);
    return true;
}

void TestU3(const Rig& rig)
{
    const int before = gFailures;
    double maxRel = 0.0;
    int nDroppedAll = 0, nEdges = 0;
    for(int seed = 0; seed < 4; seed++)
    {
        std::mt19937 rng(3000u + seed);
        const Scene sc = MakeScene(rng, rig, 6, 120);
        int nDropped = 0;
        const std::vector<double> wSub = SubsampleWeights(sc, rig, kS, kEps, nDropped);
        const std::vector<double> wAll(sc.obs.size(), 1.0);
        Window full, sub;
        BuildWindow(full, sc, wAll, kS, false, false);
        BuildWindow(sub, sc, wSub, kS, false, false);
        std::vector<double> vf, vs;
        Check(WindowVar(full, vf) && WindowVar(sub, vs), "U3 marginals of the known-geometry window");
        for(size_t k = 0; k < vf.size() && k < vs.size(); k++)
            maxRel = std::max(maxRel, std::fabs(vs[k] - vf[k]) / vf[k]);
        nDroppedAll += nDropped;
        nEdges += (int)sc.obs.size();
    }
    std::printf("U3 depth_edges=%d dropped=%d max_rel_diff Var(a_k)=%.3e\n", nEdges, nDroppedAll, maxRel);
    Check(nDroppedAll > 0, "U3 the subsample leaves edges out");
    Check(maxRel <= 1e-9, "U3 known geometry: Var(a_k) with the subsample equals Var(a_k) with every edge (1e-9)");
    Report("U3 Var(a_k) with known geometry", before);
}

void TestU4(const Rig& rig)
{
    const int before = gFailures;
    const int nSeeds = 32;
    double sum = 0.0, lo = 1e30, hi = -1e30;
    int nDroppedAll = 0, nEdges = 0, nOk = 0;
    for(int seed = 0; seed < nSeeds; seed++)
    {
        std::mt19937 rng(4000u + seed);
        const Scene sc = MakeScene(rng, rig, 6, 120);
        int nDropped = 0;
        const std::vector<double> wSub = SubsampleWeights(sc, rig, kS, kEps, nDropped);
        const std::vector<double> wAll(sc.obs.size(), 1.0);
        Window full, sub;
        BuildWindow(full, sc, wAll, kS, true, false);
        BuildWindow(sub, sc, wSub, kS, true, false);
        std::vector<double> vf, vs;
        if(!WindowVar(full, vf) || !WindowVar(sub, vs) || vf.size() != vs.size() || vf.empty())
        {
            Check(false, "U4 marginals of the free-point window");
            continue;
        }
        double r = 0.0;
        for(size_t k = 0; k < vf.size(); k++)
            r += vs[k] / vf[k];
        r /= (double)vf.size();
        sum += r;
        lo = std::min(lo, r);
        hi = std::max(hi, r);
        nOk++;
        nDroppedAll += nDropped;
        nEdges += (int)sc.obs.size();
    }
    const double mean = nOk > 0 ? sum / nOk : 0.0;
    std::printf("U4 seeds=%d depth_edges=%d dropped=%d mean Var_sub/Var_full=%.6f min=%.6f max=%.6f\n",
                nOk, nEdges, nDroppedAll, mean, lo, hi);
    Check(nOk == nSeeds, "U4 every seed gives marginals");
    Check(nDroppedAll > 0, "U4 the subsample leaves edges out");
    Check(mean >= 1.0 - 1e-3, "U4 free points: mean Var_sub / Var_full >= 1 - 1e-3");
    Report("U4 Var(a_k) with free points, 32 seeds", before);
}

// The dense Schur complement of a window with free poses (the first fixed), free a_k and marginalized
// points, full and lower-only, for U5 and U6.
void RigSchur(const Rig& rig, Eigen::MatrixXd& Sfull, Eigen::MatrixXd& Slow, std::vector<int>& cols)
{
    std::mt19937 rng(6000u);
    const Scene sc = MakeScene(rng, rig, 5, 60);
    const std::vector<double> wAll(sc.obs.size(), 1.0);
    Window w;
    BuildWindow(w, sc, wAll, kS, true, true);
    Check(UcslamDensePoseSchur(*w.solver, Sfull, false), "U6 full Schur fill of the rig");
    Check(UcslamDensePoseSchur(*w.solver, Slow, true), "U6 lower Schur fill of the rig");
    cols = ScaleColumns(w);
}

void TestU5(const Eigen::MatrixXd& Sfull, const Eigen::MatrixXd& Slow, const std::vector<int>& colsRig)
{
    const int before = gFailures;
    std::mt19937 rng(5000u);
    std::normal_distribution<double> nd(0.0, 1.0);
    const int n = 400;
    double maxRel = 0.0, maxRelLu = 0.0;
    bool bLowerSame = true;
    for(int trial = 0; trial < 3; trial++)
    {
        Eigen::MatrixXd B(n, n);
        for(int r = 0; r < n; r++)
            for(int c = 0; c < n; c++)
                B(r, c) = nd(rng);
        const Eigen::MatrixXd A = B * B.transpose() / (double)n + 0.1 * Eigen::MatrixXd::Identity(n, n);
        std::vector<int> cols;
        cols.push_back(0);
        cols.push_back(n - 1);
        std::uniform_int_distribution<int> pick(1, n - 2);
        std::set<int> used(cols.begin(), cols.end());
        while(cols.size() < 40)
        {
            const int c = pick(rng);
            if(used.insert(c).second)
                cols.push_back(c);
        }
        Eigen::VectorXd vd;
        Check(UcslamMarginalDiag(A, cols, vd), "U5 diagonal-only solve of a random SPD matrix");
        Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
        const Eigen::MatrixXd inv = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
        const Eigen::MatrixXd invLu = A.partialPivLu().inverse();
        for(size_t j = 0; j < cols.size(); j++)
        {
            const int c = cols[j];
            maxRel = std::max(maxRel, std::fabs(vd(c) - inv(c, c)) / std::fabs(inv(c, c)));
            maxRelLu = std::max(maxRelLu, std::fabs(vd(c) - invLu(c, c)) / std::fabs(invLu(c, c)));
        }
        // It reads the lower triangle only: the same result from the lower triangle alone.
        const Eigen::MatrixXd Alow = A.triangularView<Eigen::Lower>();
        Eigen::VectorXd vl;
        Check(UcslamMarginalDiag(Alow, cols, vl), "U5 diagonal-only solve from the lower triangle");
        for(size_t j = 0; j < cols.size(); j++)
            bLowerSame = bLowerSame && vl(cols[j]) == vd(cols[j]);
    }
    // The rig. (a) The same matrix Sfull: diagonal-only solve against the inverse of the same LDLT, which
    // isolates the solve (1e-12). (b) The production pairing, lower fill plus diagonal-only solve, against
    // full fill plus inverse. The lower fill may differ from the full fill in the last bits (noalias against a
    // temporary), and a relative change e of S moves Var(a_k) by about e / rcond. So (b) gets 1e-12 / rcond.
    double maxRelRigSame = 0.0, maxRelRig = 0.0, rcondRig = -1.0;
    {
        Eigen::LDLT<Eigen::MatrixXd> ldlt(Sfull);
        Check(ldlt.info() == Eigen::Success && ldlt.isPositive(), "U5 rig Schur complement is positive");
        rcondRig = ldlt.rcond();
        const Eigen::MatrixXd inv = ldlt.solve(Eigen::MatrixXd::Identity(Sfull.rows(), Sfull.cols()));
        Eigen::VectorXd vs, vd;
        Check(UcslamMarginalDiag(Sfull, colsRig, vs), "U5 diagonal-only solve of the full rig Schur complement");
        Check(UcslamMarginalDiag(Slow, colsRig, vd), "U5 diagonal-only solve of the lower rig Schur complement");
        for(size_t j = 0; j < colsRig.size(); j++)
        {
            const int c = colsRig[j];
            maxRelRigSame = std::max(maxRelRigSame, std::fabs(vs(c) - inv(c, c)) / std::fabs(inv(c, c)));
            maxRelRig = std::max(maxRelRig, std::fabs(vd(c) - inv(c, c)) / std::fabs(inv(c, c)));
        }
    }
    const double tolRig = 1e-12 / std::max(rcondRig, 1e-300);
    std::printf("U5 random SPD n=%d max_rel_diff vs LDLT inverse=%.3e vs LU inverse=%.3e, rig n=%d a_k=%d rcond=%.3e "
                "same matrix max_rel_diff=%.3e, lower fill max_rel_diff=%.3e (tol %.3e)\n",
                n, maxRel, maxRelLu, (int)Sfull.rows(), (int)colsRig.size(), rcondRig, maxRelRigSame, maxRelRig, tolRig);
    Check(maxRel <= 1e-12, "U5 diagonal-only solve equals the full inverse on random SPD matrices (1e-12)");
    Check(maxRelLu <= 1e-10, "U5 diagonal-only solve equals an LU inverse on random SPD matrices (1e-10)");
    Check(bLowerSame, "U5 the diagonal-only solve reads the lower triangle only");
    Check(rcondRig > 0.0, "U5 rig rcond is positive");
    Check(maxRelRigSame <= 1e-12, "U5 diagonal-only solve equals the inverse of the same rig matrix (1e-12)");
    Check(maxRelRig <= tolRig, "U5 diagonal-only path equals the full path on the rig (1e-12 / rcond)");
    Report("U5 diagonal-only marginal solve", before);
}

void TestU6(const Eigen::MatrixXd& Sfull, const Eigen::MatrixXd& Slow)
{
    const int before = gFailures;
    Check(Sfull.rows() == Slow.rows() && Sfull.rows() > 0, "U6 same size");
    const Eigen::MatrixXd dLow = Eigen::MatrixXd(Sfull.triangularView<Eigen::Lower>()) -
                                 Eigen::MatrixXd(Slow.triangularView<Eigen::Lower>());
    const double lowerErr = dLow.cwiseAbs().maxCoeff() / Sfull.cwiseAbs().maxCoeff();
    Eigen::LDLT<Eigen::MatrixXd> lf(Sfull), ll(Slow);
    Check(lf.info() == ll.info() && lf.isPositive() == ll.isPositive(), "U6 same LDLT status");
    const Eigen::VectorXd df = lf.vectorD(), dl = ll.vectorD();
    // Scale-aware (review): the largest difference relative to the largest D entry. With diagonal pivoting
    // |L| <= 1, so last-bit differences of S (noalias against a temporary) move each D entry by about eps
    // times max|D|, while a small late pivot can change by much more relative to itself.
    const double maxD = df.cwiseAbs().maxCoeff();
    double maxDiff = 0.0;
    for(int i = 0; i < df.size() && i < dl.size(); i++)
        maxDiff = std::max(maxDiff, std::fabs(dl(i) - df(i)));
    const double relD = maxDiff / std::max(maxD, 1e-300);
    const bool bSamePivots = lf.transpositionsP().indices() == ll.transpositionsP().indices();
    std::printf("U6 n=%d lower triangle max_rel_diff=%.3e LDLT diagonal max_abs_diff / max|D|=%.3e same_pivots=%d\n",
                (int)Sfull.rows(), lowerErr, relD, bSamePivots ? 1 : 0);
    Check(lowerErr <= 1e-12, "U6 lower fill equals the lower triangle of the full fill (1e-12 of max|S|)");
    Check(bSamePivots, "U6 same LDLT pivots");
    Check(relD <= 1e-12, "U6 lower fill gives the same LDLT diagonal as the full fill (1e-12 of max|D|)");
    Report("U6 one-triangle Schur fill", before);
}

} // namespace

int main()
{
    std::vector<float> eurocParams;
    eurocParams.push_back(458.654f); eurocParams.push_back(457.296f);
    eurocParams.push_back(367.215f); eurocParams.push_back(248.375f);
    Pinhole eurocCam(eurocParams);
    const double eurocTbc[16] = {
        0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
        0.0, 0.0, 0.0, 1.0};
    Rig rig;
    rig.cam = &eurocCam;
    rig.width = 752;
    rig.height = 480;
    {
        Eigen::Matrix3d R;
        for(int r = 0; r < 3; r++)
            for(int c = 0; c < 3; c++)
                R(r, c) = eurocTbc[4 * r + c];
        rig.Rbc = NormalizeRotation(R);
        rig.tbc = Eigen::Vector3d(eurocTbc[3], eurocTbc[7], eurocTbc[11]);
    }

    TestU1();
    TestU2(rig);
    TestU3(rig);
    TestU4(rig);
    Eigen::MatrixXd Sfull, Slow;
    std::vector<int> cols;
    RigSchur(rig, Sfull, Slow, cols);
    TestU5(Sfull, Slow, cols);
    TestU6(Sfull, Slow);

    if(gFailures == 0)
    {
        std::cout << "PASS test_depth_subsample" << std::endl;
        return 0;
    }
    std::cout << "FAIL test_depth_subsample: " << gFailures << " check(s) failed" << std::endl;
    return 1;
}
