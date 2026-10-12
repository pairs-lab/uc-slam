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

#ifndef UCSLAM_MARGINALS_H
#define UCSLAM_MARGINALS_H

#include <string>
#include <vector>

#include <Eigen/Core>

namespace ORB_SLAM3
{

class Atlas;
class LocalMapping;
class LoopClosing;

/*
 * Posterior covariance dump, docs/fork_design.md Sec. 4, format "ucslam-marginals-1".
 *
 * At shutdown, after the mapping threads have stopped and before the trajectory files are
 * saved, the full problem of the largest map is built as Optimizer::FullInertialBA builds it:
 * every keyframe with its pose and, when inertial, its velocity and bias vertices, every map
 * point with its reprojection edges, and the inertial and random walk edges. In the fork with
 * UCSLAM.UseDepthFactor on, the log-depth edges and the log-scale chain with its vertices are
 * added the way the local BA adds them. Nothing is optimized. The Gauss-Newton information is
 * evaluated at the estimate the system holds, with every Huber kernel's weight rho'(chi2) at
 * that estimate. Map points are eliminated one at a time by Schur complement. Velocities,
 * biases and log-scales are eliminated by a dense Schur complement. The pose block is inverted
 * with Eigen's LDLT. The first keyframe's pose carries a gauge prior of standard deviation
 * kGaugeSigma on all six tangent components.
 *
 * Files, in <working directory>/marginals/:
 *   keyframes.csv     kf_id,timestamp_s,px,py,pz,qx,qy,qz,qw, the world-from-body pose Twb
 *   cov_raw.npy       (6N, 6N) float64, H^-1 of the pose block
 *   cov_sandwich.npy  (6N, 6N) float64, H^-1 B H^-1 with B = sum_i J_i^T W_i r_i r_i^T W_i J_i
 *   meta.json         format, n_kf, gauge, includes_depth_edges, n_edges_by_type, wall_ms,
 *                     tree, notes, and the checks of the linear algebra
 * Block i belongs to row i of keyframes.csv. Its tangent is the VertexPose one,
 * [ur (3), ut (3)] with R_true = R_est Exp(ur) and t_true = t_est + R_est ut.
 *
 * The same source compiles in the fork and, through cpp/orbslam3_patches/0003, in the stock
 * tree. The fork compiles it with UCSLAM_FORK defined, which adds the depth factor.
 *
 * Diagnostic, off by default. With UCSLAM_MARGINALS_GAUGE4=1 in the environment the problem is
 * solved a second time with a gauge on the first keyframe's position and yaw about world z only,
 * and cov_raw_gauge4.npy and cov_sandwich_gauge4.npy are written next to the contract files.
 * With an IMU that is the exact gauge freedom, so the difference to the contract files shows
 * what holding the first keyframe's roll and pitch does to the other covariances.
 */

// Constants of the dump. Their values and reasons are in configs/default.yaml, block marginals.
extern const double kMarginalsGaugeSigma;     // marginals.gauge_sigma
extern const double kMarginalsPointRankRtol;  // marginals.point_rank_rtol
extern const double kMarginalsThreadWaitS;    // marginals.thread_wait_s

/*
 * The linear algebra of the dump, with no dependency on ORB-SLAM3 or g2o.
 *
 * A problem has three kinds of variable blocks. KEPT blocks are the ones whose joint marginal
 * covariance is returned, in the order they were added. DENSE blocks are eliminated by a dense
 * Schur complement. LOCAL blocks are eliminated one at a time. A factor may touch at most one
 * LOCAL block. KEPT and DENSE blocks are added before Freeze(), LOCAL blocks at any time.
 *
 * AddFactor takes a linearized residual r with Jacobians J_a and a weight matrix W, which is the
 * information already multiplied by the robust weight. It adds J_a^T W J_b to the information H
 * and g_a g_b^T to the meat B, with g_a = J_a^T W r.
 *
 * Solve eliminates every LOCAL block with a pseudo-inverse of its information block, which drops
 * eigenvalues at or below kMarginalsPointRankRtol times the largest. A map point seen once has
 * one such direction, and that direction carries no information on anything else. Then it
 * eliminates the DENSE blocks and returns, for the KEPT blocks x,
 *   covRaw      = S^-1,               S = H_xx - H_xy H_yy^-1 H_yx
 *   covSandwich = S^-1 A B A^T S^-1,  A = [I, -H_xy H_yy^-1]
 * which is the x block of H^-1 and of H^-1 B H^-1.
 */
class MarginalsProblem
{
public:
    enum Kind { KEPT = 0, DENSE = 1, LOCAL = 2 };

    struct Result
    {
        Eigen::MatrixXd covRaw;
        Eigen::MatrixXd covSandwich;
        bool ok;                     // both covariances were computed
        std::string error;           // why not, when ok is false
        bool spdDense;               // LDLT of H_yy after the LOCAL elimination has positive pivots
        bool spdInfo;                // LDLT of S has positive pivots
        bool spdSandwich;            // LLT of the sandwich without the leading gauge rows succeeds
        double minPivotDense;        // smallest LDLT pivot of H_yy
        double minPivotInfo;         // smallest LDLT pivot of S
        double asymInfo;             // max |S - S^T| / max |S| before symmetrization
        double asymRaw;              // the same for covRaw
        double asymSandwich;         // the same for covSandwich
        double eigMinInfo;           // extreme eigenvalues of S without the leading gauge rows
        double eigMaxInfo;
        double condInfo;             // eigMaxInfo / eigMinInfo
        int nLocal;                  // LOCAL blocks that had at least one factor
        int nLocalRankDeficient;     // of those, blocks with a dropped direction
        int nLocalEmpty;             // blocks with a zero information block
        int nDenseUntouched;         // DENSE blocks without any factor, left out
        double msLocal, msDense, msInfo, msSandwich, msEig;
        Result();
    };

    MarginalsProblem();

    int AddBlock(Kind kind, int dim);
    void Freeze();
    int NumKeptDims() const { return mnKept; }
    int NumDenseDims() const { return mnDense; }

    // blocks[a] with J[a] column major (D x dim of the block), W column major (D x D), r (D).
    void AddFactor(int nBlocks, const int* blocks, const double* const* J, int D, const double* W,
                   const double* r);

    // Information without a residual on one KEPT or DENSE block, for the gauge prior.
    void AddInformation(int block, const Eigen::MatrixXd& info);

    // gaugeDims leading KEPT rows are left out of the eigenvalue and sandwich checks.
    bool Solve(double rankRtol, int gaugeDims, Result& res);

private:
    struct Block { int kind, dim, offset, local; bool touched; };
    struct Cross { int zblock; Eigen::MatrixXd H, B; };   // H_zl and B_zl, dim(z) x dim(l)
    struct Local { int dim; Eigen::MatrixXd H, B; std::vector<Cross> cross; };

    Cross& GetCross(Local& l, int zblock);

    std::vector<Block> mvBlocks;
    std::vector<Local> mvLocal;
    bool mbFrozen;
    int mnKept, mnDense;
    Eigen::MatrixXd mH, mB;   // KEPT then DENSE, dense, both triangles
};

// UCSLAM.DumpMarginals read from a settings file with cv::FileStorage. 0 when the key or the file is absent.
int ReadDumpMarginalsFlag(const std::string& settingsFile);

/*
 * The shutdown hook. Waits up to kMarginalsThreadWaitS for local mapping, loop closing and a
 * running global BA to finish, then builds the problem of the largest map under its update
 * mutex, solves it and writes the four files into outDir, relative to the working directory.
 * Returns true when the covariance files were written. Prints one summary line on stdout.
 */
bool DumpMarginalsAtShutdown(Atlas* pAtlas, LocalMapping* pLocalMapper, LoopClosing* pLoopCloser,
                             const std::string& outDir = "marginals");

} // namespace ORB_SLAM3

#endif // UCSLAM_MARGINALS_H
