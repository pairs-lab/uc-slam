/**
* This file is part of UC-SLAM, a fork of ORB-SLAM3.
*
* Standalone test of the UC-SLAM optimizer types, docs/fork_design.md Sec. 2 and Sec. 6.
*
*   1. EdgeLogDepth (VertexPose, body-frame tangent) and EdgeLogDepthSE3 (VertexSE3Expmap):
*      analytic Jacobians against g2o's numeric linearizeOplus over one thousand random
*      configurations, half with the EuRoC pinhole camera and half with the TUM-VI fisheye,
*      both with their real intrinsics and IMU extrinsics. A second reference uses central
*      differences with a larger step than g2o's 1e-9, which isolates round-off.
*   2. EdgeLogScaleRW and EdgeLogScalePrior: analytic against numeric, and the residuals.
*   3. UcslamDensePoseSchur: the marginal variance of the a_k read from the block solver equals
*      the closed form on a bare chain, and equals a brute-force Schur complement on a graph
*      with marginalized points.
*
* Exit code 0 when every check passes, 1 otherwise. No gtest.
*
* Copyright (C) 2026 The UC-SLAM authors.
* Distributed under the GNU General Public License v3 or later, like ORB-SLAM3. See LICENSE.
*/

#include "UcslamTypes.h"
#include "G2oTypes.h"
#include "CameraModels/Pinhole.h"
#include "CameraModels/KannalaBrandt8.h"

#include "Thirdparty/g2o/g2o/core/jacobian_workspace.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

using namespace ORB_SLAM3;

namespace
{

struct Rig
{
    const char* name;
    GeometricCamera* cam;
    int width, height;
    Eigen::Matrix3d Rbc;
    Eigen::Vector3d tbc;
};

Eigen::Matrix4d ReadT(const double* v)
{
    Eigen::Matrix4d T;
    for(int r = 0; r < 4; r++)
        for(int c = 0; c < 4; c++)
            T(r, c) = v[4 * r + c];
    return T;
}

// Pixel coordinates inside the image whose ray is within 60 degrees of the optical axis.
bool SampleRay(std::mt19937& rng, const Rig& rig, Eigen::Vector3d& ray)
{
    std::uniform_real_distribution<double> uu(0.0, rig.width - 1.0), vv(0.0, rig.height - 1.0);
    for(int k = 0; k < 100; k++)
    {
        const cv::Point2f p((float)uu(rng), (float)vv(rng));
        const Eigen::Vector3f r = rig.cam->unprojectEig(p);
        if(!(r(2) > 0.f) || !std::isfinite(r(0)) || !std::isfinite(r(1)))
            continue;
        ray = r.cast<double>() / (double)r(2);
        if(std::sqrt(ray(0) * ray(0) + ray(1) * ray(1)) > std::tan(60.0 * M_PI / 180.0))
            continue;
        return true;
    }
    return false;
}

// Central differences of the one-dimensional error of an edge with respect to one vertex.
void CentralDifference(g2o::OptimizableGraph::Edge* e, g2o::OptimizableGraph::Vertex* v, int dim,
                       double h, double* out)
{
    std::vector<double> add(dim, 0.0);
    for(int d = 0; d < dim; d++)
    {
        add[d] = h;
        v->push();
        v->oplus(add.data());
        e->computeError();
        const double ep = e->errorData()[0];
        v->pop();
        v->push();
        add[d] = -h;
        v->oplus(add.data());
        e->computeError();
        const double em = e->errorData()[0];
        v->pop();
        add[d] = 0.0;
        out[d] = (ep - em) / (2.0 * h);
    }
}

double RowError(const Eigen::Matrix<double,1,EdgeLogDepthBase::kCols>& Ja,
                const Eigen::Matrix<double,1,EdgeLogDepthBase::kCols>& Jb)
{
    // Relative to the infinity norm of the analytic row, which is at least one since d r / d a = -1.
    return (Ja - Jb).cwiseAbs().maxCoeff() / std::max(1.0, Ja.cwiseAbs().maxCoeff());
}

int gFailures = 0;

void Check(bool ok, const char* what)
{
    if(!ok)
    {
        std::cout << "FAIL: " << what << std::endl;
        gFailures++;
    }
}

} // namespace

int main()
{
    // Real intrinsics and IMU extrinsics of the two YAMLs in configs/orbslam3.
    std::vector<float> eurocParams;
    eurocParams.push_back(458.654f); eurocParams.push_back(457.296f);
    eurocParams.push_back(367.215f); eurocParams.push_back(248.375f);
    Pinhole eurocCam(eurocParams);

    std::vector<float> tumviParams;
    tumviParams.push_back(190.978477f); tumviParams.push_back(190.973307f);
    tumviParams.push_back(254.931706f); tumviParams.push_back(256.897442f);
    tumviParams.push_back(0.003482389402f); tumviParams.push_back(0.000715034845f);
    tumviParams.push_back(-0.002053236141f); tumviParams.push_back(0.000202936736f);
    KannalaBrandt8 tumviCam(tumviParams);

    const double eurocTbc[16] = {
        0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
        0.0, 0.0, 0.0, 1.0};
    const double tumviTbc[16] = {
        -0.9995250378696743, 0.0075019185074052044, -0.02989013031643309, 0.045574835649698026,
        0.029615343885863205, -0.03439736061393144, -0.998969345370175, -0.071161801837997044,
        -0.008522328211654736, -0.9993800792498829, 0.03415885127385616, -0.044681254117144367,
        0.0, 0.0, 0.0, 1.0};

    Rig rigs[2];
    rigs[0].name = "euroc_pinhole"; rigs[0].cam = &eurocCam; rigs[0].width = 752; rigs[0].height = 480;
    rigs[1].name = "tumvi_kb8";     rigs[1].cam = &tumviCam; rigs[1].width = 512; rigs[1].height = 512;
    {
        Eigen::Matrix4d T = ReadT(eurocTbc);
        rigs[0].Rbc = NormalizeRotation(Eigen::Matrix3d(T.block<3,3>(0,0)));
        rigs[0].tbc = T.block<3,1>(0,3);
        T = ReadT(tumviTbc);
        rigs[1].Rbc = NormalizeRotation(Eigen::Matrix3d(T.block<3,3>(0,0)));
        rigs[1].tbc = T.block<3,1>(0,3);
    }

    std::mt19937 rng(20260924u);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> depth(0.5, 8.0);     // indoor depths, as on EuRoC and TUM-VI

    const int nTrials = 1000;
    const double tol = 1e-6;
    double maxG2o[2] = {0.0, 0.0}, maxCd[2] = {0.0, 0.0};        // per rig, EdgeLogDepth
    double maxG2oSE3[2] = {0.0, 0.0}, maxCdSE3[2] = {0.0, 0.0};  // per rig, EdgeLogDepthSE3
    double maxResidual = 0.0;
    int nDone = 0;

    for(int trial = 0; trial < nTrials; trial++)
    {
        const int ir = trial % 2;
        const Rig& rig = rigs[ir];

        // Random body pose, translation within 1.5 metres of the origin.
        const Eigen::Matrix3d Rwb = ExpSO3(Eigen::Vector3d(unit(rng), unit(rng), unit(rng)) * M_PI);
        const Eigen::Vector3d twb(1.5 * unit(rng), 1.5 * unit(rng), 1.5 * unit(rng));
        const Eigen::Matrix3d Rcb = rig.Rbc.transpose();
        const Eigen::Vector3d tcb = -Rcb * rig.tbc;
        const Eigen::Matrix3d Rbw = Rwb.transpose();
        const Eigen::Matrix3d Rcw = Rcb * Rbw;
        const Eigen::Vector3d tcw = Rcb * (-Rbw * twb) + tcb;

        Eigen::Vector3d ray;
        if(!SampleRay(rng, rig, ray))
            continue;
        const double z = depth(rng);
        const Eigen::Vector3d Xc = ray * z;
        const Eigen::Vector3d Xw = Rcw.transpose() * (Xc - tcw);

        const double a = unit(rng);                       // a_k in [-1, 1]
        const double logzhat = std::log(z) - a + 0.3 * unit(rng);
        const double sigma = 0.05 + 0.2 * std::fabs(unit(rng));

        // Body-frame pose vertex, as the inertial local BA builds it.
        ImuCamPose pose;
        pose.its = 0;
        {
            std::vector<Eigen::Matrix3d> vRcw(1, Rcw), vRbc(1, rig.Rbc);
            std::vector<Eigen::Vector3d> vtcw(1, tcw), vtbc(1, rig.tbc);
            pose.SetParam(vRcw, vtcw, vRbc, vtbc, 0.0);
            pose.pCamera.assign(1, rig.cam);
        }
        VertexPose VP;
        VP.setId(1);
        VP.setEstimate(pose);
        g2o::VertexSBAPointXYZ VX;
        VX.setId(0);
        VX.setEstimate(Xw);
        VertexLogScale VA(a);
        VA.setId(2);

        {
            EdgeLogDepth e(0);
            e.setVertex(0, &VX);
            e.setVertex(1, &VP);
            e.setVertex(2, &VA);
            e.setMeasurement(logzhat);
            Eigen::Matrix<double,1,1> info;
            info(0,0) = 1.0 / (sigma * sigma);
            e.setInformation(info);

            g2o::JacobianWorkspace jw;
            jw.updateSize(&e);
            jw.allocate();

            e.computeError();
            const double rExpected = std::log(z) - a - logzhat;
            maxResidual = std::max(maxResidual, std::fabs(e.errorData()[0] - rExpected));
            Check(e.isDepthPositive(), "EdgeLogDepth depth positive");

            e.linearizeOplus(jw);
            const Eigen::Matrix<double,1,EdgeLogDepthBase::kCols> Ja = e.GetJacobian();
            const Eigen::Matrix<double,1,EdgeLogDepthBase::kCols> Jn = e.GetJacobianNumeric();
            Eigen::Matrix<double,1,EdgeLogDepthBase::kCols> Jc;
            CentralDifference(&e, &VX, 3, 1e-6, Jc.data());
            CentralDifference(&e, &VP, 6, 1e-6, Jc.data() + 3);
            CentralDifference(&e, &VA, 1, 1e-6, Jc.data() + 9);
            maxG2o[ir] = std::max(maxG2o[ir], RowError(Ja, Jn));
            maxCd[ir] = std::max(maxCd[ir], RowError(Ja, Jc));
        }

        // Camera-frame pose vertex, as the visual local BA builds it.
        {
            g2o::VertexSE3Expmap VT;
            VT.setId(1);
            VT.setEstimate(g2o::SE3Quat(Eigen::Quaterniond(Rcw), tcw));

            EdgeLogDepthSE3 e;
            e.setVertex(0, &VX);
            e.setVertex(1, &VT);
            e.setVertex(2, &VA);
            e.setMeasurement(logzhat);
            Eigen::Matrix<double,1,1> info;
            info(0,0) = 1.0 / (sigma * sigma);
            e.setInformation(info);

            g2o::JacobianWorkspace jw;
            jw.updateSize(&e);
            jw.allocate();

            e.computeError();
            const double rExpected = std::log(z) - a - logzhat;
            maxResidual = std::max(maxResidual, std::fabs(e.errorData()[0] - rExpected));

            e.linearizeOplus(jw);
            const Eigen::Matrix<double,1,EdgeLogDepthBase::kCols> Ja = e.GetJacobian();
            const Eigen::Matrix<double,1,EdgeLogDepthBase::kCols> Jn = e.GetJacobianNumeric();
            Eigen::Matrix<double,1,EdgeLogDepthBase::kCols> Jc;
            CentralDifference(&e, &VX, 3, 1e-6, Jc.data());
            CentralDifference(&e, &VT, 6, 1e-6, Jc.data() + 3);
            CentralDifference(&e, &VA, 1, 1e-6, Jc.data() + 9);
            maxG2oSE3[ir] = std::max(maxG2oSE3[ir], RowError(Ja, Jn));
            maxCdSE3[ir] = std::max(maxCdSE3[ir], RowError(Ja, Jc));
        }
        nDone++;
    }

    std::printf("EdgeLogDepth      trials=%d residual_max_abs_err=%.3e\n", nDone, maxResidual);
    for(int ir = 0; ir < 2; ir++)
        std::printf("EdgeLogDepth      %-14s max_rel_err vs g2o numeric (h=1e-9): %.3e   vs central diff (h=1e-6): %.3e\n",
                    rigs[ir].name, maxG2o[ir], maxCd[ir]);
    for(int ir = 0; ir < 2; ir++)
        std::printf("EdgeLogDepthSE3   %-14s max_rel_err vs g2o numeric (h=1e-9): %.3e   vs central diff (h=1e-6): %.3e\n",
                    rigs[ir].name, maxG2oSE3[ir], maxCdSE3[ir]);
    Check(nDone == nTrials, "all trials sampled a valid ray");
    Check(maxResidual < 1e-12, "EdgeLogDepth residual equals log z - a - log zhat");
    // g2o differentiates with a step of 1e-9. Through SE3Quat::exp and the quaternion product of
    // VertexSE3Expmap that step leaves round-off of about 1e-6 on the pose columns for points a few
    // metres out, which the central differences with h = 1e-6 do not show (they agree to 1e-9). The
    // SE3 variant therefore gets twice the tolerance against g2o's numeric reference and the full
    // tolerance against the central differences. The design's edge, EdgeLogDepth, keeps 1e-6 for both.
    const double tolG2oSE3 = 2e-6;
    for(int ir = 0; ir < 2; ir++)
    {
        Check(maxG2o[ir] < tol, "EdgeLogDepth analytic Jacobian within 1e-6 of g2o numeric");
        Check(maxCd[ir] < tol, "EdgeLogDepth analytic Jacobian within 1e-6 of central differences");
        Check(maxG2oSE3[ir] < tolG2oSE3, "EdgeLogDepthSE3 analytic Jacobian within 2e-6 of g2o numeric");
        Check(maxCdSE3[ir] < tol, "EdgeLogDepthSE3 analytic Jacobian within 1e-6 of central differences");
    }

    // A point behind the camera contributes nothing and reports a non positive depth.
    {
        VertexLogScale VA(0.0);
        g2o::VertexSBAPointXYZ VX;
        VX.setEstimate(Eigen::Vector3d(0.0, 0.0, -3.0));
        g2o::VertexSE3Expmap VT;
        VT.setEstimate(g2o::SE3Quat());
        EdgeLogDepthSE3 e;
        e.setVertex(0, &VX);
        e.setVertex(1, &VT);
        e.setVertex(2, &VA);
        e.setMeasurement(0.0);
        g2o::JacobianWorkspace jw;
        jw.updateSize(&e);
        jw.allocate();
        e.computeError();
        e.linearizeOplus(jw);
        Check(!e.isDepthPositive(), "behind camera: depth not positive");
        Check(e.errorData()[0] == 0.0, "behind camera: zero error");
        Check(e.GetJacobian().cwiseAbs().maxCoeff() == 0.0, "behind camera: zero Jacobian");
    }

    // Chain and prior edges.
    {
        VertexLogScale V0(0.3), V1(-0.2);
        V0.setId(0);
        V1.setId(1);
        EdgeLogScaleRW e;
        e.setVertex(0, &V0);
        e.setVertex(1, &V1);
        Eigen::Matrix<double,1,1> info;
        info(0,0) = 1.0 / (0.02 * 0.02 * 0.5);
        e.setInformation(info);
        g2o::JacobianWorkspace jw;
        jw.updateSize(&e);
        jw.allocate();
        e.computeError();
        Check(std::fabs(e.errorData()[0] - (-0.2 - 0.3)) < 1e-15, "EdgeLogScaleRW residual a_next - a_prev");
        e.linearizeOplus(jw);
        const Eigen::Matrix<double,1,2> Ja = e.GetJacobian();
        const Eigen::Matrix<double,1,2> Jn = e.GetJacobianNumeric();
        std::printf("EdgeLogScaleRW    analytic [%g %g]  numeric [%g %g]  max_abs_err %.3e\n",
                    Ja(0,0), Ja(0,1), Jn(0,0), Jn(0,1), (Ja - Jn).cwiseAbs().maxCoeff());
        Check((Ja - Jn).cwiseAbs().maxCoeff() < tol, "EdgeLogScaleRW Jacobian");
        Check(Ja(0,0) == -1.0 && Ja(0,1) == 1.0, "EdgeLogScaleRW Jacobian is [-1, 1]");

        EdgeLogScalePrior p(0.1);
        p.setVertex(0, &V0);
        info(0,0) = 1.0 / 0.01;
        p.setInformation(info);
        g2o::JacobianWorkspace jwp;
        jwp.updateSize(&p);
        jwp.allocate();
        p.computeError();
        Check(std::fabs(p.errorData()[0] - (0.3 - 0.1)) < 1e-15, "EdgeLogScalePrior residual a - mean");
        p.linearizeOplus(jwp);
        const double ja = p.GetJacobian(), jn = p.GetJacobianNumeric();
        std::printf("EdgeLogScalePrior analytic %g  numeric %g  abs_err %.3e\n", ja, jn, std::fabs(ja - jn));
        Check(std::fabs(ja - jn) < tol && ja == 1.0, "EdgeLogScalePrior Jacobian is 1");
    }

    // Marginal variance of a bare chain with an absolute prior on its head:
    // Var(a_k) = sigma_0^2 + sum of the segment variances before k.
    {
        const double sigma0 = 0.1, sigmaRW = 0.02;
        const double dts[3] = {0.5, 1.0, 0.25};
        g2o::SparseOptimizer opt;
        g2o::BlockSolverX::LinearSolverType* linear = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        UcslamBlockSolverX* solver = new UcslamBlockSolverX(linear);
        opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(solver));
        std::vector<VertexLogScale*> vs;
        for(int k = 0; k < 4; k++)
        {
            VertexLogScale* v = new VertexLogScale(0.05 * k);
            v->setId(k);
            opt.addVertex(v);
            vs.push_back(v);
        }
        EdgeLogScalePrior* p = new EdgeLogScalePrior(0.0);
        p->setVertex(0, vs[0]);
        Eigen::Matrix<double,1,1> info;
        info(0,0) = 1.0 / (sigma0 * sigma0);
        p->setInformation(info);
        opt.addEdge(p);
        for(int k = 0; k < 3; k++)
        {
            EdgeLogScaleRW* e = new EdgeLogScaleRW();
            e->setVertex(0, vs[k]);
            e->setVertex(1, vs[k + 1]);
            info(0,0) = 1.0 / (sigmaRW * sigmaRW * dts[k]);
            e->setInformation(info);
            opt.addEdge(e);
        }
        opt.initializeOptimization();
        opt.optimize(10);
        opt.computeActiveErrors();
        solver->buildStructure();
        solver->buildSystem();
        Eigen::MatrixXd S;
        Check(UcslamDensePoseSchur(*solver, S), "dense pose Hessian of the bare chain");
        Check(S.rows() == 4, "bare chain Hessian is 4 by 4");
        const Eigen::MatrixXd Cov = S.inverse();
        double expected = sigma0 * sigma0, maxErr = 0.0;
        for(int k = 0; k < 4; k++)
        {
            const int idx = solver->Hpp()->colBaseOfBlock(vs[k]->hessianIndex());
            maxErr = std::max(maxErr, std::fabs(Cov(idx, idx) - expected));
            if(k < 3)
                expected += sigmaRW * sigmaRW * dts[k];
        }
        std::printf("Chain marginals   Var(a_k) vs sigma_0^2 + sum of segment variances: max_abs_err %.3e\n", maxErr);
        Check(maxErr < 1e-12, "chain marginal variances");
        for(int k = 0; k < 4; k++)
            Check(std::fabs(vs[k]->estimate()) < 1e-4, "bare chain with a zero-mean prior moves towards zero");
    }

    // Marginal variance with marginalized points: the block-solver route equals a brute-force
    // Schur complement of the dense Hessian assembled from the edge Jacobians.
    {
        const Rig& rig = rigs[0];
        g2o::SparseOptimizer opt;
        g2o::BlockSolverX::LinearSolverType* linear = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        UcslamBlockSolverX* solver = new UcslamBlockSolverX(linear);
        opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(solver));

        // Two fixed body poses one metre apart looking down z.
        std::vector<VertexPose*> poses;
        for(int k = 0; k < 2; k++)
        {
            const Eigen::Matrix3d Rwb = Eigen::Matrix3d::Identity();
            const Eigen::Vector3d twb(0.5 * k, 0.0, 0.0);
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
            VertexPose* VP = new VertexPose();
            VP->setEstimate(pose);
            VP->setId(k);
            VP->setFixed(true);
            opt.addVertex(VP);
            poses.push_back(VP);
        }
        std::vector<VertexLogScale*> as;
        for(int k = 0; k < 2; k++)
        {
            VertexLogScale* v = new VertexLogScale(0.02 * k);
            v->setId(10 + k);
            opt.addVertex(v);
            as.push_back(v);
        }
        Eigen::Matrix<double,1,1> info;
        {
            EdgeLogScalePrior* p = new EdgeLogScalePrior(0.0);
            p->setVertex(0, as[0]);
            info(0,0) = 1.0 / 0.01;
            p->setInformation(info);
            opt.addEdge(p);
            EdgeLogScaleRW* e = new EdgeLogScaleRW();
            e->setVertex(0, as[0]);
            e->setVertex(1, as[1]);
            info(0,0) = 1.0 / (0.02 * 0.02 * 0.5);
            e->setInformation(info);
            opt.addEdge(e);
        }
        std::vector<g2o::VertexSBAPointXYZ*> pts;
        std::vector<EdgeMono*> monoEdges;
        std::vector<EdgeLogDepth*> depthEdges;
        const int nPts = 6;
        for(int j = 0; j < nPts; j++)
        {
            // Points in front of both cameras. The EuRoC camera looks along body z, so spread in z.
            const Eigen::Vector3d Xw(0.4 * unit(rng), 0.4 * unit(rng), 2.0 + 0.7 * j);
            g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
            v->setId(100 + j);
            v->setEstimate(Xw + Eigen::Vector3d(0.01 * unit(rng), 0.01 * unit(rng), 0.01 * unit(rng)));
            v->setMarginalized(true);
            opt.addVertex(v);
            pts.push_back(v);
            for(int k = 0; k < 2; k++)
            {
                const ImuCamPose& P = poses[k]->estimate();
                const Eigen::Vector3d Xc = P.Rcw[0] * Xw + P.tcw[0];
                if(!(Xc(2) > 0.1))
                {
                    Check(false, "test point in front of both cameras");
                    continue;
                }
                EdgeMono* em = new EdgeMono(0);
                em->setVertex(0, v);
                em->setVertex(1, poses[k]);
                em->setMeasurement(P.Project(Xw, 0) + Eigen::Vector2d(0.3 * unit(rng), 0.3 * unit(rng)));
                em->setInformation(Eigen::Matrix2d::Identity() * (1.0 / 1.44));
                opt.addEdge(em);
                monoEdges.push_back(em);

                EdgeLogDepth* ed = new EdgeLogDepth(0);
                ed->setVertex(0, v);
                ed->setVertex(1, poses[k]);
                ed->setVertex(2, as[k]);
                ed->setMeasurement(std::log(Xc(2)) + 0.05 * unit(rng));
                info(0,0) = 1.0 / (0.1 * 0.1);
                ed->setInformation(info);
                opt.addEdge(ed);
                depthEdges.push_back(ed);
            }
        }
        opt.initializeOptimization();
        opt.optimize(2);
        opt.computeActiveErrors();
        solver->buildStructure();
        solver->buildSystem();
        Eigen::MatrixXd S;
        Check(UcslamDensePoseSchur(*solver, S), "dense pose Schur complement with points");
        Check(S.rows() == 2, "pose part holds the two log-scale vertices only");

        // Brute force at the same linearization point. Order: a_0, a_1, then the points.
        const int n = 2 + 3 * nPts;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        {
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, n);
            J(0, 0) = 1.0;
            H += J.transpose() * (1.0 / 0.01) * J;
            J.setZero();
            J(0, 0) = -1.0;
            J(0, 1) = 1.0;
            H += J.transpose() * (1.0 / (0.02 * 0.02 * 0.5)) * J;
        }
        for(size_t i = 0; i < monoEdges.size(); i++)
        {
            const Eigen::Matrix<double,2,9> Je = monoEdges[i]->GetJacobian();
            const int j = monoEdges[i]->vertex(0)->id() - 100;
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, n);
            J.block<2,3>(0, 2 + 3 * j) = Je.block<2,3>(0, 0);
            H += J.transpose() * monoEdges[i]->information() * J;
        }
        for(size_t i = 0; i < depthEdges.size(); i++)
        {
            const Eigen::Matrix<double,1,EdgeLogDepthBase::kCols> Je = depthEdges[i]->GetJacobian();
            const int j = depthEdges[i]->vertex(0)->id() - 100;
            const int k = depthEdges[i]->vertex(2)->id() - 10;
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, n);
            J.block<1,3>(0, 2 + 3 * j) = Je.block<1,3>(0, 0);
            J(0, k) = Je(0, 9);
            H += J.transpose() * depthEdges[i]->information() * J;
        }
        const Eigen::MatrixXd Haa = H.block(0, 0, 2, 2);
        const Eigen::MatrixXd Hap = H.block(0, 2, 2, n - 2);
        const Eigen::MatrixXd Hpp = H.block(2, 2, n - 2, n - 2);
        const Eigen::MatrixXd Sref = Haa - Hap * Hpp.inverse() * Hap.transpose();

        // Map the solver's ordering onto a_0, a_1.
        Eigen::MatrixXd Ssolver(2, 2);
        for(int r = 0; r < 2; r++)
            for(int c = 0; c < 2; c++)
                Ssolver(r, c) = S(solver->Hpp()->colBaseOfBlock(as[r]->hessianIndex()),
                                  solver->Hpp()->colBaseOfBlock(as[c]->hessianIndex()));
        const double err = (Ssolver - Sref).cwiseAbs().maxCoeff() / Sref.cwiseAbs().maxCoeff();
        std::printf("Schur with points block-solver route vs brute force: max_rel_err %.3e  Var(a_0)=%.6g Var(a_1)=%.6g\n",
                    err, Ssolver.inverse()(0,0), Ssolver.inverse()(1,1));
        Check(err < 1e-9, "dense pose Schur complement equals the brute-force one");
    }

    if(gFailures == 0)
    {
        std::cout << "PASS test_ucslam_types" << std::endl;
        return 0;
    }
    std::cout << "FAIL test_ucslam_types: " << gFailures << " check(s) failed" << std::endl;
    return 1;
}
