/**
* UC-SLAM. Standalone test of the log-scale chain marginalization,
* docs/fork_design.md Sec. 2, "Marginalization on culling".
*
* Ports the two chain cases of tests/test_factors.py,
*   test_cull_with_unary_terms_matches_dense_schur_node_by_node and
*   test_cull_head_transfers_a0_prior_and_preserves_head_marginal,
* together with the interior merged-segment formula and the tail rule, against
* ORB_SLAM3::LogScaleChain (src/Map.cc). Every cull, the head included, must
* leave the surviving head's marginal variance unchanged to 1e-12 against a
* direct dense Schur complement. Exit code 0 when every check holds. The last
* line prints the maximum deviations.
*
* Build: part of the fork CMake (target test_chain, binary in build/test/).
*
* Copyright (C) 2026 The UC-SLAM authors.
* Distributed under the GNU General Public License v3 or later, like ORB-SLAM3. See LICENSE.
*/

#include "Map.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using ORB_SLAM3::LogScaleChain;

namespace
{

int gChecks = 0;
int gFailures = 0;
double gMaxDevH = 0.0;        // max |H_chain - H_schur| / max |H|
double gMaxDevB = 0.0;        // max |b_chain - b_schur| / max(1, max |b|)
double gMaxDevHeadVar = 0.0;  // max |head marginal variance after - before|, absolute
double gMaxDevMean = 0.0;     // max |mean_chain - mean_schur|, absolute

void Check(bool ok, const std::string& what)
{
    gChecks++;
    if(!ok)
    {
        gFailures++;
        std::cerr << "FAIL: " << what << std::endl;
    }
}

void CheckClose(double a, double b, double tol, const std::string& what)
{
    const double dev = std::fabs(a - b);
    Check(dev <= tol, what + " (got " + std::to_string(a) + ", expected " + std::to_string(b) + ", deviation " + std::to_string(dev) + ")");
}

Eigen::MatrixXd Inverse(const Eigen::MatrixXd& H)
{
    return H.fullPivLu().inverse();
}

LogScaleChain RandomChain(std::mt19937& rng, int nNodes)
{
    std::uniform_real_distribution<double> uVar(1e-3, 1e-1);
    std::uniform_real_distribution<double> uS(5.0, 50.0);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<double> var(nNodes - 1), S(nNodes), b(nNodes);
    for(size_t i = 0; i < var.size(); i++) var[i] = uVar(rng);
    for(size_t i = 0; i < S.size(); i++) S[i] = uS(rng);
    for(size_t i = 0; i < b.size(); i++) b[i] = gauss(rng);
    return LogScaleChain(var, S, b, 0.01);
}

// One cull of node k on a chain and on its dense (H, b). Compares the chain's
// dense form with the Schur complement and the surviving head's marginal
// variance with the value before the cull.
void CullAndCompare(LogScaleChain& chain, Eigen::MatrixXd& H, Eigen::VectorXd& b, int k, const std::string& tag)
{
    const Eigen::MatrixXd SigmaBefore = Inverse(H);
    const Eigen::VectorXd meanBefore = SigmaBefore * b;
    // The head after the cull is node 0, or node 1 when the head itself is culled.
    const int headBefore = (k == 0) ? 1 : 0;
    const double headVarBefore = SigmaBefore(headBefore, headBefore);

    chain = chain.Cull(k);
    Eigen::MatrixXd Hs;
    Eigen::VectorXd bs;
    LogScaleChain::MarginalizeDense(H, b, k, Hs, bs);
    H = Hs;
    b = bs;

    Eigen::MatrixXd Hc;
    Eigen::VectorXd bc;
    chain.Dense(Hc, bc);

    const double scaleH = H.cwiseAbs().maxCoeff();
    const double scaleB = std::max(1.0, b.cwiseAbs().maxCoeff());
    const double devH = (Hc - H).cwiseAbs().maxCoeff() / scaleH;
    const double devB = (bc - b).cwiseAbs().maxCoeff() / scaleB;
    gMaxDevH = std::max(gMaxDevH, devH);
    gMaxDevB = std::max(gMaxDevB, devB);
    Check(devH <= 1e-12, tag + ": H of the chain equals the dense Schur complement, relative deviation " + std::to_string(devH));
    Check(devB <= 1e-12, tag + ": b of the chain equals the dense Schur complement, relative deviation " + std::to_string(devB));

    const Eigen::MatrixXd SigmaChain = Inverse(Hc);
    const Eigen::MatrixXd SigmaSchur = Inverse(H);
    const double devHead = std::fabs(SigmaChain(0, 0) - headVarBefore);
    const double devHeadSchur = std::fabs(SigmaChain(0, 0) - SigmaSchur(0, 0));
    gMaxDevHeadVar = std::max(gMaxDevHeadVar, std::max(devHead, devHeadSchur));
    Check(devHead <= 1e-12, tag + ": head marginal variance unchanged by the cull, deviation " + std::to_string(devHead));
    Check(devHeadSchur <= 1e-12, tag + ": head marginal variance equals the dense Schur value, deviation " + std::to_string(devHeadSchur));

    // Marginal means of the kept states are unchanged as well.
    const Eigen::VectorXd meanChain = SigmaChain * bc;
    int j = 0;
    double devMean = 0.0;
    for(int i = 0; i < meanBefore.size(); i++)
    {
        if(i == k)
            continue;
        devMean = std::max(devMean, std::fabs(meanChain(j) - meanBefore(i)));
        j++;
    }
    gMaxDevMean = std::max(gMaxDevMean, devMean);
    Check(devMean <= 1e-12, tag + ": marginal means unchanged by the cull, deviation " + std::to_string(devMean));
}

// tests/test_factors.py::test_cull_with_unary_terms_matches_dense_schur_node_by_node
void TestNodeByNode()
{
    std::mt19937 rng(5);
    const int orders[5][4] = {{2, 1, 1, 0}, {0, 0, 1, 0}, {4, 3, 2, 1}, {1, 2, 0, 1}, {3, 0, 1, 0}};
    for(int o = 0; o < 5; o++)
    {
        LogScaleChain chain = RandomChain(rng, 5);
        Eigen::MatrixXd H;
        Eigen::VectorXd b;
        chain.Dense(H, b);
        for(int step = 0; step < 4; step++)
        {
            const int k = orders[o][step];
            CullAndCompare(chain, H, b, k, "node_by_node order " + std::to_string(o) + " step " + std::to_string(step) + " cull " + std::to_string(k));
        }
        Check(chain.NumNodes() == 1, "node_by_node: one node left");
    }
}

// tests/test_factors.py::test_cull_head_transfers_a0_prior_and_preserves_head_marginal
void TestHeadTransfer()
{
    // (a) bare chain with the a_0 prior only: sigma_0^2 plus the dropped segment variance.
    {
        std::vector<double> var = {0.02, 0.03, 0.01, 0.05};
        LogScaleChain chain(var, std::vector<double>(), std::vector<double>(), 0.01);
        Eigen::MatrixXd H;
        Eigen::VectorXd b;
        chain.Dense(H, b);
        const double varNode1Before = Inverse(H)(1, 1);
        LogScaleChain culled = chain.Cull(0);
        CheckClose(culled.sigma0Sq, 0.01 + 0.02, 1e-14 * 0.03, "head (a): transferred prior variance is sigma_0^2 + variances[0]");
        Eigen::MatrixXd H2;
        Eigen::VectorXd b2;
        culled.Dense(H2, b2);
        const double dev = std::fabs(Inverse(H2)(0, 0) - varNode1Before);
        gMaxDevHeadVar = std::max(gMaxDevHeadVar, dev);
        Check(dev <= 1e-12, "head (a): marginal variance of the surviving head unchanged, deviation " + std::to_string(dev));
    }
    // (b) unary terms and an information vector on every node.
    {
        std::mt19937 rng(9);
        LogScaleChain chain = RandomChain(rng, 5);
        Eigen::MatrixXd H;
        Eigen::VectorXd b;
        chain.Dense(H, b);
        const Eigen::MatrixXd Sigma = Inverse(H);
        const Eigen::VectorXd mean = Sigma * b;
        LogScaleChain culled = chain.Cull(0);
        const double expectedS0 = chain.segVar[0] + 1.0 / (chain.S[0] + 1.0 / chain.sigma0Sq);
        CheckClose(culled.sigma0Sq, expectedS0, 1e-14 * expectedS0, "head (b): transferred prior variance is variances[0] + (S_0 + sigma_0^-2)^-1");
        Eigen::MatrixXd H2;
        Eigen::VectorXd b2;
        culled.Dense(H2, b2);
        const Eigen::MatrixXd Sigma2 = Inverse(H2);
        const double devVar = std::fabs(Sigma2(0, 0) - Sigma(1, 1));
        gMaxDevHeadVar = std::max(gMaxDevHeadVar, devVar);
        Check(devVar <= 1e-12, "head (b): marginal variance of the surviving head unchanged, deviation " + std::to_string(devVar));
        const Eigen::VectorXd mean2 = Sigma2 * b2;
        double devMean = 0.0;
        for(int i = 1; i < mean.size(); i++)
            devMean = std::max(devMean, std::fabs(mean2(i - 1) - mean(i)));
        gMaxDevMean = std::max(gMaxDevMean, devMean);
        Check(devMean <= 1e-12, "head (b): marginal means unchanged, deviation " + std::to_string(devMean));

        // (d) dropping the prior instead of transferring it makes the head less certain.
        LogScaleChain dropped(std::vector<double>(chain.segVar.begin() + 1, chain.segVar.end()),
                              std::vector<double>(chain.S.begin() + 1, chain.S.end()),
                              std::vector<double>(chain.b.begin() + 1, chain.b.end()),
                              std::numeric_limits<double>::infinity());
        Eigen::MatrixXd Hd;
        Eigen::VectorXd bd;
        dropped.Dense(Hd, bd);
        Check(Inverse(Hd)(0, 0) > Sigma(1, 1) * (1.0 + 1e-6), "head (d): dropping the prior inflates the head variance");
    }
    // (c) the prior sits on a node with both neighbours, the case LocalMapping meets.
    // Expressed through S and b of that node, as MarginalizeLogScaleOnCull does.
    {
        std::mt19937 rng(21);
        LogScaleChain chain = RandomChain(rng, 3);
        const double s0 = 0.01, m0 = 0.07;
        chain.sigma0Sq = std::numeric_limits<double>::infinity();
        chain.S[1] += 1.0 / s0;
        chain.b[1] += m0 / s0;
        Eigen::MatrixXd H;
        Eigen::VectorXd b;
        chain.Dense(H, b);
        CullAndCompare(chain, H, b, 1, "head (c): prior on an interior node");
        // The shares the neighbours receive: the next node's share as a prior has
        // variance D / (w2 S) and the previous node's share D / (w1 S).
        double w1 = 1.0 / 0.02, w2 = 1.0 / 0.05, S = 12.0, bb = 0.3, wM, Sp, bp, Sn, bn;
        LogScaleChain::CullShares(w1, w2, S, bb, wM, Sp, bp, Sn, bn);
        const double D = w1 + w2 + S;
        CheckClose(wM, w1 * w2 / D, 1e-14 * wM, "shares: merged segment information w1 w2 / D");
        CheckClose(Sp, w1 * S / D, 1e-14 * Sp, "shares: previous unary share");
        CheckClose(Sn, w2 * S / D, 1e-14 * Sn, "shares: next unary share");
        CheckClose(bp / Sp, bb / S, 1e-14, "shares: the unary mean of a share is the culled node's unary mean");
        CheckClose(bn / Sn, bb / S, 1e-14, "shares: the unary mean of a share is the culled node's unary mean");
        // With w1 = 0 the next share is the head rule of the design.
        LogScaleChain::CullShares(0.0, w2, S, bb, wM, Sp, bp, Sn, bn);
        CheckClose(1.0 / Sn, 1.0 / S + 1.0 / w2, 1e-14 * (1.0 / Sn), "shares: w1 = 0 gives variance S^-1 + sigma_rw^2 dt");
        Check(wM == 0.0 && Sp == 0.0, "shares: w1 = 0 leaves no segment and no previous share");
    }
}

// tests/test_factors.py::test_interior_cull_merged_segment_formula
void TestInteriorFormula()
{
    std::vector<double> var = {0.01, 0.04, 0.02, 0.05};
    std::vector<double> S = {0.0, 0.0, 30.0, 0.0, 0.0};
    LogScaleChain chain(var, S, std::vector<double>(), std::numeric_limits<double>::infinity());
    LogScaleChain out = chain.Cull(2);
    const double w1 = 1.0 / 0.04, w2 = 1.0 / 0.02, Sk = 30.0;
    const double expected = (w1 + w2 + Sk) / (w1 * w2);
    CheckClose(out.segVar[1], expected, 1e-14 * expected, "interior: merged segment variance (w1 + w2 + S_k) / (w1 w2)");
    Check(out.segVar[1] > 0.04 + 0.02, "interior: variance summing is optimistic");
    LogScaleChain bare(var, std::vector<double>(), std::vector<double>(), std::numeric_limits<double>::infinity());
    LogScaleChain bareOut = bare.Cull(2);
    CheckClose(bareOut.segVar[1], 0.06, 1e-14 * 0.06, "interior: bare node sums the adjacent variances");
    Check(bareOut.segVar.size() == 3 && std::fabs(bareOut.segVar[0] - 0.01) < 1e-15 && std::fabs(bareOut.segVar[2] - 0.05) < 1e-15,
          "interior: the other segments are unchanged");
}

// tests/test_factors.py::test_cull_tail_drops_segment_and_folds_unary_term
void TestTail()
{
    std::mt19937 rng(13);
    LogScaleChain chain = RandomChain(rng, 5);
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    chain.Dense(H, b);
    CullAndCompare(chain, H, b, 4, "tail: random chain");
    Check(chain.segVar.size() == 3, "tail: one segment dropped");
    std::vector<double> var = {0.01, 0.02, 0.03};
    LogScaleChain bare(var, std::vector<double>(), std::vector<double>(), std::numeric_limits<double>::infinity());
    LogScaleChain out = bare.Cull(3);
    Check(out.segVar.size() == 2 && std::fabs(out.segVar[0] - 0.01) < 1e-15 && std::fabs(out.segVar[1] - 0.02) < 1e-15,
          "tail: bare chain keeps the first two segments");
}

// Random orders on longer chains, every node type, many draws.
void TestRandomOrders()
{
    std::mt19937 rng(3);
    for(int trial = 0; trial < 200; trial++)
    {
        const int nNodes = 3 + (int)(rng() % 6);
        LogScaleChain chain = RandomChain(rng, nNodes);
        Eigen::MatrixXd H;
        Eigen::VectorXd b;
        chain.Dense(H, b);
        while(chain.NumNodes() > 1)
        {
            const int k = (int)(rng() % chain.NumNodes());
            CullAndCompare(chain, H, b, k, "random trial " + std::to_string(trial) + " cull " + std::to_string(k));
        }
    }
}

} // namespace

int main()
{
    TestNodeByNode();
    TestHeadTransfer();
    TestInteriorFormula();
    TestTail();
    TestRandomOrders();

    std::printf("test_chain: %d checks, %d failures\n", gChecks, gFailures);
    std::printf("test_chain: max deviation H (relative) %.3e, b (relative) %.3e, head marginal variance (absolute) %.3e, means (absolute) %.3e, tolerance 1e-12\n",
                gMaxDevH, gMaxDevB, gMaxDevHeadVar, gMaxDevMean);
    return gFailures == 0 ? 0 : 1;
}
