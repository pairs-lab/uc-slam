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

#ifndef UCSLAM_PRIORNET_H
#define UCSLAM_PRIORNET_H

#include "KeyFrame.h"
#include "Settings.h"

#include <opencv2/core/core.hpp>

#include <condition_variable>
#include <list>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

namespace ORB_SLAM3
{

/*
 * PriorNet, version 1 of docs/fork_design.md Sec. 2.
 *
 * A thread with a queue of keyframes. The network output is precomputed by
 * experiments/precompute_depth.py into
 *   <DepthDir>/<Dataset>/<seq>/<DepthModel>/<timestamp_ns>.npy
 * as float16 maps at a reduced resolution, one per cam0 frame, with a
 * manifest.json next to them. For each keyframe this thread loads the map
 * whose stamp is nearest to the keyframe stamp (within DepthMatchTolMs),
 * samples z_hat at every keypoint by bilinear interpolation, computes
 *   sigma_D^2 = sigma_net(log z_hat)^2 + grad(z_hat)^T Sigma_kp grad(z_hat) / z_hat^2
 * with sigma_net = c0 + c1 (log z_hat - log z_ref) and Sigma_kp the pyramid
 * covariance mvLevelSigma2[octave] I in working-image pixels, and writes
 * mvDepthPrior = z_hat exp(-LogBias) and mvDepthPriorSigma under the
 * keyframe's depth mutex, then sets mbDepthReady.
 *
 * Which pixel of the map a keypoint reads depends on the view the manifest
 * declares. "camera": the map is a block-averaged copy of the recorded cam0
 * frame, so the distorted keypoint mvKeys is scaled from the working image
 * to the map with pixel centres aligned. For a rectified stereo pair the
 * rectification maps give the raw pixel first. "pinhole": the map is a
 * rectified pinhole view (TUM-VI), so the keypoint is unprojected with the
 * keyframe's camera model and projected with the view intrinsics. The
 * Jacobian of that pixel mapping enters the edge-aware term by central
 * differences of half a pixel.
 *
 * The sequence directory is found once, from the first keyframe's stamp,
 * among the model directories under <DepthDir>/<Dataset>/. A keyframe with
 * no map within the tolerance gets no depth, which is logged once. Nothing
 * consumes the depth yet. At shutdown the thread drains its queue and
 * PrintStats reports the counts and latencies the acceptance test asks for.
 */
class PriorNet
{
public:
    struct DepthMap
    {
        int width = 0, height = 0;
        long long stampNs = 0;
        std::vector<float> z;       // row major, metres, 0 means no prediction
        std::vector<float> gx, gy;  // central differences of z in map pixels, 0 next to a hole
    };

    // workImSize is the size the tracker sees (Settings::newImSize), M1l and M2l are the
    // left rectification maps when the settings rectify, empty otherwise.
    PriorNet(const Settings::Ucslam& params, const cv::Size& workImSize,
             const cv::Mat& M1l = cv::Mat(), const cv::Mat& M2l = cv::Mat());

    void Run();
    void InsertKeyFrame(KeyFrame* pKF);
    void RequestFinish();
    bool isFinished();
    int KeyframesInQueue();

    // The [UCSLAM] priornet lines. Call after the thread finished.
    void PrintStats(std::ostream& out);

    // Exposed for testing and for the log. Thread safe copies.
    std::string SequenceDir();
    long long KeyframesWithDepth();

    // Pure helpers, no state. Public so a test can call them.
    static bool ReadNpyFloat16(const std::string& path, DepthMap& map, std::string& err);
    static void ComputeGradients(DepthMap& map);
    static bool SampleBilinear(const DepthMap& map, float u, float v, float& z, float& gx, float& gy);
    static double Percentile(std::vector<double> v, double q);

private:
    enum View { VIEW_NONE = 0, VIEW_CAMERA = 1, VIEW_PINHOLE = 2 };

    bool ProcessKeyFrame(KeyFrame* pKF);
    bool LocateSequence(long long stampNs);
    bool ReadManifest(const std::string& path);
    bool ListStamps(const std::string& dir, std::vector<long long>& stamps);
    bool NearestStamp(long long stampNs, long long& found) const;
    bool WorkToMap(KeyFrame* pKF, const cv::Point2f& p, float& um, float& vm) const;
    bool CheckFinish();

    Settings::Ucslam mParams;
    cv::Size mWorkImSize;
    cv::Mat mM1l, mM2l;
    bool mbRectified;

    // Sequence cache, set by LocateSequence
    bool mbLocated;
    bool mbGaveUp;
    std::string mSeqDir;
    View mView;
    int mnFullW, mnFullH, mnMapW, mnMapH;
    float mKview[4];
    std::vector<long long> mvStamps;   // sorted

    // Queue
    std::list<KeyFrame*> mlQueue;
    std::mutex mMutexQueue;
    std::condition_variable mCondQueue;
    size_t mnQueueMax;

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    // Statistics
    std::mutex mMutexStats;
    long long mnReceived, mnWithDepth, mnNoMap, mnNoSequence, mnBadMap;
    std::vector<double> mvLatencyMs;     // keyframe creation to mbDepthReady
    std::vector<double> mvProcessMs;     // load plus sampling, per keyframe with a map
    std::vector<double> mvCoverage;      // keypoints with depth over keypoints, per keyframe with a map
    std::vector<double> mvSigmaMedian;   // median sigma_D per keyframe
    bool mbWarnedNoMap, mbWarnedNoSequence, mbWarnedShape;
};

} // namespace ORB_SLAM3

#endif // UCSLAM_PRIORNET_H
