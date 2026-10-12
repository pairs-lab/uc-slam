/**
 * UC-SLAM node core, shared by the ROS 1 node (ros/ros1) and the ROS 2 node (ros/ros2). No ROS dependency.
 *
 * It owns the ORB_SLAM3::System and one worker thread, the only caller of Track*. The ROS layer pushes images and
 * IMU samples, receives one result per tracked frame, and forwards live depth both ways (docs: ros/README.md).
 *
 * Rules (from the benchmark example mono_inertial_euroc.cc):
 *  - IMU samples are kept only with a stamp strictly greater than the last accepted one (two equal stamps give a division
 *    by zero in the preintegration). Before a frame is tracked, the worker waits until an IMU sample at or after
 *    the image stamp has arrived (imu_wait_ms), then passes every sample with t <= t_image.
 *  - A frame older than the last tracked one is dropped (it would clear the IMU queue and start a new map).
 *  - The time passed to Track* is double(stamp_ns) / 1e9, computed once by StampToSec. The same double keys the
 *    image ring, the depth request and the depth map, so PriorNet matches them to the nanosecond.
 *  - The settings file must hold File.version "1.0", otherwise System builds no Settings and no PriorNet and the
 *    depth factor stays off without an error.
 */
#ifndef UCSLAM_NODE_CORE_H
#define UCSLAM_NODE_CORE_H

#include "System.h"
#include "ImuTypes.h"
#include "PriorNet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sophus/se3.hpp>

#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ucslam_node
{

inline double StampToSec(int64_t stamp_ns) { return double(stamp_ns) / 1e9; }

// File.version "1.0" and an existing file. Returns an error message, empty when the file is fine.
inline std::string CheckSettingsFile(const std::string& path)
{
    std::ifstream f(path.c_str());
    if(!f)
        return "cannot open settings file " + path;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return "cannot parse settings file " + path;
    cv::FileNode v = fs["File.version"];
    if(v.empty() || !v.isString() || v.string() != "1.0")
        return "settings file " + path + " has no File.version: \"1.0\"; UC-SLAM would run without its depth factor";
    return std::string();
}

// IMU.T_b_c1 of the settings file: the left camera in the IMU (body) frame.
inline bool ReadTbc(const std::string& path, Sophus::SE3f& Tbc)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    cv::Mat T;
    fs["IMU.T_b_c1"] >> T;
    if(T.empty() || T.rows != 4 || T.cols != 4)
        return false;
    T.convertTo(T, CV_32F);
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    for(int r = 0; r < 3; r++)
    {
        for(int c = 0; c < 3; c++)
            R(r, c) = T.at<float>(r, c);
        t(r) = T.at<float>(r, 3);
    }
    Eigen::Quaternionf q(R);
    q.normalize();
    Tbc = Sophus::SE3f(q, t);
    return true;
}

// One scalar of the settings file as text ("" when missing). Strings come without quotes.
inline std::string ReadSettingString(const std::string& path, const std::string& key)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return std::string();
    cv::FileNode n = fs[key];
    if(n.empty())
        return std::string();
    if(n.isString())
        return n.string();
    if(n.isInt())
        return std::to_string((int)n);
    if(n.isReal())
    {
        std::ostringstream o;
        o << (double)n;
        return o.str();
    }
    return std::string();
}

// An absolute path: a relative one is taken from the current directory. The node changes into output_dir before
// System opens its files, so every path must be absolute by then.
inline std::string AbsolutePath(const std::string& p)
{
    if(p.empty() || p[0] == '/')
        return p;
    char buf[4096];
    if(!getcwd(buf, sizeof(buf)))
        return p;
    return std::string(buf) + "/" + p;
}

// Map time after which the inertial refinements of the base system have run: VIBA 2 at 15 s for both modes, and the
// scale refinements up to 75.5 s for monocular-inertial (LocalMapping::Run). Until then scale and gravity can move.
inline double SettlingMapTime(bool stereo) { return stereo ? 16.0 : 76.0; }

// mkdir -p. Returns an error message, empty on success.
inline std::string MakeDirs(const std::string& path)
{
    if(path.empty())
        return std::string();
    std::string cur;
    std::stringstream ss(path);
    std::string part;
    if(path[0] == '/')
        cur = "/";
    while(std::getline(ss, part, '/'))
    {
        if(part.empty())
            continue;
        cur += part + "/";
        if(::mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST)
            return "cannot create directory " + cur + ": " + std::strerror(errno);
    }
    return std::string();
}

// Copies the settings file to out with the given scalar keys replaced: their "Key: value" lines are dropped and
// the new lines appended at the end. Values are YAML text (strings in double quotes). The node runs System on
// this copy, so the file that was used stays next to the trajectories. Returns an error message.
inline std::string WriteSettingsWithOverrides(const std::string& in,
                                              const std::vector<std::pair<std::string, std::string> >& kv,
                                              const std::string& out)
{
    std::ifstream f(in.c_str());
    if(!f)
        return "cannot open settings file " + in;
    std::ofstream o(out.c_str());
    if(!o)
        return "cannot write settings file " + out;
    std::string line;
    while(std::getline(f, line))
    {
        size_t b = line.find_first_not_of(" \t");
        bool drop = false;
        for(size_t i = 0; i < kv.size() && b != std::string::npos; i++)
        {
            const std::string& k = kv[i].first;
            if(line.compare(b, k.size(), k) == 0)
            {
                size_t c = line.find_first_not_of(" \t", b + k.size());
                if(c != std::string::npos && line[c] == ':')
                    drop = true;
            }
        }
        if(!drop)
            o << line << "\n";
    }
    o << "\n# Set by the UC-SLAM ROS node\n";
    for(size_t i = 0; i < kv.size(); i++)
        o << kv[i].first << ": " << kv[i].second << "\n";
    return o.good() ? std::string() : "write error on " + out;
}

struct Options
{
    std::string vocabulary;               // ORBvoc.txt
    std::string settings;                 // UC-SLAM settings YAML (File.version "1.0")
    bool stereo = false;                  // false: mono-inertial, true: stereo-inertial
    bool viewer = false;
    double imu_wait_ms = 50.0;            // wait for an IMU sample at or after the image stamp
    double imu_time_offset_s = 0.0;       // added to every IMU stamp (camera-IMU offset from calibration)
    bool drop_oldest = true;              // live: drop the oldest frame when the queue is full. Bag replay: false
    size_t frame_queue = 3;               // frames waiting for the worker when drop_oldest
    double image_ring_s = 2.0;            // images kept for depth requests
    bool clahe = false;                   // CLAHE(3.0, 8x8) before tracking, as the TUM-VI benchmark examples do.
                                          // The depth request keeps the raw image, like the depth cache
    bool collect_points = false;          // fill Result::points with the map points of the tracked frame
    std::string output_dir;               // where Stop() saves the trajectories. The node also chdirs there
};

struct Result
{
    double t = 0.0;
    int64_t stamp_ns = 0;
    int state = -1;                       // Tracking::eTrackingState: 2 OK, 3 RECENTLY_LOST
    bool imu_initialized = false;
    double time_from_imu_init = 0.0;      // System::GetTimeFromIMUInit: map time, the keyframe time since the first
                                          // keyframe of the map once the IMU is initialized (0 before). It is the
                                          // clock of the inertial refinements, see SettlingMapTime
    bool map_changed = false;             // a loop closure or a map merge changed the map
    Sophus::SE3f Tcw;                     // left camera in the world, as Track* returns it
    Sophus::SE3f Twb;                     // body (IMU) in the world: Twc * Tbc^-1
    double track_ms = 0.0;
    std::vector<Eigen::Vector3f> points;  // world positions of the tracked map points (collect_points only)
};

struct Counters
{
    std::atomic<uint64_t> frames{0}, dropped_frames{0}, old_frames{0}, imu_rejected{0}, imu_wait_timeouts{0};
    std::atomic<uint64_t> depth_requests{0}, depth_pushed{0}, depth_refused{0};
};

class NodeCore
{
public:
    using ResultCallback = std::function<void(const Result&)>;
    // t, stamp_ns, keyframe id, the keyframe image (left)
    using DepthRequestCallback = std::function<void(double, int64_t, unsigned long, const cv::Mat&)>;

    explicit NodeCore(const Options& o) : opt_(o) {}
    ~NodeCore() { Stop(); }

    // Builds the System (this can take seconds: the vocabulary is loaded) and starts the worker.
    // Returns an error message, empty on success.
    std::string Start(ResultCallback onResult, DepthRequestCallback onDepthRequest)
    {
        std::string err = CheckSettingsFile(opt_.settings);
        if(!err.empty())
            return err;
        if(!ReadTbc(opt_.settings, Tbc_))
            return "settings file " + opt_.settings + " has no 4x4 IMU.T_b_c1";
        onResult_ = onResult;
        onDepthRequest_ = onDepthRequest;
        system_.reset(new ORB_SLAM3::System(opt_.vocabulary, opt_.settings,
                                            opt_.stereo ? ORB_SLAM3::System::IMU_STEREO
                                                        : ORB_SLAM3::System::IMU_MONOCULAR,
                                            opt_.viewer));
        imageScale_ = system_->GetImageScale();
        // The keyframe callback runs in the worker (inside Track*), right after a keyframe is made.
        system_->SetKeyFrameCallback([this](double t, unsigned long id) { OnKeyFrame(t, id); });
        stop_ = false;
        worker_ = std::thread(&NodeCore::Run, this);
        return std::string();
    }

    void PushImu(int64_t stamp_ns, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyr)
    {
        const double t = StampToSec(stamp_ns) + opt_.imu_time_offset_s;
        {
            std::unique_lock<std::mutex> lock(mImu_);
            if(!(t > lastImuT_))
            {
                counters.imu_rejected++;
                return;
            }
            imu_.push_back(ORB_SLAM3::IMU::Point(acc.x(), acc.y(), acc.z(), gyr.x(), gyr.y(), gyr.z(), t));
            lastImuT_ = t;
        }
        cvImu_.notify_all();
    }

    void PushFrame(int64_t stamp_ns, const cv::Mat& left, const cv::Mat& right = cv::Mat())
    {
        {
            std::unique_lock<std::mutex> lock(mFrames_);
            if(opt_.drop_oldest && frames_.size() >= std::max<size_t>(1, opt_.frame_queue))
            {
                frames_.pop_front();
                counters.dropped_frames++;
            }
            frames_.push_back(Frame{stamp_ns, left, right});
        }
        cvFrames_.notify_one();
    }

    // Live depth for the keyframe image at stamp_ns. kind/full size/K/model as PriorNet::DepthView.
    bool PushDepth(int64_t stamp_ns, const cv::Mat& z, const ORB_SLAM3::PriorNet::DepthView& view)
    {
        if(!system_)
            return false;
        const bool ok = system_->InsertDepthMap(StampToSec(stamp_ns), z, view);
        if(ok) counters.depth_pushed++; else counters.depth_refused++;
        return ok;
    }

    // Services. They run between two frames in the worker.
    void RequestReset() { resetRequested_ = true; }
    void RequestResetActiveMap() { resetMapRequested_ = true; }
    void SetLocalizationMode(bool on) { localization_ = on ? 1 : 0; }

    // Stops the worker, shuts the System down (waits for local mapping and PriorNet) and saves the trajectories
    // f_<name>.txt and kf_<name>.txt in output_dir, the files the benchmark scores.
    void Stop(const std::string& name = "ucslam")
    {
        if(!system_ || stopped_)
            return;
        stop_ = true;
        cvFrames_.notify_all();
        cvImu_.notify_all();
        if(worker_.joinable())
            worker_.join();
        system_->Shutdown();
        const std::string dir = opt_.output_dir.empty() ? std::string(".") : opt_.output_dir;
        system_->SaveTrajectoryEuRoC(dir + "/f_" + name + ".txt");
        system_->SaveKeyFrameTrajectoryEuRoC(dir + "/kf_" + name + ".txt");
        stopped_ = true;
    }

    ORB_SLAM3::System* system() { return system_.get(); }
    const Sophus::SE3f& Tbc() const { return Tbc_; }
    Counters counters;

private:
    struct Frame { int64_t stamp_ns; cv::Mat left, right; };

    void OnKeyFrame(double t, unsigned long id)
    {
        if(!onDepthRequest_)
            return;
        cv::Mat img;
        int64_t ns = 0;
        {
            std::unique_lock<std::mutex> lock(mRing_);
            // The keyframe stamp is the image stamp passed to Track*, so the ring holds it exactly.
            for(std::deque<std::pair<int64_t, cv::Mat> >::const_reverse_iterator it = ring_.rbegin(); it != ring_.rend(); ++it)
                if(std::llabs(it->first - (int64_t)std::llround(t * 1e9)) <= 1000)   // 1 us: the same image
                {
                    img = it->second;
                    ns = it->first;
                    break;
                }
        }
        if(img.empty())
            return;
        counters.depth_requests++;
        onDepthRequest_(t, ns, id, img);
    }

    void Run()
    {
        double lastT = -1.0;
        while(!stop_)
        {
            Frame f;
            {
                std::unique_lock<std::mutex> lock(mFrames_);
                cvFrames_.wait_for(lock, std::chrono::milliseconds(50), [this] { return stop_ || !frames_.empty(); });
                if(stop_)
                    break;
                if(frames_.empty())
                    continue;
                f = frames_.front();
                frames_.pop_front();
            }
            const double t = StampToSec(f.stamp_ns);
            if(!(t > lastT))
            {
                counters.old_frames++;
                continue;
            }
            HandleServices();

            // IMU up to the image stamp
            std::vector<ORB_SLAM3::IMU::Point> vImu;
            {
                std::unique_lock<std::mutex> lock(mImu_);
                const bool got = cvImu_.wait_for(lock, std::chrono::microseconds((long long)(opt_.imu_wait_ms * 1000.0)),
                                                 [this, t] { return stop_ || (!imu_.empty() && imu_.back().t >= t); });
                if(!got)
                    counters.imu_wait_timeouts++;
                while(!imu_.empty() && imu_.front().t <= t)
                {
                    vImu.push_back(imu_.front());
                    imu_.pop_front();
                }
            }
            if(stop_)
                break;
            {
                std::unique_lock<std::mutex> lock(mRing_);
                ring_.push_back(std::make_pair(f.stamp_ns, f.left));
                while(!ring_.empty() && StampToSec(f.stamp_ns - ring_.front().first) > opt_.image_ring_s)
                    ring_.pop_front();
            }

            // The steps of the benchmark examples before Track*: CLAHE (TUM-VI examples), then the resize by
            // GetImageScale. With File.version 1.0 settings that scale is 1 and Track* itself resizes to
            // Camera.newWidth x newHeight. The ring above keeps the raw image.
            cv::Mat left = f.left, right = f.right;
            if(opt_.clahe)
            {
                left = left.clone();
                clahe_->apply(left, left);
                if(!right.empty())
                {
                    right = right.clone();
                    clahe_->apply(right, right);
                }
            }
            if(imageScale_ != 1.f)
            {
                cv::resize(left, left, cv::Size(int(left.cols * imageScale_), int(left.rows * imageScale_)));
                if(!right.empty())
                    cv::resize(right, right, cv::Size(int(right.cols * imageScale_), int(right.rows * imageScale_)));
            }

            const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            Result r;
            r.Tcw = opt_.stereo ? system_->TrackStereo(left, right, t, vImu)
                                : system_->TrackMonocular(left, t, vImu);
            r.track_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            r.t = t;
            r.stamp_ns = f.stamp_ns;
            r.state = system_->GetTrackingState();
            r.time_from_imu_init = system_->GetTimeFromIMUInit();
            r.imu_initialized = r.time_from_imu_init > 0.0;
            r.map_changed = system_->MapChanged();
            r.Twb = r.Tcw.inverse() * Tbc_.inverse();
            if(opt_.collect_points)
            {
                const std::vector<ORB_SLAM3::MapPoint*> vMP = system_->GetTrackedMapPoints();
                for(size_t i = 0; i < vMP.size(); i++)
                    if(vMP[i] && !vMP[i]->isBad())
                        r.points.push_back(vMP[i]->GetWorldPos());
            }
            counters.frames++;
            lastT = t;
            if(onResult_)
                onResult_(r);
        }
    }

    void HandleServices()
    {
        if(resetRequested_.exchange(false))
            system_->Reset();
        if(resetMapRequested_.exchange(false))
            system_->ResetActiveMap();
        const int loc = localization_.exchange(-1);
        if(loc == 1) system_->ActivateLocalizationMode();
        if(loc == 0) system_->DeactivateLocalizationMode();
    }

    Options opt_;
    std::unique_ptr<ORB_SLAM3::System> system_;
    float imageScale_ = 1.f;
    cv::Ptr<cv::CLAHE> clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
    Sophus::SE3f Tbc_;
    ResultCallback onResult_;
    DepthRequestCallback onDepthRequest_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    bool stopped_ = false;
    std::atomic<bool> resetRequested_{false}, resetMapRequested_{false};
    std::atomic<int> localization_{-1};

    std::mutex mImu_;
    std::condition_variable cvImu_;
    std::deque<ORB_SLAM3::IMU::Point> imu_;
    double lastImuT_ = -1e300;            // stamp of the last accepted IMU sample

    std::mutex mFrames_;
    std::condition_variable cvFrames_;
    std::deque<Frame> frames_;

    std::mutex mRing_;
    std::deque<std::pair<int64_t, cv::Mat> > ring_;
};

}  // namespace ucslam_node

#endif  // UCSLAM_NODE_CORE_H
