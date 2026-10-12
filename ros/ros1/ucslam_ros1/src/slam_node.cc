/**
 * UC-SLAM SLAM node for ROS 1 (Noetic). The SLAM logic lives in ros/common/ucslam_node_core.h, shared with the
 * ROS 2 node. This file converts messages, publishes the results and serves the services.
 *
 * In:  image (mono8), image_right (stereo only), imu, depth_prior (live depth from depth_node)
 * Out: ~odom, ~pose, tf map_frame -> body_frame, ~tracked_points (optional), ~status (1 Hz), depth_request
 * Services: ~reset, ~reset_active_map (std_srvs/Trigger), ~localization_mode (std_srvs/SetBool)
 * See ros/README.md for the parameters.
 */
#include "ucslam_node_core.h"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <diagnostic_msgs/DiagnosticArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_srvs/SetBool.h>
#include <std_srvs/Trigger.h>
#include <tf2_ros/transform_broadcaster.h>
#include <ucslam_ros1/DepthPrior.h>
#include <ucslam_ros1/DepthRequest.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace
{

typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> StereoPolicy;

cv::Mat HighByte(const cv::Mat& im16)
{
    cv::Mat out(im16.rows, im16.cols, CV_8UC1);
    for(int r = 0; r < im16.rows; r++)
    {
        const uint16_t* a = im16.ptr<uint16_t>(r);
        uint8_t* b = out.ptr<uint8_t>(r);
        for(int c = 0; c < im16.cols; c++)
            b[c] = (uint8_t)(a[c] >> 8);
    }
    return out;
}

double Percentile(std::vector<double> v, double q)
{
    if(v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
}

class SlamNode
{
public:
    SlamNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh), pnh_(pnh) {}

    bool Init()
    {
        ucslam_node::Options o;
        std::string settings, frameDrop, depthSource, depthDir;
        pnh_.param<std::string>("vocabulary", o.vocabulary, "");
        pnh_.param<std::string>("settings", settings, "");
        pnh_.param<bool>("stereo", o.stereo, false);
        pnh_.param<bool>("viewer", o.viewer, false);
        pnh_.param<double>("imu_wait_ms", o.imu_wait_ms, 50.0);
        pnh_.param<double>("imu_time_offset", o.imu_time_offset_s, 0.0);
        pnh_.param<std::string>("frame_drop", frameDrop, "oldest");
        int queue = 3;
        pnh_.param<int>("frame_queue", queue, 3);
        o.frame_queue = (size_t)std::max(1, queue);
        pnh_.param<double>("image_ring_s", o.image_ring_s, 2.0);
        pnh_.param<bool>("clahe", o.clahe, false);
        pnh_.param<bool>("publish_points", publishPoints_, false);
        o.collect_points = publishPoints_;
        pnh_.param<std::string>("output_dir", o.output_dir, "");
        pnh_.param<std::string>("trajectory_name", trajName_, "ucslam");
        pnh_.param<std::string>("map_frame", mapFrame_, "map");
        pnh_.param<std::string>("body_frame", bodyFrame_, "imu");
        pnh_.param<bool>("publish_tf", publishTf_, true);
        pnh_.param<std::string>("depth_source", depthSource, "live");
        pnh_.param<std::string>("depth_dir", depthDir, "");
        pnh_.param<std::string>("depth_backend", backend_, "torch-fp32");
        double depthWaitMs = 2000.0, dppWaitMs = 0.0;
        int dppDeferMaxKFs = 0;
        pnh_.param<double>("depth_wait_ms", depthWaitMs, 2000.0);
        pnh_.param<double>("dpp_wait_ms", dppWaitMs, 0.0);
        pnh_.param<int>("dpp_defer_max_kfs", dppDeferMaxKFs, 0);
        pnh_.param<double>("pose_std_m", poseStdM_, 0.01);
        pnh_.param<double>("pose_std_rad", poseStdRad_, 0.01);

        // Init changes into output_dir before System opens its files, so every path becomes absolute here.
        o.vocabulary = ucslam_node::AbsolutePath(o.vocabulary);
        settings = ucslam_node::AbsolutePath(settings);
        o.output_dir = ucslam_node::AbsolutePath(o.output_dir);
        depthDir = ucslam_node::AbsolutePath(depthDir);
        stereo_ = o.stereo;
        if(frameDrop != "oldest" && frameDrop != "none")
            return Fail("parameter frame_drop must be oldest or none, got " + frameDrop);
        o.drop_oldest = (frameDrop == "oldest");
        if(depthSource != "live" && depthSource != "file")
            return Fail("parameter depth_source must be live or file, got " + depthSource);
        if(o.vocabulary.empty() || access(o.vocabulary.c_str(), R_OK) != 0)
            return Fail("cannot read the vocabulary " + o.vocabulary + " (parameter vocabulary)");
        std::string err = ucslam_node::CheckSettingsFile(settings);
        if(!err.empty())
            return Fail(err);
        if(o.output_dir.empty())
            return Fail("parameter output_dir is required (trajectories, settings copy, Atlas)");
        err = ucslam_node::MakeDirs(o.output_dir);
        if(!err.empty())
            return Fail(err);

        // The settings copy that System reads: the depth source and the late-depth rules of this run.
        domain_ = ucslam_node::ReadSettingString(settings, "UCSLAM.Dataset");
        std::vector<std::pair<std::string, std::string> > kv;
        kv.push_back(std::make_pair("UCSLAM.DepthSource", "\"" + depthSource + "\""));
        if(depthSource == "live")
        {
            kv.push_back(std::make_pair("UCSLAM.DepthWaitMs", std::to_string(depthWaitMs)));
            kv.push_back(std::make_pair("UCSLAM.DppWaitMs", std::to_string(dppWaitMs)));
            kv.push_back(std::make_pair("UCSLAM.DppDeferMaxKFs", std::to_string(dppDeferMaxKFs)));
        }
        else
        {
            // The node runs in output_dir, so a relative cache path of the benchmark settings would not resolve.
            const std::string dir = depthDir.empty() ? ucslam_node::ReadSettingString(settings, "UCSLAM.DepthDir") : depthDir;
            if(dir.empty() || dir[0] != '/')
                return Fail("depth_source file needs an absolute depth cache path: set the parameter depth_dir");
            kv.push_back(std::make_pair("UCSLAM.DepthDir", "\"" + dir + "\""));
        }
        o.settings = o.output_dir + "/ucslam_settings.yaml";
        err = ucslam_node::WriteSettingsWithOverrides(settings, kv, o.settings);
        if(!err.empty())
            return Fail(err);
        // System writes the Atlas, ucslam_track.csv and marginals/ relative to the working directory.
        if(chdir(o.output_dir.c_str()) != 0)
            return Fail("cannot chdir to " + o.output_dir);

        pubOdom_ = pnh_.advertise<nav_msgs::Odometry>("odom", 10);
        pubPose_ = pnh_.advertise<geometry_msgs::PoseStamped>("pose", 10);
        pubStatus_ = pnh_.advertise<diagnostic_msgs::DiagnosticArray>("status", 10);
        if(publishPoints_)
            pubPoints_ = pnh_.advertise<sensor_msgs::PointCloud2>("tracked_points", 1);
        pubRequest_ = nh_.advertise<ucslam_ros1::DepthRequest>("depth_request", 16);

        core_.reset(new ucslam_node::NodeCore(o));
        ROS_INFO("UC-SLAM: loading the vocabulary and starting the system (settings %s)", o.settings.c_str());
        err = core_->Start([this](const ucslam_node::Result& r) { OnResult(r); },
                           [this](double t, int64_t ns, unsigned long id, const cv::Mat& img) { OnDepthRequest(t, ns, id, img); });
        if(!err.empty())
            return Fail(err);

        // Subscriptions only after the System exists.
        subImu_ = nh_.subscribe("imu", 2000, &SlamNode::OnImu, this, ros::TransportHints().tcpNoDelay());
        subDepth_ = nh_.subscribe("depth_prior", 16, &SlamNode::OnDepthPrior, this);
        if(o.stereo)
        {
            subLeft_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh_, "image", 5));
            subRight_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh_, "image_right", 5));
            sync_.reset(new message_filters::Synchronizer<StereoPolicy>(StereoPolicy(10), *subLeft_, *subRight_));
            sync_->registerCallback(boost::bind(&SlamNode::OnStereo, this, _1, _2));
        }
        else
            subImage_ = nh_.subscribe("image", 5, &SlamNode::OnImage, this);

        srvReset_ = pnh_.advertiseService("reset", &SlamNode::OnReset, this);
        srvResetMap_ = pnh_.advertiseService("reset_active_map", &SlamNode::OnResetMap, this);
        srvLoc_ = pnh_.advertiseService("localization_mode", &SlamNode::OnLocalization, this);
        timer_ = nh_.createTimer(ros::Duration(1.0), &SlamNode::OnTimer, this);
        ROS_INFO("UC-SLAM: running (%s-inertial, depth %s, domain %s)", o.stereo ? "stereo" : "mono",
                 depthSource.c_str(), domain_.c_str());
        return true;
    }

    // After ros::shutdown: stop the worker, shut the System down and save the trajectories.
    void Stop()
    {
        stopping_ = true;
        if(core_)
        {
            ROS_INFO("UC-SLAM: shutting down, saving f_%s.txt and kf_%s.txt", trajName_.c_str(), trajName_.c_str());
            core_->Stop(trajName_);
        }
    }

private:
    bool Fail(const std::string& msg)
    {
        ROS_FATAL("UC-SLAM: %s", msg.c_str());
        return false;
    }

    static cv::Mat ToMono(const sensor_msgs::ImageConstPtr& msg)
    {
        try
        {
            // 16-bit images (TUM-VI bags): keep the high byte, as cv::imread with IMREAD_GRAYSCALE does for the
            // benchmark examples and the depth cache.
            if(msg->encoding == "mono16" || msg->encoding == "16UC1")
                return HighByte(cv_bridge::toCvShare(msg)->image);
            return cv_bridge::toCvCopy(msg, "mono8")->image;
        }
        catch(const cv_bridge::Exception& e)
        {
            ROS_ERROR_THROTTLE(5.0, "UC-SLAM: image conversion failed: %s", e.what());
            return cv::Mat();
        }
    }

    void OnImu(const sensor_msgs::ImuConstPtr& m)
    {
        core_->PushImu((int64_t)m->header.stamp.toNSec(),
                       Eigen::Vector3f(m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z),
                       Eigen::Vector3f(m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z));
    }

    void OnImage(const sensor_msgs::ImageConstPtr& m)
    {
        cv::Mat im = ToMono(m);
        if(!im.empty())
            core_->PushFrame((int64_t)m->header.stamp.toNSec(), im);
    }

    void OnStereo(const sensor_msgs::ImageConstPtr& l, const sensor_msgs::ImageConstPtr& r)
    {
        cv::Mat a = ToMono(l), b = ToMono(r);
        if(!a.empty() && !b.empty())
            core_->PushFrame((int64_t)l->header.stamp.toNSec(), a, b);
    }

    void OnDepthPrior(const ucslam_ros1::DepthPriorConstPtr& m)
    {
        // The per-domain constants were fitted for one camera domain and one network backend.
        if(m->domain != domain_ || m->backend != backend_)
        {
            refusedPriors_++;
            ROS_WARN_ONCE("UC-SLAM: refused a depth prior with domain '%s' and backend '%s' (expected '%s' and '%s')",
                          m->domain.c_str(), m->backend.c_str(), domain_.c_str(), backend_.c_str());
            return;
        }
        const sensor_msgs::Image& d = m->depth;
        if(d.encoding != "32FC1" || d.height == 0 || d.width == 0 || d.step < d.width * 4 ||
           d.data.size() < (size_t)d.step * d.height || d.is_bigendian)
        {
            refusedPriors_++;
            ROS_WARN_ONCE("UC-SLAM: refused a depth prior that is not a little-endian 32FC1 image");
            return;
        }
        cv::Mat z(d.height, d.width, CV_32F);
        for(uint32_t r = 0; r < d.height; r++)
            std::memcpy(z.ptr<float>(r), &d.data[(size_t)r * d.step], (size_t)d.width * 4);
        ORB_SLAM3::PriorNet::DepthView view;
        view.kind = m->view;
        view.fullW = (int)m->full_width;
        view.fullH = (int)m->full_height;
        for(int i = 0; i < 4; i++)
            view.K[i] = m->k_view[i];
        view.model = m->model;
        core_->PushDepth((int64_t)m->header.stamp.toNSec(), z, view);
    }

    void OnDepthRequest(double, int64_t ns, unsigned long id, const cv::Mat& img)
    {
        if(stopping_ || !ros::ok())
            return;
        ucslam_ros1::DepthRequest req;
        req.header.stamp.fromNSec((uint64_t)ns);
        req.kf_id = id;
        std_msgs::Header h;
        h.stamp = req.header.stamp;
        cv_bridge::CvImage(h, "mono8", img).toImageMsg(req.image);
        pubRequest_.publish(req);
    }

    void OnResult(const ucslam_node::Result& r)
    {
        {
            std::lock_guard<std::mutex> lock(mStatus_);
            lastState_ = r.state;
            imuInit_ = r.imu_initialized;
            timeFromInit_ = r.time_from_imu_init;
            trackMs_.push_back(r.track_ms);
            if(r.map_changed)
                mapChanges_++;
        }
        // No pose before IMU init (no metric scale, no gravity) and none while lost.
        if(stopping_ || !ros::ok() || !r.imu_initialized || (r.state != 2 && r.state != 3))
            return;
        ros::Time stamp;
        stamp.fromNSec((uint64_t)r.stamp_ns);
        const Eigen::Quaternionf q = r.Twb.unit_quaternion();
        const Eigen::Vector3f p = r.Twb.translation();

        geometry_msgs::PoseStamped ps;
        ps.header.stamp = stamp;
        ps.header.frame_id = mapFrame_;
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        ps.pose.position.z = p.z();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        ps.pose.orientation.w = q.w();
        pubPose_.publish(ps);

        nav_msgs::Odometry od;
        od.header = ps.header;
        od.child_frame_id = bodyFrame_;
        od.pose.pose = ps.pose;
        // A fixed diagonal: the system gives no per-frame covariance. RECENTLY_LOST inflates it.
        const double k = (r.state == 3) ? 100.0 : 1.0;
        for(int i = 0; i < 3; i++)
        {
            od.pose.covariance[i * 7] = k * poseStdM_ * poseStdM_;
            od.pose.covariance[(i + 3) * 7] = k * poseStdRad_ * poseStdRad_;
        }
        od.twist.covariance[0] = -1.0;   // twist not provided
        pubOdom_.publish(od);

        if(publishTf_)
        {
            geometry_msgs::TransformStamped tf;
            tf.header = ps.header;
            tf.child_frame_id = bodyFrame_;
            tf.transform.translation.x = p.x();
            tf.transform.translation.y = p.y();
            tf.transform.translation.z = p.z();
            tf.transform.rotation = ps.pose.orientation;
            tfb_.sendTransform(tf);
        }

        if(publishPoints_)
        {
            sensor_msgs::PointCloud2 pc;
            pc.header = ps.header;
            sensor_msgs::PointCloud2Modifier mod(pc);
            mod.setPointCloud2FieldsByString(1, "xyz");
            mod.resize(r.points.size());
            if(!r.points.empty())
            {
                sensor_msgs::PointCloud2Iterator<float> x(pc, "x"), y(pc, "y"), z(pc, "z");
                for(size_t i = 0; i < r.points.size(); i++, ++x, ++y, ++z)
                {
                    *x = r.points[i].x();
                    *y = r.points[i].y();
                    *z = r.points[i].z();
                }
            }
            pubPoints_.publish(pc);
        }
    }

    void OnTimer(const ros::TimerEvent&)
    {
        diagnostic_msgs::DiagnosticStatus s;
        s.name = "ucslam";
        s.hardware_id = "ucslam";
        int state;
        bool init;
        double fromInit, p50, p90;
        unsigned long changes;
        {
            std::lock_guard<std::mutex> lock(mStatus_);
            state = lastState_;
            init = imuInit_;
            fromInit = timeFromInit_;
            p50 = Percentile(trackMs_, 0.5);
            p90 = Percentile(trackMs_, 0.9);
            trackMs_.clear();
            changes = mapChanges_;
        }
        // Scale and gravity can still move until the last inertial refinement of the base system.
        const bool settling = init && fromInit < ucslam_node::SettlingMapTime(stereo_);
        if(state == 2 && init && !settling)
        {
            s.level = diagnostic_msgs::DiagnosticStatus::OK;
            s.message = "tracking";
        }
        else if(state == 2 || state == 3 || state == 1)
        {
            s.level = diagnostic_msgs::DiagnosticStatus::WARN;
            s.message = !init ? "waiting for IMU initialization" : state == 3 ? "recently lost" : "settling";
        }
        else
        {
            s.level = diagnostic_msgs::DiagnosticStatus::ERROR;
            s.message = state == 4 ? "lost" : "no images yet";
        }
        const ucslam_node::Counters& c = core_->counters;
        Add(s, "state", std::to_string(state));
        Add(s, "imu_initialized", init ? "true" : "false");
        Add(s, "map_time_s", std::to_string(fromInit));
        Add(s, "settling", settling ? "true" : "false");
        Add(s, "map_changes", std::to_string(changes));
        Add(s, "frames", std::to_string(c.frames.load()));
        Add(s, "dropped_frames", std::to_string(c.dropped_frames.load()));
        Add(s, "old_frames", std::to_string(c.old_frames.load()));
        Add(s, "imu_rejected", std::to_string(c.imu_rejected.load()));
        Add(s, "imu_wait_timeouts", std::to_string(c.imu_wait_timeouts.load()));
        Add(s, "depth_requests", std::to_string(c.depth_requests.load()));
        Add(s, "depth_pushed", std::to_string(c.depth_pushed.load()));
        Add(s, "depth_refused", std::to_string(c.depth_refused.load() + refusedPriors_.load()));
        Add(s, "track_ms_p50", std::to_string(p50));
        Add(s, "track_ms_p90", std::to_string(p90));
        diagnostic_msgs::DiagnosticArray a;
        a.header.stamp = ros::Time::now();
        a.status.push_back(s);
        pubStatus_.publish(a);
    }

    static void Add(diagnostic_msgs::DiagnosticStatus& s, const std::string& k, const std::string& v)
    {
        diagnostic_msgs::KeyValue kv;
        kv.key = k;
        kv.value = v;
        s.values.push_back(kv);
    }

    bool OnReset(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res)
    {
        core_->RequestReset();
        res.success = true;
        res.message = "reset of the whole Atlas requested, runs before the next frame";
        return true;
    }

    bool OnResetMap(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res)
    {
        core_->RequestResetActiveMap();
        res.success = true;
        res.message = "reset of the active map requested, runs before the next frame";
        return true;
    }

    bool OnLocalization(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res)
    {
        core_->SetLocalizationMode(req.data);
        res.success = true;
        res.message = req.data ? "localization only mode requested" : "SLAM mode requested";
        return true;
    }

    ros::NodeHandle nh_, pnh_;
    std::unique_ptr<ucslam_node::NodeCore> core_;
    std::string mapFrame_, bodyFrame_, trajName_, domain_, backend_;
    bool publishTf_ = true, publishPoints_ = false, stereo_ = false;
    double poseStdM_ = 0.01, poseStdRad_ = 0.01;
    std::atomic<bool> stopping_{false};
    std::atomic<unsigned long> refusedPriors_{0};

    ros::Publisher pubOdom_, pubPose_, pubStatus_, pubPoints_, pubRequest_;
    ros::Subscriber subImu_, subImage_, subDepth_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image> > subLeft_, subRight_;
    std::unique_ptr<message_filters::Synchronizer<StereoPolicy> > sync_;
    ros::ServiceServer srvReset_, srvResetMap_, srvLoc_;
    ros::Timer timer_;
    tf2_ros::TransformBroadcaster tfb_;

    std::mutex mStatus_;
    int lastState_ = -1;
    bool imuInit_ = false;
    double timeFromInit_ = 0.0;
    std::vector<double> trackMs_;
    unsigned long mapChanges_ = 0;
};

}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "slam_node");
    ros::NodeHandle nh, pnh("~");
    SlamNode node(nh, pnh);
    if(!node.Init())
        return 1;
    // Callbacks of IMU, images, depth priors and services run in parallel. Track* runs in the core's worker.
    ros::AsyncSpinner spinner(4);
    spinner.start();
    ros::waitForShutdown();
    spinner.stop();
    node.Stop();
    return 0;
}
