/**
 * UC-SLAM SLAM node for ROS 2 (Humble). The SLAM logic lives in ros/common/ucslam_node_core.h, shared with the
 * ROS 1 node. This file converts messages, publishes the results and serves the services.
 *
 * In:  image (mono8), image_right (stereo only), imu, depth_prior (live depth from depth_node)
 * Out: ~/odom, ~/pose, tf map_frame -> body_frame, ~/tracked_points (optional), ~/status (1 Hz), depth_request
 * Services: ~/reset, ~/reset_active_map (std_srvs/Trigger), ~/localization_mode (std_srvs/SetBool)
 * See ros/README.md for the parameters.
 */
#include "ucslam_node_core.h"

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <ucslam_msgs/msg/depth_prior.hpp>
#include <ucslam_msgs/msg/depth_request.hpp>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace
{

using sensor_msgs::msg::Image;
typedef message_filters::sync_policies::ExactTime<Image, Image> StereoPolicy;

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

int64_t StampNs(const builtin_interfaces::msg::Time& t)
{
    return (int64_t)t.sec * 1000000000LL + (int64_t)t.nanosec;
}

builtin_interfaces::msg::Time ToStamp(int64_t ns)
{
    builtin_interfaces::msg::Time t;
    t.sec = (int32_t)(ns / 1000000000LL);
    t.nanosec = (uint32_t)(ns % 1000000000LL);
    return t;
}

class SlamNode : public rclcpp::Node
{
public:
    SlamNode() : rclcpp::Node("slam_node") {}

    bool Init()
    {
        ucslam_node::Options o;
        o.vocabulary = declare_parameter<std::string>("vocabulary", "");
        std::string settings = declare_parameter<std::string>("settings", "");
        o.stereo = declare_parameter<bool>("stereo", false);
        o.viewer = declare_parameter<bool>("viewer", false);
        o.imu_wait_ms = declare_parameter<double>("imu_wait_ms", 50.0);
        o.imu_time_offset_s = declare_parameter<double>("imu_time_offset", 0.0);
        const std::string frameDrop = declare_parameter<std::string>("frame_drop", "oldest");
        o.frame_queue = (size_t)std::max<int64_t>(1, declare_parameter<int64_t>("frame_queue", 3));
        o.image_ring_s = declare_parameter<double>("image_ring_s", 2.0);
        o.clahe = declare_parameter<bool>("clahe", false);
        publishPoints_ = declare_parameter<bool>("publish_points", false);
        o.collect_points = publishPoints_;
        o.output_dir = declare_parameter<std::string>("output_dir", "");
        trajName_ = declare_parameter<std::string>("trajectory_name", "ucslam");
        mapFrame_ = declare_parameter<std::string>("map_frame", "map");
        bodyFrame_ = declare_parameter<std::string>("body_frame", "imu");
        publishTf_ = declare_parameter<bool>("publish_tf", true);
        const std::string depthSource = declare_parameter<std::string>("depth_source", "live");
        std::string depthDir = declare_parameter<std::string>("depth_dir", "");
        backend_ = declare_parameter<std::string>("depth_backend", "torch-fp32");
        const double depthWaitMs = declare_parameter<double>("depth_wait_ms", 2000.0);
        const double dppWaitMs = declare_parameter<double>("dpp_wait_ms", 0.0);
        const int64_t dppDeferMaxKFs = declare_parameter<int64_t>("dpp_defer_max_kfs", 0);
        poseStdM_ = declare_parameter<double>("pose_std_m", 0.01);
        poseStdRad_ = declare_parameter<double>("pose_std_rad", 0.01);
        const bool imuReliable = declare_parameter<bool>("imu_qos_reliable", false);

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

        const rclcpp::QoS reliable10 = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
        pubOdom_ = create_publisher<nav_msgs::msg::Odometry>("~/odom", reliable10);
        pubPose_ = create_publisher<geometry_msgs::msg::PoseStamped>("~/pose", reliable10);
        pubStatus_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("~/status", reliable10);
        if(publishPoints_)
            pubPoints_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/tracked_points", rclcpp::QoS(1).best_effort());
        pubRequest_ = create_publisher<ucslam_msgs::msg::DepthRequest>("depth_request", rclcpp::QoS(16).reliable());
        tfb_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        core_.reset(new ucslam_node::NodeCore(o));
        RCLCPP_INFO(get_logger(), "UC-SLAM: loading the vocabulary and starting the system (settings %s)", o.settings.c_str());
        err = core_->Start([this](const ucslam_node::Result& r) { OnResult(r); },
                           [this](double t, int64_t ns, unsigned long id, const cv::Mat& img) { OnDepthRequest(t, ns, id, img); });
        if(!err.empty())
            return Fail(err);

        // Subscriptions only after the System exists. One callback group per input, so a slow image conversion
        // never delays the IMU queue.
        cbImu_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cbImage_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cbDepth_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cbSrv_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions soImu, soImage, soDepth;
        soImu.callback_group = cbImu_;
        soImage.callback_group = cbImage_;
        soDepth.callback_group = cbDepth_;
        rclcpp::QoS imuQos = rclcpp::QoS(rclcpp::KeepLast(2000));
        if(imuReliable)
            imuQos.reliable();
        else
            imuQos.best_effort();   // matches a best-effort and a reliable publisher
        subImu_ = create_subscription<sensor_msgs::msg::Imu>("imu", imuQos,
            [this](sensor_msgs::msg::Imu::ConstSharedPtr m) { OnImu(m); }, soImu);
        subDepth_ = create_subscription<ucslam_msgs::msg::DepthPrior>("depth_prior", rclcpp::QoS(16).reliable(),
            [this](ucslam_msgs::msg::DepthPrior::ConstSharedPtr m) { OnDepthPrior(m); }, soDepth);
        if(o.stereo)
        {
            subLeft_.subscribe(this, "image", rmw_qos_profile_sensor_data);
            subRight_.subscribe(this, "image_right", rmw_qos_profile_sensor_data);
            sync_ = std::make_shared<message_filters::Synchronizer<StereoPolicy> >(StereoPolicy(10), subLeft_, subRight_);
            sync_->registerCallback(std::bind(&SlamNode::OnStereo, this, std::placeholders::_1, std::placeholders::_2));
        }
        else
            subImage_ = create_subscription<Image>("image", rclcpp::SensorDataQoS(),
                [this](Image::ConstSharedPtr m) { OnImage(m); }, soImage);

        srvReset_ = create_service<std_srvs::srv::Trigger>("~/reset",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>, std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                core_->RequestReset();
                res->success = true;
                res->message = "reset of the whole Atlas requested, runs before the next frame";
            }, rmw_qos_profile_services_default, cbSrv_);
        srvResetMap_ = create_service<std_srvs::srv::Trigger>("~/reset_active_map",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>, std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                core_->RequestResetActiveMap();
                res->success = true;
                res->message = "reset of the active map requested, runs before the next frame";
            }, rmw_qos_profile_services_default, cbSrv_);
        srvLoc_ = create_service<std_srvs::srv::SetBool>("~/localization_mode",
            [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> req, std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
                core_->SetLocalizationMode(req->data);
                res->success = true;
                res->message = req->data ? "localization only mode requested" : "SLAM mode requested";
            }, rmw_qos_profile_services_default, cbSrv_);
        timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { OnTimer(); }, cbSrv_);
        RCLCPP_INFO(get_logger(), "UC-SLAM: running (%s-inertial, depth %s, domain %s)", o.stereo ? "stereo" : "mono",
                    depthSource.c_str(), domain_.c_str());
        return true;
    }

    // After rclcpp::shutdown: stop the worker, shut the System down and save the trajectories.
    void Stop()
    {
        stopping_ = true;
        if(core_)
        {
            RCLCPP_INFO(get_logger(), "UC-SLAM: shutting down, saving f_%s.txt and kf_%s.txt", trajName_.c_str(), trajName_.c_str());
            core_->Stop(trajName_);
        }
    }

private:
    bool Fail(const std::string& msg)
    {
        RCLCPP_FATAL(get_logger(), "UC-SLAM: %s", msg.c_str());
        return false;
    }

    cv::Mat ToMono(const Image::ConstSharedPtr& msg)
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
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "UC-SLAM: image conversion failed: %s", e.what());
            return cv::Mat();
        }
    }

    void OnImu(const sensor_msgs::msg::Imu::ConstSharedPtr& m)
    {
        core_->PushImu(StampNs(m->header.stamp),
                       Eigen::Vector3f(m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z),
                       Eigen::Vector3f(m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z));
    }

    void OnImage(const Image::ConstSharedPtr& m)
    {
        cv::Mat im = ToMono(m);
        if(!im.empty())
            core_->PushFrame(StampNs(m->header.stamp), im);
    }

    void OnStereo(const Image::ConstSharedPtr& l, const Image::ConstSharedPtr& r)
    {
        cv::Mat a = ToMono(l), b = ToMono(r);
        if(!a.empty() && !b.empty())
            core_->PushFrame(StampNs(l->header.stamp), a, b);
    }

    void OnDepthPrior(const ucslam_msgs::msg::DepthPrior::ConstSharedPtr& m)
    {
        // The per-domain constants were fitted for one camera domain and one network backend.
        if(m->domain != domain_ || m->backend != backend_)
        {
            refusedPriors_++;
            if(!warnedDomain_.exchange(true))
                RCLCPP_WARN(get_logger(), "UC-SLAM: refused a depth prior with domain '%s' and backend '%s' (expected '%s' and '%s')",
                            m->domain.c_str(), m->backend.c_str(), domain_.c_str(), backend_.c_str());
            return;
        }
        const Image& d = m->depth;
        if(d.encoding != "32FC1" || d.height == 0 || d.width == 0 || d.step < d.width * 4 ||
           d.data.size() < (size_t)d.step * d.height || d.is_bigendian)
        {
            refusedPriors_++;
            if(!warnedFormat_.exchange(true))
                RCLCPP_WARN(get_logger(), "UC-SLAM: refused a depth prior that is not a little-endian 32FC1 image");
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
        core_->PushDepth(StampNs(m->header.stamp), z, view);
    }

    void OnDepthRequest(double, int64_t ns, unsigned long id, const cv::Mat& img)
    {
        if(stopping_ || !rclcpp::ok())
            return;
        ucslam_msgs::msg::DepthRequest req;
        req.header.stamp = ToStamp(ns);
        req.kf_id = id;
        std_msgs::msg::Header h;
        h.stamp = req.header.stamp;
        cv_bridge::CvImage(h, "mono8", img).toImageMsg(req.image);
        pubRequest_->publish(req);
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
        if(stopping_ || !rclcpp::ok() || !r.imu_initialized || (r.state != 2 && r.state != 3))
            return;
        const Eigen::Quaternionf q = r.Twb.unit_quaternion();
        const Eigen::Vector3f p = r.Twb.translation();

        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = ToStamp(r.stamp_ns);
        ps.header.frame_id = mapFrame_;
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        ps.pose.position.z = p.z();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        ps.pose.orientation.w = q.w();
        pubPose_->publish(ps);

        nav_msgs::msg::Odometry od;
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
        pubOdom_->publish(od);

        if(publishTf_)
        {
            geometry_msgs::msg::TransformStamped tf;
            tf.header = ps.header;
            tf.child_frame_id = bodyFrame_;
            tf.transform.translation.x = p.x();
            tf.transform.translation.y = p.y();
            tf.transform.translation.z = p.z();
            tf.transform.rotation = ps.pose.orientation;
            tfb_->sendTransform(tf);
        }

        if(publishPoints_)
        {
            sensor_msgs::msg::PointCloud2 pc;
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
            pubPoints_->publish(pc);
        }
    }

    void OnTimer()
    {
        diagnostic_msgs::msg::DiagnosticStatus s;
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
            s.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
            s.message = "tracking";
        }
        else if(state == 2 || state == 3 || state == 1)
        {
            s.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            s.message = !init ? "waiting for IMU initialization" : state == 3 ? "recently lost" : "settling";
        }
        else
        {
            s.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
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
        diagnostic_msgs::msg::DiagnosticArray a;
        a.header.stamp = now();
        a.status.push_back(s);
        pubStatus_->publish(a);
    }

    static void Add(diagnostic_msgs::msg::DiagnosticStatus& s, const std::string& k, const std::string& v)
    {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = k;
        kv.value = v;
        s.values.push_back(kv);
    }

    std::unique_ptr<ucslam_node::NodeCore> core_;
    std::string mapFrame_, bodyFrame_, trajName_, domain_, backend_;
    bool publishTf_ = true, publishPoints_ = false, stereo_ = false;
    double poseStdM_ = 0.01, poseStdRad_ = 0.01;
    std::atomic<bool> stopping_{false}, warnedDomain_{false}, warnedFormat_{false};
    std::atomic<unsigned long> refusedPriors_{0};

    rclcpp::CallbackGroup::SharedPtr cbImu_, cbImage_, cbDepth_, cbSrv_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdom_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pubPose_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pubStatus_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPoints_;
    rclcpp::Publisher<ucslam_msgs::msg::DepthRequest>::SharedPtr pubRequest_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu_;
    rclcpp::Subscription<Image>::SharedPtr subImage_;
    rclcpp::Subscription<ucslam_msgs::msg::DepthPrior>::SharedPtr subDepth_;
    message_filters::Subscriber<Image> subLeft_, subRight_;
    std::shared_ptr<message_filters::Synchronizer<StereoPolicy> > sync_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srvReset_, srvResetMap_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srvLoc_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfb_;

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
    rclcpp::init(argc, argv);
    std::shared_ptr<SlamNode> node = std::make_shared<SlamNode>();
    if(!node->Init())
    {
        rclcpp::shutdown();
        return 1;
    }
    // Callbacks of IMU, images, depth priors and services run in parallel. Track* runs in the core's worker.
    rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
    exec.add_node(node);
    exec.spin();
    node->Stop();
    rclcpp::shutdown();
    return 0;
}
