/**
 * Image message to 8-bit gray, without cv_bridge and without ROS. Used by the ROS 2 slam_node.
 *
 * The input is the content of a sensor_msgs/Image: encoding, width, height, step, is_bigendian and the data bytes.
 * The output is a CV_8UC1 cv::Mat that owns its data, so it outlives the message (the node keeps frames in a queue
 * and a ring). The gray values are the ones the node got from cv_bridge before:
 *  - mono8, 8UC1: a copy. (cv_bridge refused 8UC1 to mono8, now it is taken as gray.)
 *  - mono16, 16UC1: byte swap when the data is big endian, then the high byte of each pixel. This is what
 *    cv::imread with IMREAD_GRAYSCALE does to a 16-bit PNG, the input of the benchmark examples and of the depth cache.
 *  - rgb8, bgr8, rgba8, bgra8: cv::cvtColor with RGB2GRAY, BGR2GRAY, RGBA2GRAY, BGRA2GRAY, as cv_bridge does.
 *  - yuv422 (UYVY), yuv422_yuy2 (YUYV): cv::cvtColor with YUV2GRAY_UYVY and YUV2GRAY_YUY2, as cv_bridge does.
 * Any other encoding, a step shorter than one row, a 16-bit image with an odd step, or too few data bytes give an
 * empty Mat and an error message. ToMono8 never throws: an OpenCV or allocation error also gives an empty Mat.
 */
#ifndef UCSLAM_IMAGE_CONVERT_H
#define UCSLAM_IMAGE_CONVERT_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

namespace ucslam_node
{

// High byte of every pixel of a CV_16UC1 image (values in native byte order).
inline cv::Mat HighByte(const cv::Mat& im16)
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

inline bool HostIsBigEndian()
{
    const uint16_t one = 1;
    return *reinterpret_cast<const uint8_t*>(&one) == 0;
}

// See the file comment. err (optional) gets the reason of an empty result.
inline cv::Mat ToMono8(const std::string& encoding, uint32_t width, uint32_t height, uint32_t step,
                       bool is_bigendian, const uint8_t* data, size_t data_size, std::string* err = nullptr)
{
    int type = -1, code = -1;
    if(encoding == "mono8" || encoding == "8UC1")
        type = CV_8UC1;
    else if(encoding == "mono16" || encoding == "16UC1")
        type = CV_16UC1;
    else if(encoding == "rgb8")
    {
        type = CV_8UC3;
        code = cv::COLOR_RGB2GRAY;
    }
    else if(encoding == "bgr8")
    {
        type = CV_8UC3;
        code = cv::COLOR_BGR2GRAY;
    }
    else if(encoding == "rgba8")
    {
        type = CV_8UC4;
        code = cv::COLOR_RGBA2GRAY;
    }
    else if(encoding == "bgra8")
    {
        type = CV_8UC4;
        code = cv::COLOR_BGRA2GRAY;
    }
    else if(encoding == "yuv422")
    {
        type = CV_8UC2;
        code = cv::COLOR_YUV2GRAY_UYVY;
    }
    else if(encoding == "yuv422_yuy2")
    {
        type = CV_8UC2;
        code = cv::COLOR_YUV2GRAY_YUY2;
    }
    if(type < 0)
    {
        if(err)
            *err = "unsupported image encoding '" + encoding +
                   "' (mono8, 8UC1, mono16, 16UC1, rgb8, bgr8, rgba8, bgra8, yuv422, yuv422_yuy2)";
        return cv::Mat();
    }
    const size_t rowBytes = (size_t)width * CV_ELEM_SIZE(type);
    // cv::Mat needs a step that is a multiple of the channel size (2 bytes for 16-bit) and sizes that fit an int.
    const size_t maxSide = 1u << 20;
    if(width == 0 || height == 0 || width > maxSide || height > maxSide || data == nullptr || step < rowBytes ||
       step % CV_ELEM_SIZE1(type) != 0 || data_size < (size_t)step * height)
    {
        if(err)
            *err = "bad image size: " + std::to_string(width) + "x" + std::to_string(height) + " " + encoding +
                   ", step " + std::to_string(step) + ", " + std::to_string(data_size) + " data bytes";
        return cv::Mat();
    }
    try
    {
        // A header on the message data. Every branch below writes into a new Mat, so the result owns its data.
        const cv::Mat in((int)height, (int)width, type, const_cast<uint8_t*>(data), (size_t)step);
        if(type == CV_8UC1)
            return in.clone();
        if(type == CV_16UC1)
        {
            if(is_bigendian == HostIsBigEndian())
                return HighByte(in);
            cv::Mat sw((int)height, (int)width, CV_16UC1);
            for(uint32_t r = 0; r < height; r++)
            {
                const uint16_t* a = in.ptr<uint16_t>((int)r);
                uint16_t* b = sw.ptr<uint16_t>((int)r);
                for(uint32_t c = 0; c < width; c++)
                    b[c] = (uint16_t)((a[c] >> 8) | (a[c] << 8));
            }
            return HighByte(sw);
        }
        cv::Mat out;
        cv::cvtColor(in, out, code);
        return out;
    }
    catch(const std::exception& e)   // cv::Exception, std::bad_alloc
    {
        if(err)
            *err = "conversion of a " + std::to_string(width) + "x" + std::to_string(height) + " " + encoding +
                   " image failed: " + e.what();
        return cv::Mat();
    }
}

}  // namespace ucslam_node

#endif  // UCSLAM_IMAGE_CONVERT_H
