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

#include "PriorNet.h"

#include <opencv2/core/persistence.hpp>

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace std;

namespace ORB_SLAM3
{

namespace
{

bool IsDir(const string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool IsFile(const string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

vector<string> ListDir(const string& dir)
{
    vector<string> out;
    DIR* d = opendir(dir.c_str());
    if(!d)
        return out;
    while(struct dirent* e = readdir(d))
    {
        string name(e->d_name);
        if(name != "." && name != "..")
            out.push_back(name);
    }
    closedir(d);
    sort(out.begin(), out.end());
    return out;
}

double NowMsSince(const chrono::steady_clock::time_point& t0)
{
    return chrono::duration_cast<chrono::duration<double, milli> >(chrono::steady_clock::now() - t0).count();
}

// Bilinear read of a CV_32F map at (x, y). The caller checks the bounds.
float BilinearMat(const cv::Mat& m, float x, float y)
{
    int x0 = (int)floor(x), y0 = (int)floor(y);
    int x1 = min(x0 + 1, m.cols - 1), y1 = min(y0 + 1, m.rows - 1);
    x0 = max(x0, 0); y0 = max(y0, 0);
    float fx = x - x0, fy = y - y0;
    return (1 - fx) * (1 - fy) * m.at<float>(y0, x0) + fx * (1 - fy) * m.at<float>(y0, x1)
         + (1 - fx) * fy * m.at<float>(y1, x0) + fx * fy * m.at<float>(y1, x1);
}

} // namespace

PriorNet::PriorNet(const Settings::Ucslam& params, const cv::Size& workImSize,
                   const cv::Mat& M1l, const cv::Mat& M2l):
    mParams(params), mWorkImSize(workImSize), mM1l(M1l), mM2l(M2l),
    mbRectified(!M1l.empty() && !M2l.empty()),
    mbLocated(false), mbGaveUp(false), mView(VIEW_NONE),
    mnFullW(0), mnFullH(0), mnMapW(0), mnMapH(0), mnQueueMax(0),
    mbFinishRequested(false), mbFinished(false),
    mnReceived(0), mnWithDepth(0), mnNoMap(0), mnNoSequence(0), mnBadMap(0),
    mbWarnedNoMap(false), mbWarnedNoSequence(false), mbWarnedShape(false)
{
    mKview[0] = mKview[1] = mKview[2] = mKview[3] = 0.f;
    if(mbRectified && (mM1l.type() != CV_32F || mM2l.type() != CV_32F))
    {
        cerr << "[UCSLAM] PriorNet: rectification maps are not CV_32F, ignoring them" << endl;
        mbRectified = false;
    }
}

// ------------------------------------------------------------------ queue --

void PriorNet::InsertKeyFrame(KeyFrame* pKF)
{
    {
        unique_lock<mutex> lock(mMutexQueue);
        mlQueue.push_back(pKF);
        mnQueueMax = max(mnQueueMax, mlQueue.size());
    }
    {
        unique_lock<mutex> lock(mMutexStats);
        mnReceived++;
    }
    mCondQueue.notify_one();
}

int PriorNet::KeyframesInQueue()
{
    unique_lock<mutex> lock(mMutexQueue);
    return (int)mlQueue.size();
}

void PriorNet::RequestFinish()
{
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }
    mCondQueue.notify_all();
}

bool PriorNet::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

bool PriorNet::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void PriorNet::Run()
{
    while(true)
    {
        KeyFrame* pKF = NULL;
        {
            unique_lock<mutex> lock(mMutexQueue);
            while(mlQueue.empty() && !CheckFinish())
                mCondQueue.wait_for(lock, chrono::milliseconds(20));
            if(mlQueue.empty())
                break;                        // finish was requested and the queue is drained
            pKF = mlQueue.front();
            mlQueue.pop_front();
        }
        ProcessKeyFrame(pKF);
    }
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

// ------------------------------------------------------------ sequence dir --

bool PriorNet::ListStamps(const string& dir, vector<long long>& stamps)
{
    stamps.clear();
    for(const string& name : ListDir(dir))
    {
        if(name.size() < 5 || name.compare(name.size() - 4, 4, ".npy") != 0)
            continue;
        char* end = NULL;
        long long v = strtoll(name.c_str(), &end, 10);
        if(end && string(end) == ".npy")
            stamps.push_back(v);
    }
    sort(stamps.begin(), stamps.end());
    return !stamps.empty();
}

bool PriorNet::NearestStamp(long long stampNs, long long& found) const
{
    if(mvStamps.empty())
        return false;
    vector<long long>::const_iterator it = lower_bound(mvStamps.begin(), mvStamps.end(), stampNs);
    long long best = 0;
    bool have = false;
    if(it != mvStamps.end()) { best = *it; have = true; }
    if(it != mvStamps.begin())
    {
        long long prev = *(it - 1);
        if(!have || llabs(prev - stampNs) < llabs(best - stampNs))
            best = prev;
        have = true;
    }
    if(!have)
        return false;
    const double tolNs = (double)mParams.depthMatchTolMs * 1e6;
    if((double)llabs(best - stampNs) > tolNs)
        return false;
    found = best;
    return true;
}

bool PriorNet::ReadManifest(const string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        cerr << "[UCSLAM] PriorNet: cannot read " << path << endl;
        return false;
    }
    string view = (string)fs["view"];
    string model = (string)fs["model"];
    if(!mParams.depthModel.empty() && model != mParams.depthModel)
    {
        cerr << "[UCSLAM] PriorNet: manifest model " << model << " differs from UCSLAM.DepthModel "
             << mParams.depthModel << endl;
        return false;
    }
    cv::FileNode fullSize = fs["full_size"], mapSize = fs["map_size"];
    if(fullSize.size() != 2 || mapSize.size() != 2)
    {
        cerr << "[UCSLAM] PriorNet: manifest without full_size or map_size: " << path << endl;
        return false;
    }
    mnFullW = (int)fullSize[0]; mnFullH = (int)fullSize[1];
    mnMapW = (int)mapSize[0];   mnMapH = (int)mapSize[1];
    if(view == "camera")
        mView = VIEW_CAMERA;
    else if(view == "pinhole")
    {
        cv::FileNode K = fs["K_view"];
        if(K.size() != 4)
        {
            cerr << "[UCSLAM] PriorNet: pinhole view without K_view: " << path << endl;
            return false;
        }
        for(int i = 0; i < 4; i++)
            mKview[i] = (float)(double)K[i];
        mView = VIEW_PINHOLE;
    }
    else
    {
        cerr << "[UCSLAM] PriorNet: unknown view '" << view << "' in " << path << endl;
        return false;
    }
    return mnFullW > 0 && mnFullH > 0 && mnMapW > 0 && mnMapH > 0;
}

bool PriorNet::LocateSequence(long long stampNs)
{
    if(mParams.depthDir.empty() || mParams.depthModel.empty())
        return false;
    vector<string> roots;
    if(!mParams.dataset.empty())
        roots.push_back(mParams.depthDir + "/" + mParams.dataset);
    else
        for(const string& ds : ListDir(mParams.depthDir))
            if(IsDir(mParams.depthDir + "/" + ds))
                roots.push_back(mParams.depthDir + "/" + ds);
    for(const string& root : roots)
    {
        for(const string& seq : ListDir(root))
        {
            const string dir = root + "/" + seq + "/" + mParams.depthModel;
            if(!IsDir(dir) || !IsFile(dir + "/manifest.json"))
                continue;
            vector<long long> stamps;
            if(!ListStamps(dir, stamps))
                continue;
            // Cheap range test first, then the exact nearest-stamp test.
            const long long tolNs = (long long)(mParams.depthMatchTolMs * 1e6);
            if(stampNs < stamps.front() - tolNs || stampNs > stamps.back() + tolNs)
                continue;
            mvStamps.swap(stamps);
            long long found;
            if(!NearestStamp(stampNs, found))
            {
                mvStamps.clear();
                continue;
            }
            if(!ReadManifest(dir + "/manifest.json"))
            {
                mvStamps.clear();
                return false;
            }
            mSeqDir = dir;
            cout << "[UCSLAM] PriorNet: depth cache " << dir << " (" << mvStamps.size() << " maps, view "
                 << (mView == VIEW_PINHOLE ? "pinhole" : "camera") << ", full " << mnFullW << "x" << mnFullH
                 << ", map " << mnMapW << "x" << mnMapH << ", working image " << mWorkImSize.width << "x"
                 << mWorkImSize.height << (mbRectified ? ", rectified keypoints" : "") << ")" << endl;
            return true;
        }
    }
    return false;
}

string PriorNet::SequenceDir()
{
    unique_lock<mutex> lock(mMutexStats);
    return mSeqDir;
}

long long PriorNet::KeyframesWithDepth()
{
    unique_lock<mutex> lock(mMutexStats);
    return mnWithDepth;
}

// -------------------------------------------------------------- npy reader --

bool PriorNet::ReadNpyFloat16(const string& path, DepthMap& map, string& err)
{
    ifstream f(path.c_str(), ios::binary);
    if(!f)
    {
        err = "cannot open";
        return false;
    }
    char magic[6];
    f.read(magic, 6);
    if(f.gcount() != 6 || memcmp(magic, "\x93NUMPY", 6) != 0)
    {
        err = "not an npy file";
        return false;
    }
    unsigned char ver[2];
    f.read((char*)ver, 2);
    unsigned int headerLen = 0;
    if(ver[0] == 1)
    {
        unsigned char b[2];
        f.read((char*)b, 2);
        headerLen = b[0] | (b[1] << 8);
    }
    else if(ver[0] == 2 || ver[0] == 3)
    {
        unsigned char b[4];
        f.read((char*)b, 4);
        headerLen = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
    }
    else
    {
        err = "unknown npy version";
        return false;
    }
    string header(headerLen, '\0');
    f.read(&header[0], headerLen);
    if(header.find("'descr': '<f2'") == string::npos && header.find("'descr': '|f2'") == string::npos)
    {
        err = "dtype is not little-endian float16: " + header;
        return false;
    }
    if(header.find("'fortran_order': False") == string::npos)
    {
        err = "fortran order is not supported";
        return false;
    }
    size_t s = header.find("'shape': (");
    if(s == string::npos)
    {
        err = "no shape in header";
        return false;
    }
    int h = 0, w = 0;
    if(sscanf(header.c_str() + s, "'shape': (%d, %d)", &h, &w) != 2 || h <= 0 || w <= 0)
    {
        err = "shape is not (H, W): " + header;
        return false;
    }
    vector<unsigned short> raw((size_t)h * w);
    f.read((char*)raw.data(), raw.size() * 2);
    if((size_t)f.gcount() != raw.size() * 2)
    {
        err = "file is shorter than its shape";
        return false;
    }
    cv::Mat half(h, w, CV_16F, raw.data());
    cv::Mat z32;
    half.convertTo(z32, CV_32F);
    map.width = w;
    map.height = h;
    map.z.assign((float*)z32.data, (float*)z32.data + (size_t)h * w);
    for(float& v : map.z)
        if(!(v > 0.f) || !std::isfinite(v))
            v = 0.f;
    ComputeGradients(map);
    return true;
}

void PriorNet::ComputeGradients(DepthMap& map)
{
    const int w = map.width, h = map.height;
    map.gx.assign((size_t)w * h, 0.f);
    map.gy.assign((size_t)w * h, 0.f);
    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            const size_t i = (size_t)y * w + x;
            if(map.z[i] <= 0.f)
                continue;
            // Central difference where both neighbours are valid, one sided otherwise, 0 at a hole.
            float l = x > 0 ? map.z[i - 1] : 0.f, r = x + 1 < w ? map.z[i + 1] : 0.f;
            if(l > 0.f && r > 0.f) map.gx[i] = 0.5f * (r - l);
            else if(l > 0.f)       map.gx[i] = map.z[i] - l;
            else if(r > 0.f)       map.gx[i] = r - map.z[i];
            float u = y > 0 ? map.z[i - w] : 0.f, d = y + 1 < h ? map.z[i + w] : 0.f;
            if(u > 0.f && d > 0.f) map.gy[i] = 0.5f * (d - u);
            else if(u > 0.f)       map.gy[i] = map.z[i] - u;
            else if(d > 0.f)       map.gy[i] = d - map.z[i];
        }
    }
}

bool PriorNet::SampleBilinear(const DepthMap& map, float u, float v, float& z, float& gx, float& gy)
{
    // Pixel centres at integer coordinates, the convention of ucslam.depthnet.sample_depth.
    // A sample whose four neighbours are not all valid is refused.
    if(!(u >= 0.f) || !(v >= 0.f) || !(u <= map.width - 1) || !(v <= map.height - 1))
        return false;
    int x0 = (int)floor(u), y0 = (int)floor(v);
    if(x0 >= map.width - 1) x0 = map.width - 2;
    if(y0 >= map.height - 1) y0 = map.height - 2;
    if(map.width < 2 || map.height < 2)
        return false;
    const int x1 = x0 + 1, y1 = y0 + 1;
    const float fx = u - x0, fy = v - y0;
    const size_t i00 = (size_t)y0 * map.width + x0, i01 = i00 + 1, i10 = i00 + map.width, i11 = i10 + 1;
    if(map.z[i00] <= 0.f || map.z[i01] <= 0.f || map.z[i10] <= 0.f || map.z[i11] <= 0.f)
        return false;
    const float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
    z  = w00 * map.z[i00]  + w01 * map.z[i01]  + w10 * map.z[i10]  + w11 * map.z[i11];
    gx = w00 * map.gx[i00] + w01 * map.gx[i01] + w10 * map.gx[i10] + w11 * map.gx[i11];
    gy = w00 * map.gy[i00] + w01 * map.gy[i01] + w10 * map.gy[i10] + w11 * map.gy[i11];
    (void)x1; (void)y1;
    return z > 0.f;
}

// --------------------------------------------------------- pixel mapping --

bool PriorNet::WorkToMap(KeyFrame* pKF, const cv::Point2f& p, float& um, float& vm) const
{
    float uf, vf;   // pixel in the image the network saw, at its full resolution
    if(mView == VIEW_PINHOLE)
    {
        if(!pKF->mpCamera)
            return false;
        Eigen::Vector3f b = pKF->mpCamera->unprojectEig(p);
        if(!(b(2) > 1e-6f) || !std::isfinite(b(0)) || !std::isfinite(b(1)))
            return false;
        uf = mKview[0] * b(0) / b(2) + mKview[2];
        vf = mKview[1] * b(1) / b(2) + mKview[3];
    }
    else if(mbRectified)
    {
        // The rectification maps hold, for every rectified pixel, its raw pixel in the recorded frame.
        if(!(p.x >= 0.f) || !(p.y >= 0.f) || !(p.x <= mM1l.cols - 1) || !(p.y <= mM1l.rows - 1))
            return false;
        uf = BilinearMat(mM1l, p.x, p.y);
        vf = BilinearMat(mM2l, p.x, p.y);
    }
    else
    {
        // The working image is the recorded frame resized. Pixel centres stay aligned.
        uf = (p.x + 0.5f) * (float)mnFullW / (float)mWorkImSize.width - 0.5f;
        vf = (p.y + 0.5f) * (float)mnFullH / (float)mWorkImSize.height - 0.5f;
    }
    um = (uf + 0.5f) * (float)mnMapW / (float)mnFullW - 0.5f;
    vm = (vf + 0.5f) * (float)mnMapH / (float)mnFullH - 0.5f;
    return std::isfinite(um) && std::isfinite(vm);
}

// ------------------------------------------------------------- per keyframe --

bool PriorNet::ProcessKeyFrame(KeyFrame* pKF)
{
    const chrono::steady_clock::time_point tStart = chrono::steady_clock::now();
    const long long stampNs = llround(pKF->mTimeStamp * 1e9);

    if(!mbLocated)
    {
        if(mbGaveUp)
        {
            unique_lock<mutex> lock(mMutexStats);
            mnNoSequence++;
            return false;
        }
        mbLocated = LocateSequence(stampNs);
        if(!mbLocated)
        {
            unique_lock<mutex> lock(mMutexStats);
            mnNoSequence++;
            if(!mbWarnedNoSequence)
            {
                cerr << "[UCSLAM] PriorNet: no depth cache holds a map within " << mParams.depthMatchTolMs
                     << " ms of keyframe stamp " << stampNs << " ns under " << mParams.depthDir << "/"
                     << mParams.dataset << "/*/" << mParams.depthModel << ". No keyframe gets depth in this run."
                     << endl;
                mbWarnedNoSequence = true;
            }
            mbGaveUp = true;
            return false;
        }
    }

    long long mapStamp;
    if(!NearestStamp(stampNs, mapStamp))
    {
        unique_lock<mutex> lock(mMutexStats);
        mnNoMap++;
        if(!mbWarnedNoMap)
        {
            cerr << "[UCSLAM] PriorNet: no map within " << mParams.depthMatchTolMs << " ms of keyframe stamp "
                 << stampNs << " ns, this keyframe gets no depth (logged once)" << endl;
            mbWarnedNoMap = true;
        }
        return false;
    }
    DepthMap map;
    string err;
    {
        char name[64];
        snprintf(name, sizeof(name), "/%lld.npy", mapStamp);
        if(!ReadNpyFloat16(mSeqDir + name, map, err))
        {
            unique_lock<mutex> lock(mMutexStats);
            mnBadMap++;
            cerr << "[UCSLAM] PriorNet: " << mSeqDir << name << ": " << err << endl;
            return false;
        }
    }
    map.stampNs = mapStamp;
    if((map.width != mnMapW || map.height != mnMapH) && !mbWarnedShape)
    {
        cerr << "[UCSLAM] PriorNet: map is " << map.width << "x" << map.height << " but the manifest says "
             << mnMapW << "x" << mnMapH << ", using the map's own size" << endl;
        mbWarnedShape = true;
    }
    const float mapW = (float)map.width, mapH = (float)map.height;
    const float sx = mapW / (float)mnMapW, sy = mapH / (float)mnMapH;   // 1 unless the shapes disagree

    const int N = pKF->N;
    const int nLeft = (pKF->NLeft == -1) ? N : pKF->NLeft;   // right-image keypoints of a fisheye pair have no cam0 pixel
    vector<float> z(N, -1.f), sig(N, 0.f);
    vector<float> sigmas;
    sigmas.reserve(nLeft);
    const float logZref = log(max(mParams.zRef, 1e-6f));
    const float biasScale = exp(-mParams.logBias);
    int nGot = 0;
    for(int i = 0; i < nLeft && i < (int)pKF->mvKeys.size(); i++)
    {
        const cv::KeyPoint& kp = (mView == VIEW_PINHOLE) ? pKF->mvKeysUn[i] : pKF->mvKeys[i];
        float um, vm;
        if(!WorkToMap(pKF, kp.pt, um, vm))
            continue;
        um = (um + 0.5f) * sx - 0.5f;
        vm = (vm + 0.5f) * sy - 0.5f;
        float zr, gxm, gym;
        if(!SampleBilinear(map, um, vm, zr, gxm, gym))
            continue;
        // Jacobian of the map pixel with respect to the working pixel, central differences of half a pixel.
        float uxp, vxp, uxm, vxm, uyp, vyp, uym, vym;
        if(!WorkToMap(pKF, cv::Point2f(kp.pt.x + 0.5f, kp.pt.y), uxp, vxp) ||
           !WorkToMap(pKF, cv::Point2f(kp.pt.x - 0.5f, kp.pt.y), uxm, vxm) ||
           !WorkToMap(pKF, cv::Point2f(kp.pt.x, kp.pt.y + 0.5f), uyp, vyp) ||
           !WorkToMap(pKF, cv::Point2f(kp.pt.x, kp.pt.y - 0.5f), uym, vym))
            continue;
        const float J00 = (uxp - uxm) * sx, J01 = (uyp - uym) * sx;   // d um / d u_work, d um / d v_work
        const float J10 = (vxp - vxm) * sy, J11 = (vyp - vym) * sy;   // d vm / d u_work, d vm / d v_work
        const float gwu = J00 * gxm + J10 * gym;                       // grad z in working pixels
        const float gwv = J01 * gxm + J11 * gym;
        const int octave = max(0, min(kp.octave, (int)pKF->mvLevelSigma2.size() - 1));
        const float s2 = pKF->mvLevelSigma2.empty() ? 1.f : pKF->mvLevelSigma2[octave];
        const float logz = log(zr);
        float sn = mParams.sigmaNetC0 + mParams.sigmaNetC1 * (logz - logZref);
        sn = max(sn, 1e-3f);
        const float edge = s2 * (gwu * gwu + gwv * gwv) / (zr * zr);
        z[i] = zr * biasScale;
        sig[i] = sqrt(sn * sn + edge);
        sigmas.push_back(sig[i]);
        nGot++;
    }

    const double latencyMs = NowMsSince(pKF->mtCreated);
    pKF->SetDepthPrior(z, sig, latencyMs);

    unique_lock<mutex> lock(mMutexStats);
    mnWithDepth++;
    mvLatencyMs.push_back(latencyMs);
    mvProcessMs.push_back(NowMsSince(tStart));
    mvCoverage.push_back(nLeft > 0 ? (double)nGot / (double)nLeft : 0.0);
    if(!sigmas.empty())
    {
        nth_element(sigmas.begin(), sigmas.begin() + sigmas.size() / 2, sigmas.end());
        mvSigmaMedian.push_back(sigmas[sigmas.size() / 2]);
    }
    return true;
}

// ------------------------------------------------------------------ stats --

double PriorNet::Percentile(vector<double> v, double q)
{
    if(v.empty())
        return NAN;
    sort(v.begin(), v.end());
    const double pos = q * (v.size() - 1);
    const size_t lo = (size_t)floor(pos);
    const size_t hi = min(lo + 1, v.size() - 1);
    return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}

void PriorNet::PrintStats(ostream& out)
{
    unique_lock<mutex> lock(mMutexStats);
    out.precision(6);
    out << "[UCSLAM] priornet keyframes_received=" << mnReceived << " with_depth=" << mnWithDepth
        << " no_map=" << mnNoMap << " no_sequence=" << mnNoSequence << " bad_map=" << mnBadMap
        << " queue_max=" << mnQueueMax << " left_in_queue=" << mlQueue.size() << endl;
    out << "[UCSLAM] priornet sequence_dir=" << (mSeqDir.empty() ? string("none") : mSeqDir) << endl;
    out << "[UCSLAM] priornet latency_ms creation_to_ready p50=" << Percentile(mvLatencyMs, 0.5)
        << " p90=" << Percentile(mvLatencyMs, 0.9) << " p99=" << Percentile(mvLatencyMs, 0.99)
        << " max=" << Percentile(mvLatencyMs, 1.0) << " n=" << mvLatencyMs.size() << endl;
    out << "[UCSLAM] priornet process_ms load_and_sample p50=" << Percentile(mvProcessMs, 0.5)
        << " p99=" << Percentile(mvProcessMs, 0.99) << " max=" << Percentile(mvProcessMs, 1.0) << endl;
    out << "[UCSLAM] priornet coverage keypoints_with_depth_over_keypoints p50=" << Percentile(mvCoverage, 0.5)
        << " p10=" << Percentile(mvCoverage, 0.1) << " min=" << Percentile(mvCoverage, 0.0) << endl;
    out << "[UCSLAM] priornet sigma_D per_keyframe_median p50=" << Percentile(mvSigmaMedian, 0.5)
        << " p90=" << Percentile(mvSigmaMedian, 0.9) << endl;
}

} // namespace ORB_SLAM3
