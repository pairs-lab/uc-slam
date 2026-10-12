/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
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


#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"
#include "PriorNet.h"
#include "System.h"

#include <chrono>
#include <iomanip>

#include<mutex>
#include<chrono>
#include<algorithm>
#include<cmath>

namespace ORB_SLAM3
{

LocalMapping::LocalMapping(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9,9))
{
    mpPriorNet = static_cast<PriorNet*>(NULL);
    mnLogScaleCulls = 0;
    mnLogScaleCullsWithState = 0;
    mnLogScaleHeadTransfers = 0;
    mnLogScaleDepthEdgesFolded = 0;
    mnLogScaleDepthEdgesGated = 0;
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;

    // UC-SLAM: initialized, upstream leaves them undefined until the first keyframe. System::GetTimeFromIMUInit reads
    // both, which the ROS nodes call after every frame (the benchmark examples never call it).
    mpCurrentKeyFrame = static_cast<KeyFrame*>(NULL);
    mFirstTs = 0.0;

    mNumLM = 0;
    mNumKFCulling=0;

#ifdef REGISTER_TIMES
    nLBA_exec = 0;
    nLBA_abort = 0;
#endif

}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::SetPriorNet(PriorNet* pPriorNet)
{
    mpPriorNet = pPriorNet;
}

void LocalMapping::SetUcslamSettings(const Settings::Ucslam& u)
{
    Optimizer::SetUcslamParams(u);
}

const Settings::Ucslam& LocalMapping::UcslamSettings() const
{
    return Optimizer::UcslamParams();
}

// UC-SLAM depth-prior map points, defined after CreateNewMapPoints below.
static int CreateDepthPriorMapPoints(KeyFrame* pKF, Atlas* pAtlas, std::list<MapPoint*>& lRecentAdded,
                                     const Settings::Ucslam& u);

void LocalMapping::Run()
{
    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames() && !mbBadImu)
        {
#ifdef REGISTER_TIMES
            double timeLBA_ms = 0;
            double timeKFCulling_ms = 0;

            std::chrono::steady_clock::time_point time_StartProcessKF = std::chrono::steady_clock::now();
#endif
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndProcessKF = std::chrono::steady_clock::now();

            double timeProcessKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndProcessKF - time_StartProcessKF).count();
            vdKFInsert_ms.push_back(timeProcessKF);
#endif

            // Check recent MapPoints
            MapPointCulling();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCulling = std::chrono::steady_clock::now();

            double timeMPCulling = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCulling - time_EndProcessKF).count();
            vdMPCulling_ms.push_back(timeMPCulling);
#endif

            // Triangulate new MapPoints
            CreateNewMapPoints();

            // UC-SLAM: map points from the depth prior, only with the depth factor on and the depth ready.
            {
                const Settings::Ucslam& u = Optimizer::UcslamParams();
                const bool bDpp = u.useDepthFactor && u.maxDepthPriorPoints > 0;
                // Live depth, rule UCSLAM.DppWaitMs: wait a bounded time for the depth of this keyframe. Off by default.
                if(bDpp && u.dppWaitMs > 0.f && !mpCurrentKeyFrame->DepthPriorReady())
                {
                    const std::chrono::steady_clock::time_point tEnd = std::chrono::steady_clock::now() +
                        std::chrono::microseconds((long long)(u.dppWaitMs * 1000.0f));
                    while(!mpCurrentKeyFrame->DepthPriorReady() && std::chrono::steady_clock::now() < tEnd)
                        usleep(1000);
                }
                // Live depth, rule UCSLAM.DppDeferMaxKFs: a keyframe without depth yet gets its points in a later step.
                if(bDpp && u.dppDeferMaxKFs > 0 && !mpCurrentKeyFrame->DepthPriorReady())
                    mlDppPending.push_back(std::make_pair(mpCurrentKeyFrame, 0));
                else
                    CreateDepthPriorMapPoints(mpCurrentKeyFrame, mpAtlas, mlpRecentAddedMapPoints, u);
                if(!mlDppPending.empty())
                {
                    Map* pCurrentMap = mpAtlas->GetCurrentMap();
                    for(std::list<std::pair<KeyFrame*, int> >::iterator it = mlDppPending.begin(); it != mlDppPending.end();)
                    {
                        KeyFrame* pKF = it->first;
                        if(pKF == mpCurrentKeyFrame)
                        {
                            ++it;
                            continue;
                        }
                        it->second++;
                        if(pKF->isBad() || pKF->GetMap() != pCurrentMap || it->second > u.dppDeferMaxKFs)
                        {
                            cout << "[UCSLAM] dpp_deferred kf=" << pKF->mnId << " age_kfs=" << it->second << " created=0 dropped=1" << endl;
                            it = mlDppPending.erase(it);
                        }
                        else if(pKF->DepthPriorReady())
                        {
                            const int n = CreateDepthPriorMapPoints(pKF, mpAtlas, mlpRecentAddedMapPoints, u);
                            cout << "[UCSLAM] dpp_deferred kf=" << pKF->mnId << " age_kfs=" << it->second << " created=" << n << endl;
                            it = mlDppPending.erase(it);
                        }
                        else
                            ++it;
                    }
                }
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCreation = std::chrono::steady_clock::now();

            double timeMPCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCreation - time_EndMPCulling).count();
            vdMPCreation_ms.push_back(timeMPCreation);
#endif

            bool b_doneLBA = false;
            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                if(mpAtlas->KeyFramesInMap()>2)
                {

                    if(mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
                    {
                        float dist = (mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()).norm() +
                                (mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter()).norm();

                        if(dist>0.05)
                            mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            if((mTinit<10.f) && (dist<0.02))
                            {
                                cout << "Not enough motion for initializing. Reseting..." << endl;
                                unique_lock<mutex> lock(mMutexReset);
                                mbResetRequestedActiveMap = true;
                                mpMapToReset = mpCurrentKeyFrame->GetMap();
                                mbBadImu = true;
                            }
                        }

                        bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);
                        const std::chrono::steady_clock::time_point tUcslamLBA0 = std::chrono::steady_clock::now();
                        Optimizer::sUcslamInertialLbaKf = 0;
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                        b_doneLBA = true;
                        if(UcslamSettings().diag)
                            cout << "[UCSLAM] diag lba_ms kf=" << mpCurrentKeyFrame->mnId << " ms="
                                 << std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(std::chrono::steady_clock::now() - tUcslamLBA0).count()
                                 << " abort=" << (mbAbortBA ? 1 : 0) << " large=" << (bLarge ? 1 : 0) << endl;
                        // UC-SLAM (UCSLAM.YoungMapScaleGate tau > 0, attempts 06 and 07): the first local inertial BA of the run
                        // that printed its lba line checks the log-scale a_cur of the current keyframe. |a_cur| > tau means the
                        // first IMU initialization disagrees with the depth prior by more than a factor e^tau. With
                        // UCSLAM.YoungMapBiasGate beta > 0 the gate also needs |bgv| > beta, bgv = bg0 . up0 the vertical gyro
                        // bias of the first init of the map. Pending covers the local-mapping reset flags, an active-map reset
                        // requested at the System and not yet run, and a tracker state other than OK. The pending read, the
                        // line and the bad-IMU write share one mMutexReset scope, so the line precedes the tracker's reset.
                        // In shadow mode the gate never writes the flag. At most one check per run.
                        const double dUcslamGate = UcslamSettings().youngMapScaleGate;
                        if(dUcslamGate > 0.0 && !mbUcslamGateChecked && Optimizer::sUcslamInertialLbaKf == mpCurrentKeyFrame->mnId + 1)
                        {
                            mbUcslamGateChecked = true;
                            const double dBeta = UcslamSettings().youngMapBiasGate;
                            const bool bShadow = UcslamSettings().youngMapGateShadow != 0;
                            const double aCur = mpCurrentKeyFrame->mLogScale;
                            double aGeo = 0.0;
                            int nGeo = 0;
                            const bool bGeo = Optimizer::UcslamGeometricLogScale(mpCurrentKeyFrame, aGeo, nGeo);
                            const Eigen::Vector3d bg0 = mbUcslamInit0 ? mUcslamBg0 : Eigen::Vector3d::Zero();
                            const Eigen::Vector3d up0 = mbUcslamInit0 ? mUcslamUp0 : Eigen::Vector3d::Zero();
                            const double bgv = bg0.dot(up0);
                            bool bAct = false;
                            {
                                unique_lock<mutex> lock(mMutexReset);
                                const bool bPending = mbBadImu || mbResetRequestedActiveMap
                                                      || (mpSystem && mpSystem->UcslamResetActiveMapPending())
                                                      || mpTracker->mState != Tracking::OK;
                                const bool bWould = !bPending && std::fabs(aCur) > dUcslamGate && (dBeta <= 0.0 || std::fabs(bgv) > dBeta);
                                bAct = bWould && !bShadow;
                                char bufGate[512];
                                snprintf(bufGate, sizeof(bufGate),
                                         "[UCSLAM] young_map_gate kf=%lu t=%.6f a_cur=%.9f a_geo=%.9f n_geo=%d bg0=%.9f,%.9f,%.9f "
                                         "up0=%.9f,%.9f,%.9f bgv=%.9f tau=%.4f beta=%.4f pending=%d shadow=%d would=%d act=%d",
                                         mpCurrentKeyFrame->mnId, mpCurrentKeyFrame->mTimeStamp, aCur, bGeo ? aGeo : std::nan(""), nGeo,
                                         bg0(0), bg0(1), bg0(2), up0(0), up0(1), up0(2), bgv, dUcslamGate, dBeta,
                                         bPending ? 1 : 0, bShadow ? 1 : 0, bWould ? 1 : 0, bAct ? 1 : 0);
                                cout << bufGate << endl;
                                if(bAct)
                                {
                                    // Only the bad-IMU flag: ResetIfRequested() then leaves it set, local mapping idles, and the
                                    // tracker resets the active map on its next frame (Tracking.cc, "TRACK: Reset map because
                                    // local mapper set the bad imu flag"), which also resets local mapping.
                                    cout << "Young map scale disagrees with the depth prior. Reseting..." << endl;
                                    mbBadImu = true;
                                }
                            }
                        }
                    }
                    else
                    {
                        const std::chrono::steady_clock::time_point tUcslamLBA0 = std::chrono::steady_clock::now();
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                        b_doneLBA = true;
                        if(UcslamSettings().diag)
                            cout << "[UCSLAM] diag lba_ms kf=" << mpCurrentKeyFrame->mnId << " ms="
                                 << std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(std::chrono::steady_clock::now() - tUcslamLBA0).count()
                                 << " abort=" << (mbAbortBA ? 1 : 0) << " visual=1" << endl;
                    }

                }
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndLBA = std::chrono::steady_clock::now();

                if(b_doneLBA)
                {
                    timeLBA_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLBA - time_EndMPCreation).count();
                    vdLBA_ms.push_back(timeLBA_ms);

                    nLBA_exec += 1;
                    if(mbAbortBA)
                    {
                        nLBA_abort += 1;
                    }
                    vnLBA_edges.push_back(num_edges_BA);
                    vnLBA_KFopt.push_back(num_OptKF_BA);
                    vnLBA_KFfixed.push_back(num_FixedKF_BA);
                    vnLBA_MPs.push_back(num_MPs_BA);
                }

#endif

                // Initialize IMU here
                if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
                {
                    if (mbMonocular)
                        InitializeIMU(1e2, 1e10, true);
                    else
                        InitializeIMU(1e2, 1e5, true);
                }


                // Check redundant local Keyframes
                KeyFrameCulling();

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndKFCulling = std::chrono::steady_clock::now();

                timeKFCulling_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndKFCulling - time_EndLBA).count();
                vdKFCulling_ms.push_back(timeKFCulling_ms);
#endif

                if ((mTinit<50.0f) && mbInertial)
                {
                    if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK) // Enter here everytime local-mapping is called
                    {
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA1()){
                            if (mTinit>5.0f)
                            {
                                cout << "start VIBA 1" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA1();
                                if (mbMonocular)
                                    InitializeIMU(1.f, 1e5, true);
                                else
                                    InitializeIMU(1.f, 1e5, true);

                                cout << "end VIBA 1" << endl;
                            }
                        }
                        else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()){
                            if (mTinit>15.0f){
                                cout << "start VIBA 2" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA2();
                                if (mbMonocular)
                                    InitializeIMU(0.f, 0.f, true);
                                else
                                    InitializeIMU(0.f, 0.f, true);

                                cout << "end VIBA 2" << endl;
                            }
                        }

                        // scale refinement
                        if (((mpAtlas->KeyFramesInMap())<=200) &&
                                ((mTinit>25.0f && mTinit<25.5f)||
                                (mTinit>35.0f && mTinit<35.5f)||
                                (mTinit>45.0f && mTinit<45.5f)||
                                (mTinit>55.0f && mTinit<55.5f)||
                                (mTinit>65.0f && mTinit<65.5f)||
                                (mTinit>75.0f && mTinit<75.5f))){
                            if (mbMonocular)
                                ScaleRefinement();
                        }
                    }
                }
            }

#ifdef REGISTER_TIMES
            vdLBASync_ms.push_back(timeKFCulling_ms);
            vdKFCullingSync_ms.push_back(timeKFCulling_ms);
#endif

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLocalMap = std::chrono::steady_clock::now();

            double timeLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLocalMap - time_StartProcessKF).count();
            vdLMTotal_ms.push_back(timeLocalMap);
#endif
        }
        else if(Stop() && !mbBadImu)
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mlNewKeyFrames.push_back(pKF);
        mbAbortBA=true;
    }
    // UC-SLAM live depth: ask for the depth of this keyframe now, so that the network runs while the keyframe waits
    // in the mapper queue. Returns at once with UCSLAM.DepthSource file.
    if(mpPriorNet)
        mpPriorNet->NotifyKeyFrame(pKF->mTimeStamp, pKF->mnId);
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // UC-SLAM: hand the keyframe to the PriorNet thread. It attaches the depth prior asynchronously.
    if(mpPriorNet)
        mpPriorNet->InsertKeyFrame(mpCurrentKeyFrame);

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    int borrar = mlpRecentAddedMapPoints.size();

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;

        if(pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if(pMP->GetFoundRatio()<0.25f)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
        {
            lit++;
            borrar--;
        }
    }
}


void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    // For stereo inertial case
    if(mbMonocular)
        nn=30;
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    if (mbInertial)
    {
        KeyFrame* pKF = mpCurrentKeyFrame;
        int count=0;
        while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
        {
            vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
            if(it==vpNeighKFs.end())
                vpNeighKFs.push_back(pKF->mPrevKF);
            pKF = pKF->mPrevKF;
        }
    }

    float th = 0.6f;

    ORBmatcher matcher(th,false);

    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
    int countStereo = 0;
    int countStereoGoodProj = 0;
    int countStereoAttempt = 0;
    int totalStereoPts = 0;
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        const float baseline = vBaseline.norm();

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();

        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);

        Sophus::SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = sophTcw2.translation();

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame -> NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                         : (idx1 < mpCurrentKeyFrame -> NLeft) ? mpCurrentKeyFrame -> mvKeys[idx1]
                                                                                                               : mpCurrentKeyFrame -> mvKeysRight[idx1 - mpCurrentKeyFrame -> NLeft];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0);
            const bool bRight1 = (mpCurrentKeyFrame -> NLeft == -1 || idx1 < mpCurrentKeyFrame -> NLeft) ? false
                                                                                                         : true;

            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];

            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0);
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2){
                if(bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else{
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1 = eigTcw1.block<3,3>(0,0);
                Rwc1 = Rcw1.transpose();
                tcw1 = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2 = eigTcw2.block<3,3>(0,0);
                Rwc2 = Rcw2.transpose();
                tcw2 = sophTcw2.translation();
            }

            // Check parallax between rays
            Eigen::Vector3f xn1 = pCamera1->unprojectEig(kp1.pt);
            Eigen::Vector3f xn2 = pCamera2->unprojectEig(kp2.pt);

            Eigen::Vector3f ray1 = Rwc1 * xn1;
            Eigen::Vector3f ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm() * ray2.norm());

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            if (bStereo1 || bStereo2) totalStereoPts++;
            
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            Eigen::Vector3f x3D;

            bool goodProj = false;
            bool bPointStereo = false;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
                                                                          (cosParallaxRays<0.9996 && mbInertial) || (cosParallaxRays<0.9998 && !mbInertial)))
            {
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if(!goodProj)
                    continue;
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            if(goodProj && bPointStereo)
                countStereoGoodProj++;

            if(!goodProj)
                continue;

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0);
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;

                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0);
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            Eigen::Vector3f normal1 = x3D - Ow1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D - Ow2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());
            if (bPointStereo)
                countStereo++;
            
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }    
}

/*
 * UC-SLAM depth-prior map points, docs/fork_design.md Sec. 2, "Depth-prior map points".
 *
 * For each keypoint of the keyframe without a map point and with a depth prior, a map point is
 * created at depth z_hat exp(a_k), flagged mbFromDepthPrior, in order of increasing depth and
 * capped by UCSLAM.MaxDepthPriorPoints. a_k is the keyframe's estimate when the BA has set one,
 * else its predecessor's, the random walk prediction. With UCSLAM.LogScaleSeed 1 a keyframe
 * without an estimate uses its geometric log-scale instead (docs/fork_design.md Sec. 7). The points join
 * mlpRecentAddedMapPoints, so MapPointCulling treats them like the stereo points Tracking
 * creates. They enter the local BA through their depth edge and any later reprojection edge.
 * Nothing happens with the factor off, or before PriorNet has written the depth of the keyframe.
 *
 * This is a free function so that LocalMapping.h stays as it is. It could become the member
 * LocalMapping::CreateDepthPriorMapPoints() at the placeholder left there.
 */
static int CreateDepthPriorMapPoints(KeyFrame* pKF, Atlas* pAtlas, std::list<MapPoint*>& lRecentAdded,
                                     const Settings::Ucslam& u)
{
    if(!u.useDepthFactor || u.maxDepthPriorPoints <= 0 || !pKF || pKF->isBad())
        return 0;
    std::vector<float> vDepth, vSigma;
    if(!pKF->GetDepthPrior(vDepth, vSigma))
    {
        cout << "[UCSLAM] dpp kf=" << pKF->mnId << " ready=0 created=0" << endl;
        return 0;
    }

    const int nLeft = (pKF->NLeft == -1) ? pKF->N : pKF->NLeft;
    std::vector<std::pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(nLeft);
    for(int i = 0; i < nLeft && i < (int)vDepth.size(); i++)
        if(vDepth[i] > 0.f && !pKF->GetMapPoint(i))
            vDepthIdx.push_back(std::make_pair(vDepth[i], i));
    std::sort(vDepthIdx.begin(), vDepthIdx.end());

    // a_k of the keyframe, or the seed of a keyframe without a state (Optimizer::UcslamSeedLogScale).
    const double a = Optimizer::UcslamSeedLogScale(pKF);
    const float scale = (float)std::exp(a);
    const Sophus::SE3f Twc = pKF->GetPoseInverse();
    Map* pMap = pAtlas->GetCurrentMap();

    int nCreated = 0;
    for(size_t j = 0; j < vDepthIdx.size() && nCreated < u.maxDepthPriorPoints; j++)
    {
        const int i = vDepthIdx[j].second;
        // mvKeysUn are the undistorted pixels of a pinhole camera and the raw ones of a fisheye,
        // which is what unprojectEig of each model expects.
        const Eigen::Vector3f ray = pKF->mpCamera->unprojectEig(pKF->mvKeysUn[i].pt);
        if(!(ray(2) > 1e-6f))
            continue;
        const Eigen::Vector3f x3Dc = ray * (vDepthIdx[j].first * scale / ray(2));
        const Eigen::Vector3f x3Dw = Twc * x3Dc;

        MapPoint* pMP = new MapPoint(x3Dw, pKF, pMap);
        pMP->mbFromDepthPrior = true;
        pMP->AddObservation(pKF, i);
        pKF->AddMapPoint(pMP, i);
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
        pAtlas->AddMapPoint(pMP);
        lRecentAdded.push_back(pMP);
        nCreated++;
    }
    cout << "[UCSLAM] dpp kf=" << pKF->mnId << " ready=1 candidates=" << vDepthIdx.size()
         << " created=" << nCreated << " a=" << a << endl;
    return nCreated;
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=30;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++)
    {
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    if(mbInertial)
    {
        KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
        while(vpTargetKFs.size()<20 && pKFi)
        {
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
            pKFi = pKFi->mPrevKF;
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
        if(pKFi->NLeft != -1) matcher.Fuse(pKFi,vpMapPointMatches,true);
    }


    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);
    if(mpCurrentKeyFrame->NLeft != -1) matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21;
    mpCurrentKeyFrame->UpdateBestCovisibles();
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    if(!mbInertial)
        redundant_th = 0.9;
    else if (mbMonocular)
        redundant_th = 0.9;
    else
        redundant_th = 0.5;

    const bool bInitImu = mpAtlas->isImuInitialized();
    int count=0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial)
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame;
        while(count<Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }



    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        count++;
        KeyFrame* pKF = *vit;

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                     : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                          : pKF -> mvKeysRight[i].octave;
                        const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            tuple<int,int> indexes = mit->second;
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = -1;
                            if(pKFi -> NLeft == -1)
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                            else {
                                if (leftIndex != -1) {
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                                }
                                if (rightIndex != -1) {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                  : scaleLeveli;
                                }
                            }

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>thObs)
                                    break;
                            }
                        }
                        if(nObs>thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>redundant_th*nMPs)
        {
            if (mbInertial)
            {
                if (mpAtlas->KeyFramesInMap()<=Nd)
                    continue;

                if(pKF->mnId>(mpCurrentKeyFrame->mnId-2))
                    continue;

                if(pKF->mPrevKF && pKF->mNextKF)
                {
                    const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;

                    if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5))
                    {
                        if(UcslamSettings().useDepthFactor)
                            MarginalizeLogScaleOnCull(pKF);   // UC-SLAM, before the relink of mPrevKF and mNextKF
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                    else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && ((pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<3))
                    {
                        if(UcslamSettings().useDepthFactor)
                            MarginalizeLogScaleOnCull(pKF);   // UC-SLAM, before the relink of mPrevKF and mNextKF
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                }
            }
            else
            {
                pKF->SetBadFlag();
            }
        }
        if((count > 20 && mbAbortBA) || count>100)
        {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// UC-SLAM. Log-scale chain marginalization on keyframe culling.
// docs/fork_design.md Sec. 2, "Marginalization on culling", and ucslam/factors.py.
// ---------------------------------------------------------------------------

double LocalMapping::LogScaleSegmentInfo(KeyFrame* pKF)
{
    if(!pKF || !pKF->mPrevKF)
        return 0.0;
    Map::LogScaleChainTerms terms;
    if(Map::GetLogScaleChainTerms(pKF, terms) && terms.segInfo > 0.0)
        return terms.segInfo;
    const double sigmaRW = UcslamSettings().sigmaRW;
    if(sigmaRW <= 0.0)
        return 0.0;
    // EdgeLogScaleRW carries 1 / (sigma_rw^2 dt). Keyframe stamps grow along the chain.
    const double dt = std::max(pKF->mTimeStamp - pKF->mPrevKF->mTimeStamp, 1e-6);
    return 1.0 / (sigmaRW * sigmaRW * dt);
}

void LocalMapping::DepthInformationOfKeyFrame(KeyFrame* pKF, double& S, double& b, int& nInside, int& nGated)
{
    S = 0.0;
    b = 0.0;
    nInside = 0;
    nGated = 0;

    vector<float> vDepth, vSigma;
    if(!pKF->GetDepthPrior(vDepth, vSigma))
        return;
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
    const Sophus::SE3f Tcw = pKF->GetPose();
    const Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
    const Eigen::Vector3f tcw = Tcw.translation();
    const double a = pKF->mLogScale;
    // The depth edge carries 1 / sigma_D^2, scaled by N_eff / N under ESS deflation.
    const double infoScale = UcslamSettings().essDeflation ? (double)UcslamSettings().neffOverN : 1.0;
    // Huber delta of EdgeLogDepth, the square root of the chi-square 95 percent
    // quantile with one degree of freedom. Inside it the edge is quadratic.
    const double delta = 1.96;

    const size_t n = std::min(vpMPs.size(), std::min(vDepth.size(), vSigma.size()));
    for(size_t i = 0; i < n; i++)
    {
        const double zhat = vDepth[i];
        const double sigma = vSigma[i];
        if(zhat <= 0.0 || sigma <= 0.0)
            continue;
        MapPoint* pMP = vpMPs[i];
        if(!pMP || pMP->isBad())
            continue;
        // The observations the local BA gives no depth edge carry no depth information here either
        // (UCSLAM.DepthEdgesStereoObs and UCSLAM.DepthEdgesStereoPoints, docs/fork_design.md Sec. 7).
        if(!UcslamSettings().depthEdgesStereoObs &&
           ((pKF->NLeft == -1 && i < pKF->mvuRight.size() && pKF->mvuRight[i] >= 0.f) ||
            (pKF->NLeft != -1 && i < pKF->mvLeftToRightMatch.size() && pKF->mvLeftToRightMatch[i] >= 0)))
            continue;
        if(!UcslamSettings().depthEdgesStereoPoints && Optimizer::UcslamHasStereoObservation(pMP))
            continue;
        const Eigen::Vector3f Xc = Rcw * pMP->GetWorldPos() + tcw;
        const double z = Xc(2);
        if(z <= 0.0)
            continue;
        const double m = log(z) - log(zhat);       // the edge is W (a_k - m)^2 with this m
        const double r = m - a;                    // residual log z - a_k - log z_hat
        if(fabs(r) > delta * sigma)
        {
            nGated++;
            continue;
        }
        const double W = infoScale / (sigma * sigma);
        S += W;
        b += W * m;
        nInside++;
    }
}

void LocalMapping::MarginalizeLogScaleOnCull(KeyFrame* pKF)
{
    KeyFrame* pPrev = pKF->mPrevKF;
    KeyFrame* pNext = pKF->mNextKF;
    mnLogScaleCulls++;
    if(!pPrev && !pNext)
    {
        Map::EraseLogScaleChainTerms(pKF);
        return;
    }

    // Chain informations of the two segments, 0 for a missing neighbour. In the
    // inertial culling branch both neighbours exist, so this is the interior
    // case of factors.py. With w1 = 0 the same shares are the head rule, with
    // w2 = 0 the tail rule.
    const double w1 = pPrev ? LogScaleSegmentInfo(pKF) : 0.0;
    const double w2 = pNext ? LogScaleSegmentInfo(pNext) : 0.0;

    // Total unary term of the culled node: its depth edges inside the quadratic
    // region at the current estimate, the unary terms inherited from earlier
    // culls, and the absolute prior when it carries it. A keyframe without a
    // log-scale state never entered a BA with the chain, so its depth edges
    // carry no information about a_k yet.
    const bool bHasState = pKF->mLogScaleVar > 0.0;
    double S = 0.0, b = 0.0;
    int nInside = 0, nGated = 0;
    if(bHasState)
    {
        DepthInformationOfKeyFrame(pKF, S, b, nInside, nGated);
        mnLogScaleCullsWithState++;
        mnLogScaleDepthEdgesFolded += nInside;
        mnLogScaleDepthEdgesGated += nGated;
    }
    Map::LogScaleChainTerms own;
    if(Map::GetLogScaleChainTerms(pKF, own))
    {
        S += own.unaryInfo;
        b += own.unaryB;
    }
    const bool bHead = pKF->mbLogScalePrior;
    if(bHead && pKF->mLogScalePriorVar > 0.0)
    {
        S += 1.0 / pKF->mLogScalePriorVar;
        b += pKF->mLogScalePriorMean / pKF->mLogScalePriorVar;
    }

    double wMerged, Sp, bp, Sn, bn;
    LogScaleChain::CullShares(w1, w2, S, b, wMerged, Sp, bp, Sn, bn);

    // The absolute prior moves to the next keyframe, or to the previous one when
    // there is no next. Its share of the culled node's information becomes the new
    // prior: with w1 = 0 that is variance 1 / (S_0 + sigma_0^-2) + sigma_rw^2 dt,
    // the rule of the design. The other neighbour's share is an inherited unary term.
    KeyFrame* pNewHead = bHead ? (pNext ? pNext : pPrev) : static_cast<KeyFrame*>(NULL);

    if(pPrev)
    {
        Map::LogScaleChainTerms t;
        Map::GetLogScaleChainTerms(pPrev, t);
        if(pNewHead == pPrev && Sp > 0.0)
        {
            pPrev->mbLogScalePrior = true;
            pPrev->mLogScalePriorVar = 1.0 / Sp;
            pPrev->mLogScalePriorMean = bp / Sp;
        }
        else
        {
            t.unaryInfo += Sp;
            t.unaryB += bp;
        }
        Map::SetLogScaleChainTerms(pPrev, t);
    }
    if(pNext)
    {
        Map::LogScaleChainTerms t;
        Map::GetLogScaleChainTerms(pNext, t);
        // The relink makes pPrev the mPrevKF of pNext. The merged segment is theirs.
        t.segInfo = (pPrev && wMerged > 0.0) ? wMerged : -1.0;
        if(pNewHead == pNext && Sn > 0.0)
        {
            pNext->mbLogScalePrior = true;
            pNext->mLogScalePriorVar = 1.0 / Sn;
            pNext->mLogScalePriorMean = bn / Sn;
        }
        else
        {
            t.unaryInfo += Sn;
            t.unaryB += bn;
        }
        Map::SetLogScaleChainTerms(pNext, t);
    }

    if(bHead)
    {
        mnLogScaleHeadTransfers++;
        cout << "[UCSLAM] chain cull: keyframe " << pKF->mnId << " carried the absolute log-scale prior (mean "
             << pKF->mLogScalePriorMean << " var " << pKF->mLogScalePriorVar << "), transferred to keyframe "
             << (pNewHead ? (long)pNewHead->mnId : -1L) << " (mean " << (pNewHead ? pNewHead->mLogScalePriorMean : 0.0)
             << " var " << (pNewHead ? pNewHead->mLogScalePriorVar : 0.0) << ")" << endl;
        pKF->mbLogScalePrior = false;
        pKF->mLogScalePriorMean = 0.0;
        pKF->mLogScalePriorVar = 0.0;
    }
    Map::EraseLogScaleChainTerms(pKF);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    cout << "LM: Map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
    cout << "LM: Active map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Active map reset, Done!!!" << endl;
}

void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mlDppPending.clear();
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mUcslamRefineLogSum = 0.0;
            mbUcslamInit0 = false;
            mUcslamBg0.setZero();
            mUcslamUp0.setZero();
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mIdxInit=0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if(mbResetRequestedActiveMap) {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mlDppPending.clear();

            // Inertial parameters
            mTinit = 0.f;
            mUcslamRefineLogSum = 0.0;
            mbUcslamInit0 = false;
            mUcslamBg0.setZero();
            mUcslamUp0.setZero();
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mbResetRequested = false;
            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if(executed_reset)
        cout << "LM: Reset free the mutex" << endl;

}

void LocalMapping::RequestFinish()
{
    // UC-SLAM: one summary line of the chain marginalization, printed here because
    // System::Shutdown does not wait for this thread to finish. Silent with the factor off.
    if(UcslamSettings().useDepthFactor)
        cout << "[UCSLAM] chain culls=" << mnLogScaleCulls << " with_state=" << mnLogScaleCullsWithState
             << " head_transfers=" << mnLogScaleHeadTransfers << " depth_edges_folded=" << mnLogScaleDepthEdgesFolded
             << " depth_edges_gated=" << mnLogScaleDepthEdgesGated << " stored_terms=" << Map::CountLogScaleChainTerms()
             << " merge_priors=" << Map::GetLogScaleMergePriors().size() << endl;

    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if (mbResetRequested)
        return;

    float minTime;
    int nMinKF;
    if (mbMonocular)
    {
        minTime = 2.0;
        nMinKF = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF = 10;
    }


    if(mpAtlas->KeyFramesInMap()<nMinKF)
        return;

    // Retrieve all keyframe in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    if(vpKF.size()<nMinKF)
        return;

    mFirstTs=vpKF.front()->mTimeStamp;
    if(mpCurrentKeyFrame->mTimeStamp-mFirstTs<minTime)
        return;

    bInitializing = true;

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);

    // Compute and KF velocities mRwg estimation
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        Eigen::Matrix3f Rwg;
        Eigen::Vector3f dirG;
        dirG.setZero();
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;

            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }

        dirG = dirG/dirG.norm();
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f v = gI.cross(dirG);
        const float nv = v.norm();
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        Eigen::Vector3f vzg = v*ang/nv;
        Rwg = Sophus::SO3f::exp(vzg).matrix();
        mRwg = Rwg.cast<double>();
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs;
        mUcslamRefineLogSum = 0.0;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = mpCurrentKeyFrame->GetGyroBias().cast<double>();
        mba = mpCurrentKeyFrame->GetAccBias().cast<double>();
    }

    mScale=1.0;

    mInitTime = mpTracker->mLastFrame.mTimeStamp-vpKF.front()->mTimeStamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, mbMonocular, infoInertial, false, false, priorG, priorA);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if(UcslamSettings().diag || UcslamSettings().logInitState)
        cout << "[UCSLAM] diag init kf=" << mpCurrentKeyFrame->mnId << " t=" << std::fixed << std::setprecision(6)
             << mpCurrentKeyFrame->mTimeStamp << std::defaultfloat << " priorG=" << priorG << " priorA=" << priorA
             << " imu_init=" << (mpAtlas->isImuInitialized() ? 1 : 0) << " kfs=" << N << " scale=" << mScale
             << " a_cur=" << mpCurrentKeyFrame->mLogScale << " rot_deg=" << std::acos(std::max(-1.0, std::min(1.0, 0.5 * (mRwg.trace() - 1.0)))) * 180.0 / M_PI
             << " bg=" << mbg.transpose() << " ba=" << mba.transpose()
             << " opt_ms=" << std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count()
             << " queue=" << KeyframesInQueue() << endl;

    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }

    // UC-SLAM (attempt 07): the first accepted IMU init of this map stores its gyro bias and the up vector, the unit mean
    // accelerometer (specific force, body frame) of the keyframe preintegrations of the 2 s before the init keyframe.
    if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && (UcslamSettings().youngMapScaleGate > 0.f || UcslamSettings().logInitState))
    {
        Eigen::Vector3d accSum = Eigen::Vector3d::Zero();
        int nAcc = 0;
        const double tInit = mpCurrentKeyFrame->mTimeStamp;
        for(KeyFrame* pKFi : vpKF)
        {
            if(!pKFi || !pKFi->mpImuPreintegrated || pKFi->mTimeStamp < tInit - 2.0 || pKFi->mTimeStamp > tInit)
                continue;
            int nKF = 0;
            accSum += pKFi->mpImuPreintegrated->UcslamSumAcc(nKF);
            nAcc += nKF;
        }
        mUcslamBg0 = mbg;
        mUcslamUp0 = (nAcc > 0 && accSum.norm() > 0.0) ? Eigen::Vector3d(accSum / accSum.norm()) : Eigen::Vector3d::Zero();
        mbUcslamInit0 = true;
        char bufInit[256];
        snprintf(bufInit, sizeof(bufInit),
                 "[UCSLAM] young_map_init kf=%lu t=%.6f bg0=%.9f,%.9f,%.9f up0=%.9f,%.9f,%.9f bgv=%.9f n_acc=%d",
                 mpCurrentKeyFrame->mnId, mpCurrentKeyFrame->mTimeStamp, mUcslamBg0(0), mUcslamBg0(1), mUcslamBg0(2),
                 mUcslamUp0(0), mUcslamUp0(1), mUcslamUp0(2), mUcslamBg0.dot(mUcslamUp0), nAcc);
        cout << bufInit << endl;
    }

    // UC-SLAM, UCSLAM.ImuInitHandoff (docs/fork_design.md Sec. 7). While bInitializing is set the tracker
    // may insert keyframes, and the inertial optimization above takes long enough for one to arrive.
    // Stock then hands the tracker mpCurrentKeyFrame, which is older than the tracker's own last
    // keyframe, while the tracker's preintegration already starts at the newer one. The tracker's IMU
    // prediction is then wrong by the motion between the two keyframes. With the key on, the queue is
    // drained again under the map mutex, which the tracker holds for a whole frame, so the newest
    // keyframe is in the map when it is rotated and scaled and is the one handed to the tracker.
    const bool bHandoff = UcslamSettings().imuInitHandoff != 0;

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        int nLate = 0;
        if(bHandoff)
        {
            const bool bFirstInit = !mpAtlas->isImuInitialized();
            const IMU::Bias bNew = vpKF[0]->GetImuBias();
            while(CheckNewKeyFrames())
            {
                ProcessNewKeyFrame();
                KeyFrame* pLate = mpCurrentKeyFrame;
                vpKF.push_back(pLate);
                lpKF.push_back(pLate);
                nLate++;
                // The state InertialOptimization gives the keyframes it saw: the new biases, and at the
                // first initialization the velocity from the positions, as in the loop at the top.
                if((pLate->GetGyroBias() - Eigen::Vector3f(bNew.bwx, bNew.bwy, bNew.bwz)).norm() > 0.01f)
                {
                    pLate->SetNewBias(bNew);
                    if(pLate->mpImuPreintegrated)
                        pLate->mpImuPreintegrated->Reintegrate();
                }
                else
                    pLate->SetNewBias(bNew);
                if(bFirstInit && pLate->mPrevKF && pLate->mpImuPreintegrated && pLate->mpImuPreintegrated->dT > 0.f)
                    pLate->SetVelocity((pLate->GetImuPosition() - pLate->mPrevKF->GetImuPosition()) / pLate->mpImuPreintegrated->dT);
            }
        }
        if(UcslamSettings().diag)
            cout << "[UCSLAM] diag init_handoff on=" << (bHandoff ? 1 : 0) << " late_kfs=" << nLate
                 << " lm_kf=" << mpCurrentKeyFrame->mnId << endl;

        if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true);
            mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
        }

        // Check if initialization OK
        if (!mpAtlas->isImuInitialized())
            for (int i = 0; i < (int)vpKF.size(); i++) {
                KeyFrame *pKF2 = vpKF[i];
                pKF2->bImu = true;
            }

        if(bHandoff)
        {
            mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
            if (!mpAtlas->isImuInitialized())
            {
                mpAtlas->SetImuInitialized();
                mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
                mpCurrentKeyFrame->bImu = true;
            }
        }
    }

    if(!bHandoff)
    {
        mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
        if (!mpAtlas->isImuInitialized())
        {
            mpAtlas->SetImuInitialized();
            mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
            mpCurrentKeyFrame->bImu = true;
        }
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (bFIBA)
    {
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    Verbose::PrintMess("Global Bundle Adjustment finished\nUpdating map ...", Verbose::VERBOSITY_NORMAL);

    // Get Map Mutex
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

    unsigned long GBAid = mpCurrentKeyFrame->mnId;

    // Process keyframes in the queue
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Correct keyframes starting at map first keyframe
    list<KeyFrame*> lpKFtoCheck(mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.begin(),mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.end());

    while(!lpKFtoCheck.empty())
    {
        KeyFrame* pKF = lpKFtoCheck.front();
        const set<KeyFrame*> sChilds = pKF->GetChilds();
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
        {
            KeyFrame* pChild = *sit;
            if(!pChild || pChild->isBad())
                continue;

            if(pChild->mnBAGlobalForKF!=GBAid)
            {
                Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;

                Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                if(pChild->isVelocitySet()){
                    pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                }
                else {
                    Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);
                }

                pChild->mBiasGBA = pChild->GetImuBias();
                pChild->mnBAGlobalForKF = GBAid;

            }
            lpKFtoCheck.push_back(pChild);
        }

        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);

        if(pKF->bImu)
        {
            pKF->mVwbBefGBA = pKF->GetVelocity();
            pKF->SetVelocity(pKF->mVwbGBA);
            pKF->SetNewBias(pKF->mBiasGBA);
        } else {
            cout << "KF " << pKF->mnId << " not set to inertial!! \n";
        }

        lpKFtoCheck.pop_front();
    }

    // Correct MapPoints
    const vector<MapPoint*> vpMPs = mpAtlas->GetCurrentMap()->GetAllMapPoints();

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        if(pMP->mnBAGlobalForKF==GBAid)
        {
            // If optimized by Global BA, just update
            pMP->SetWorldPos(pMP->mPosGBA);
        }
        else
        {
            // Update according to the correction of its reference keyframe
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

            if(pRefKF->mnBAGlobalForKF!=GBAid)
                continue;

            // Map to non-corrected camera
            Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

            // Backproject using corrected camera
            pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
        }
    }

    Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);

    mnKFs=vpKF.size();
    mIdxInit++;

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    mpTracker->mState=Tracking::OK;
    bInitializing = false;

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);

    if(UcslamSettings().diag)
        cout << "[UCSLAM] diag scale_refinement kf=" << mpCurrentKeyFrame->mnId << " t=" << std::fixed
             << std::setprecision(6) << mpCurrentKeyFrame->mTimeStamp << std::defaultfloat << " scale=" << mScale
             << " a_cur=" << mpCurrentKeyFrame->mLogScale << endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }

    // UC-SLAM (UCSLAM.ScaleRefineMaxLog c > 0, protocol A9): trust region on the refinements of this map. The fitted
    // factor is clamped so that the sum of the applied logs stays within [-c, c]. The dead band below is unchanged.
    // With the cap on, each refinement prints its fitted and applied factor, with or without Diag.
    const double dUcslamMaxLog = UcslamSettings().scaleRefineMaxLog;
    const double dScaleFitted = mScale;
    if(dUcslamMaxLog > 0.0)
    {
        const double l = std::log(mScale);
        const double lc = std::min(std::max(l, -dUcslamMaxLog - mUcslamRefineLogSum), dUcslamMaxLog - mUcslamRefineLogSum);
        mScale = std::exp(lc);
    }
    const bool bApplyRefinement = (fabs(mScale-1.f)>0.002)||!mbMonocular;
    if(dUcslamMaxLog > 0.0 && bApplyRefinement)
        mUcslamRefineLogSum += std::log(mScale);
    if(dUcslamMaxLog > 0.0)
        cout << "[UCSLAM] scale_refine_cap kf=" << mpCurrentKeyFrame->mnId << " fitted=" << dScaleFitted
             << " applied=" << (bApplyRefinement ? mScale : 1.0) << " logsum=" << mUcslamRefineLogSum
             << " cap=" << dUcslamMaxLog << endl;

    Sophus::SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if (bApplyRefinement)
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Tgw,mScale,true);
        mpTracker->UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}



bool LocalMapping::IsInitializing()
{
    return bInitializing;
}


double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

KeyFrame* LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

} //namespace ORB_SLAM
