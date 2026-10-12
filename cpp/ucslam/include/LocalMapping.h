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


#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Atlas.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "Settings.h"

#include <mutex>


namespace ORB_SLAM3
{

class System;
class Tracking;
class LoopClosing;
class Atlas;
class PriorNet;

class LocalMapping
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LocalMapping(System* pSys, Atlas* pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName=std::string());

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // UC-SLAM: every keyframe this thread processes is handed to the PriorNet queue. NULL disables the hook.
    void SetPriorNet(PriorNet* pPriorNet);

    // UC-SLAM settings, the UCSLAM.* keys. One block for every thread, held by
    // Optimizer::UcslamParams(). System registers it once after parsing the
    // settings, and until it does the optimizer reads the keys from the settings
    // file on the command line. The defaults are the stock behaviour, so nothing
    // below changes until UseDepthFactor is 1.
    void SetUcslamSettings(const Settings::Ucslam& u);
    const Settings::Ucslam& UcslamSettings() const;

    // UC-SLAM counters of the log-scale marginalization on culling, printed at finish.
    int mnLogScaleCulls;             // culled keyframes that went through the marginalization
    int mnLogScaleCullsWithState;    // of those, keyframes with a log-scale state (mLogScaleVar > 0)
    int mnLogScaleHeadTransfers;     // of those, keyframes that carried the absolute a_0 prior
    long mnLogScaleDepthEdgesFolded; // depth edges inside the quadratic region folded into the neighbours
    long mnLogScaleDepthEdgesGated;  // depth edges outside it, dropped

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);
    void EmptyQueue();

    // Thread Synch
    void RequestStop();
    void RequestReset();
    void RequestResetActiveMap(Map* pMap);
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    bool IsInitializing();
    double GetCurrKFTime();
    KeyFrame* GetCurrKF();

    std::mutex mMutexImuInit;

    Eigen::MatrixXd mcovInertial;
    Eigen::Matrix3d mRwg;
    Eigen::Vector3d mbg;
    Eigen::Vector3d mba;
    double mScale;
    double mInitTime;
    double mCostTime;

    unsigned int mInitSect;
    unsigned int mIdxInit;
    unsigned int mnKFs;
    double mFirstTs;
    int mnMatchesInliers;

    // For debugging (erase in normal mode)
    int mInitFr;
    int mIdxIteration;
    string strSequence;

    bool mbNotBA1;
    bool mbNotBA2;
    bool mbBadImu;

    bool mbWriteStats;

    // not consider far points (clouds)
    bool mbFarPoints;
    float mThFarPoints;

#ifdef REGISTER_TIMES
    vector<double> vdKFInsert_ms;
    vector<double> vdMPCulling_ms;
    vector<double> vdMPCreation_ms;
    vector<double> vdLBA_ms;
    vector<double> vdKFCulling_ms;
    vector<double> vdLMTotal_ms;


    vector<double> vdLBASync_ms;
    vector<double> vdKFCullingSync_ms;
    vector<int> vnLBA_edges;
    vector<int> vnLBA_KFopt;
    vector<int> vnLBA_KFfixed;
    vector<int> vnLBA_MPs;
    int nLBA_exec;
    int nLBA_abort;
#endif
protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    // UC-SLAM: the declaration of CreateDepthPriorMapPoints() goes here. It is
    // added by the depth-prior points step, docs/fork_design.md Sec. 2.

    void MapPointCulling();
    void SearchInNeighbors();
    void KeyFrameCulling();

    // UC-SLAM log-scale chain marginalization on culling, docs/fork_design.md
    // Sec. 2 and ucslam/factors.py. Called from KeyFrameCulling before the
    // relink of mPrevKF and mNextKF, only when UseDepthFactor is 1.
    void MarginalizeLogScaleOnCull(KeyFrame* pKF);
    // Chain information of the edge mPrevKF -> pKF: the merged value a cull
    // stored in the Map, else the random walk 1 / (sigma_rw^2 dt). 0 without mPrevKF.
    double LogScaleSegmentInfo(KeyFrame* pKF);
    // Unary information S and information-vector entry b of the depth edges of
    // pKF at the current estimate, inside the Huber quadratic region only.
    void DepthInformationOfKeyFrame(KeyFrame* pKF, double& S, double& b, int& nInside, int& nGated);

    System *mpSystem;

    bool mbMonocular;
    bool mbInertial;

    void ResetIfRequested();
    bool mbResetRequested;
    bool mbResetRequestedActiveMap;
    Map* mpMapToReset;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Atlas* mpAtlas;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;
    PriorNet* mpPriorNet;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;

    void InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);
    void ScaleRefinement();

    bool bInitializing;

    Eigen::MatrixXd infoInertial;
    int mNumLM;
    int mNumKFCulling;

    float mTinit;
    // UC-SLAM (UCSLAM.ScaleRefineMaxLog): sum of the logs of the scale refinements applied to the current map
    // since its IMU initialization. Reset wherever mTinit is reset.
    double mUcslamRefineLogSum = 0.0;
    // UC-SLAM (UCSLAM.YoungMapScaleGate): set after the first printed local inertial BA of the run, never reset, so the
    // gate decides at most once per run.
    bool mbUcslamGateChecked = false;
    // UC-SLAM (attempt 07): gyro bias bg0 and unit up vector up0 (mean accelerometer over the 2 s before the init keyframe)
    // of the first accepted IMU init of the current map. Cleared wherever mTinit is reset.
    bool mbUcslamInit0 = false;
    Eigen::Vector3d mUcslamBg0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d mUcslamUp0 = Eigen::Vector3d::Zero();

    int countRefinement;

    //DEBUG
    ofstream f_lm;

    };

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
