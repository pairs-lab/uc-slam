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


#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"

#include <set>
#include <map>
#include <vector>
#include <pangolin/pangolin.h>
#include <mutex>
#include <Eigen/Core>

#include <boost/serialization/base_object.hpp>


namespace ORB_SLAM3
{

class MapPoint;
class KeyFrame;
class Atlas;
class KeyFrameDatabase;

/*
 * UC-SLAM. Random-walk chain of log-scale states a_0 .. a_n with unary terms,
 * the C++ port of ucslam/factors.py (ChainModel, cull_keyframe). Segment k joins
 * node k and node k+1 with variance segVar[k]. Node k carries unary information
 * S[k] and information-vector entry b[k]. The absolute a_0 prior N(0, sigma0Sq)
 * sits on node 0 only. A prior with a nonzero mean m enters through b[0] as
 * m / sigma0Sq, as factors.py does. test/test_chain.cc checks every branch of
 * Cull against a dense Schur complement. docs/fork_design.md Sec. 2.
 */
struct LogScaleChain
{
    std::vector<double> segVar;
    std::vector<double> S;
    std::vector<double> b;
    double sigma0Sq;

    LogScaleChain();
    LogScaleChain(const std::vector<double>& vSegVar, const std::vector<double>& vS,
                  const std::vector<double>& vb, double s0Sq);
    int NumNodes() const {return (int)segVar.size() + 1;}

    // Dense (H, b) of the chain with the unary terms and the a_0 prior.
    void Dense(Eigen::MatrixXd& H, Eigen::VectorXd& bvec) const;

    // Exact Schur elimination of node k on the chain representation. Any node,
    // the head that carries the a_0 prior and the tail included.
    LogScaleChain Cull(int k) const;

    // The primitive every branch uses. w1 is the chain information to the previous
    // node and w2 to the next one, 0 when that neighbour does not exist. S and b
    // are the culled node's total unary terms. With D = w1 + w2 + S the merged
    // segment information is w1 w2 / D, the previous node receives the unary
    // share (w1 S / D, w1 b / D) and the next node (w2 S / D, w2 b / D).
    static void CullShares(double w1, double w2, double S, double b,
                           double& wMerged, double& Sprev, double& bprev, double& Snext, double& bnext);

    // Dense reference. Schur complement of (H, b) that removes state idx.
    static void MarginalizeDense(const Eigen::MatrixXd& H, const Eigen::VectorXd& b, int idx,
                                 Eigen::MatrixXd& Hred, Eigen::VectorXd& bred);
};

class Map
{
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & mnId;
        ar & mnInitKFid;
        ar & mnMaxKFid;
        ar & mnBigChangeIdx;

        // Save/load a set structure, the set structure is broken in libboost 1.58 for ubuntu 16.04, a vector is serializated
        //ar & mspKeyFrames;
        //ar & mspMapPoints;
        ar & mvpBackupKeyFrames;
        ar & mvpBackupMapPoints;

        ar & mvBackupKeyFrameOriginsId;

        ar & mnBackupKFinitialID;
        ar & mnBackupKFlowerID;

        ar & mbImuInitialized;
        ar & mbIsInertial;
        ar & mbIMU_BA1;
        ar & mbIMU_BA2;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Map();
    Map(int initKFid);
    ~Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetId();

    long unsigned int GetInitKFid();
    void SetInitKFid(long unsigned int initKFif);
    long unsigned int GetMaxKFid();

    KeyFrame* GetOriginKF();

    void SetCurrentMap();
    void SetStoredMap();

    bool HasThumbnail();
    bool IsInUse();

    void SetBad();
    bool IsBad();

    void clear();

    int GetMapChangeIndex();
    void IncreaseChangeIndex();
    int GetLastMapChange();
    void SetLastMapChange(int currentChangeId);

    void SetImuInitialized();
    bool isImuInitialized();

    void ApplyScaledRotation(const Sophus::SE3f &T, const float s, const bool bScaledVel=false);

    void SetInertialSensor();
    bool IsInertial();
    void SetIniertialBA1();
    void SetIniertialBA2();
    bool GetIniertialBA1();
    bool GetIniertialBA2();

    void PrintEssentialGraph();
    bool CheckEssentialGraph();
    void ChangeId(long unsigned int nId);

    unsigned int GetLowerKFID();

    void PreSave(std::set<GeometricCamera*> &spCams);
    void PostLoad(KeyFrameDatabase* pKFDB, ORBVocabulary* pORBVoc/*, map<long unsigned int, KeyFrame*>& mpKeyFrameId*/, map<unsigned int, GeometricCamera*> &mpCams);

    void printReprojectionError(list<KeyFrame*> &lpLocalWindowKFs, KeyFrame* mpCurrentKF, string &name, string &name_folder);

    /*
     * UC-SLAM log-scale chain terms that the KeyFrame fields cannot hold,
     * docs/fork_design.md Sec. 2, "Marginalization on culling". Written by
     * LocalMapping::KeyFrameCulling. The local BA reads them when it rebuilds
     * the chain: the chain edge from mPrevKF to a keyframe takes segInfo when
     * it is positive and 1 / (sigma_rw^2 dt) otherwise, and a keyframe with
     * unaryInfo > 0 gets one more unary prior edge with mean unaryB / unaryInfo
     * and information unaryInfo, next to the absolute prior of the head. The
     * table is keyed by the keyframe pointer and shared by all maps, because a
     * keyframe keeps its chain edge when a merge moves it to another map.
     * The marginals mLogScale and mLogScaleVar of the neighbours are not
     * touched by a cull. Exact marginalization leaves them unchanged.
     */
    struct LogScaleChainTerms
    {
        double segInfo;    // information of the chain edge mPrevKF -> this keyframe, <= 0 means the stock 1 / (sigma_rw^2 dt)
        double unaryInfo;  // unary information inherited from culled neighbours, 0 when none
        double unaryB;     // its information-vector entry, the unary mean is unaryB / unaryInfo
        LogScaleChainTerms(): segInfo(-1.0), unaryInfo(0.0), unaryB(0.0) {}
    };
    static bool GetLogScaleChainTerms(KeyFrame* pKF, LogScaleChainTerms& terms);
    static void SetLogScaleChainTerms(KeyFrame* pKF, const LogScaleChainTerms& terms);
    static void EraseLogScaleChainTerms(KeyFrame* pKF);
    static size_t CountLogScaleChainTerms();

    /*
     * Relative log-scale prior a_absorbed - a_surviving ~ N(mean, var) that
     * re-anchors the chain head of an absorbed Atlas map to the head of the
     * surviving one. Written by LoopClosing on a merge, read by the BA as one
     * EdgeLogScaleRW-like binary edge. Each map then carries exactly one
     * absolute prior, docs/fork_design.md Sec. 2.
     */
    struct LogScaleMergePrior
    {
        KeyFrame* pAbsorbedHead;
        KeyFrame* pSurvivingHead;
        double mean;
        double var;
    };
    static void AddLogScaleMergePrior(const LogScaleMergePrior& prior);
    static std::vector<LogScaleMergePrior> GetLogScaleMergePriors();

    // Keyframes of this map that carry the absolute log-scale prior (mbLogScalePrior).
    // Exactly one is expected once the chain is active. GetLogScaleHead returns the
    // one with the lowest id, or NULL when there is none.
    int CountLogScalePriors();
    KeyFrame* GetLogScaleHead();

    vector<KeyFrame*> mvpKeyFrameOrigins;
    vector<unsigned long int> mvBackupKeyFrameOriginsId;
    KeyFrame* mpFirstRegionKF;
    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    bool mbFail;

    // Size of the thumbnail (always in power of 2)
    static const int THUMB_WIDTH = 512;
    static const int THUMB_HEIGHT = 512;

    static long unsigned int nNextId;

    // DEBUG: show KFs which are used in LBA
    std::set<long unsigned int> msOptKFs;
    std::set<long unsigned int> msFixedKFs;

protected:

    long unsigned int mnId;

    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;

    // Save/load, the set structure is broken in libboost 1.58 for ubuntu 16.04, a vector is serializated
    std::vector<MapPoint*> mvpBackupMapPoints;
    std::vector<KeyFrame*> mvpBackupKeyFrames;

    KeyFrame* mpKFinitial;
    KeyFrame* mpKFlowerID;

    unsigned long int mnBackupKFinitialID;
    unsigned long int mnBackupKFlowerID;

    std::vector<MapPoint*> mvpReferenceMapPoints;

    bool mbImuInitialized;

    int mnMapChange;
    int mnMapChangeNotified;

    long unsigned int mnInitKFid;
    long unsigned int mnMaxKFid;
    //long unsigned int mnLastLoopKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;


    // View of the map in aerial sight (for the AtlasViewer)
    GLubyte* mThumbnail;

    bool mIsInUse;
    bool mHasTumbnail;
    bool mbBad = false;

    bool mbIsInertial;
    bool mbIMU_BA1;
    bool mbIMU_BA2;

    // Mutex
    std::mutex mMutexMap;

    // UC-SLAM chain bookkeeping shared by all maps, see LogScaleChainTerms.
    static std::mutex msMutexLogScale;
    static std::map<KeyFrame*, LogScaleChainTerms> msLogScaleChainTerms;
    static std::vector<LogScaleMergePrior> msLogScaleMergePriors;

};

} //namespace ORB_SLAM3

#endif // MAP_H
