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


#include "Map.h"

#include<mutex>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ORB_SLAM3
{

long unsigned int Map::nNextId=0;

// ---------------------------------------------------------------------------
// UC-SLAM. LogScaleChain, the C++ port of ucslam/factors.py (ChainModel,
// chain_information, marginalize_middle, cull_keyframe). Every formula below is
// the one certified in tests/test_factors.py and again in test/test_chain.cc.
// ---------------------------------------------------------------------------

LogScaleChain::LogScaleChain(): sigma0Sq(std::numeric_limits<double>::infinity())
{
    S.assign(1, 0.0);
    b.assign(1, 0.0);
}

LogScaleChain::LogScaleChain(const std::vector<double>& vSegVar, const std::vector<double>& vS,
                             const std::vector<double>& vb, double s0Sq):
    segVar(vSegVar), S(vS), b(vb), sigma0Sq(s0Sq)
{
    const size_t n = segVar.size() + 1;
    if(S.empty())
        S.assign(n, 0.0);
    if(b.empty())
        b.assign(n, 0.0);
    if(S.size() != n || b.size() != n)
        throw std::runtime_error("LogScaleChain: S and b need one entry per node");
}

void LogScaleChain::Dense(Eigen::MatrixXd& H, Eigen::VectorXd& bvec) const
{
    const int n = NumNodes();
    H = Eigen::MatrixXd::Zero(n, n);
    for(int k = 0; k < n - 1; k++)
    {
        const double w = 1.0 / segVar[k];
        H(k, k) += w;
        H(k + 1, k + 1) += w;
        H(k, k + 1) -= w;
        H(k + 1, k) -= w;
    }
    for(int k = 0; k < n; k++)
        H(k, k) += S[k];
    if(std::isfinite(sigma0Sq))
        H(0, 0) += 1.0 / sigma0Sq;
    bvec = Eigen::Map<const Eigen::VectorXd>(b.data(), n);
}

void LogScaleChain::CullShares(double w1, double w2, double S, double b,
                               double& wMerged, double& Sprev, double& bprev, double& Snext, double& bnext)
{
    const double D = w1 + w2 + S;
    if(D <= 0.0)
    {
        // A node without any information. Nothing to hand over.
        wMerged = 0.0;
        Sprev = bprev = Snext = bnext = 0.0;
        return;
    }
    wMerged = w1 * w2 / D;
    Sprev = w1 * S / D;
    bprev = w1 * b / D;
    Snext = w2 * S / D;
    bnext = w2 * b / D;
}

LogScaleChain LogScaleChain::Cull(int k) const
{
    const int n = NumNodes();
    if(k < 0 || k >= n)
        throw std::runtime_error("LogScaleChain::Cull: node index out of range");
    if(n < 2)
        throw std::runtime_error("LogScaleChain::Cull: cannot cull the only node of a chain");

    std::vector<double> var(segVar), Sn(S), bn(b);
    double s0 = sigma0Sq;
    double wMerged, Sp, bp, Sx, bx;

    if(k == 0)
    {
        // Head. The whole absolute information of the head, w2 (S_0 + 1/sigma0Sq) / D,
        // becomes the a_0 prior of node 1, together with its information-vector share.
        const double w2 = 1.0 / var[0];
        const double priorInfo = std::isfinite(s0) ? 1.0 / s0 : 0.0;
        CullShares(0.0, w2, Sn[0] + priorInfo, bn[0], wMerged, Sp, bp, Sx, bx);
        const double newS0 = (Sx > 0.0) ? 1.0 / Sx : std::numeric_limits<double>::infinity();
        bn[1] += bx;
        return LogScaleChain(std::vector<double>(var.begin() + 1, var.end()),
                             std::vector<double>(Sn.begin() + 1, Sn.end()),
                             std::vector<double>(bn.begin() + 1, bn.end()), newS0);
    }

    if(k == n - 1)
    {
        // Tail. The last segment is dropped and the unary share goes to node n-1.
        const double w1 = 1.0 / var[n - 2];
        CullShares(w1, 0.0, Sn[n - 1], bn[n - 1], wMerged, Sp, bp, Sx, bx);
        Sn[n - 2] += Sp;
        bn[n - 2] += bp;
        var.pop_back();
        Sn.pop_back();
        bn.pop_back();
        return LogScaleChain(var, Sn, bn, s0);
    }

    // Interior node. Merged segment w1 w2 / D, unary and information-vector shares to both neighbours.
    const double w1 = 1.0 / var[k - 1];
    const double w2 = 1.0 / var[k];
    CullShares(w1, w2, Sn[k], bn[k], wMerged, Sp, bp, Sx, bx);
    Sn[k - 1] += Sp;
    Sn[k + 1] += Sx;
    bn[k - 1] += bp;
    bn[k + 1] += bx;
    var[k - 1] = 1.0 / wMerged;
    var.erase(var.begin() + k);
    Sn.erase(Sn.begin() + k);
    bn.erase(bn.begin() + k);
    return LogScaleChain(var, Sn, bn, s0);
}

void LogScaleChain::MarginalizeDense(const Eigen::MatrixXd& H, const Eigen::VectorXd& b, int idx,
                                     Eigen::MatrixXd& Hred, Eigen::VectorXd& bred)
{
    const int n = H.rows();
    std::vector<int> keep;
    for(int i = 0; i < n; i++)
        if(i != idx)
            keep.push_back(i);
    const int m = (int)keep.size();
    Eigen::MatrixXd Haa(m, m);
    Eigen::VectorXd Hab(m), ba(m);
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < m; j++)
            Haa(i, j) = H(keep[i], keep[j]);
        Hab(i) = H(keep[i], idx);
        ba(i) = b(keep[i]);
    }
    const double Hbb = H(idx, idx);
    Hred = Haa - (Hab * Hab.transpose()) / Hbb;
    bred = ba - Hab * (b(idx) / Hbb);
}

// UC-SLAM chain bookkeeping shared by all maps.
std::mutex Map::msMutexLogScale;
std::map<KeyFrame*, Map::LogScaleChainTerms> Map::msLogScaleChainTerms;
std::vector<Map::LogScaleMergePrior> Map::msLogScaleMergePriors;

bool Map::GetLogScaleChainTerms(KeyFrame* pKF, LogScaleChainTerms& terms)
{
    unique_lock<mutex> lock(msMutexLogScale);
    std::map<KeyFrame*, LogScaleChainTerms>::const_iterator it = msLogScaleChainTerms.find(pKF);
    if(it == msLogScaleChainTerms.end())
    {
        terms = LogScaleChainTerms();
        return false;
    }
    terms = it->second;
    return true;
}

void Map::SetLogScaleChainTerms(KeyFrame* pKF, const LogScaleChainTerms& terms)
{
    unique_lock<mutex> lock(msMutexLogScale);
    msLogScaleChainTerms[pKF] = terms;
}

void Map::EraseLogScaleChainTerms(KeyFrame* pKF)
{
    unique_lock<mutex> lock(msMutexLogScale);
    msLogScaleChainTerms.erase(pKF);
}

size_t Map::CountLogScaleChainTerms()
{
    unique_lock<mutex> lock(msMutexLogScale);
    return msLogScaleChainTerms.size();
}

void Map::AddLogScaleMergePrior(const LogScaleMergePrior& prior)
{
    unique_lock<mutex> lock(msMutexLogScale);
    msLogScaleMergePriors.push_back(prior);
}

std::vector<Map::LogScaleMergePrior> Map::GetLogScaleMergePriors()
{
    unique_lock<mutex> lock(msMutexLogScale);
    return msLogScaleMergePriors;
}

int Map::CountLogScalePriors()
{
    // Copy first. KeyFrame::isBad locks the keyframe, and SetBadFlag locks the
    // keyframe before it asks the map for its id, so the map mutex is not held here.
    vector<KeyFrame*> vpKFs = GetAllKeyFrames();
    int n = 0;
    for(size_t i = 0; i < vpKFs.size(); i++)
        if(vpKFs[i] && vpKFs[i]->mbLogScalePrior && !vpKFs[i]->isBad())
            n++;
    return n;
}

KeyFrame* Map::GetLogScaleHead()
{
    vector<KeyFrame*> vpKFs = GetAllKeyFrames();
    KeyFrame* pHead = static_cast<KeyFrame*>(NULL);
    for(size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(!pKF || !pKF->mbLogScalePrior || pKF->isBad())
            continue;
        if(!pHead || pKF->mnId < pHead->mnId)
            pHead = pKF;
    }
    return pHead;
}

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0), mbImuInitialized(false), mnMapChange(0), mpFirstRegionKF(static_cast<KeyFrame*>(NULL)),
mbFail(false), mIsInUse(false), mHasTumbnail(false), mbBad(false), mnMapChangeNotified(0), mbIsInertial(false), mbIMU_BA1(false), mbIMU_BA2(false)
{
    mnId=nNextId++;
    mThumbnail = static_cast<GLubyte*>(NULL);
}

Map::Map(int initKFid):mnInitKFid(initKFid), mnMaxKFid(initKFid),/*mnLastLoopKFid(initKFid),*/ mnBigChangeIdx(0), mIsInUse(false),
                       mHasTumbnail(false), mbBad(false), mbImuInitialized(false), mpFirstRegionKF(static_cast<KeyFrame*>(NULL)),
                       mnMapChange(0), mbFail(false), mnMapChangeNotified(0), mbIsInertial(false), mbIMU_BA1(false), mbIMU_BA2(false)
{
    mnId=nNextId++;
    mThumbnail = static_cast<GLubyte*>(NULL);
}

Map::~Map()
{
    //TODO: erase all points from memory
    mspMapPoints.clear();

    //TODO: erase all keyframes from memory
    mspKeyFrames.clear();

    if(mThumbnail)
        delete mThumbnail;
    mThumbnail = static_cast<GLubyte*>(NULL);

    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    if(mspKeyFrames.empty()){
        cout << "First KF:" << pKF->mnId << "; Map init KF:" << mnInitKFid << endl;
        mnInitKFid = pKF->mnId;
        mpKFinitial = pKF;
        mpKFlowerID = pKF;
    }
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
    {
        mnMaxKFid=pKF->mnId;
    }
    if(pKF->mnId<mpKFlowerID->mnId)
    {
        mpKFlowerID = pKF;
    }
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::SetImuInitialized()
{
    unique_lock<mutex> lock(mMutexMap);
    mbImuInitialized = true;
}

bool Map::isImuInitialized()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbImuInitialized;
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
    if(mspKeyFrames.size()>0)
    {
        if(pKF->mnId == mpKFlowerID->mnId)
        {
            vector<KeyFrame*> vpKFs = vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
            sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
            mpKFlowerID = vpKFs[0];
        }
    }
    else
    {
        mpKFlowerID = 0;
    }

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetId()
{
    return mnId;
}
long unsigned int Map::GetInitKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnInitKFid;
}

void Map::SetInitKFid(long unsigned int initKFif)
{
    unique_lock<mutex> lock(mMutexMap);
    mnInitKFid = initKFif;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

KeyFrame* Map::GetOriginKF()
{
    return mpKFinitial;
}

void Map::SetCurrentMap()
{
    mIsInUse = true;
}

void Map::SetStoredMap()
{
    mIsInUse = false;
}

void Map::clear()
{
//    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
//        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
    {
        KeyFrame* pKF = *sit;
        pKF->UpdateMap(static_cast<Map*>(NULL));
        EraseLogScaleChainTerms(pKF);   // UC-SLAM: the chain of a cleared map is gone
//        delete *sit;
    }

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = mnInitKFid;
    mbImuInitialized = false;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
    mbIMU_BA1 = false;
    mbIMU_BA2 = false;
}

bool Map::IsInUse()
{
    return mIsInUse;
}

void Map::SetBad()
{
    mbBad = true;
}

bool Map::IsBad()
{
    return mbBad;
}


void Map::ApplyScaledRotation(const Sophus::SE3f &T, const float s, const bool bScaledVel)
{
    unique_lock<mutex> lock(mMutexMap);

    // Body position (IMU) of first keyframe is fixed to (0,0,0)
    Sophus::SE3f Tyw = T;
    Eigen::Matrix3f Ryw = Tyw.rotationMatrix();
    Eigen::Vector3f tyw = Tyw.translation();

    const bool bShiftLogScale = (s > 0.f) && (fabs(s - 1.f) > 1e-7f);
    const double logS = bShiftLogScale ? log((double)s) : 0.0;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(); sit!=mspKeyFrames.end(); sit++)
    {
        KeyFrame* pKF = *sit;
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Twc.translation() *= s;
        Sophus::SE3f Tyc = Tyw*Twc;
        Sophus::SE3f Tcy = Tyc.inverse();
        pKF->SetPose(Tcy);
        Eigen::Vector3f Vw = pKF->GetVelocity();
        if(!bScaledVel)
            pKF->SetVelocity(Ryw*Vw);
        else
            pKF->SetVelocity(Ryw*Vw*s);

        // UC-SLAM. Every depth in the map is now s times larger, so a keyframe
        // with a log-scale state keeps its depth residuals log z - a_k - log z_hat
        // only if a_k moves by log s. The inherited unary means of earlier culls
        // are in map units too and move with it. The absolute prior mean is a
        // statement about the network's calibration in a metric map and stays.
        // Keyframes without a state (mLogScaleVar == 0) are untouched, so the
        // stock path with the depth factor off never enters here.
        if(bShiftLogScale && pKF->mLogScaleVar > 0.0)
        {
            pKF->mLogScale += logS;
            LogScaleChainTerms terms;
            if(GetLogScaleChainTerms(pKF, terms) && terms.unaryInfo > 0.0)
            {
                terms.unaryB += terms.unaryInfo * logS;
                SetLogScaleChainTerms(pKF, terms);
            }
        }
    }
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(); sit!=mspMapPoints.end(); sit++)
    {
        MapPoint* pMP = *sit;
        pMP->SetWorldPos(s * Ryw * pMP->GetWorldPos() + tyw);
        pMP->UpdateNormalAndDepth();
    }
    mnMapChange++;
}

void Map::SetInertialSensor()
{
    unique_lock<mutex> lock(mMutexMap);
    mbIsInertial = true;
}

bool Map::IsInertial()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbIsInertial;
}

void Map::SetIniertialBA1()
{
    unique_lock<mutex> lock(mMutexMap);
    mbIMU_BA1 = true;
}

void Map::SetIniertialBA2()
{
    unique_lock<mutex> lock(mMutexMap);
    mbIMU_BA2 = true;
}

bool Map::GetIniertialBA1()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbIMU_BA1;
}

bool Map::GetIniertialBA2()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbIMU_BA2;
}

void Map::ChangeId(long unsigned int nId)
{
    mnId = nId;
}

unsigned int Map::GetLowerKFID()
{
    unique_lock<mutex> lock(mMutexMap);
    if (mpKFlowerID) {
        return mpKFlowerID->mnId;
    }
    return 0;
}

int Map::GetMapChangeIndex()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMapChange;
}

void Map::IncreaseChangeIndex()
{
    unique_lock<mutex> lock(mMutexMap);
    mnMapChange++;
}

int Map::GetLastMapChange()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMapChangeNotified;
}

void Map::SetLastMapChange(int currentChangeId)
{
    unique_lock<mutex> lock(mMutexMap);
    mnMapChangeNotified = currentChangeId;
}

void Map::PreSave(std::set<GeometricCamera*> &spCams)
{
    int nMPWithoutObs = 0;
    for(MapPoint* pMPi : mspMapPoints)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        if(pMPi->GetObservations().size() == 0)
        {
            nMPWithoutObs++;
        }
        map<KeyFrame*, std::tuple<int,int>> mpObs = pMPi->GetObservations();
        for(map<KeyFrame*, std::tuple<int,int>>::iterator it= mpObs.begin(), end=mpObs.end(); it!=end; ++it)
        {
            if(it->first->GetMap() != this || it->first->isBad())
            {
                pMPi->EraseObservation(it->first);
            }

        }
    }

    // Saves the id of KF origins
    mvBackupKeyFrameOriginsId.clear();
    mvBackupKeyFrameOriginsId.reserve(mvpKeyFrameOrigins.size());
    for(int i = 0, numEl = mvpKeyFrameOrigins.size(); i < numEl; ++i)
    {
        mvBackupKeyFrameOriginsId.push_back(mvpKeyFrameOrigins[i]->mnId);
    }


    // Backup of MapPoints
    mvpBackupMapPoints.clear();
    for(MapPoint* pMPi : mspMapPoints)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        mvpBackupMapPoints.push_back(pMPi);
        pMPi->PreSave(mspKeyFrames,mspMapPoints);
    }

    // Backup of KeyFrames
    mvpBackupKeyFrames.clear();
    for(KeyFrame* pKFi : mspKeyFrames)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        mvpBackupKeyFrames.push_back(pKFi);
        pKFi->PreSave(mspKeyFrames,mspMapPoints, spCams);
    }

    mnBackupKFinitialID = -1;
    if(mpKFinitial)
    {
        mnBackupKFinitialID = mpKFinitial->mnId;
    }

    mnBackupKFlowerID = -1;
    if(mpKFlowerID)
    {
        mnBackupKFlowerID = mpKFlowerID->mnId;
    }

}

void Map::PostLoad(KeyFrameDatabase* pKFDB, ORBVocabulary* pORBVoc/*, map<long unsigned int, KeyFrame*>& mpKeyFrameId*/, map<unsigned int, GeometricCamera*> &mpCams)
{
    std::copy(mvpBackupMapPoints.begin(), mvpBackupMapPoints.end(), std::inserter(mspMapPoints, mspMapPoints.begin()));
    std::copy(mvpBackupKeyFrames.begin(), mvpBackupKeyFrames.end(), std::inserter(mspKeyFrames, mspKeyFrames.begin()));

    map<long unsigned int,MapPoint*> mpMapPointId;
    for(MapPoint* pMPi : mspMapPoints)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        pMPi->UpdateMap(this);
        mpMapPointId[pMPi->mnId] = pMPi;
    }

    map<long unsigned int, KeyFrame*> mpKeyFrameId;
    for(KeyFrame* pKFi : mspKeyFrames)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateMap(this);
        pKFi->SetORBVocabulary(pORBVoc);
        pKFi->SetKeyFrameDatabase(pKFDB);
        mpKeyFrameId[pKFi->mnId] = pKFi;
    }

    // References reconstruction between different instances
    for(MapPoint* pMPi : mspMapPoints)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        pMPi->PostLoad(mpKeyFrameId, mpMapPointId);
    }

    for(KeyFrame* pKFi : mspKeyFrames)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->PostLoad(mpKeyFrameId, mpMapPointId, mpCams);
        pKFDB->add(pKFi);
    }


    if(mnBackupKFinitialID != -1)
    {
        mpKFinitial = mpKeyFrameId[mnBackupKFinitialID];
    }

    if(mnBackupKFlowerID != -1)
    {
        mpKFlowerID = mpKeyFrameId[mnBackupKFlowerID];
    }

    mvpKeyFrameOrigins.clear();
    mvpKeyFrameOrigins.reserve(mvBackupKeyFrameOriginsId.size());
    for(int i = 0; i < mvBackupKeyFrameOriginsId.size(); ++i)
    {
        mvpKeyFrameOrigins.push_back(mpKeyFrameId[mvBackupKeyFrameOriginsId[i]]);
    }

    mvpBackupMapPoints.clear();
}


} //namespace ORB_SLAM3
