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

#ifndef ORB_SLAM3_SETTINGS_H
#define ORB_SLAM3_SETTINGS_H


// Flag to activate the measurement of time in each process (track,localmap, place recognition).
//#define REGISTER_TIMES

#include "CameraModels/GeometricCamera.h"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

namespace ORB_SLAM3 {

    class System;

    //TODO: change to double instead of float

    class Settings {
    public:
        /*
         * Enum for the different camera types implemented
         */
        enum CameraType {
            PinHole = 0,
            Rectified = 1,
            KannalaBrandt = 2
        };

        /*
         * Delete default constructor
         */
        Settings() = delete;

        /*
         * Constructor from file
         */
        Settings(const std::string &configFile, const int& sensor);

        /*
         * Ostream operator overloading to dump settings to the terminal
         */
        friend std::ostream &operator<<(std::ostream &output, const Settings &s);

        /*
         * Getter methods
         */
        CameraType cameraType() {return cameraType_;}
        GeometricCamera* camera1() {return calibration1_;}
        GeometricCamera* camera2() {return calibration2_;}
        cv::Mat camera1DistortionCoef() {return cv::Mat(vPinHoleDistorsion1_.size(),1,CV_32F,vPinHoleDistorsion1_.data());}
        cv::Mat camera2DistortionCoef() {return cv::Mat(vPinHoleDistorsion2_.size(),1,CV_32F,vPinHoleDistorsion1_.data());}

        Sophus::SE3f Tlr() {return Tlr_;}
        float bf() {return bf_;}
        float b() {return b_;}
        float thDepth() {return thDepth_;}

        bool needToUndistort() {return bNeedToUndistort_;}

        cv::Size newImSize() {return newImSize_;}
        float fps() {return fps_;}
        bool rgb() {return bRGB_;}
        bool needToResize() {return bNeedToResize1_;}
        bool needToRectify() {return bNeedToRectify_;}

        float noiseGyro() {return noiseGyro_;}
        float noiseAcc() {return noiseAcc_;}
        float gyroWalk() {return gyroWalk_;}
        float accWalk() {return accWalk_;}
        float imuFrequency() {return imuFrequency_;}
        Sophus::SE3f Tbc() {return Tbc_;}
        bool insertKFsWhenLost() {return insertKFsWhenLost_;}

        float depthMapFactor() {return depthMapFactor_;}

        int nFeatures() {return nFeatures_;}
        int nLevels() {return nLevels_;}
        float initThFAST() {return initThFAST_;}
        float minThFAST() {return minThFAST_;}
        float scaleFactor() {return scaleFactor_;}

        float keyFrameSize() {return keyFrameSize_;}
        float keyFrameLineWidth() {return keyFrameLineWidth_;}
        float graphLineWidth() {return graphLineWidth_;}
        float pointSize() {return pointSize_;}
        float cameraSize() {return cameraSize_;}
        float cameraLineWidth() {return cameraLineWidth_;}
        float viewPointX() {return viewPointX_;}
        float viewPointY() {return viewPointY_;}
        float viewPointZ() {return viewPointZ_;}
        float viewPointF() {return viewPointF_;}
        float imageViewerScale() {return imageViewerScale_;}

        std::string atlasLoadFile() {return sLoadFrom_;}
        std::string atlasSaveFile() {return sSaveto_;}

        float thFarPoints() {return thFarPoints_;}

        cv::Mat M1l() {return M1l_;}
        cv::Mat M2l() {return M2l_;}
        cv::Mat M1r() {return M1r_;}
        cv::Mat M2r() {return M2r_;}

        cv::Size originalImSize() {return originalImSize_;}

        /*
         * UC-SLAM parameters, the UCSLAM.* keys of the settings file.
         * docs/fork_design.md Sec. 2 to 5. scripts/gen_ucslam_yaml.py writes
         * them from configs/default.yaml. Every default below reproduces stock
         * ORB-SLAM3 when the key is absent: the flags are off and the thresholds
         * are the manuscript values, which nothing reads while the flags are off.
         */
        struct Ucslam {
            std::string dataset;                // UCSLAM.Dataset, sensor domain of the per-domain constants
            int priorNet = 0;                   // UCSLAM.PriorNet, 1 starts the PriorNet thread
            std::string depthDir;               // UCSLAM.DepthDir, root of the precomputed depth cache
            std::string depthModel;             // UCSLAM.DepthModel, subdirectory name of the network
            float depthMatchTolMs = 1.0f;       // UCSLAM.DepthMatchTolMs, stamp tolerance for a cached map
            int useDepthFactor = 0;             // UCSLAM.UseDepthFactor, 1 adds the log-depth edges
            float sigmaNetC0 = 0.0f;            // UCSLAM.SigmaNetC0, Eq. (signet)
            float sigmaNetC1 = 0.0f;            // UCSLAM.SigmaNetC1
            float logBias = 0.0f;               // UCSLAM.LogBias, per-domain median of log z_hat minus log z_ref
            float neffOverN = 1.0f;             // UCSLAM.NeffOverN, effective sample ratio per keyframe
            float zRef = 1.0f;                  // UCSLAM.ZRef [m]
            float sigma0 = 0.1f;                // UCSLAM.Sigma0, a_0 prior std
            float sigmaRW = 0.02f;              // UCSLAM.SigmaRW, a_k random walk density [s^-1/2]
            int essDeflation = 0;               // UCSLAM.ESSDeflation, 1 scales depth information by neffOverN
            int maxDepthPriorPoints = 0;        // UCSLAM.MaxDepthPriorPoints, 0 creates none
            std::string initPolicy = "timer";   // UCSLAM.InitPolicy: timer (stock), ours, wait
            float initMinWindow = 2.0f;         // UCSLAM.InitMinWindow [s]
            float initMaxWindow = 15.0f;        // UCSLAM.InitMaxWindow [s]
            float tauS = 0.15f;                 // UCSLAM.TauS, commit threshold on std of log s
            float tauG = 1.0f;                  // UCSLAM.TauG, commit threshold on tilt std [deg]
            float tauBa = 0.05f;                // UCSLAM.TauBa, commit threshold on ba std [m/s^2]
            int dumpMarginals = 0;              // UCSLAM.DumpMarginals, 1 writes joint pose marginals
            std::string covSource = "pyramid";  // UCSLAM.CovSource: pyramid (stock weights), learned
            float sigmaMerge = 0.1f;            // UCSLAM.SigmaMerge, std of the relative prior on an Atlas merge
            // Diagnostics and the switches of the divergence study (docs/fork_design.md Sec. 7). The
            // defaults keep the behaviour of the depth factor as first built.
            int diag = 0;                       // UCSLAM.Diag, 1 writes ucslam_track.csv and extra [UCSLAM] lines
            int depthEdgeMinObs = 1;            // UCSLAM.DepthEdgeMinObs, depth edge only on points seen by this many keyframes
            int depthEdgesFixedKFs = 1;         // UCSLAM.DepthEdgesFixedKFs, 0 drops the depth edges of fixed keyframes
            int chainFixedBoundary = 1;         // UCSLAM.ChainFixedBoundary, 0 gives fixed keyframes no a_k node
            int imuInitHandoff = 0;             // UCSLAM.ImuInitHandoff, 1 hands the newest keyframe to the tracker at IMU init
            int logScaleFixed = 0;              // UCSLAM.LogScaleFixed, 1 fixes every a_k at 0, an ablation that takes the
                                                // bias-corrected network depth as metric (docs/fork_design.md Sec. 8)
            int depthGateMode = 0;              // UCSLAM.DepthGateMode, 0 gate after the solve and solve again, 1 gate at the initial estimate,
                                                // 2 gate at the initial estimate after each free a_k is moved to its geometric seed
            // The stereo study (docs/fork_design.md Sec. 7, stereo-inertial). Defaults keep the depth_fix behaviour.
            int depthEdgesStereoPoints = 1;     // UCSLAM.DepthEdgesStereoPoints, 0 gives a point with a stereo observation no depth edge
            int logScaleSeed = 0;               // UCSLAM.LogScaleSeed, 1 seeds a_k of a keyframe without a state from its own geometric points
            int depthEdgesStereoObs = 1;        // UCSLAM.DepthEdgesStereoObs, 0 gives an observation with a stereo depth no depth edge
            int diagLbaDelayMs = 0;             // UCSLAM.DiagLbaDelayMs, diagnostic: the local inertial BA waits this long before its write back
            float scaleRefineMaxLog = 0.0f;     // UCSLAM.ScaleRefineMaxLog, c > 0: the applied scale refinements of a map stay within exp(+-c) in total (protocol A9)
            float youngMapScaleGate = 0.0f;     // UCSLAM.YoungMapScaleGate, tau > 0: once per run, after the first printed local inertial BA, reset the
                                                // young map if |a_cur| > tau (the IMU scale disagrees with the depth prior), attempt 06
            int logInitState = 0;               // UCSLAM.LogInitState, 1 prints the IMU init line (scale, bg, ba) without Diag
            float youngMapBiasGate = 0.0f;      // UCSLAM.YoungMapBiasGate beta [rad/s], > 0: the gate also needs |bg0 . up0| > beta, the
                                                // vertical gyro bias of the first IMU init of the map (attempt 07)
            int youngMapGateShadow = 0;         // UCSLAM.YoungMapGateShadow, 1: the gate prints its line with act=0 and never resets
            // Attempt 08, a faster local BA. Every default keeps the computation above unchanged.
            int depthEdgeGateAtAdd = 0;         // UCSLAM.DepthEdgeGateAtAdd, 1 with DepthGateMode 1 does not create a depth edge the gate at the initial estimate rejects
            int lbaMargDiagOnly = 0;            // UCSLAM.LbaMargDiagOnly, 1 solves the LDLT of the local BA marginals for the free a_k columns only
            float depthEdgeSubsampleEps = 0.0f; // UCSLAM.DepthEdgeSubsampleEps eps > 0: the inertial local BA keeps K of the n candidate depth edges of each
                                                // free keyframe, weighted to keep their information (needs ESSDeflation, DepthGateMode and DepthEdgeGateAtAdd 1)
            int diagMargCheck = 0;              // UCSLAM.DiagMargCheck, 1 computes both marginal paths, marg_check in the diag lba line
            int diagSubsampleCheck = 0;         // UCSLAM.DiagSubsampleCheck, 1 prints one diag subsample line per free keyframe of the subsample
            // Live depth for a robot (ROS 1 and ROS 2 nodes). Every default keeps the depth cache path of the benchmark.
            std::string depthSource = "file";   // UCSLAM.DepthSource: file (the cache under DepthDir), live (maps pushed through System::InsertDepthMap)
            float depthWaitMs = 0.0f;           // UCSLAM.DepthWaitMs, live: how long PriorNet waits for the map of a keyframe that has not arrived
            int depthBufferMaps = 32;           // UCSLAM.DepthBufferMaps, live: pushed maps kept in a ring
            float dppWaitMs = 0.0f;             // UCSLAM.DppWaitMs, LocalMapping waits up to this long for the depth before it makes depth-prior points
            int dppDeferMaxKFs = 0;             // UCSLAM.DppDeferMaxKFs, a keyframe whose depth was late gets its depth-prior points in a later
                                                // mapper step, up to this many keyframes later. Such points skip the fuse into the neighbour
                                                // keyframes and meet the culling test one mapper step sooner than normal ones
            std::vector<std::string> keysFound; // the UCSLAM.* keys present in the file, in read order
        };
        const Ucslam& ucslam() const {return ucslam_;}

    private:
        template<typename T>
        T readParameter(cv::FileStorage& fSettings, const std::string& name, bool& found,const bool required = true){
            cv::FileNode node = fSettings[name];
            if(node.empty()){
                if(required){
                    std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                    exit(-1);
                }
                else{
                    std::cerr << name << " optional parameter does not exist..." << std::endl;
                    found = false;
                    return T();
                }

            }
            else{
                found = true;
                return (T) node;
            }
        }

        void readCamera1(cv::FileStorage& fSettings);
        void readCamera2(cv::FileStorage& fSettings);
        void readImageInfo(cv::FileStorage& fSettings);
        void readIMU(cv::FileStorage& fSettings);
        void readRGBD(cv::FileStorage& fSettings);
        void readORB(cv::FileStorage& fSettings);
        void readViewer(cv::FileStorage& fSettings);
        void readLoadAndSave(cv::FileStorage& fSettings);
        void readOtherParameters(cv::FileStorage& fSettings);
        void readUcslam(cv::FileStorage& fSettings);

        void precomputeRectificationMaps();

        int sensor_;
        CameraType cameraType_;     //Camera type

        /*
         * Visual stuff
         */
        GeometricCamera* calibration1_, *calibration2_;   //Camera calibration
        GeometricCamera* originalCalib1_, *originalCalib2_;
        std::vector<float> vPinHoleDistorsion1_, vPinHoleDistorsion2_;

        cv::Size originalImSize_, newImSize_;
        float fps_;
        bool bRGB_;

        bool bNeedToUndistort_;
        bool bNeedToRectify_;
        bool bNeedToResize1_, bNeedToResize2_;

        Sophus::SE3f Tlr_;
        float thDepth_;
        float bf_, b_;

        /*
         * Rectification stuff
         */
        cv::Mat M1l_, M2l_;
        cv::Mat M1r_, M2r_;

        /*
         * Inertial stuff
         */
        float noiseGyro_, noiseAcc_;
        float gyroWalk_, accWalk_;
        float imuFrequency_;
        Sophus::SE3f Tbc_;
        bool insertKFsWhenLost_;

        /*
         * RGBD stuff
         */
        float depthMapFactor_;

        /*
         * ORB stuff
         */
        int nFeatures_;
        float scaleFactor_;
        int nLevels_;
        int initThFAST_, minThFAST_;

        /*
         * Viewer stuff
         */
        float keyFrameSize_;
        float keyFrameLineWidth_;
        float graphLineWidth_;
        float pointSize_;
        float cameraSize_;
        float cameraLineWidth_;
        float viewPointX_, viewPointY_, viewPointZ_, viewPointF_;
        float imageViewerScale_;

        /*
         * Save & load maps
         */
        std::string sLoadFrom_, sSaveto_;

        /*
         * Other stuff
         */
        float thFarPoints_;

        /*
         * UC-SLAM stuff
         */
        Ucslam ucslam_;

    };
};


#endif //ORB_SLAM3_SETTINGS_H
