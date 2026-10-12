# Modifications to ORB-SLAM3

This directory is a modified version of ORB-SLAM3 (https://github.com/UZ-SLAMLab/ORB_SLAM3),
commit 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4, released under the GNU General Public License, version 3 or (at your option) any later version.
It was modified by the UC-SLAM authors in 2026. The modified version is distributed under the same
license (see LICENSE). This notice follows section 5(a) of the GPLv3.

The estimator changes are switched on by `UCSLAM.*` settings keys.

## Files added

| File | Purpose |
|---|---|
| `include/PriorNet.h` | metric depth prior service: per-keyframe depth from the precomputed cache |
| `include/UcslamMarginals.h` | optional dump of the full-problem posterior covariance at shutdown |
| `include/UcslamTypes.h` | g2o vertex and edge types: log-scale vertex, log-depth edge, log-scale random-walk edge |
| `src/PriorNet.cc` | metric depth prior service: per-keyframe depth from the precomputed cache |
| `src/UcslamMarginals.cc` | optional dump of the full-problem posterior covariance at shutdown |
| `src/UcslamTypes.cc` | g2o vertex and edge types: log-scale vertex, log-depth edge, log-scale random-walk edge |
| `test/test_chain.cc` | unit test of the log-scale chain and its marginalization |
| `test/test_depth_subsample.cc` | unit test of the depth-edge subsample budget (key `UCSLAM.DepthEdgeSubsampleEps`, off in the reported variant) |
| `test/test_ucslam_types.cc` | unit test of the analytic Jacobians against numeric ones |
| `Thirdparty/DBoW2/DUtils/LICENSE_DLib.txt` | DLib license text for DUtils, copied from https://github.com/dorian3d/DLib |
| `Thirdparty/DBoW2/LICENSE.txt` | DBoW2 license text, copied from https://github.com/dorian3d/DBoW2 |

## Files modified

In the estimator files, the depth factor, the log-scale chain and their marginalization are wired into local
and global bundle adjustment, keyframe culling, map merging and loop closing, and the settings reader parses
the `UCSLAM.*` keys.

| File | Change |
|---|---|
| `CMakeLists.txt` | new sources, the unit tests, OpenCV 4.2 accepted |
| `Examples/Monocular-Inertial/mono_inertial_euroc.cc` | runs without the viewer (headless) |
| `Examples/Monocular-Inertial/mono_inertial_tum_vi.cc` | runs without the viewer (headless) |
| `.gitignore` | keeps LICENSE and the Thirdparty timing sources tracked, and ignores the upstream data lists, the vocabulary and build products |
| `include/ImuTypes.h` | read-only accessor `UcslamSumAcc` to the summed IMU accelerations, used by the young-map gate |
| `include/KeyFrame.h` | estimator integration |
| `include/LocalMapping.h` | estimator integration |
| `include/LoopClosing.h` | estimator integration |
| `include/Map.h` | estimator integration |
| `include/MapPoint.h` | estimator integration |
| `include/Optimizer.h` | estimator integration |
| `include/Settings.h` | estimator integration |
| `include/System.h` | estimator integration |
| `include/Tracking.h` | estimator integration |
| `README.md` | fork notice at the top, upstream text unchanged below it |
| `src/KeyFrame.cc` | estimator integration |
| `src/LocalMapping.cc` | estimator integration |
| `src/LoopClosing.cc` | estimator integration |
| `src/Map.cc` | estimator integration |
| `src/Optimizer.cc` | estimator integration |
| `src/Settings.cc` | estimator integration |
| `src/System.cc` | estimator integration |
| `src/Tracking.cc` | estimator integration |

Every other upstream file present here is identical to the upstream commit above.

## Files not included

503 files of the upstream commit are not in this directory. The public release leaves out 4 more, named after the table.

| Upstream path | Files | Reason |
|---|---|---|
| `Examples*/*/EuRoC_TimeStamps/`, `Examples*/*/EuRoC_IMU/`, `Examples*/*/TUM_TimeStamps/`, `Examples*/*/TUM_IMU/` | 482 | timestamp lists and IMU files derived from EuRoC and TUM-VI. The runner reads them from the stock clone made by `scripts/build_orbslam3.sh` |
| `evaluation/Ground_truth/` | 17 | ground truth of EuRoC |
| `Vocabulary/ORBvoc.txt.tar.gz` | 1 | the ORB vocabulary (42 MB), taken from the same stock clone |
| `Calibration_Tutorial.pdf` | 1 | calibration tutorial, not used |
| `Examples_old/REAMDME.md`, `Examples_old/ROS/ORB_SLAM3/manifest.xml` | 2 | notes and a ROS manifest of the old examples, not used |

The public release of UC-SLAM also leaves out `build.sh` and `build_ros.sh` (replaced by `scripts/build_orbslam3.sh` and
`scripts/build_ucslam.sh`) and `evaluation/associate.py` and `evaluation/evaluate_ate_scale.py` (Python 2 scripts, replaced by `eval/run_orbslam3.py` and evo).
