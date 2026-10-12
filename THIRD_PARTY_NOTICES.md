# Third-party notices

UC-SLAM is distributed under the GNU General Public License, version 3 or (at your option) any later version (see [LICENSE](LICENSE)).
It contains or uses the components below, which keep their own licenses.

## Contained in this repository

| Component | Where | License | Source |
|---|---|---|---|
| ORB-SLAM3, commit 4452a3c (modified, see `cpp/ucslam/MODIFICATIONS.md`) | `cpp/ucslam/` | GPLv3 or later | https://github.com/UZ-SLAMLab/ORB_SLAM3 |
| DBoW2, with Random and Timestamp from DLib | `cpp/ucslam/Thirdparty/DBoW2/` | BSD-style with a notification clause (`LICENSE.txt`, `DUtils/LICENSE_DLib.txt`) | https://github.com/dorian3d/DBoW2, https://github.com/dorian3d/DLib |
| g2o | `cpp/ucslam/Thirdparty/g2o/` | BSD (`license-bsd.txt`). `g2o/types/se3mat.*` come from ORB-SLAM3 without a license header | https://github.com/RainerKuemmerle/g2o |
| Sophus | `cpp/ucslam/Thirdparty/Sophus/` | MIT (`LICENSE.txt`) | https://github.com/strasdat/Sophus |
| ORB extractor, a modified `orb.cpp` of OpenCV | `cpp/ucslam/src/ORBextractor.cc` | BSD, notice kept in the file | https://github.com/opencv/opencv |
| MLPnP solver | `cpp/ucslam/include/MLPnPsolver.h`, `cpp/ucslam/src/MLPnPsolver.cpp` | BSD, notice kept in the files | https://github.com/urbste/opengv |
| `ORBmatcher::DescriptorDistance` | `cpp/ucslam/src/ORBmatcher.cc` | public domain | http://graphics.stanford.edu/~seander/bithacks.html |
| XFeat detection, adapted with an added sub-pixel refinement | `ucslam/backbone.py` (`XFeatBackbone.detect`) | Apache-2.0 ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)) | https://github.com/verlab/accelerated_features |
| TUM RGB-D association lists shipped with the ORB-SLAM3 examples | `cpp/ucslam/Examples*/RGB-D/associations/` | the dataset terms (CC BY 4.0), see below | https://cvg.cit.tum.de/data/datasets/rgbd-dataset |

The DBoW2 and DLib licenses both contain this clause 3: "The original author of the work must be notified of any redistribution of source code or in binary form."
The clause comes with the copies that ORB-SLAM3 ships. DBoW2 asks academic users to cite D. Galvez-Lopez and J. D. Tardos, "Bags of Binary Words for Fast Place Recognition in Image Sequences", IEEE Transactions on Robotics, 2012.

## Downloaded at build or run time, not redistributed

| Component | Used for | License | Source |
|---|---|---|---|
| ORB-SLAM3 at commit 4452a3c (git submodule) and its ORB vocabulary | `third_party/ORB_SLAM3/` (vocabulary, timestamp lists, TUM-VI IMU files) | GPLv3 or later | https://github.com/UZ-SLAMLab/ORB_SLAM3 |
| Timestamp lists and TUM-VI IMU files of that clone | inputs of the example binaries | derived from EuRoC and TUM-VI, see the dataset terms below | https://github.com/UZ-SLAMLab/ORB_SLAM3 |
| Pangolin v0.6 | viewer library linked by ORB-SLAM3 | MIT | https://github.com/stevenlovegrove/Pangolin |
| Depth Anything V2, metric indoor ViT-B weights | the metric depth prior of UC-SLAM | the model card states no license. The upstream README lists the Base and Large models under CC-BY-NC-4.0, so treat it as non-commercial | https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf |
| Depth Anything V2, metric indoor ViT-S weights | the depth-prior benchmark only (`experiments/out/depth_prior_bench.*`) | the model card states no license. The upstream README lists the Small model under Apache-2.0 | https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf |
| Metric3D v2 ViT-S (code and weights through `torch.hub`) | the depth-prior benchmark only | code BSD-2-Clause, weights under their release terms | https://github.com/YvanYin/Metric3D |
| XFeat weights (through `torch.hub`) | stereo reference depth (`ucslam/stereo.py`) | Apache-2.0 | https://github.com/verlab/accelerated_features |
| evo | trajectory evaluation | GPLv3 or later | https://github.com/MichaelGrupp/evo |
| Python packages in `uv.lock` | | their own licenses | PyPI |

## Datasets, not redistributed

| Dataset | Terms | Citation |
|---|---|---|
| EuRoC MAV | see the ETH Research Collection record, https://doi.org/10.3929/ethz-b-000690084 | M. Burri et al., "The EuRoC micro aerial vehicle datasets", IJRR 2016 |
| TUM-VI | CC BY 4.0 | D. Schubert et al., "The TUM VI Benchmark for Evaluating Visual-Inertial Odometry", IROS 2018 |
| TUM RGB-D (association lists only, shipped by ORB-SLAM3) | CC BY 4.0 | J. Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012 |

Depth maps made with the network from these datasets are derived data and are not distributed here.
