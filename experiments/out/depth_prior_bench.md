# Depth-prior characterization (Table tab:depthprior)

Device NVIDIA GeForce RTX 4070 Ti. Code `experiments/depth_prior_bench.py`. Every number here is produced by that code from the run this file describes. The JSON next to this file holds full precision.

This is a standalone measurement of released depth networks against a stereo reference, in Python. It is not the C++ system.

## Setup

- Sequences: 17 (12 fit, 5 held-out). Held-out: euroc/MH_04_difficult, euroc/V1_03_difficult, euroc/V2_03_difficult, tumvi/room5, tumvi/room6. 40 keyframes per sequence, spread evenly over the cam0 frames with a cam1 partner within 5 ms.
- Networks: Depth Anything V2 metric indoor, ViT-S, Depth Anything V2 metric indoor, ViT-B, Metric3D v2, ViT-S.
- Reference: `ucslam.stereo.stereo_landmarks` on XFeat top-1024 detections with the per-sequence stereo pixel noise of `data/corpus/manifest.json`. Landmarks with a first-order log-depth noise above 0.3 are dropped.
- Input, EuRoC: grayscale 752x480 frame replicated to three channels, fed as recorded with the radial-tangential distortion left in, cam0 intrinsics passed to Metric3D
- Input, TUM-VI: 512x512 equidistant fisheye rectified with cv2.fisheye.initUndistortRectifyMap, R = I, to a 512x512 pinhole view of 90 degrees, f = 256.0 px, then replicated to three channels. Keypoints are unprojected with ucslam.calib and projected into that view.
- Wall time 8.3 min. Points: 126052 in `experiments/out/depth_prior_points.npz`, 120000 subsampled rows in `experiments/out/depth_prior_points.csv`.

## Definitions

- r = log z_hat minus log z_ref per point. z_ref is the stereo depth in the cam0 frame at the cam0 keypoint.
- Bias is the median of r over a split. Spread is 1.4826 times the median absolute deviation of r. The deflated spread subtracts the mean reference variance, sqrt(spread^2 minus mean(sigma_ref^2)), and is the network's own spread.
- (c0, c1): the per-keyframe median of r is removed first, because it is the a_k of Eq. (achain) the chain absorbs. |r minus median_k| is binned into quantile bins of log z_hat. Per bin the robust spread is 1.4826 times the median, the bin's mean reference variance is subtracted, and a count-weighted line through the bin spreads gives c0 and c1 with z_ref = 1 m. Fitted on the fit split only. The held-out fit is printed as a diagnostic and is not used.
- Coverage of the fitted model on a split: the fraction of |r minus median_k| inside one and two sigma_tot, with sigma_tot = sqrt(sigma_net(log z_hat)^2 + sigma_ref^2). A calibrated Gaussian gives 0.683 and 0.954, and a median ratio of 1.
- sigma_0 (std) is the standard deviation across sequences of the per-sequence median of r. It is the global calibration error a_0 must cover after a per-domain bias is removed. sigma_0 (rms) is the root mean square of the same medians about zero, which a zero-mean prior must cover without a bias correction.
- Keyframe-median spread is the robust spread of the per-keyframe medians within a sequence, reported as the median over sequences. It is what sigma_rw covers. The random-walk upper bound is 1.4826 times the median of |median_{k+1} minus median_k| / sqrt(dt) over consecutive keyframes. The median noise adds to it, so it is an upper bound on sigma_rw.
- N_eff / N per keyframe: N_eff = N / (1 + (N minus 1) rho), rho = (N Var(mean_k) / Var(r) minus 1) / (N minus 1) per sequence, N the mean point count of the used keyframes, residuals clipped at the sequence median plus or minus 3 robust spreads before the means. Keyframes with fewer than 20 points are not used. Median, min and max over sequences.

## Main table, per dataset and network

Held-out values first, fit-set values in parentheses. (c0, c1) from the fit set. The reference log-noise column is the median first-order stereo noise sigma_z / z of the points.

| Dataset | Network | N held-out (fit) | bias | spread raw | spread deflated | ref. log-noise | (c0, c1) deflated | (c0, c1) raw | sigma_0 std, n seq | sigma_0 rms | N_eff/N median [min, max] | N per kf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EuRoC | Depth Anything V2 metric indoor, ViT-S | 24190 (75613) | -0.244 (-0.136) | 0.533 (0.442) | 0.516 (0.425) | 0.106 (0.087) | (0.144, 0.005) | (0.146, 0.041) | 0.305, 8 (held-out 0.400, 3) | 0.324 (held-out 0.410) | 0.006 [0.006, 0.013] (0.007 [0.003, 0.035]) | 202 (236) |
| EuRoC | Depth Anything V2 metric indoor, ViT-B | 24190 (75613) | -0.000 (-0.015) | 0.371 (0.356) | 0.346 (0.334) | 0.106 (0.087) | (0.075, 0.055) | (0.068, 0.092) | 0.207, 8 (held-out 0.296, 3) | 0.197 (held-out 0.246) | 0.008 [0.007, 0.019] (0.008 [0.003, 0.056]) | 202 (236) |
| EuRoC | Metric3D v2, ViT-S | 24190 (75613) | -0.268 (-0.237) | 0.344 (0.377) | 0.317 (0.357) | 0.106 (0.087) | (0.115, 0.025) | (0.112, 0.066) | 0.121, 8 (held-out 0.038, 3) | 0.256 (held-out 0.255) | 0.007 [0.006, 0.025] (0.006 [0.005, 0.034]) | 202 (236) |
| TUM-VI | Depth Anything V2 metric indoor, ViT-S | 9246 (17003) | 0.775 (0.791) | 0.245 (0.269) | 0.208 (0.242) | 0.117 (0.100) | (0.083, 0.012) | (0.088, 0.048) | 0.061, 4 (held-out 0.042, 2) | 0.778 (held-out 0.781) | 0.019 [0.013, 0.024] (0.019 [0.014, 0.030]) | 116 (107) |
| TUM-VI | Depth Anything V2 metric indoor, ViT-B | 9246 (17003) | 0.786 (0.796) | 0.246 (0.246) | 0.208 (0.217) | 0.117 (0.100) | (0.078, 0.014) | (0.085, 0.049) | 0.038, 4 (held-out 0.014, 2) | 0.789 (held-out 0.784) | 0.025 [0.015, 0.036] (0.022 [0.019, 0.052]) | 116 (107) |
| TUM-VI | Metric3D v2, ViT-S | 9246 (17003) | 0.051 (0.072) | 0.214 (0.220) | 0.170 (0.186) | 0.117 (0.100) | (0.109, -0.010) | (0.135, 0.033) | 0.027, 4 (held-out 0.008, 2) | 0.073 (held-out 0.052) | 0.038 [0.024, 0.052] (0.033 [0.016, 0.095]) | 116 (107) |

## Coverage of the fitted (c0, c1) and the held-out diagnostic fit

| Dataset | Network | split | n | inside 1 sigma (0.683) | inside 2 sigma (0.954) | median ratio (1.0) | floored | held-out own fit (c0, c1), diagnostic |
|---|---|---|---|---|---|---|---|---|
| EuRoC | Depth Anything V2 metric indoor, ViT-S | heldout | 24190 | 0.651 | 0.873 | 0.991 | 0.000 | (0.205, -0.061) |
| EuRoC | Depth Anything V2 metric indoor, ViT-S | fit | 75613 | 0.635 | 0.859 | 1.033 | 0.000 |  |
| EuRoC | Depth Anything V2 metric indoor, ViT-B | heldout | 24190 | 0.667 | 0.883 | 0.951 | 0.000 | (0.054, 0.057) |
| EuRoC | Depth Anything V2 metric indoor, ViT-B | fit | 75613 | 0.639 | 0.863 | 1.025 | 0.000 |  |
| EuRoC | Metric3D v2, ViT-S | heldout | 24190 | 0.656 | 0.875 | 1.000 | 0.000 | (0.088, 0.050) |
| EuRoC | Metric3D v2, ViT-S | fit | 75613 | 0.640 | 0.869 | 1.028 | 0.000 |  |
| TUM-VI | Depth Anything V2 metric indoor, ViT-S | heldout | 9246 | 0.647 | 0.873 | 1.025 | 0.000 | (0.167, -0.047) |
| TUM-VI | Depth Anything V2 metric indoor, ViT-S | fit | 16987 | 0.634 | 0.851 | 1.021 | 0.000 |  |
| TUM-VI | Depth Anything V2 metric indoor, ViT-B | heldout | 9246 | 0.637 | 0.844 | 1.020 | 0.000 | (0.174, -0.054) |
| TUM-VI | Depth Anything V2 metric indoor, ViT-B | fit | 16987 | 0.630 | 0.841 | 1.026 | 0.000 |  |
| TUM-VI | Metric3D v2, ViT-S | heldout | 9246 | 0.643 | 0.838 | 0.982 | 0.000 | (0.229, -0.157) |
| TUM-VI | Metric3D v2, ViT-S | fit | 16987 | 0.639 | 0.845 | 1.006 | 0.000 |  |

## Bin table of the (c0, c1) fit, fit split

EuRoC, Depth Anything V2 metric indoor, ViT-S. Weighted RMS bin misfit 0.0107 (deflated), 0.0147 (raw).

| bin log z_hat range | mean log z_hat | n | sigma raw | ref. sigma | sigma deflated |
|---|---|---|---|---|---|
| [-1.50, 0.46] | 0.19 | 9452 | 0.182 | 0.082 | 0.162 |
| [0.46, 0.74] | 0.60 | 9451 | 0.151 | 0.082 | 0.126 |
| [0.74, 0.96] | 0.86 | 9452 | 0.174 | 0.092 | 0.148 |
| [0.96, 1.14] | 1.05 | 9451 | 0.184 | 0.103 | 0.152 |
| [1.14, 1.31] | 1.22 | 9452 | 0.192 | 0.114 | 0.154 |
| [1.31, 1.48] | 1.39 | 9451 | 0.197 | 0.128 | 0.149 |
| [1.48, 1.73] | 1.59 | 9452 | 0.206 | 0.147 | 0.144 |
| [1.73, 2.86] | 2.06 | 9452 | 0.250 | 0.188 | 0.165 |

EuRoC, Depth Anything V2 metric indoor, ViT-B. Weighted RMS bin misfit 0.0146 (deflated), 0.0175 (raw).

| bin log z_hat range | mean log z_hat | n | sigma raw | ref. sigma | sigma deflated |
|---|---|---|---|---|---|
| [-1.01, 0.63] | 0.39 | 9452 | 0.137 | 0.059 | 0.124 |
| [0.63, 0.91] | 0.78 | 9451 | 0.127 | 0.071 | 0.105 |
| [0.91, 1.12] | 1.01 | 9452 | 0.139 | 0.085 | 0.110 |
| [1.12, 1.30] | 1.21 | 9451 | 0.174 | 0.103 | 0.141 |
| [1.30, 1.45] | 1.38 | 9452 | 0.186 | 0.117 | 0.144 |
| [1.45, 1.63] | 1.54 | 9451 | 0.199 | 0.130 | 0.151 |
| [1.63, 1.87] | 1.74 | 9452 | 0.230 | 0.151 | 0.174 |
| [1.87, 2.94] | 2.16 | 9452 | 0.288 | 0.199 | 0.208 |

EuRoC, Metric3D v2, ViT-S. Weighted RMS bin misfit 0.0125 (deflated), 0.0130 (raw).

| bin log z_hat range | mean log z_hat | n | sigma raw | ref. sigma | sigma deflated |
|---|---|---|---|---|---|
| [-0.52, 0.45] | 0.24 | 9452 | 0.151 | 0.055 | 0.141 |
| [0.45, 0.71] | 0.58 | 9451 | 0.143 | 0.072 | 0.123 |
| [0.71, 0.89] | 0.80 | 9452 | 0.154 | 0.087 | 0.126 |
| [0.89, 1.05] | 0.97 | 9451 | 0.160 | 0.101 | 0.124 |
| [1.05, 1.23] | 1.14 | 9452 | 0.177 | 0.116 | 0.134 |
| [1.23, 1.45] | 1.33 | 9451 | 0.203 | 0.132 | 0.155 |
| [1.45, 1.76] | 1.59 | 9452 | 0.231 | 0.153 | 0.174 |
| [1.76, 3.66] | 2.21 | 9452 | 0.260 | 0.197 | 0.170 |

TUM-VI, Depth Anything V2 metric indoor, ViT-S. Weighted RMS bin misfit 0.0417 (deflated), 0.0346 (raw).

| bin log z_hat range | mean log z_hat | n | sigma raw | ref. sigma | sigma deflated |
|---|---|---|---|---|---|
| [-1.06, 0.76] | 0.44 | 2124 | 0.164 | 0.067 | 0.150 |
| [0.76, 1.02] | 0.90 | 2123 | 0.113 | 0.081 | 0.079 |
| [1.02, 1.30] | 1.16 | 2123 | 0.119 | 0.096 | 0.070 |
| [1.30, 1.46] | 1.39 | 2123 | 0.133 | 0.109 | 0.077 |
| [1.46, 1.58] | 1.52 | 2124 | 0.130 | 0.115 | 0.061 |
| [1.58, 1.72] | 1.65 | 2123 | 0.144 | 0.126 | 0.070 |
| [1.72, 1.95] | 1.81 | 2123 | 0.176 | 0.141 | 0.104 |
| [1.95, 2.82] | 2.18 | 2124 | 0.252 | 0.171 | 0.186 |

TUM-VI, Depth Anything V2 metric indoor, ViT-B. Weighted RMS bin misfit 0.0417 (deflated), 0.0339 (raw).

| bin log z_hat range | mean log z_hat | n | sigma raw | ref. sigma | sigma deflated |
|---|---|---|---|---|---|
| [-1.11, 0.75] | 0.42 | 2124 | 0.157 | 0.069 | 0.141 |
| [0.75, 1.07] | 0.92 | 2123 | 0.126 | 0.080 | 0.098 |
| [1.07, 1.28] | 1.18 | 2123 | 0.109 | 0.094 | 0.055 |
| [1.28, 1.43] | 1.37 | 2123 | 0.120 | 0.105 | 0.058 |
| [1.43, 1.57] | 1.50 | 2124 | 0.131 | 0.116 | 0.062 |
| [1.57, 1.72] | 1.64 | 2123 | 0.139 | 0.124 | 0.064 |
| [1.72, 1.97] | 1.84 | 2123 | 0.196 | 0.145 | 0.132 |
| [1.97, 2.76] | 2.21 | 2124 | 0.243 | 0.173 | 0.171 |

TUM-VI, Metric3D v2, ViT-S. Weighted RMS bin misfit 0.0497 (deflated), 0.0407 (raw).

| bin log z_hat range | mean log z_hat | n | sigma raw | ref. sigma | sigma deflated |
|---|---|---|---|---|---|
| [-1.39, 0.06] | -0.26 | 2124 | 0.189 | 0.063 | 0.179 |
| [0.06, 0.36] | 0.22 | 2123 | 0.122 | 0.076 | 0.095 |
| [0.36, 0.59] | 0.49 | 2123 | 0.122 | 0.096 | 0.075 |
| [0.59, 0.74] | 0.67 | 2123 | 0.133 | 0.109 | 0.076 |
| [0.74, 0.87] | 0.80 | 2124 | 0.131 | 0.116 | 0.060 |
| [0.87, 1.02] | 0.95 | 2123 | 0.136 | 0.127 | 0.050 |
| [1.02, 1.23] | 1.12 | 2123 | 0.171 | 0.143 | 0.093 |
| [1.23, 2.32] | 1.42 | 2124 | 0.257 | 0.171 | 0.192 |

## sigma_0 verdict against the configured value

Configured `depth_prior.sigma_0` = 0.1. A reading "yes" means the measured quantity is at or below the configured value.

| Scope | Network | split | n seq | bias | sigma_0 std | covers after bias removal | sigma_0 rms | covers zero-mean | max abs median |
|---|---|---|---|---|---|---|---|---|---|
| EuRoC | Depth Anything V2 metric indoor, ViT-S | fit | 8 | -0.136 | 0.305 | no | 0.324 | no | 0.668 |
| EuRoC | Depth Anything V2 metric indoor, ViT-S | heldout | 3 | -0.244 | 0.400 | no | 0.410 | no | 0.710 |
| EuRoC | Depth Anything V2 metric indoor, ViT-B | fit | 8 | -0.015 | 0.207 | no | 0.197 | no | 0.345 |
| EuRoC | Depth Anything V2 metric indoor, ViT-B | heldout | 3 | -0.000 | 0.296 | no | 0.246 | no | 0.388 |
| EuRoC | Metric3D v2, ViT-S | fit | 8 | -0.237 | 0.121 | no | 0.256 | no | 0.401 |
| EuRoC | Metric3D v2, ViT-S | heldout | 3 | -0.268 | 0.038 | yes | 0.255 | no | 0.297 |
| TUM-VI | Depth Anything V2 metric indoor, ViT-S | fit | 4 | 0.791 | 0.061 | yes | 0.778 | no | 0.858 |
| TUM-VI | Depth Anything V2 metric indoor, ViT-S | heldout | 2 | 0.775 | 0.042 | yes | 0.781 | no | 0.811 |
| TUM-VI | Depth Anything V2 metric indoor, ViT-B | fit | 4 | 0.796 | 0.038 | yes | 0.789 | no | 0.817 |
| TUM-VI | Depth Anything V2 metric indoor, ViT-B | heldout | 2 | 0.786 | 0.014 | yes | 0.784 | no | 0.794 |
| TUM-VI | Metric3D v2, ViT-S | fit | 4 | 0.072 | 0.027 | yes | 0.073 | yes | 0.108 |
| TUM-VI | Metric3D v2, ViT-S | heldout | 2 | 0.051 | 0.008 | yes | 0.052 | yes | 0.058 |
| both datasets | Depth Anything V2 metric indoor, ViT-S | fit | 12 | -0.016 | 0.519 | no | 0.521 | no | 0.858 |
| both datasets | Depth Anything V2 metric indoor, ViT-S | heldout | 5 | -0.021 | 0.631 | no | 0.587 | no | 0.811 |
| both datasets | Depth Anything V2 metric indoor, ViT-B | fit | 12 | 0.074 | 0.437 | no | 0.483 | no | 0.817 |
| both datasets | Depth Anything V2 metric indoor, ViT-B | heldout | 5 | 0.132 | 0.502 | no | 0.531 | no | 0.794 |
| both datasets | Metric3D v2, ViT-S | fit | 12 | -0.167 | 0.177 | no | 0.214 | no | 0.401 |
| both datasets | Metric3D v2, ViT-S | heldout | 5 | -0.173 | 0.169 | no | 0.200 | no | 0.297 |

## Drift within a sequence, what sigma_rw covers

Configured `depth_prior.sigma_rw` = 0.02 s^-1/2.

| Dataset | Network | split | keyframe-median spread, median over seq | random-walk upper bound [s^-1/2] | n increments |
|---|---|---|---|---|---|
| EuRoC | Depth Anything V2 metric indoor, ViT-S | heldout | 0.212 | 0.2087 | 117 |
| EuRoC | Depth Anything V2 metric indoor, ViT-S | fit | 0.217 | 0.1201 | 312 |
| EuRoC | Depth Anything V2 metric indoor, ViT-B | heldout | 0.129 | 0.0942 | 117 |
| EuRoC | Depth Anything V2 metric indoor, ViT-B | fit | 0.205 | 0.1003 | 312 |
| EuRoC | Metric3D v2, ViT-S | heldout | 0.187 | 0.1306 | 117 |
| EuRoC | Metric3D v2, ViT-S | fit | 0.261 | 0.1414 | 312 |
| TUM-VI | Depth Anything V2 metric indoor, ViT-S | heldout | 0.159 | 0.0921 | 78 |
| TUM-VI | Depth Anything V2 metric indoor, ViT-S | fit | 0.189 | 0.1447 | 155 |
| TUM-VI | Depth Anything V2 metric indoor, ViT-B | heldout | 0.178 | 0.1203 | 78 |
| TUM-VI | Depth Anything V2 metric indoor, ViT-B | fit | 0.143 | 0.1407 | 155 |
| TUM-VI | Metric3D v2, ViT-S | heldout | 0.097 | 0.0621 | 78 |
| TUM-VI | Metric3D v2, ViT-S | fit | 0.100 | 0.0940 | 155 |

## Per sequence

| Sequence | split | Network | n points | n kf used | median r | spread r | ref. log-noise | kf-median spread | N_eff/N | rho | N per kf | median z_hat | median z_ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| euroc/MH_01_easy | fit | Depth Anything V2 metric indoor, ViT-S | 10123 | 40/40 | -0.140 | 0.439 | 0.118 | 0.374 | 0.008 | 0.495 | 253 | 4.02 | 5.01 |
| euroc/MH_02_easy | fit | Depth Anything V2 metric indoor, ViT-S | 9782 | 40/40 | -0.081 | 0.383 | 0.116 | 0.206 | 0.009 | 0.429 | 245 | 4.41 | 4.99 |
| euroc/MH_03_medium | fit | Depth Anything V2 metric indoor, ViT-S | 11520 | 40/40 | -0.569 | 0.560 | 0.113 | 0.455 | 0.005 | 0.652 | 288 | 2.75 | 4.84 |
| euroc/MH_05_difficult | fit | Depth Anything V2 metric indoor, ViT-S | 10874 | 40/40 | -0.668 | 0.417 | 0.156 | 0.248 | 0.006 | 0.566 | 272 | 3.47 | 7.02 |
| euroc/V1_01_easy | fit | Depth Anything V2 metric indoor, ViT-S | 12176 | 40/40 | -0.027 | 0.259 | 0.055 | 0.207 | 0.004 | 0.795 | 304 | 2.36 | 2.42 |
| euroc/V1_02_medium | fit | Depth Anything V2 metric indoor, ViT-S | 12270 | 40/40 | -0.003 | 0.268 | 0.069 | 0.228 | 0.003 | 1.000 | 307 | 2.68 | 2.77 |
| euroc/V2_01_easy | fit | Depth Anything V2 metric indoor, ViT-S | 4607 | 40/40 | 0.187 | 0.194 | 0.065 | 0.097 | 0.035 | 0.243 | 115 | 3.17 | 2.47 |
| euroc/V2_02_medium | fit | Depth Anything V2 metric indoor, ViT-S | 4261 | 40/40 | 0.083 | 0.253 | 0.082 | 0.157 | 0.034 | 0.272 | 107 | 3.29 | 3.05 |
| euroc/MH_04_difficult | heldout | Depth Anything V2 metric indoor, ViT-S | 10849 | 40/40 | -0.710 | 0.441 | 0.158 | 0.281 | 0.006 | 0.653 | 271 | 3.59 | 7.26 |
| euroc/V1_03_difficult | heldout | Depth Anything V2 metric indoor, ViT-S | 9379 | 40/40 | -0.022 | 0.238 | 0.094 | 0.184 | 0.006 | 0.714 | 234 | 2.65 | 2.73 |
| euroc/V2_03_difficult | heldout | Depth Anything V2 metric indoor, ViT-S | 3962 | 40/40 | -0.014 | 0.308 | 0.097 | 0.212 | 0.013 | 0.802 | 99 | 2.61 | 2.66 |
| euroc/MH_01_easy | fit | Depth Anything V2 metric indoor, ViT-B | 10123 | 40/40 | -0.050 | 0.348 | 0.118 | 0.220 | 0.009 | 0.415 | 253 | 4.63 | 5.01 |
| euroc/MH_02_easy | fit | Depth Anything V2 metric indoor, ViT-B | 9782 | 40/40 | -0.030 | 0.361 | 0.116 | 0.218 | 0.007 | 0.547 | 245 | 4.81 | 4.99 |
| euroc/MH_03_medium | fit | Depth Anything V2 metric indoor, ViT-B | 11520 | 40/40 | -0.345 | 0.451 | 0.113 | 0.354 | 0.006 | 0.575 | 288 | 3.65 | 4.84 |
| euroc/MH_05_difficult | fit | Depth Anything V2 metric indoor, ViT-B | 10874 | 40/40 | -0.337 | 0.406 | 0.156 | 0.311 | 0.009 | 0.426 | 272 | 4.90 | 7.02 |
| euroc/V1_01_easy | fit | Depth Anything V2 metric indoor, ViT-B | 12176 | 40/40 | 0.048 | 0.237 | 0.055 | 0.192 | 0.004 | 0.783 | 304 | 2.58 | 2.42 |
| euroc/V1_02_medium | fit | Depth Anything V2 metric indoor, ViT-B | 12270 | 40/40 | 0.105 | 0.212 | 0.069 | 0.182 | 0.003 | 1.000 | 307 | 2.99 | 2.77 |
| euroc/V2_01_easy | fit | Depth Anything V2 metric indoor, ViT-B | 4607 | 40/40 | 0.206 | 0.166 | 0.065 | 0.109 | 0.048 | 0.172 | 115 | 3.14 | 2.47 |
| euroc/V2_02_medium | fit | Depth Anything V2 metric indoor, ViT-B | 4261 | 40/40 | 0.133 | 0.205 | 0.082 | 0.108 | 0.056 | 0.159 | 107 | 3.54 | 3.05 |
| euroc/MH_04_difficult | heldout | Depth Anything V2 metric indoor, ViT-B | 10849 | 40/40 | -0.388 | 0.385 | 0.158 | 0.256 | 0.008 | 0.461 | 271 | 4.97 | 7.26 |
| euroc/V1_03_difficult | heldout | Depth Anything V2 metric indoor, ViT-B | 9379 | 40/40 | 0.149 | 0.182 | 0.094 | 0.129 | 0.007 | 0.588 | 234 | 3.22 | 2.73 |
| euroc/V2_03_difficult | heldout | Depth Anything V2 metric indoor, ViT-B | 3962 | 40/40 | 0.093 | 0.209 | 0.097 | 0.124 | 0.019 | 0.537 | 99 | 3.07 | 2.66 |
| euroc/MH_01_easy | fit | Metric3D v2, ViT-S | 10123 | 40/40 | -0.350 | 0.408 | 0.118 | 0.365 | 0.008 | 0.502 | 253 | 3.45 | 5.01 |
| euroc/MH_02_easy | fit | Metric3D v2, ViT-S | 9782 | 40/40 | -0.302 | 0.409 | 0.116 | 0.366 | 0.006 | 0.643 | 245 | 3.60 | 4.99 |
| euroc/MH_03_medium | fit | Metric3D v2, ViT-S | 11520 | 40/40 | -0.401 | 0.389 | 0.113 | 0.260 | 0.005 | 0.666 | 288 | 3.32 | 4.84 |
| euroc/MH_05_difficult | fit | Metric3D v2, ViT-S | 10874 | 40/40 | -0.243 | 0.543 | 0.156 | 0.508 | 0.006 | 0.643 | 272 | 5.04 | 7.02 |
| euroc/V1_01_easy | fit | Metric3D v2, ViT-S | 12176 | 40/40 | -0.151 | 0.312 | 0.055 | 0.263 | 0.005 | 0.628 | 304 | 2.08 | 2.42 |
| euroc/V1_02_medium | fit | Metric3D v2, ViT-S | 12270 | 40/40 | -0.203 | 0.276 | 0.069 | 0.208 | 0.005 | 0.592 | 307 | 2.26 | 2.77 |
| euroc/V2_01_easy | fit | Metric3D v2, ViT-S | 4607 | 40/40 | -0.025 | 0.224 | 0.065 | 0.169 | 0.026 | 0.327 | 115 | 2.43 | 2.47 |
| euroc/V2_02_medium | fit | Metric3D v2, ViT-S | 4261 | 40/40 | -0.168 | 0.267 | 0.082 | 0.183 | 0.034 | 0.267 | 107 | 2.56 | 3.05 |
| euroc/MH_04_difficult | heldout | Metric3D v2, ViT-S | 10849 | 40/40 | -0.235 | 0.574 | 0.158 | 0.569 | 0.006 | 0.593 | 271 | 5.42 | 7.26 |
| euroc/V1_03_difficult | heldout | Metric3D v2, ViT-S | 9379 | 40/40 | -0.297 | 0.241 | 0.094 | 0.187 | 0.007 | 0.616 | 234 | 2.08 | 2.73 |
| euroc/V2_03_difficult | heldout | Metric3D v2, ViT-S | 3962 | 40/40 | -0.228 | 0.219 | 0.097 | 0.180 | 0.025 | 0.392 | 99 | 2.10 | 2.66 |
| tumvi/room1 | fit | Depth Anything V2 metric indoor, ViT-S | 4765 | 40/40 | 0.787 | 0.229 | 0.109 | 0.149 | 0.016 | 0.536 | 119 | 4.45 | 2.00 |
| tumvi/room2 | fit | Depth Anything V2 metric indoor, ViT-S | 4246 | 40/40 | 0.858 | 0.222 | 0.107 | 0.123 | 0.030 | 0.310 | 106 | 4.87 | 1.99 |
| tumvi/room3 | fit | Depth Anything V2 metric indoor, ViT-S | 3705 | 39/40 | 0.721 | 0.357 | 0.079 | 0.307 | 0.023 | 0.446 | 95 | 3.14 | 1.54 |
| tumvi/room4 | fit | Depth Anything V2 metric indoor, ViT-S | 4287 | 40/40 | 0.740 | 0.279 | 0.101 | 0.228 | 0.014 | 0.671 | 107 | 4.25 | 1.93 |
| tumvi/room5 | heldout | Depth Anything V2 metric indoor, ViT-S | 4589 | 40/40 | 0.811 | 0.267 | 0.108 | 0.195 | 0.013 | 0.657 | 115 | 4.45 | 1.96 |
| tumvi/room6 | heldout | Depth Anything V2 metric indoor, ViT-S | 4657 | 40/40 | 0.750 | 0.212 | 0.125 | 0.123 | 0.024 | 0.346 | 116 | 4.65 | 2.26 |
| tumvi/room1 | fit | Depth Anything V2 metric indoor, ViT-B | 4765 | 40/40 | 0.812 | 0.218 | 0.109 | 0.106 | 0.024 | 0.347 | 119 | 4.52 | 2.00 |
| tumvi/room2 | fit | Depth Anything V2 metric indoor, ViT-B | 4246 | 40/40 | 0.817 | 0.203 | 0.107 | 0.103 | 0.052 | 0.172 | 106 | 4.49 | 1.99 |
| tumvi/room3 | fit | Depth Anything V2 metric indoor, ViT-B | 3705 | 39/40 | 0.734 | 0.337 | 0.079 | 0.280 | 0.020 | 0.527 | 95 | 3.08 | 1.54 |
| tumvi/room4 | fit | Depth Anything V2 metric indoor, ViT-B | 4287 | 40/40 | 0.789 | 0.256 | 0.101 | 0.181 | 0.019 | 0.483 | 107 | 4.31 | 1.93 |
| tumvi/room5 | heldout | Depth Anything V2 metric indoor, ViT-B | 4589 | 40/40 | 0.774 | 0.289 | 0.108 | 0.237 | 0.015 | 0.578 | 115 | 4.39 | 1.96 |
| tumvi/room6 | heldout | Depth Anything V2 metric indoor, ViT-B | 4657 | 40/40 | 0.794 | 0.208 | 0.125 | 0.119 | 0.036 | 0.235 | 116 | 5.03 | 2.26 |
| tumvi/room1 | fit | Metric3D v2, ViT-S | 4765 | 40/40 | 0.108 | 0.201 | 0.109 | 0.098 | 0.042 | 0.194 | 119 | 2.28 | 2.00 |
| tumvi/room2 | fit | Metric3D v2, ViT-S | 4246 | 40/40 | 0.062 | 0.184 | 0.107 | 0.075 | 0.095 | 0.091 | 106 | 2.16 | 1.99 |
| tumvi/room3 | fit | Metric3D v2, ViT-S | 3705 | 39/40 | 0.044 | 0.227 | 0.079 | 0.101 | 0.025 | 0.417 | 95 | 1.62 | 1.54 |
| tumvi/room4 | fit | Metric3D v2, ViT-S | 4287 | 40/40 | 0.064 | 0.273 | 0.101 | 0.223 | 0.016 | 0.577 | 107 | 2.15 | 1.93 |
| tumvi/room5 | heldout | Metric3D v2, ViT-S | 4589 | 40/40 | 0.058 | 0.244 | 0.108 | 0.126 | 0.024 | 0.364 | 115 | 2.11 | 1.96 |
| tumvi/room6 | heldout | Metric3D v2, ViT-S | 4657 | 40/40 | 0.046 | 0.189 | 0.125 | 0.068 | 0.052 | 0.159 | 116 | 2.46 | 2.26 |

## Latency per frame on this GPU

NVIDIA GeForce RTX 4070 Ti. Wall time of `MetricDepth.predict`, host to device copy and resample included, after 3 warm-up calls. Another process shared the GPU.

| Dataset | input h x w | Network | n frames | median ms | p90 ms |
|---|---|---|---|---|---|
| EuRoC | 480 x 752 | Depth Anything V2 metric indoor, ViT-S | 440 | 94.9 | 186.8 |
| EuRoC | 480 x 752 | Depth Anything V2 metric indoor, ViT-B | 440 | 130.1 | 216.4 |
| EuRoC | 480 x 752 | Metric3D v2, ViT-S | 440 | 240.1 | 693.3 |
| TUM-VI | 512 x 512 | Depth Anything V2 metric indoor, ViT-S | 240 | 77.6 | 140.6 |
| TUM-VI | 512 x 512 | Depth Anything V2 metric indoor, ViT-B | 240 | 95.5 | 173.3 |
| TUM-VI | 512 x 512 | Metric3D v2, ViT-S | 240 | 245.7 | 390.4 |

## Reference and rejection counts

| Sequence | sigma_px | frames | stereo landmarks | dropped, ref. noise | dropped, outside view | points |
|---|---|---|---|---|---|---|
| euroc/MH_01_easy | 0.681 | 40 | 10491 | 368 | 0 | 10123 |
| euroc/MH_02_easy | 0.676 | 40 | 10093 | 311 | 0 | 9782 |
| euroc/MH_03_medium | 0.657 | 40 | 11899 | 379 | 0 | 11520 |
| euroc/MH_05_difficult | 0.628 | 40 | 11163 | 289 | 0 | 10874 |
| euroc/V1_01_easy | 0.643 | 40 | 12179 | 3 | 0 | 12176 |
| euroc/V1_02_medium | 0.674 | 40 | 12277 | 7 | 0 | 12270 |
| euroc/V2_01_easy | 0.682 | 40 | 4609 | 2 | 0 | 4607 |
| euroc/V2_02_medium | 0.764 | 40 | 4272 | 11 | 0 | 4261 |
| tumvi/room1 | 0.563 | 40 | 9278 | 598 | 3915 | 4765 |
| tumvi/room2 | 0.566 | 40 | 8851 | 768 | 3837 | 4246 |
| tumvi/room3 | 0.555 | 40 | 8069 | 273 | 4091 | 3705 |
| tumvi/room4 | 0.539 | 40 | 8984 | 490 | 4207 | 4287 |
| euroc/MH_04_difficult | 0.618 | 40 | 11125 | 276 | 0 | 10849 |
| euroc/V1_03_difficult | 0.932 | 40 | 9396 | 17 | 0 | 9379 |
| euroc/V2_03_difficult | 0.964 | 40 | 3969 | 7 | 0 | 3962 |
| tumvi/room5 | 0.564 | 40 | 9325 | 577 | 4159 | 4589 |
| tumvi/room6 | 0.559 | 40 | 9555 | 753 | 4145 | 4657 |

## Not done

- UniDepth was not run. The manuscript table lists it, and no wrapper exists for it.
- The reference is stereo triangulation with a first-order covariance, not a depth sensor. Its noise is subtracted in variance. A stereo extrinsic error would be a shared error that this deflation does not remove.
