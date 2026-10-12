# Results

The UC-SLAM results behind the table of the top-level README.

## Score files (`scores/ucslam/`)

| File | Content |
|---|---|
| `euroc_mono_inertial.json` | EuRoC, 11 sequences, monocular-inertial, 10 runs each |
| `euroc_stereo_inertial.json` | EuRoC, 11 sequences, stereo-inertial, 10 runs each |
| `tumvi_rooms_mono_inertial.json` | TUM-VI room1 to room6, monocular-inertial, 10 runs each |
| `tumvi_long_mono_inertial.json` | TUM-VI corridor1 to corridor5 and magistrale1 to magistrale6, 3 runs each |

Each file has three parts:
- `meta`: how the runs were made and scored (settings, trajectory type, alignment).
- `summary`: per sequence, the run and failure counts and the statistics of `ate_kf_m`, the frame ATE (`ate_frame_m`), the Sim(3) scale error and, on the long sequences, the end drift (`end_drift_m`, `end_drift_pct`).
- `runs`: one record per run.

`ate_kf_m` is the RMSE in meters of the keyframe trajectory after SE(3) alignment with evo (`--align --t_max_diff 0.02`). A run fails when its ATE is above 1 m on EuRoC or above 5 percent of the path length on TUM-VI, when it creates a second map, loses tracking for more than 1 s, crashes or times out. Dates and times of day are removed from the released records.

## Trajectories (`trajectories/`)

One estimated keyframe trajectory per sequence, in TUM format: the median run by the scoring metric. `MANIFEST.csv` names the run, its metric value and the sha256 of each file.
