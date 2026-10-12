"""UC-SLAM evaluation tooling.

Modules
-------
run_orbslam3     runs the ORB-SLAM3 examples of a tree (stock or the UC-SLAM fork) on
                 EuRoC and TUM-VI, converts the trajectories and scores them: ATE with
                 evo, Sim(3) scale error, TUM-VI end drift, and the failure rule.
nees             relative-pose NEES over fixed time horizons (gauge-free consistency
                 metric) plus a clearly-secondary absolute NEES with an explicit gauge fix.
nees_from_dump   relative-pose NEES from the posterior covariance dump of one run.
stats            median/IQR cells, success counts, paired Wilcoxon, bootstrap CI on
                 the median paired difference, Holm correction.
tables           LaTeX table bodies (booktabs rows only) generated from results.

The evaluation protocol these tools implement is docs/protocol.md.
Nothing in this package imports torch at import time; matplotlib is imported lazily.
"""

__all__ = ["run_orbslam3", "nees", "nees_from_dump", "stats", "tables"]
