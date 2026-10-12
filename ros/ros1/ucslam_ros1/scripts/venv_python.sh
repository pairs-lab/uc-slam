#!/usr/bin/env bash
# Runs a ROS 1 Python node in the repository's Python environment (Python 3.11 with torch) instead of the
# system Python 3.8 of Noetic. rospy and the generated messages are pure Python and import there once rospkg and
# catkin_pkg are on the path. One-time setup, from the repository root:
#   uv pip install --python .venv/bin/python --target ros/ros1/pydeps rospkg catkin_pkg
# Used as launch-prefix in launch/ucslam.launch. UCSLAM_PYTHON and UCSLAM_ROS1_PYDEPS override the defaults.
set -e
REPO=${UCSLAM_REPO:-$(cd "$(dirname "$(readlink -f "$0")")/../../../.." && pwd)}
PY=${UCSLAM_PYTHON:-$REPO/.venv/bin/python}
DEPS=${UCSLAM_ROS1_PYDEPS:-$REPO/ros/ros1/pydeps}
if [ ! -x "$PY" ]; then
  echo "venv_python.sh: no Python at $PY (set UCSLAM_PYTHON)" >&2
  exit 1
fi
export UCSLAM_REPO="$REPO"
export PYTHONPATH="$DEPS:$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$@"
