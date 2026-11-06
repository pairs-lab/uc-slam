"""Tests of the ROS-free bridge core (pytest, no ROS needed):
  python3 -m pytest ros/ros2/ucslam_px4/test -q
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ucslam_px4.bridge_core import (IDENTITY as IDENT, BridgeConfig, BridgeCore, Pose,  # noqa: E402
                                    StatusReport, mat_mul, mat_t, mat_to_rpy, mat_vec, mount_from_params,
                                    percentile, quat_to_mat, rotate_cov6, rot_z, rpy_to_mat, yaw_of)

HALF_PI = math.pi / 2
OPTICAL = (HALF_PI, -HALF_PI, 0.0)   # base_roll, base_pitch, base_yaw of the D455 default (config yaml)
DT = 1.0 / 30.0


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def mat_close(a, b, tol=1e-9):
    return all(close(a[i], b[i], tol) for i in range(3))


def cov(k=1.0, var=1e-4):
    c = [0.0] * 36
    for i in range(6):
        c[i * 7] = k * var
    return c


def status(ok=True, settling=False, init=True, state=2, map_changes=0, message=None):
    level = 0 if ok and init and not settling and state == 2 else 1
    msg = message or ("tracking" if level == 0 else "settling")
    return StatusReport(level, msg, {"state": str(state), "imu_initialized": "true" if init else "false",
                                     "settling": "true" if settling else "false",
                                     "map_changes": str(map_changes)})


def core_ok(mount=Pose(), **kw):
    """A core with an OK status at t=0 and the default start_frames (3)."""
    c = BridgeCore(BridgeConfig(**kw), mount)
    c.on_status(status(), 0.0)
    return c


def level_pose(x, y, z, yaw, roll=0.0, pitch=0.0):
    return Pose(rpy_to_mat(roll, pitch, yaw), (x, y, z))


# ---------------------------------------------------------------- frames

def test_optical_mount_matrix():
    # base x = IMU z (forward), base y = -IMU x (left), base z = -IMU y (up)
    r = rpy_to_mat(*OPTICAL)
    assert mat_close(r, ((0, -1, 0), (0, 0, -1), (1, 0, 0)))
    assert close(mat_vec(r, (1, 0, 0)), (0, 0, 1))
    assert close(mat_vec(r, (0, 1, 0)), (-1, 0, 0))
    assert close(mat_vec(r, (0, 0, 1)), (0, -1, 0))


def test_quaternion_round_trip():
    for rpy in [(0.1, -0.2, 0.3), (HALF_PI, -HALF_PI, 0.0), (math.pi, 0.0, 0.0), (0.0, 0.0, -3.0), (2.0, 1.0, -2.5)]:
        r = rpy_to_mat(*rpy)
        assert mat_close(quat_to_mat(Pose(r).quat()), r, 1e-9)


def test_identity_mount_passes_pose_through():
    c = BridgeCore(BridgeConfig(require_status_ok=False, start_frames=1), mount_from_params(0, 0, 0, 0, 0, 0))
    raw = level_pose(1.0, -2.0, 3.0, 0.4, roll=0.1, pitch=-0.05)
    out = c.on_odom(10.0, raw, cov(), 10.01)
    assert out is not None and out.stamp == 10.0
    assert close(out.pose.p, raw.p) and mat_close(out.pose.R, raw.R)
    assert close(out.covariance, cov())


def test_optical_imu_to_flu():
    lever = (0.02, 0.05, -0.10)   # base_link origin in IMU axes
    mount = mount_from_params(*lever, *OPTICAL)
    c = BridgeCore(BridgeConfig(require_status_ok=False, start_frames=1), mount)
    # vehicle level, heading 30 deg: the IMU orientation in the map is R_world_base * R_imu_base^T
    r_wb = rot_z(math.radians(30.0))
    r_wi = mat_mul(r_wb, mat_t(mount.R))
    assert close(mat_vec(r_wi, (0, 1, 0)), (0, 0, -1))   # IMU y (down) points down in the z-up map
    raw = Pose(r_wi, (1.0, 2.0, 0.5))
    out = c.on_odom(1.0, raw, cov(), 1.0)
    roll, pitch, yaw = mat_to_rpy(out.pose.R)
    assert abs(roll) < 1e-9 and abs(pitch) < 1e-9 and abs(math.degrees(yaw) - 30.0) < 1e-9
    expect = tuple(raw.p[i] + mat_vec(r_wi, lever)[i] for i in range(3))
    assert close(out.pose.p, expect)
    # forward motion of the camera (IMU z) is +x of base_link
    assert close(mat_vec(out.pose.R, (1, 0, 0)), mat_vec(r_wi, (0, 0, 1)))


def test_rotate_cov6():
    c = [0.0] * 36
    c[0], c[7], c[14] = 4.0, 1.0, 9.0      # anisotropic position block
    r = rot_z(HALF_PI)
    out = rotate_cov6(c, r)
    assert abs(out[0] - 1.0) < 1e-12 and abs(out[7] - 4.0) < 1e-12 and abs(out[14] - 9.0) < 1e-12
    assert close(rotate_cov6(cov(), r), cov(), 1e-15)   # isotropic stays the same


# ---------------------------------------------------------------- gating

def test_gating_on_status():
    c = BridgeCore(BridgeConfig(), Pose())
    t = 100.0
    assert c.on_odom(t, level_pose(0, 0, 0, 0), cov(), t) is None
    assert c.state == "hold: no status yet"
    c.on_status(status(settling=True), t)
    t += DT
    assert c.on_odom(t, level_pose(0, 0, 0, 0), cov(), t) is None
    assert c.state.startswith("hold: status: settling")
    c.on_status(status(init=False, state=2, message="waiting for IMU initialization"), t)
    t += DT
    assert c.on_odom(t, level_pose(0, 0, 0, 0), cov(), t) is None
    c.on_status(status(), t)
    sent = []
    for _ in range(4):
        t += DT
        sent.append(c.on_odom(t, level_pose(0, 0, 0, 0), cov(), t) is not None)
    assert sent == [False, False, True, True]   # start_frames = 3 in a row
    assert c.counters["held"] == 5 and c.counters["sent"] == 2 and c.state == "sending"
    # a stale status holds the output
    t += 4.0
    assert c.on_odom(t, level_pose(0, 0, 0, 0), cov(), t) is None
    assert "status older" in c.state


def test_gating_on_covariance():
    c = core_ok()
    t = 0.0
    for _ in range(3):
        t += DT
        c.on_odom(t, level_pose(0, 0, 0, 0), cov(), t)
    t += DT
    assert c.on_odom(t, level_pose(0, 0, 0, 0), cov(k=100.0), t) is None
    assert c.state == "hold: recently lost (covariance)"
    t += DT   # back to OK within the gap: the segment goes on, no new start run needed
    assert c.on_odom(t, level_pose(0, 0, 0, 0), cov(), t) is not None
    assert c.counters["resets"] == 0


def test_status_gate_off():
    c = BridgeCore(BridgeConfig(require_status_ok=False), Pose())
    out = [c.on_odom(i * DT, level_pose(0, 0, 0, 0), cov(), i * DT) for i in range(1, 4)]
    assert out[-1] is not None


def test_mount_unknown_holds():
    c = BridgeCore(BridgeConfig(require_status_ok=False, start_frames=1), None)
    assert c.on_odom(1.0, level_pose(0, 0, 0, 0), cov(), 1.0) is None
    assert "mount" in c.state
    c.mount = Pose()
    assert c.on_odom(1.1, level_pose(0, 0, 0, 0), cov(), 1.1) is not None


def test_old_stamps_dropped():
    c = core_ok()
    for i in range(1, 4):
        c.on_odom(i * DT, level_pose(0, 0, 0, 0), cov(), i * DT)
    assert c.on_odom(2 * DT, level_pose(0, 0, 0, 0), cov(), 4 * DT) is None
    assert c.on_odom(3 * DT, level_pose(0, 0, 0, 0), cov(), 4 * DT) is None
    assert c.counters["old"] == 2 and c.counters["sent"] == 1


# ---------------------------------------------------------------- jumps and segments

def fly(c, t0, n, start, v=1.0, yaw=0.0, roll=0.0, pitch=0.0, jump=None):
    """n frames of straight flight at v m/s along heading yaw. jump: (dx, dy, dyaw) map jump of the raw pose."""
    outs = []
    for i in range(n):
        t = t0 + i * DT
        x = start[0] + v * i * DT * math.cos(yaw)
        y = start[1] + v * i * DT * math.sin(yaw)
        true = level_pose(x, y, start[2], yaw, roll, pitch)
        raw = true if jump is None else Pose(rot_z(jump[2]), (jump[0], jump[1], 0.0)).compose(true)
        outs.append((t, raw, c.on_odom(t, raw, cov(), t)))
    return outs


def test_no_false_jump_below_limits():
    c = core_ok()
    outs = fly(c, DT, 60, (0, 0, 1), v=4.9, yaw=0.3)
    # yaw rate 170 deg/s, below 180
    t, raw, yaw = outs[-1][0], outs[-1][1], 0.3
    for i in range(1, 30):
        yaw += math.radians(170.0) * DT
        assert c.on_odom(t + i * DT, level_pose(*raw.p, yaw), cov(), t + i * DT) is not None
    assert c.counters["jumps"] == 0 and c.counters["resets"] == 0


def test_jump_reanchors_and_keeps_roll_pitch():
    c = core_ok()
    roll, pitch = math.radians(5.0), math.radians(-3.0)
    before = fly(c, DT, 30, (0, 0, 1), v=1.0, roll=roll, pitch=pitch)
    last = before[-1][2].pose
    # loop closure: the map moves 2 m in y and turns 20 deg
    after = fly(c, DT * 31, 10, (30 * DT, 0, 1), v=1.0, roll=roll, pitch=pitch,
                jump=(0.0, 2.0, math.radians(20.0)))
    assert c.counters["jumps"] == 1 and c.counters["resets"] == 0
    first = after[0][2].pose
    assert close(first.p, last.p, 1e-9)
    assert abs(yaw_of(first.R) - yaw_of(last.R)) < 1e-9
    r0, p0, _ = mat_to_rpy(first.R)
    assert abs(r0 - roll) < 1e-9 and abs(p0 - pitch) < 1e-9
    # the output goes on with the true motion: 1 m/s along the last heading
    for k in range(1, len(after)):
        a, b = after[k - 1][2].pose, after[k][2].pose
        step = math.dist(a.p, b.p)
        assert abs(step - DT) < 1e-9
        assert abs(math.atan2(b.p[1] - a.p[1], b.p[0] - a.p[0]) - yaw_of(last.R)) < 1e-9
    assert any(level == "warn" and text.startswith("jump: pose step") for level, text in c.pop_events())


def test_yaw_only_jump_detected():
    c = core_ok()
    fly(c, DT, 10, (0, 0, 0), v=0.0)
    fly(c, DT * 11, 3, (0, 0, 0), v=0.0, jump=(0.0, 0.0, math.radians(15.0)))
    assert c.counters["jumps"] == 1
    assert abs(yaw_of(c.last_out[1].R)) < 1e-9


def test_status_map_change_reanchors():
    c = core_ok()
    before = fly(c, DT, 10, (0, 0, 0), v=1.0)
    c.on_status(status(map_changes=1), 10 * DT)
    # a small map correction that the step test does not catch (5 cm)
    after = fly(c, DT * 11, 3, (10 * DT, 0, 0), v=1.0, jump=(0.0, 0.05, 0.0))
    assert c.counters["jumps"] == 1
    assert close(after[0][2].pose.p, before[-1][2].pose.p, 1e-9)
    assert abs(math.dist(after[1][2].pose.p, after[2][2].pose.p) - DT) < 1e-9


def test_imu_init_flip_counts_jump():
    c = core_ok()
    fly(c, DT, 5, (0, 0, 0))
    c.on_status(status(init=False, message="waiting for IMU initialization"), 1.0)
    assert c.counters["jumps"] == 1 and c.pending_jump is not None


def test_jump_during_short_hold_keeps_held_motion():
    c = core_ok()
    before = fly(c, DT, 10, (0, 0, 0), v=1.0)
    assert close(before[-1][2].pose.p, (9 * DT, 0, 0), 1e-12)
    t = 11 * DT
    # RECENTLY_LOST frames (IMU prediction) go on moving, then the map jumps while held
    c.on_odom(t, level_pose(10 * DT, 0, 0, 0), cov(k=100.0), t)
    c.on_odom(t + DT, level_pose(11 * DT, 0, 0, 0), cov(k=100.0), t + DT)
    c.on_odom(t + 2 * DT, level_pose(3.0, 1.0, 0, 0), cov(k=100.0), t + 2 * DT)
    out = c.on_odom(t + 3 * DT, level_pose(3.0 + DT, 1.0, 0, 0), cov(), t + 3 * DT)
    assert out is not None and c.counters["jumps"] == 1 and c.counters["resets"] == 0
    # anchored to the held frame before the jump (x 11 DT), plus one frame of motion. Only the jump frame
    # interval is lost: 12 DT instead of 13 DT. Anchoring to the last sent pose would give 9 DT.
    assert close(out.pose.p, (12 * DT, 0, 0), 1e-9)


def test_status_jump_after_step_jump_is_one_event():
    c = core_ok()
    fly(c, DT, 30, (0, 0, 1), v=1.0)
    after = fly(c, 31 * DT, 15, (30 * DT, 0, 1), v=1.0, jump=(0.0, 1.0, 0.3))
    assert c.counters["jumps"] == 1
    c.on_status(status(map_changes=1), 46 * DT)   # the 1 Hz status reports the same loop closure later
    assert c.counters["jumps"] == 1 and c.pending_jump is None
    more = fly(c, 46 * DT, 5, (45 * DT, 0, 1), v=1.0, jump=(0.0, 1.0, 0.3))
    outs = [o.pose.p[0] for _, _, o in after + more]
    assert all(abs((b - a) - DT) < 1e-9 for a, b in zip(outs, outs[1:]))   # no frame of motion lost
    # a status jump long after the step jump still counts
    c.on_status(status(map_changes=2), 46 * DT + 2.0)
    assert c.counters["jumps"] == 2 and c.pending_jump is not None


def test_status_state3_passes_and_covariance_gates():
    c = core_ok()
    fly(c, DT, 10, (0, 0, 0), v=0.0)
    # the 1 Hz status samples one RECENTLY_LOST frame, tracking is back at the next frame
    c.on_status(status(state=3, message="recently lost"), 11 * DT)
    assert c.status.ok()
    outs = fly(c, 11 * DT, 30, (0, 0, 0), v=0.0)
    assert all(o is not None for _, _, o in outs) and c.counters["resets"] == 0
    t = 41 * DT
    assert c.on_odom(t, level_pose(0, 0, 0, 0), cov(k=100.0), t) is None
    assert c.state == "hold: recently lost (covariance)"
    # settling, no IMU init and LOST still hold
    assert not status(state=3, settling=True).ok() and not status(state=3, init=False).ok()
    assert not status(state=4).ok()


def test_state3_without_baseline_holds():
    # bridge started while UC-SLAM is RECENTLY_LOST: no state-2 frame seen, so no covariance baseline
    c = BridgeCore(BridgeConfig(), Pose())
    c.on_status(status(state=3, message="recently lost"), 0.0)
    outs = [c.on_odom(i * DT, level_pose(0, 0, 0, 0), cov(k=100.0), i * DT) for i in range(1, 6)]
    assert all(o is None for o in outs) and c.min_pos_var is None
    assert c.state == "hold: status: recently lost, no covariance baseline yet"
    c.on_status(status(), 6 * DT)
    outs = [c.on_odom(i * DT, level_pose(0, 0, 0, 0), cov(), i * DT) for i in range(6, 9)]
    assert outs[-1] is not None and c.min_pos_var == pytest.approx(1e-4)


def test_zero_covariance_warns_once():
    c = BridgeCore(BridgeConfig(require_status_ok=False), Pose())
    for i in range(1, 5):
        c.on_odom(i * DT, level_pose(0, 0, 0, 0), [0.0] * 36, i * DT)
    warns = [t for lvl, t in c.pop_events() if lvl == "warn" and "variance is 0" in t]
    assert len(warns) == 1 and c.counters["sent"] == 2


def test_non_finite_pose_is_held_and_does_not_poison():
    nan = float("nan")
    with pytest.raises(ValueError):
        quat_to_mat((nan, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        quat_to_mat((float("inf"), 0.0, 0.0, 1.0))
    c = core_ok()
    before = fly(c, DT, 10, (0, 0, 0), v=1.0)
    c.on_status(status(map_changes=1), 10 * DT)   # a status jump is pending
    t = 11 * DT
    assert c.on_odom(t, Pose(IDENT, (nan, 0.0, 0.0)), cov(), t) is None
    bad_cov = cov()
    bad_cov[7] = nan
    assert c.on_odom(t + DT, level_pose(11 * DT, 0, 0, 0), bad_cov, t + DT) is None
    c.on_invalid("quaternion is not finite", t + 2 * DT)
    assert c.counters["invalid"] == 3 and c.state == "hold: invalid pose"
    assert sum(1 for lvl, txt in c.pop_events() if txt.startswith("odometry skipped")) == 1
    outs = fly(c, t + 3 * DT, 5, (12 * DT, 0, 0), v=1.0)
    ps = [o.pose.p for _, _, o in outs if o is not None]
    assert len(ps) >= 2 and all(math.isfinite(v) for p in ps for v in p)
    assert all(math.isfinite(v) for v in c.correction.p)
    assert close(ps[0], before[-1][2].pose.p, 1e-9)   # the pending status jump re-anchored a valid frame


def test_stamp_going_back_restarts_stream():
    c = BridgeCore(BridgeConfig(require_status_ok=False), Pose())
    for i in range(30):
        c.on_odom(100.0 + i * DT, level_pose(0, 0, 0, 0), cov(), 100.0 + i * DT)
    # a small step back is an out-of-order message: dropped
    assert c.on_odom(100.0 + 28 * DT, level_pose(0, 0, 0, 0), cov(), 101.0) is None and c.counters["old"] == 1
    # bag loop: the clock goes back 50 s
    outs = [c.on_odom(50.0 + i * DT, level_pose(1.0, 0, 0, 0), cov(), 50.0 + i * DT) for i in range(10)]
    assert [o is None for o in outs[:3]] == [True, True, False]
    assert sum(1 for o in outs if o is not None) == 8
    assert c.counters["resets"] == 1 and c.counters["jumps"] == 0 and c.state == "sending"
    assert any(txt.startswith("reset: stamp went back") for _, txt in c.pop_events())


def test_long_gap_starts_new_segment_without_reanchor():
    c = core_ok()
    fly(c, DT, 10, (0, 0, 0), v=1.0)
    c.on_status(status(), 1.0)
    # 1 s gap, then the pose is 5 m away (for example a new map)
    after = fly(c, 10 * DT + 1.0, 4, (5.0, 0.0, 0.0), v=1.0)
    assert [o is None for _, _, o in after] == [True, True, False, False]
    assert c.counters["resets"] == 1 and c.counters["jumps"] == 0 and c.counters["segments"] == 2
    t, raw, out = after[2]
    assert close(out.pose.p, raw.p) and mat_close(out.pose.R, raw.R)   # no re-anchoring
    assert any(text.startswith("reset:") for _, text in c.pop_events())


def test_correction_kept_across_new_segment():
    c = core_ok()
    fly(c, DT, 10, (0, 0, 0), v=0.0)
    fly(c, 11 * DT, 3, (0, 0, 0), v=0.0, jump=(1.0, 0.0, 0.0))
    assert c.counters["jumps"] == 1
    c.on_status(status(), 1.0)
    after = fly(c, 14 * DT + 1.0, 4, (0, 0, 0), v=0.0, jump=(1.0, 0.0, 0.0))
    assert close(after[-1][2].pose.p, (0.0, 0.0, 0.0), 1e-9)   # same correction as before the gap


# ---------------------------------------------------------------- statistics

def test_percentile_matches_slam_node_rule():
    v = [10.0 * i for i in range(1, 11)]
    assert percentile(v, 0.5) == 60.0 and percentile(v, 0.9) == 90.0 and percentile([], 0.5) == 0.0


def test_latency_and_rate():
    c = core_ok()
    c.stats(0.0)   # starts the window
    for i in range(10):
        stamp = 1.0 + i * DT
        c.on_odom(stamp, level_pose(0, 0, 0, 0), cov(), stamp + 0.010 * (i + 1))
    s = c.stats(2.0)
    assert s["latency_ms_p50"] == pytest.approx(60.0) and s["latency_ms_p90"] == pytest.approx(90.0)
    assert s["sent"] == 8 and s["rate_hz"] == pytest.approx(4.0)
    s2 = c.stats(3.0)
    assert s2["latency_ms_p50"] == 0.0 and s2["rate_hz"] == 0.0 and s2["sent"] == 8


def test_tick_reports_stopped_stream():
    c = core_ok()
    fly(c, DT, 5, (0, 0, 0))
    c.tick(5 * DT + 0.1)
    assert c.state == "sending"
    c.tick(5 * DT + 2.0)
    assert c.state.startswith("hold: no odometry")
