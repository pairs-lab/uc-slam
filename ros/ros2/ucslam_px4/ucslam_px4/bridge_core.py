"""ROS-free core of the bridge from UC-SLAM to PX4 (through MAVROS).

Frames
  UC-SLAM publishes the pose of its IMU body frame ("imu") in its map frame ("map"). After IMU initialization
  the map is gravity aligned with z up, and its yaw is arbitrary. The body axes are the IMU axes of the settings
  (IMU.T_b_c1), not FLU. The bridge applies the static pose of base_link in the IMU frame (the mount) to get the
  pose of base_link (FLU) in the map. It then treats the map as the ENU world that MAVROS expects. The x axis of
  that world is arbitrary, so PX4 must fuse vision yaw with the magnetometer off (see ros/README.md).

Gating
  A pose is sent only when it is finite, the last slam_node status says IMU initialized and not settling with
  state 2 (or state 3, RECENTLY_LOST, which the covariance gate then judges per frame), the status is fresh,
  the pose covariance is not the inflated RECENTLY_LOST value (x100), and the stream is continuous (a new
  segment needs a few messages in a row at short intervals).

Jumps
  UC-SLAM can jump on loop closure, map merge, relocalization or a new map. MAVROS cannot send PX4 a
  reset_counter. So the bridge detects a jump from the pose step against physical limits (max speed, max yaw
  rate) and from a change of map_changes or imu_initialized in the status. A step jump updates a correction
  (yaw and position only) at once, so the jumped pose has the position and yaw of the corrected pose just
  before it. Held frames count too, so motion during a short hold is kept. A status jump re-anchors the next
  frame the same way, unless a step jump was already handled within status_jump_window_s. Roll and pitch always
  come from the gravity-aligned estimate. After a gap longer than gap_reset_s, or a stamp that goes back by
  more than gap_reset_s, the bridge starts a new segment without re-anchoring and counts a reset. PX4 stops
  vision fusion after 0.4 s without data and resets to the vision pose when it starts again.

Rotations are 3x3 tuples (row major). Quaternions use the ROS order (x, y, z, w).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]
Quat = Tuple[float, float, float, float]

IDENTITY: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# ---------------------------------------------------------------- small rotation helpers (no numpy)

def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))  # type: ignore


def mat_t(a: Mat3) -> Mat3:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))  # type: ignore


def mat_vec(a: Mat3, v: Sequence[float]) -> Vec3:
    return tuple(a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2] for i in range(3))  # type: ignore


def rot_x(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def rot_y(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def rot_z(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def rpy_to_mat(roll: float, pitch: float, yaw: float) -> Mat3:
    """Rz(yaw) Ry(pitch) Rx(roll), the convention of tf2 setRPY and static_transform_publisher."""
    return mat_mul(rot_z(yaw), mat_mul(rot_y(pitch), rot_x(roll)))


def mat_to_rpy(r: Mat3) -> Vec3:
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    return (math.atan2(r[2][1], r[2][2]), pitch, math.atan2(r[1][0], r[0][0]))


def yaw_of(r: Mat3) -> float:
    """Heading of the body x axis about the world z axis (the ZYX yaw)."""
    return math.atan2(r[1][0], r[0][0])


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_mat(q: Sequence[float]) -> Mat3:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n):
        raise ValueError("quaternion is not finite")
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def mat_to_quat(r: Mat3) -> Quat:
    t = r[0][0] + r[1][1] + r[2][2]
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = ((r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, 0.25 * s)
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2])
        q = (0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s)
    elif r[1][1] > r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2])
        q = ((r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1])
        q = ((r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s, (r[1][0] - r[0][1]) / s)
    if q[3] < 0.0:   # one sign for the same rotation
        q = (-q[0], -q[1], -q[2], -q[3])
    return q


@dataclass(frozen=True)
class Pose:
    """Rigid transform: maps points of the child frame into the parent frame (x_parent = R x_child + p)."""
    R: Mat3 = IDENTITY
    p: Vec3 = (0.0, 0.0, 0.0)

    @staticmethod
    def from_quat(p: Sequence[float], q: Sequence[float]) -> "Pose":
        return Pose(quat_to_mat(q), (float(p[0]), float(p[1]), float(p[2])))

    def quat(self) -> Quat:
        return mat_to_quat(self.R)

    def compose(self, other: "Pose") -> "Pose":
        """self * other."""
        v = mat_vec(self.R, other.p)
        return Pose(mat_mul(self.R, other.R), (v[0] + self.p[0], v[1] + self.p[1], v[2] + self.p[2]))

    def inverse(self) -> "Pose":
        rt = mat_t(self.R)
        v = mat_vec(rt, self.p)
        return Pose(rt, (-v[0], -v[1], -v[2]))


def mount_from_params(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> Pose:
    """Pose of base_link in the UC-SLAM IMU body frame (metres, radians, tf2 RPY convention)."""
    return Pose(rpy_to_mat(roll, pitch, yaw), (float(x), float(y), float(z)))


def rotate_cov6(cov: Sequence[float], r: Mat3) -> List[float]:
    """C' = B C B^T with B = diag(r, r), for a row-major 6x6 pose covariance (x, y, z, rx, ry, rz)."""
    b = [[0.0] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            b[i][j] = b[i + 3][j + 3] = r[i][j]
    c = [[float(cov[6 * i + j]) for j in range(6)] for i in range(6)]
    bc = [[sum(b[i][k] * c[k][j] for k in range(6)) for j in range(6)] for i in range(6)]
    return [sum(bc[i][k] * b[j][k] for k in range(6)) for i in range(6) for j in range(6)]


def percentile(values: Sequence[float], q: float) -> float:
    """Same rule as Percentile() in slam_node.cc: nearest rank after sorting, 0 when empty."""
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


# ---------------------------------------------------------------- status of slam_node

LEVEL_OK = 0


@dataclass
class StatusReport:
    """The 'ucslam' entry of /ucslam/slam_node/status."""
    level: int
    message: str
    values: Dict[str, str] = field(default_factory=dict)

    def flag(self, key: str) -> Optional[bool]:
        v = self.values.get(key)
        return None if v is None else v.strip().lower() == "true"

    def int_value(self, key: str) -> Optional[int]:
        try:
            return int(float(self.values[key]))
        except (KeyError, ValueError):
            return None

    def ok(self) -> bool:
        """True for OK level with state 2 (or no state key), or for state 3 (RECENTLY_LOST, WARN level) after
        IMU init and settling. Keys that are present must agree. slam_node sends no odometry in other states.
        State 3 is accepted because the 1 Hz status can sample one RECENTLY_LOST frame and then hold a whole
        second of good frames. The per-frame covariance gate of BridgeCore holds the RECENTLY_LOST frames."""
        state = self.int_value("state")
        init, settling = self.flag("imu_initialized"), self.flag("settling")
        if init is False or settling is True:
            return False
        if state == 3:
            return init is True and settling is False
        return self.level == LEVEL_OK and (state is None or state == 2)

    def recently_lost(self) -> bool:
        return self.int_value("state") == 3


# ---------------------------------------------------------------- bridge core

@dataclass
class BridgeConfig:
    max_speed_mps: float = 5.0          # a larger position step per frame interval is a jump
    max_yaw_rate_dps: float = 180.0     # a larger yaw step per frame interval is a jump
    jump_margin_m: float = 0.05         # slack added to the position limit
    jump_margin_deg: float = 3.0        # slack added to the yaw limit
    gap_reset_s: float = 0.5            # a longer gap between sent poses starts a new segment
    require_status_ok: bool = True
    status_timeout_s: float = 3.0       # an older status holds the output
    start_frames: int = 3               # messages in a row needed to start a segment
    max_interval_s: float = 0.2         # longest interval that keeps a run "in a row" (PX4 EV_MAX_INTERVAL)
    lost_cov_ratio: float = 10.0        # position variance above ratio x the smallest seen: RECENTLY_LOST
    status_jump_window_s: float = 1.5   # a status jump this soon after a step jump is the same event


@dataclass
class Output:
    stamp: float                 # the image stamp of the UC-SLAM message, unchanged
    pose: Pose                   # base_link (FLU) in the ENU world
    covariance: List[float]      # 6x6 row major, in the world frame


class BridgeCore:
    """Gating, frame conversion, jump re-anchoring and statistics. Times are seconds."""

    def __init__(self, config: BridgeConfig, mount: Optional[Pose] = Pose()):
        self.cfg = config
        self.mount = mount                    # None: not known yet (TF lookup pending), output held
        self.correction = Pose()              # world <- UC-SLAM map, yaw and translation only
        self.prev_raw: Optional[Tuple[float, Pose]] = None   # last base_link pose in the map, any message
        self.last_out: Optional[Tuple[float, Pose]] = None   # last published pose in the world
        self.segment_active = False
        self.segment_start: Optional[float] = None   # receive time of the first pose of the segment
        self.pending_jump: Optional[str] = None      # a status jump, re-anchored at the next frame
        self.last_step_jump_rx: Optional[float] = None   # receive time of the last step jump
        self.run = 0
        self.status: Optional[StatusReport] = None
        self.status_rx: Optional[float] = None
        self.min_pos_var: Optional[float] = None
        self.last_odom_rx: Optional[float] = None
        self.state = "waiting for odometry"
        self.counters: Dict[str, int] = dict(received=0, sent=0, held=0, old=0, invalid=0, jumps=0, resets=0,
                                             segments=0)
        self._invalid_run = False
        self._warned_zero_cov = False
        self.events: List[Tuple[str, str]] = []   # (level, text), drained by the node with pop_events()
        self._lat_ms: List[float] = []
        self._sent_window = 0
        self._window_start: Optional[float] = None

    # ---- inputs

    def on_status(self, report: StatusReport, now: float) -> None:
        old = self.status
        self.status, self.status_rx = report, now
        if old is None or self.last_out is None:
            return
        mc_old, mc_new = old.int_value("map_changes"), report.int_value("map_changes")
        if mc_old is not None and mc_new is not None and mc_new > mc_old:
            self._status_jump(f"map_changes {mc_old} -> {mc_new} in the status", now)
        ii_old, ii_new = old.flag("imu_initialized"), report.flag("imu_initialized")
        if ii_old is not None and ii_new is not None and ii_old != ii_new:
            self._status_jump(f"imu_initialized {ii_old} -> {ii_new} in the status", now)

    def on_invalid(self, reason: str, now: float) -> None:
        """A message the node could not turn into a Pose (for example a NaN or zero quaternion)."""
        self.counters["received"] += 1
        self.last_odom_rx = now
        return self._invalid(reason)

    def on_odom(self, stamp: float, raw_imu: Pose, cov: Sequence[float], now: float) -> Optional[Output]:
        """raw_imu: pose of the UC-SLAM IMU body in the UC-SLAM map. Returns what to publish, or None."""
        c = self.counters
        c["received"] += 1
        self.last_odom_rx = now
        # A NaN would pass every comparison below (nan > limit is False) and could poison the correction.
        if not (math.isfinite(stamp) and _pose_finite(raw_imu) and all(math.isfinite(float(v)) for v in cov)):
            return self._invalid("pose, stamp or covariance is not finite")
        self._invalid_run = False
        self._lat_ms.append(1000.0 * (now - stamp))
        if self.mount is None:
            return self._hold("waiting for the base_link mount (TF)")
        if self.prev_raw is not None and stamp <= self.prev_raw[0]:
            back = self.prev_raw[0] - stamp
            if back <= self.cfg.gap_reset_s:   # out of order or repeated: drop it
                c["old"] += 1
                return None
            # bag loop, sim time restart or driver restart: start again instead of dropping every message
            self._restart_stream(f"stamp went back {back:.2f} s")
        base = raw_imu.compose(self.mount)
        interval = None if self.prev_raw is None else stamp - self.prev_raw[0]
        if interval is not None and interval <= self.cfg.gap_reset_s:
            if not self._check_step(self.prev_raw[1], base, interval, now) and self.pending_jump is not None:
                self._reanchor(self.prev_raw[1], base, self.pending_jump)
        self.prev_raw = (stamp, base)

        reason = self._gate(cov, now)
        if reason:
            self.run = 0
            return self._hold(reason)
        self.run = self.run + 1 if interval is not None and interval <= self.cfg.max_interval_s else 1

        if self.segment_active and self.last_out is not None and stamp - self.last_out[0] > self.cfg.gap_reset_s:
            self.segment_active = False
        if not self.segment_active:
            if self.run < self.cfg.start_frames:
                return self._hold("starting stream")
            if self.last_out is not None:
                c["resets"] += 1
                self.events.append(("warn", f"reset: {stamp - self.last_out[0]:.2f} s since the last sent pose, "
                                            "new segment without re-anchoring"))
            self.segment_active = True
            self.segment_start = now
            self.pending_jump = None
            c["segments"] += 1

        out = self.correction.compose(base)
        self.last_out = (stamp, out)
        c["sent"] += 1
        self._sent_window += 1
        self.state = "sending"
        return Output(stamp, out, rotate_cov6(cov, self.correction.R))

    def tick(self, now: float) -> None:
        """Called by the node timer: notices a stream that stopped."""
        if self.last_odom_rx is not None and now - self.last_odom_rx > self.cfg.gap_reset_s:
            self.state = f"hold: no odometry for {now - self.last_odom_rx:.1f} s"

    # ---- outputs

    def pop_events(self) -> List[Tuple[str, str]]:
        e, self.events = self.events, []
        return e

    def stats(self, now: float) -> Dict[str, float]:
        """Statistics since the last call (rate, latency) and running counters."""
        span = None if self._window_start is None else now - self._window_start
        rate = self._sent_window / span if span and span > 0 else 0.0
        s: Dict[str, float] = dict(self.counters)
        s.update(rate_hz=rate, latency_ms_p50=percentile(self._lat_ms, 0.5),
                 latency_ms_p90=percentile(self._lat_ms, 0.9),
                 correction_yaw_deg=math.degrees(yaw_of(self.correction.R)),
                 segment_age_s=0.0 if self.segment_start is None else now - self.segment_start)
        self._lat_ms, self._sent_window, self._window_start = [], 0, now
        return s

    # ---- internals

    def _hold(self, reason: str) -> None:
        self.counters["held"] += 1
        self.state = "hold: " + reason
        return None

    def _invalid(self, reason: str) -> None:
        self.counters["invalid"] += 1
        self.run = 0
        if not self._invalid_run:   # one WARN per run of bad messages
            self.events.append(("warn", f"odometry skipped: {reason}"))
            self._invalid_run = True
        return self._hold("invalid pose")

    def _restart_stream(self, reason: str) -> None:
        """Forget the stream (not the correction). The next poses start a new segment."""
        if self.last_out is not None:
            self.counters["resets"] += 1
        self.events.append(("warn", f"reset: {reason}, new segment without re-anchoring"))
        self.prev_raw = self.last_out = None
        self.segment_active = False
        self.run = 0
        self.pending_jump = self.last_step_jump_rx = None

    def _status_reason(self, now: float) -> Optional[str]:
        if self.status is None:
            return "no status yet"
        if now - self.status_rx > self.cfg.status_timeout_s:
            return f"status older than {self.cfg.status_timeout_s:.1f} s"
        if not self.status.ok():
            return "status: " + (self.status.message or "not OK")
        return None

    def _gate(self, cov: Sequence[float], now: float) -> Optional[str]:
        var = float(cov[0]) if len(cov) >= 36 else 0.0
        status_reason = self._status_reason(now) if self.cfg.require_status_ok else None
        # With the status gate on, state 3 passes the status. Then only the covariance tells RECENTLY_LOST
        # frames apart, so its baseline is learned only from frames that arrive while the status says state 2.
        lost_status = self.cfg.require_status_ok and self.status is not None and self.status.recently_lost()
        if var > 0.0:
            if status_reason is None and not lost_status and (self.min_pos_var is None or var < self.min_pos_var):
                self.min_pos_var = var
            if self.min_pos_var is not None and var > self.cfg.lost_cov_ratio * self.min_pos_var:
                return "recently lost (covariance)"
        elif not self._warned_zero_cov:
            self.events.append(("warn", "odometry position variance is 0 (pose_std_m 0?): the RECENTLY_LOST "
                                        "covariance gate is off, only the status gate is left"))
            self._warned_zero_cov = True
        if status_reason:
            return status_reason
        if lost_status and self.min_pos_var is None:
            return "status: recently lost, no covariance baseline yet"
        return None

    def _check_step(self, prev: Pose, cur: Pose, interval: float, now: float) -> bool:
        """Re-anchors at once and returns True when the step from prev to cur is a jump."""
        d = math.sqrt(sum((cur.p[i] - prev.p[i]) ** 2 for i in range(3)))
        dyaw = abs(wrap_pi(yaw_of(cur.R) - yaw_of(prev.R)))
        lim_d = self.cfg.max_speed_mps * interval + self.cfg.jump_margin_m
        lim_y = math.radians(self.cfg.max_yaw_rate_dps * interval + self.cfg.jump_margin_deg)
        if not (d > lim_d or dyaw > lim_y) or self.last_out is None:   # nothing sent yet: nothing to keep
            return False
        reason = (f"pose step {d:.3f} m, {math.degrees(dyaw):.1f} deg in {1000.0 * interval:.0f} ms "
                  f"(limits {lim_d:.3f} m, {math.degrees(lim_y):.1f} deg)")
        self.counters["jumps"] += 1
        self.events.append(("warn", "jump: " + reason))
        self.last_step_jump_rx = now
        self._reanchor(prev, cur, reason)
        return True

    def _status_jump(self, reason: str, now: float) -> None:
        """A jump seen in the status. The status is 1 Hz, so a step jump just before is the same event."""
        if self.last_step_jump_rx is not None and 0.0 <= now - self.last_step_jump_rx <= self.cfg.status_jump_window_s:
            self.events.append(("info", f"{reason}: same event as the step jump "
                                        f"{now - self.last_step_jump_rx:.2f} s ago, no second re-anchor"))
            return
        self.counters["jumps"] += 1
        self.pending_jump = reason
        self.events.append(("warn", "jump: " + reason))

    def _reanchor(self, prev: Pose, cur: Pose, reason: str) -> None:
        """New correction so that correction * cur has the position and yaw of old correction * prev.
        prev is the frame just before cur, held or sent, so motion during a short hold is kept."""
        anchor = self.correction.compose(prev)
        rz = rot_z(wrap_pi(yaw_of(anchor.R) - yaw_of(cur.R)))
        v = mat_vec(rz, cur.p)
        self.correction = Pose(rz, (anchor.p[0] - v[0], anchor.p[1] - v[1], anchor.p[2] - v[2]))
        self.events.append(("info", f"re-anchored after the jump ({reason}): correction yaw "
                                    f"{math.degrees(yaw_of(rz)):.1f} deg, position "
                                    f"({self.correction.p[0]:.2f}, {self.correction.p[1]:.2f}, "
                                    f"{self.correction.p[2]:.2f}) m"))
        self.pending_jump = None


def _pose_finite(p: Pose) -> bool:
    return all(math.isfinite(v) for v in p.p) and all(math.isfinite(v) for row in p.R for v in row)
