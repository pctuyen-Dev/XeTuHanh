"""
Black Box (Hop Den) module for XeTuHanh.

Records synchronized telemetry, lane perception diagnostics, traffic sign detections,
and camera/pipeline frames during runs so issues can be observed and fixed afterwards.
Includes session management, anomaly summarization, and offline lane re-verification.
"""

from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import zipfile

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
BLACKBOX_DIR = BASE_DIR / "blackbox_logs"

DEFAULT_TELEMETRY_HZ = 20.0
DEFAULT_FRAME_HZ = 10.0
MIN_EVENT_FRAME_GAP_S = 0.04
DEFAULT_MAX_SESSIONS = 30
JPEG_QUALITY = 65
MAX_ANOMALIES_IN_META = 400

MOTION_CHECK_INTERVAL_S = 0.10
MOTION_MAD_THRESH = 2.3
MOTION_FRAC_THRESH = 0.015
MOTION_START_HITS = 2
MOTION_STOP_IDLE_S = 1.20

_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def is_safe_name(name):
    return bool(name and _SAFE_SESSION_RE.match(name) and ".." not in name)


def _resize_max_width(img, max_w=480):
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    new_h = max(2, int(round(h * (max_w / float(w)))))
    return cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_jpeg_file(path, img, quality=JPEG_QUALITY):
    if img is None or img.size == 0:
        return False
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def _build_pipeline_strip(lane_thres, lane_warp):
    """Tao anh ghep gon [Threshold | Bird's-Eye Warp] de chuan doan loi lan duong."""
    has_t = lane_thres is not None and lane_thres.size > 0
    has_w = lane_warp is not None and lane_warp.size > 0
    if not has_t and not has_w:
        return None

    cell_w, cell_h = 240, 120
    if has_t:
        t_vis = np.full((lane_thres.shape[0], lane_thres.shape[1], 3), (18, 15, 13), dtype=np.uint8)
        t_vis[lane_thres > 0] = (153, 211, 52)
        t_cell = cv2.resize(t_vis, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    else:
        t_cell = np.full((cell_h, cell_w, 3), (18, 15, 13), dtype=np.uint8)

    if has_w:
        w_bgr = cv2.cvtColor(lane_warp, cv2.COLOR_GRAY2BGR) if lane_warp.ndim == 2 else lane_warp
        w_cell = cv2.resize(w_bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    else:
        w_cell = np.full((cell_h, cell_w, 3), (18, 15, 13), dtype=np.uint8)

    strip = np.hstack([t_cell, w_cell])
    cv2.line(strip, (cell_w, 0), (cell_w, cell_h - 1), (70, 76, 86), 1)
    cv2.putText(strip, "THRES", (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 220, 230), 1, cv2.LINE_AA)
    cv2.putText(strip, "BEV", (cell_w + 6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 220, 230), 1, cv2.LINE_AA)
    return strip


def _extract_lane_summary(lane_info):
    if not isinstance(lane_info, dict):
        return {}
    corner = lane_info.get("corner") or {}
    junction = lane_info.get("junction") or {}
    lighting = lane_info.get("lighting") or {}
    ctrl_dbg = lane_info.get("ctrl_debug") or {}

    compact_ctrl = {}
    for k in ("pp_deg", "stanley_deg", "steer_deg", "kappa_ff", "turn_prog", "ld_used"):
        if k in ctrl_dbg and ctrl_dbg[k] is not None:
            try:
                compact_ctrl[k] = round(float(ctrl_dbg[k]), 3)
            except Exception:
                pass

    return {
        "observed": bool(lane_info.get("observed", False)),
        "state": str(lane_info.get("lane_state", "UNKNOWN")),
        "conf": round(float(lane_info.get("confidence", 0.0)), 3),
        "e_px": round(float(lane_info.get("center_offset_px", 0.0)), 1),
        "e_norm": round(float(lane_info.get("e_norm", 0.0)), 3),
        "heading_deg": round(float(lane_info.get("heading_error_deg", 0.0)), 2),
        "missing_frames": int(lane_info.get("missing_frames", 0)),
        "lighting": str(lighting.get("mode", "NORMAL")),
        "corner": {
            "state": str(corner.get("state", "NONE")),
            "dir": str(corner.get("direction", "UNKNOWN")),
            "cand": str(corner.get("candidate", "UNKNOWN")),
            "intent": str(corner.get("intent_direction", "UNKNOWN")),
            "zone": str(corner.get("distance_zone", "NONE")),
            "turn_now": bool(corner.get("turn_now", False)),
            "confirmed": bool(corner.get("confirmed", False)),
        },
        "junction": {
            "detected": bool(junction.get("detected", False)),
            "type": str(junction.get("type", "NONE")),
            "phase": str(junction.get("phase", "NONE")),
        },
        "ctrl": compact_ctrl,
    }


def _describe_event(ev, row):
    lane = row.get("lane") or {}
    if ev == "LANE_LOST":
        return f"Mat lan duong (state={lane.get('state')}, miss={lane.get('missing_frames', 0)})"
    if ev == "LANE_RECOVERED":
        return f"Tim lai lan duong (state={lane.get('state')})"
    if ev == "ONE_EDGE_LONG":
        return f"Chi thay 1 vach keo dai (err={row.get('err', 0):+.2f})"
    if ev == "WATCHDOG_STOP":
        return "Watchdog phanh khan cap (mat cam/tre > 250ms)"
    if ev == "RECOVERY":
        return f"Lui phuc hoi K-turn ({row.get('cmd')}: {row.get('sign_action', '')})"
    if ev == "SHARP_TURN":
        return f"Cua gat {row.get('cmd')} (err={row.get('err', 0):+.2f})"
    if ev == "HIGH_STEER_FLIP":
        return f"Dao lai dot ngot (err={row.get('err', 0):+.2f})"
    if ev == "MANUAL_OVERRIDE":
        return "Chuyen che do lai thu cong (MANUAL)"
    if ev.startswith("SIGN:"):
        return f"Phat hien bien bao {ev[5:]}"
    return ev


class MotionDetector:
    """Phat hien xe co thuc su di chuyen hay dang dung yen de tranh luu video rac."""

    def __init__(
        self,
        check_interval_s=MOTION_CHECK_INTERVAL_S,
        mad_thresh=MOTION_MAD_THRESH,
        frac_thresh=MOTION_FRAC_THRESH,
        start_hits=MOTION_START_HITS,
        stop_idle_s=MOTION_STOP_IDLE_S,
    ):
        self.check_interval_s = float(check_interval_s)
        self.mad_thresh = float(mad_thresh)
        self.frac_thresh = float(frac_thresh)
        self.start_hits = max(1, int(start_hits))
        self.stop_idle_s = float(stop_idle_s)

        self.last_check_t = 0.0
        self._prev_small = None
        self._hits = 0
        self._is_moving = False
        self._last_motion_t = 0.0
        self.last_mad = 0.0
        self.last_frac = 0.0

    @property
    def is_moving(self):
        return self._is_moving

    def reset(self):
        self.last_check_t = 0.0
        self._prev_small = None
        self._hits = 0
        self._is_moving = False
        self._last_motion_t = 0.0
        self.last_mad = 0.0
        self.last_frac = 0.0

    def needs_frame(self, now=None):
        t = time.time() if now is None else float(now)
        return (t - self.last_check_t) >= self.check_interval_s

    def update(self, frame_bgr, speed, cmd, now=None):
        t = time.time() if now is None else float(now)
        if (t - self.last_check_t) < self.check_interval_s:
            return self._is_moving
        self.last_check_t = t

        if frame_bgr is None or frame_bgr.size == 0:
            self._prev_small = None
            self._hits = 0
            if self._is_moving and (t - self._last_motion_t) >= self.stop_idle_s:
                self._is_moving = False
            return self._is_moving

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        small = cv2.resize(gray, (120, 60), interpolation=cv2.INTER_AREA)

        if self._prev_small is None:
            self._prev_small = small
            return self._is_moving

        diff = cv2.absdiff(small, self._prev_small)
        self._prev_small = small
        mad = float(np.mean(diff))
        frac = float(np.mean(diff > 12))
        self.last_mad = round(mad, 2)
        self.last_frac = round(frac, 4)

        visual_move = (mad >= self.mad_thresh and frac >= self.frac_thresh) or (frac >= 0.025)
        cmd_move = (int(speed) != 0) or (str(cmd) != "STOP")
        instant_move = bool(visual_move and cmd_move)

        if instant_move:
            self._last_motion_t = t
            self._hits += 1
            if not self._is_moving and self._hits >= self.start_hits:
                self._is_moving = True
        else:
            self._hits = 0
            if self._is_moving and (t - self._last_motion_t) >= self.stop_idle_s:
                self._is_moving = False

        return self._is_moving


class BlackBoxRecorder:
    """Bo ghi hop den da luong an toan, khong lam tre luong dieu khien."""

    def __init__(
        self,
        base_dir=BLACKBOX_DIR,
        telemetry_hz=DEFAULT_TELEMETRY_HZ,
        frame_hz=DEFAULT_FRAME_HZ,
        max_sessions=DEFAULT_MAX_SESSIONS,
    ):
        self.base_dir = Path(base_dir)
        self.telemetry_period = 1.0 / max(1.0, float(telemetry_hz))
        self.frame_period = 1.0 / max(1.0, float(frame_hz))
        self.max_sessions = max(1, int(max_sessions))

        self._lock = threading.Lock()
        self._active = False
        self._session_id = None
        self._session_dir = None
        self._frames_dir = None
        self._tel_file = None
        self._meta = {}

        self._t0 = 0.0
        self._seq = 0
        self._frame_seq = 0
        self._last_flush_t = 0.0
        self._last_frame_t = 0.0
        self._last_aux_frame_t = 0.0

        self._cmd_counts = {}
        self._event_counts = {}
        self._anomalies = []
        self._fps_lane_sum = 0.0
        self._fps_cnt = 0

        # Trang thai so sanh de phat hien su kien tu dong
        self._prev_observed = True
        self._prev_mode = "AUTO"
        self._prev_cmd = "STOP"
        self._prev_watchdog = False
        self._prev_recovery = False
        self._prev_signs_key = ""
        self._prev_err = 0.0
        self._prev_err_t = 0.0
        self._one_edge_since = None
        self._one_edge_emitted = False

    def is_recording(self):
        with self._lock:
            return self._active

    def get_status(self):
        with self._lock:
            if not self._active:
                return {
                    "recording": False,
                    "session_id": None,
                    "duration_s": 0.0,
                    "ticks": 0,
                    "frames": 0,
                    "anomalies": 0,
                }
            return {
                "recording": True,
                "session_id": self._session_id,
                "duration_s": round(max(0.0, time.time() - self._t0), 1),
                "ticks": self._seq,
                "frames": self._frame_seq,
                "anomalies": len(self._anomalies),
            }

    def start_session(self, config_meta=None):
        with self._lock:
            if self._active:
                return self._session_id

            self.base_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            sid = stamp
            suffix = 1
            while (self.base_dir / sid).exists():
                sid = f"{stamp}_{suffix:02d}"
                suffix += 1

            sdir = self.base_dir / sid
            fdir = sdir / "frames"
            fdir.mkdir(parents=True, exist_ok=True)

            self._session_id = sid
            self._session_dir = sdir
            self._frames_dir = fdir
            self._tel_file = open(sdir / "telemetry.jsonl", "w", encoding="utf-8")

            now = time.time()
            self._t0 = now
            self._seq = 0
            self._frame_seq = 0
            self._last_flush_t = now
            self._last_frame_t = 0.0
            self._last_aux_frame_t = 0.0

            self._cmd_counts = {}
            self._event_counts = {}
            self._anomalies = []
            self._fps_lane_sum = 0.0
            self._fps_cnt = 0

            self._prev_observed = True
            self._prev_mode = str((config_meta or {}).get("drive_mode_initial", "AUTO"))
            self._prev_cmd = "STOP"
            self._prev_watchdog = False
            self._prev_recovery = False
            self._prev_signs_key = ""
            self._prev_err = 0.0
            self._prev_err_t = now
            self._one_edge_since = None
            self._one_edge_emitted = False

            self._meta = {
                "session_id": sid,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "ended_at": None,
                "duration_s": 0.0,
                "status": "RECORDING",
                "notes": "",
                "config": dict(config_meta or {}),
                "summary": {},
            }
            self._write_meta_locked()
            self._active = True
            self._cleanup_old_sessions_locked()
            return sid

    def stop_session(self, reason="stopped"):
        with self._lock:
            if not self._active:
                return None
            sid = self._session_id
            now = time.time()
            duration_s = round(max(0.0, now - self._t0), 2)

            if self._tel_file is not None:
                try:
                    self._tel_file.flush()
                    self._tel_file.close()
                except Exception:
                    pass
                self._tel_file = None

            avg_lane_fps = round(self._fps_lane_sum / self._fps_cnt, 1) if self._fps_cnt > 0 else 0.0
            self._meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
            self._meta["duration_s"] = duration_s
            self._meta["status"] = "DONE"
            self._meta["stop_reason"] = str(reason)
            self._meta["summary"] = {
                "ticks": self._seq,
                "frames": self._frame_seq,
                "cmd_counts": dict(self._cmd_counts),
                "event_counts": dict(self._event_counts),
                "anomalies_count": len(self._anomalies),
                "anomalies": list(self._anomalies),
                "avg_lane_fps": avg_lane_fps,
            }
            self._write_meta_locked()

            self._active = False
            self._session_id = None
            self._session_dir = None
            self._frames_dir = None
            return sid

    def _write_meta_locked(self):
        if self._session_dir is None:
            return
        tmp_path = self._session_dir / "meta.json.tmp"
        dst_path = self._session_dir / "meta.json"
        try:
            tmp_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, dst_path)
        except Exception:
            pass

    def _cleanup_old_sessions_locked(self):
        if not self.base_dir.exists():
            return
        dirs = [
            d for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_") and d.name != self._session_id
        ]
        dirs.sort(key=lambda p: p.name)
        excess = len(dirs) + (1 if self._session_id else 0) - self.max_sessions
        for old in dirs[:max(0, excess)]:
            try:
                shutil.rmtree(old, ignore_errors=True)
            except Exception:
                pass

    def record_step(
        self,
        *,
        running,
        mode,
        cmd,
        speed,
        err,
        servo_deg,
        packet,
        sign_action,
        curve_val,
        lane_info,
        lane_age_ms,
        signs,
        fps_info,
        lane_vis=None,
        raw_lane=None,
        sign_vis=None,
        lane_thres=None,
        lane_warp=None,
        extra_events=None,
    ):
        with self._lock:
            if not self._active or self._tel_file is None:
                return None

            now = time.time()
            rel_t = round(max(0.0, now - self._t0), 3)
            self._seq += 1

            lane_summary = _extract_lane_summary(lane_info)
            observed = bool(lane_summary.get("observed", False))
            lane_state = lane_summary.get("state", "UNKNOWN")
            missing_frames = int(lane_summary.get("missing_frames", 0))

            events = list(extra_events or [])
            if not observed and (self._prev_observed or missing_frames == 3):
                events.append("LANE_LOST")
            elif observed and not self._prev_observed:
                events.append("LANE_RECOVERED")
            self._prev_observed = observed

            if lane_state == "ONE_EDGE":
                if self._one_edge_since is None:
                    self._one_edge_since = now
                    self._one_edge_emitted = False
                elif not self._one_edge_emitted and (now - self._one_edge_since) >= 1.0:
                    events.append("ONE_EDGE_LONG")
                    self._one_edge_emitted = True
            else:
                self._one_edge_since = None
                self._one_edge_emitted = False

            is_watchdog = (sign_action == "WATCHDOG STOP")
            if is_watchdog and not self._prev_watchdog:
                events.append("WATCHDOG_STOP")
            self._prev_watchdog = is_watchdog

            is_recovery = (cmd in ("REV_L", "REV_R")) or any(
                kw in str(sign_action or "")
                for kw in ("CHUAN BI LUI", "MAT VACH", "PHUC HOI", "MAT LAN DUONG")
            )
            if is_recovery and not self._prev_recovery:
                events.append("RECOVERY")
            self._prev_recovery = is_recovery

            if cmd in ("SHARP_L", "SHARP_R", "SIGN_L", "SIGN_R") and cmd != self._prev_cmd:
                events.append("SHARP_TURN")

            if mode == "MANUAL" and self._prev_mode != "MANUAL":
                events.append("MANUAL_OVERRIDE")
            self._prev_mode = mode

            if running and mode == "AUTO" and (now - self._prev_err_t) <= 0.28:
                if abs(float(err) - self._prev_err) >= 0.70:
                    events.append("HIGH_STEER_FLIP")
            self._prev_err = float(err)
            self._prev_err_t = now
            self._prev_cmd = cmd

            signs_list = list(signs or [])
            signs_key = "|".join(s.get("cls", "?") for s in signs_list)
            if signs_key and signs_key != self._prev_signs_key:
                for s in signs_list:
                    events.append(f"SIGN:{s.get('cls', '?')}")
            self._prev_signs_key = signs_key

            # Quyet dinh co luu frame anh dong bo tai thoi diem nay hay khong
            has_urgent_event = any(
                ev in ("LANE_LOST", "WATCHDOG_STOP", "RECOVERY", "SHARP_TURN", "HIGH_STEER_FLIP")
                or ev.startswith("SIGN:")
                for ev in events
            )
            frame_due = (now - self._last_frame_t) >= self.frame_period
            event_frame_ok = has_urgent_event and ((now - self._last_frame_t) >= MIN_EVENT_FRAME_GAP_S)

            frame_tag = None
            saved_types = []
            if (frame_due or event_frame_ok) and self._frames_dir is not None:
                self._frame_seq += 1
                frame_tag = f"{self._frame_seq:06d}"
                self._last_frame_t = now

                if _encode_jpeg_file(self._frames_dir / f"c1_{frame_tag}.jpg", _resize_max_width(lane_vis, 480)):
                    saved_types.append("c1")
                if _encode_jpeg_file(self._frames_dir / f"r1_{frame_tag}.jpg", _resize_max_width(raw_lane, 480)):
                    saved_types.append("r1")

                aux_due = has_urgent_event or bool(signs_list) or ((now - self._last_aux_frame_t) >= (self.frame_period * 2.0))
                if aux_due:
                    self._last_aux_frame_t = now
                    if _encode_jpeg_file(self._frames_dir / f"c2_{frame_tag}.jpg", _resize_max_width(sign_vis, 480)):
                        saved_types.append("c2")
                    strip = _build_pipeline_strip(lane_thres, lane_warp)
                    if strip is not None and _encode_jpeg_file(self._frames_dir / f"wp_{frame_tag}.jpg", strip):
                        saved_types.append("wp")

            row = {
                "seq": self._seq,
                "t": rel_t,
                "ts": round(now, 3),
                "running": bool(running),
                "mode": str(mode),
                "cmd": str(cmd),
                "speed": int(speed),
                "err": round(float(err), 3),
                "servo": int(servo_deg),
                "packet": str(packet),
                "sign_action": str(sign_action or ""),
                "curve": round(float(curve_val), 3),
                "lane_age_ms": round(float(lane_age_ms), 1),
                "lane": lane_summary,
                "signs": signs_list,
                "fps": fps_info,
                "events": events,
            }
            if frame_tag and saved_types:
                row["frame"] = frame_tag
                row["imgs"] = saved_types

            self._cmd_counts[cmd] = self._cmd_counts.get(cmd, 0) + 1
            lane_fps = float((fps_info or {}).get("lane", 0.0))
            if lane_fps > 0:
                self._fps_lane_sum += lane_fps
                self._fps_cnt += 1

            for ev in events:
                self._event_counts[ev] = self._event_counts.get(ev, 0) + 1
                if ev != "LANE_RECOVERED" and len(self._anomalies) < MAX_ANOMALIES_IN_META:
                    self._anomalies.append({
                        "seq": self._seq,
                        "t": rel_t,
                        "type": ev,
                        "desc": _describe_event(ev, row),
                        "frame": frame_tag,
                    })

            self._tel_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if events or (now - self._last_flush_t) >= 0.5:
                self._tel_file.flush()
                self._last_flush_t = now

            return row


# =============================================================================
# HAM DOC / PHAN TICH / QUAN LY PHIEN HOP DEN (CHO WEB UI & CLI)
# =============================================================================

def _reconstruct_meta_from_telemetry(session_dir):
    tel_path = session_dir / "telemetry.jsonl"
    ticks = 0
    cmd_counts = {}
    event_counts = {}
    anomalies = []
    last_t = 0.0
    frame_count = 0
    if tel_path.exists():
        with open(tel_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ticks += 1
                last_t = float(row.get("t", last_t))
                cmd = str(row.get("cmd", "STOP"))
                cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1
                if row.get("frame"):
                    frame_count += 1
                for ev in row.get("events", ()):
                    event_counts[ev] = event_counts.get(ev, 0) + 1
                    if ev != "LANE_RECOVERED" and len(anomalies) < MAX_ANOMALIES_IN_META:
                        anomalies.append({
                            "seq": row.get("seq", ticks),
                            "t": last_t,
                            "type": ev,
                            "desc": _describe_event(ev, row),
                            "frame": row.get("frame"),
                        })
    return {
        "session_id": session_dir.name,
        "started_at": session_dir.name.replace("run_", ""),
        "ended_at": None,
        "duration_s": round(last_t, 2),
        "status": "UNFINALIZED",
        "notes": "",
        "config": {},
        "summary": {
            "ticks": ticks,
            "frames": frame_count,
            "cmd_counts": cmd_counts,
            "event_counts": event_counts,
            "anomalies_count": len(anomalies),
            "anomalies": anomalies,
        },
    }


def list_sessions(base_dir=BLACKBOX_DIR):
    bdir = Path(base_dir)
    if not bdir.exists():
        return []
    sessions = []
    for d in sorted(bdir.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir() or not d.name.startswith("run_"):
            continue
        meta_path = d / "meta.json"
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None
        if not isinstance(meta, dict):
            meta = _reconstruct_meta_from_telemetry(d)
        elif meta.get("status") == "RECORDING" and not (d / "meta.json.tmp").exists():
            # Neu truoc do tat dot ngot ma chua dong session -> bo sung summary tu telemetry
            rebuilt = _reconstruct_meta_from_telemetry(d)
            if rebuilt["summary"]["ticks"] > (meta.get("summary") or {}).get("ticks", 0):
                meta["duration_s"] = rebuilt["duration_s"]
                meta["summary"] = rebuilt["summary"]
        sessions.append(meta)
    return sessions


def load_session(session_id, base_dir=BLACKBOX_DIR):
    if not is_safe_name(session_id):
        raise ValueError("Invalid session id")
    sdir = Path(base_dir) / session_id
    if not sdir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_id}")

    meta_path = sdir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = _reconstruct_meta_from_telemetry(sdir)
    else:
        meta = _reconstruct_meta_from_telemetry(sdir)

    telemetry = []
    tel_path = sdir / "telemetry.jsonl"
    if tel_path.exists():
        with open(tel_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    telemetry.append(json.loads(line))
                except Exception:
                    continue

    # Dam bao moi frame co duong dan anh fallback gan nhat de scrub muot ma
    last_imgs = {}
    for row in telemetry:
        ftag = row.get("frame")
        if ftag:
            imgs = row.get("imgs") or []
            for k in imgs:
                last_imgs[k] = f"{k}_{ftag}.jpg"
        row["resolved_imgs"] = dict(last_imgs)

    anomalies = (meta.get("summary") or {}).get("anomalies")
    if anomalies is None:
        anomalies = _reconstruct_meta_from_telemetry(sdir)["summary"]["anomalies"]

    return {"meta": meta, "telemetry": telemetry, "anomalies": anomalies}


def save_session_note(session_id, note, base_dir=BLACKBOX_DIR):
    if not is_safe_name(session_id):
        raise ValueError("Invalid session id")
    sdir = Path(base_dir) / session_id
    if not sdir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_id}")
    meta_path = sdir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = _reconstruct_meta_from_telemetry(sdir)
    meta["notes"] = str(note or "")
    tmp_path = sdir / "meta.json.tmp"
    tmp_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, meta_path)
    return meta


def delete_session(session_id, active_session_id=None, base_dir=BLACKBOX_DIR):
    if not is_safe_name(session_id):
        raise ValueError("Invalid session id")
    if active_session_id and session_id == active_session_id:
        raise RuntimeError("Cannot delete an active recording session")
    sdir = Path(base_dir) / session_id
    if not sdir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_id}")
    shutil.rmtree(sdir)
    return True


def build_session_zip_bytes(session_id, base_dir=BLACKBOX_DIR):
    if not is_safe_name(session_id):
        raise ValueError("Invalid session id")
    sdir = Path(base_dir) / session_id
    if not sdir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_id}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(sdir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=f"{session_id}/{p.relative_to(sdir).as_posix()}")
    return buf.getvalue()


def recheck_session_lane(session_id, base_dir=BLACKBOX_DIR):
    """Chay lai LaneDetector hien tai tren cac anh goc (r1_*.jpg) cua phien de kiem chung ban fix."""
    p1_dir = str(BASE_DIR / "P1.NhanDienLanDuong")
    if p1_dir not in sys.path:
        sys.path.insert(0, p1_dir)
    from LaneDetectionModule import LaneDetector, PRESET_ROBOT_TOY
    import utils as lane_utils

    data = load_session(session_id, base_dir=base_dir)
    meta = data["meta"]
    roi_cfg = (meta.get("config") or {}).get("roi") or lane_utils.load_roi_config()
    points = lane_utils.roi_cfg_to_points(roi_cfg, wT=480, hT=240)
    hsv_range = PRESET_ROBOT_TOY["hsv_range"]

    detector = LaneDetector(avg_window=6)
    sdir = Path(base_dir) / session_id / "frames"

    checked = 0
    lost_old = 0
    lost_new = 0
    diffs = []
    for row in data["telemetry"]:
        ftag = row.get("frame")
        if not ftag:
            continue
        raw_path = sdir / f"r1_{ftag}.jpg"
        if not raw_path.exists():
            continue
        img = cv2.imread(str(raw_path))
        if img is None:
            continue

        checked += 1
        old_obs = bool((row.get("lane") or {}).get("observed", False))
        old_curve = float(row.get("curve", 0.0))
        if not old_obs:
            lost_old += 1

        new_curve, _, _, _, _, _ = detector.process_frame(
            img, points, hsv_range=hsv_range, use_dual_edge=True, profile="robot",
            fresh_observation=True, restrict_roi=True,
        )
        new_obs = bool(detector.last_detection.get("observed", False))
        new_state = str(detector.last_detection.get("lane_state", "UNKNOWN"))
        if not new_obs:
            lost_new += 1

        d_curve = abs(float(new_curve) - old_curve)
        if old_obs != new_obs or d_curve >= 0.15:
            diffs.append((row["t"], ftag, old_obs, new_obs, old_curve, float(new_curve), new_state))

    print(f"[RECHECK] Session: {session_id} | Frames da kiem tra: {checked}")
    print(f"  - Mat lan (cu): {lost_old} frames | Mat lan (code moi): {lost_new} frames")
    print(f"  - So frame thay doi trang thai / lech curve >= 0.15: {len(diffs)}")
    for t, ftag, o_obs, n_obs, o_c, n_c, n_st in diffs[:20]:
        print(f"    t={t:6.2f}s [{ftag}] obs:{o_obs}->{n_obs} ({n_st}) | curve:{o_c:+.2f}->{n_c:+.2f}")


def main():
    args = sys.argv[1:]
    if "--list" in args:
        sessions = list_sessions()
        if not sessions:
            print("[BLACKBOX] Chua co phien ghi nao trong blackbox_logs/.")
            return
        for s in sessions:
            sm = s.get("summary") or {}
            print(
                f"{s['session_id']} | {s.get('duration_s', 0):5.1f}s | "
                f"ticks={sm.get('ticks', 0)} frames={sm.get('frames', 0)} "
                f"anomalies={sm.get('anomalies_count', 0)}"
            )
        return

    if "--recheck" in args:
        idx = args.index("--recheck")
        if idx + 1 >= len(args):
            print("Usage: python BlackBoxModule.py --recheck <session_id>")
            return
        recheck_session_lane(args[idx + 1])
        return

    if "--inspect" in args:
        idx = args.index("--inspect")
        if idx + 1 >= len(args):
            print("Usage: python BlackBoxModule.py --inspect <session_id>")
            return
        data = load_session(args[idx + 1])
        meta = data["meta"]
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    print("BlackBoxModule CLI:")
    print("  python BlackBoxModule.py --list")
    print("  python BlackBoxModule.py --inspect <session_id>")
    print("  python BlackBoxModule.py --recheck <session_id>")


if __name__ == "__main__":
    main()
