import importlib
import os
import subprocess
import sys
import threading
import time

# BUOC 1: TU DONG KIEM TRA & CAI DAT THU VIEN THIEU TREN RASPBERRY PI
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("serial", "pyserial"),
]


def ensure_dependencies():
    """Kiem tra cac thu vien cot loi; neu thieu se tu dong cai dat qua pip."""
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"[AUTO-FIX] Phat hien thieu thu vien '{module_name}'. Dang tu dong cai dat '{pip_name}'...")
            installed = False
            for extra_args in ([], ["--break-system-packages"]):
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", pip_name] + extra_args
                    )
                    installed = True
                    print(f"[AUTO-FIX] Da cai dat thanh cong '{pip_name}'!")
                    break
                except Exception:
                    continue
            if not installed:
                print(f"[ERROR] Khong the tu dong cai '{pip_name}'. Hay chay: sudo apt install python3-{module_name}")


ensure_dependencies()

import cv2
import numpy as np

# Them duong dan thu muc goc, P1 va P2 de chay duoc tu bat ky thu muc nao
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
P1_DIR = os.path.join(BASE_DIR, "P1.NhanDienLanDuong")
P2_DIR = os.path.join(BASE_DIR, "P2.NhanDienBienBao")
for d in (BASE_DIR, P1_DIR, P2_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import cameraModule
from BlackBoxModule import BlackBoxRecorder, MotionDetector
from LaneDetectionModule import LaneDetector, PRESET_ROBOT_TOY
import utils as lane_utils
from MotorModule import Esp32Controller, decide_pi_to_esp32_command, resolve_sign_directives
from RobotUI import RobotDashboardUI

try:
    from TrafficSignDetectionModule import detect_traffic_signs, load_sign_model, SignTracker, SIGN_INFO
    SIGN_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Khong the nap module bien bao ({e}). Xe se chi chay bam lan duong.")
    SIGN_AVAILABLE = False
    SIGN_INFO = {}


DEFAULT_ROI_CFG = {"w_top": 123, "h_top": 105, "w_bot": 5, "h_bot": 222}


# BUOC 2: BO NHO DUNG CHUNG AN TOAN DA LUONG (2 CAMERA DOC LAP)
class RobotSharedState:
    def __init__(self, shared_cam=False):
        self.running = True

        # 1. Du lieu 2 Camera (Cam 1: Lan duong | Cam 2: Bien bao YOLO11n)
        self.cam_lock = threading.Lock()
        self.lane_latest_frame = None
        self.lane_frame_id = 0
        self.lane_frame_time = 0.0
        self.lane_cam_ok = False
        self.lane_cam_fps = 0.0

        self.sign_latest_frame = None
        self.sign_frame_id = 0
        self.sign_frame_time = 0.0
        self.sign_cam_ok = False
        self.sign_cam_fps = 0.0

        self.shared_cam = bool(shared_cam)
        self.cam_fov_pct = 100  # Mac dinh 100% = Full-FOV cua camera
        lane_utils.DEFAULT_ROI_CFG.update(DEFAULT_ROI_CFG)
        self.roi_cfg = lane_utils.load_roi_config()

        # 2. Du lieu Lan duong (Luong T2 -> Luong T4 & GUI)
        self.lane_lock = threading.Lock()
        self.curve_val = 0.0
        self.lane_observed = False
        self.lane_state = "UNKNOWN"
        self.corner_info = {}
        self.lighting_mode = "NORMAL"
        self.lane_info = {}
        self.lane_time = 0.0
        self.lane_fps = 0.0
        self.debug_lane_img = None
        self.lane_thres_img = None
        self.lane_warp_img = None
        self.lane_hist_img = None

        # 3. Du lieu Bien bao YOLO11n (Luong T3 -> Luong T4 & GUI)
        self.sign_lock = threading.Lock()
        self.detected_signs = []
        self.sign_debug = {}
        self.sign_time = 0.0
        self.sign_fps = 0.0

        # 4. Trang thai Dieu khien UART (Luong T4 -> Terminal/GUI)
        self.ctrl_lock = threading.Lock()
        self.last_packet = "<0,+0.00,STOP>"
        self.last_cmd = "STOP"
        self.last_speed = 0
        self.last_err = 0.0
        self.last_sign_action = ""
        self.esp32_connected = False
        self.esp32_port = ""

        # 5. Trang thai chay tu hanh (da tat Web Dieu Khien)
        self.car_running = True
        self.drive_mode = "AUTO"
        self.max_speed = 100

        # 6. Trang thai Hop den (Black Box)
        self.blackbox_auto = "--no-blackbox" not in sys.argv
        self.blackbox_manual = "--blackbox-rec" in sys.argv
        self.blackbox_status = {
            "recording": False,
            "session_id": None,
            "duration_s": 0.0,
            "ticks": 0,
            "frames": 0,
            "anomalies": 0,
        }


def can_use_display():
    """Tu dong kiem tra co man hinh (GUI) hay dang chay SSH khong man hinh (Headless)."""
    if "--headless" in sys.argv:
        return False
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _parse_int_arg(flag_names):
    for flag in flag_names:
        if flag in sys.argv:
            try:
                ci = sys.argv.index(flag)
                if ci + 1 < len(sys.argv):
                    return int(sys.argv[ci + 1])
            except Exception:
                pass
    return None


def thread_1a_lane_camera(state: RobotSharedState, cam_rig: cameraModule.DualCameraRig):
    """Luong doc Camera 1 (Lan duong). Neu may chi co 1 cam, tu dong chia se frame cho Cam 2."""
    fps_cnt = 0
    t0 = time.time()

    while state.running:
        with state.cam_lock:
            fov_pct = state.cam_fov_pct
        img = cam_rig.read_lane_frame(fov_pct=fov_pct)
        now = time.time()
        if img is not None:
            fps_cnt += 1
            cam_fps = None
            if fps_cnt >= 15:
                dt = now - t0
                cam_fps = fps_cnt / dt if dt > 0 else 0.0
                t0 = now
                fps_cnt = 0

            with state.cam_lock:
                state.lane_latest_frame = img
                state.lane_frame_id += 1
                state.lane_frame_time = now
                state.lane_cam_ok = True
                if cam_fps is not None:
                    state.lane_cam_fps = cam_fps
                state.shared_cam = cam_rig.shared_mode
                if cam_rig.shared_mode:
                    state.sign_latest_frame = img
                    state.sign_frame_id += 1
                    state.sign_frame_time = now
                    state.sign_cam_ok = True
                    if cam_fps is not None:
                        state.sign_cam_fps = cam_fps
            time.sleep(0.002)
        else:
            with state.cam_lock:
                state.lane_cam_ok = False
            time.sleep(0.03)


def thread_1b_sign_camera(state: RobotSharedState, cam_rig: cameraModule.DualCameraRig):
    """Luong doc Camera 2 (Bien bao YOLO11n) doc lap voi Camera 1."""
    fps_cnt = 0
    t0 = time.time()

    while state.running:
        if cam_rig.shared_mode:
            time.sleep(0.05)
            continue

        with state.cam_lock:
            fov_pct = state.cam_fov_pct
        img = cam_rig.read_sign_frame(fov_pct=fov_pct)
        now = time.time()
        if img is not None:
            fps_cnt += 1
            cam_fps = None
            if fps_cnt >= 15:
                dt = now - t0
                cam_fps = fps_cnt / dt if dt > 0 else 0.0
                t0 = now
                fps_cnt = 0

            with state.cam_lock:
                state.sign_latest_frame = img
                state.sign_frame_id += 1
                state.sign_frame_time = now
                state.sign_cam_ok = True
                if cam_fps is not None:
                    state.sign_cam_fps = cam_fps
            time.sleep(0.002)
        else:
            with state.cam_lock:
                state.sign_cam_ok = False
            time.sleep(0.03)


def thread_2_lane_perception(state: RobotSharedState, lane_size=(480, 240)):
    wT, hT = lane_size
    detector = LaneDetector(avg_window=6)
    hsv_range = PRESET_ROBOT_TOY["hsv_range"]

    last_processed_id = -1
    fps_cnt = 0
    t0 = time.time()
    latched_intent = "AUTO"
    latched_intent_time = 0.0

    while state.running:
        with state.cam_lock:
            fid = state.lane_frame_id
            frame = state.lane_latest_frame.copy() if (state.lane_latest_frame is not None and fid != last_processed_id) else None
            roi_cfg = dict(state.roi_cfg)

        if frame is None:
            time.sleep(0.004)
            continue

        points = lane_utils.roi_cfg_to_points(roi_cfg, wT=wT, hT=hT)
        last_processed_id = fid

        now_j = time.time()
        with state.sign_lock:
            sign_fresh = (state.sign_time > 0.0) and ((now_j - state.sign_time) <= 0.50)
            active_signs = list(state.detected_signs) if sign_fresh else []

        if active_signs:
            directives = resolve_sign_directives(active_signs, SIGN_INFO)
            junction_intent = directives["junction_intent"]
            if junction_intent in ("STRAIGHT", "RIGHT", "LEFT") or directives["forbid_straight"]:
                latched_intent = junction_intent
                latched_intent_time = now_j
            else:
                latched_intent = "AUTO"
                latched_intent_time = 0.0
        else:
            junction_intent = "AUTO"
            if latched_intent_time > 0.0 and (now_j - latched_intent_time) <= 2.60:
                last_det = detector.last_detection or {}
                j_cur = last_det.get("junction") or {}
                c_cur = last_det.get("corner") or {}
                exited_to_straight = (
                    (now_j - latched_intent_time) >= 1.20
                    and (not j_cur.get("detected", False))
                    and (not c_cur.get("confirmed", False))
                    and last_det.get("lane_state") == "TWO_EDGES"
                    and abs(float(last_det.get("center_offset_norm", 0.0))) < 0.25
                )
                if exited_to_straight:
                    latched_intent = "AUTO"
                    latched_intent_time = 0.0
                else:
                    junction_intent = latched_intent
            else:
                latched_intent = "AUTO"
                latched_intent_time = 0.0

        try:
            curve_val, _, img_thres, img_warp_vis, img_hist_combined, img_result = detector.process_frame(
                frame, points, hsv_range=hsv_range, use_dual_edge=True, profile="robot",
                fresh_observation=True, restrict_roi=True, junction_intent=junction_intent
            )
            det_info = dict(detector.last_detection)
            now = time.time()
            fps_cnt += 1
            lane_fps = None
            if fps_cnt >= 15:
                dt = now - t0
                lane_fps = fps_cnt / dt if dt > 0 else 0.0
                t0 = now
                fps_cnt = 0

            with state.lane_lock:
                state.curve_val = float(curve_val)
                state.lane_observed = bool(det_info.get("observed", False))
                state.lane_state = str(det_info.get("lane_state", "UNKNOWN"))
                state.corner_info = dict(det_info.get("corner", {}))
                state.lighting_mode = str(det_info.get("lighting", {}).get("mode", "NORMAL"))
                state.lane_info = det_info
                state.lane_time = now
                state.debug_lane_img = img_result
                state.lane_thres_img = img_thres
                state.lane_warp_img = img_warp_vis
                state.lane_hist_img = img_hist_combined
                if lane_fps is not None:
                    state.lane_fps = lane_fps
        except Exception as e:
            print(f"[WARNING] Loi xu ly Luong 2 - Lan duong: {e}")
            time.sleep(0.01)


def thread_3_sign_perception(state: RobotSharedState, sign_model, sign_tracker):
    if sign_model is None:
        return

    last_processed_id = -1
    fps_cnt = 0
    t0 = time.time()
    target_period = 1.0 / 20.0

    while state.running:
        t_start = time.time()
        with state.cam_lock:
            fid = state.sign_frame_id
            frame = state.sign_latest_frame.copy() if (state.sign_latest_frame is not None and fid != last_processed_id) else None

        if frame is None:
            time.sleep(0.01)
            continue

        last_processed_id = fid
        try:
            dets, sign_dbg = detect_traffic_signs(
                frame, model=sign_model, tracker=sign_tracker, conf_thresh=0.50,
                return_scores=True, return_debug=True
            )
            now = time.time()
            fps_cnt += 1
            sign_fps = None
            if fps_cnt >= 10:
                dt = now - t0
                sign_fps = fps_cnt / dt if dt > 0 else 0.0
                t0 = now
                fps_cnt = 0

            with state.sign_lock:
                state.detected_signs = list(dets)
                state.sign_debug = sign_dbg if isinstance(sign_dbg, dict) else {}
                state.sign_time = now
                if sign_fps is not None:
                    state.sign_fps = sign_fps
        except Exception as e:
            print(f"[WARNING] Loi xu ly Luong 3 - Bien bao YOLO11n: {e}")

        elapsed = time.time() - t_start
        if elapsed < target_period:
            time.sleep(target_period - elapsed)


def thread_4_control_uart(state: RobotSharedState, esp32: Esp32Controller):
    control_period = 0.020  # 50 Hz (20ms)
    lane_watchdog_timeout = 0.250  # 250ms
    sign_watchdog_timeout = 0.500  # 500ms

    while state.running:
        loop_start = time.time()

        with state.cam_lock:
            lane_cam_ok = state.lane_cam_ok

        with state.lane_lock:
            curve_val = state.curve_val
            lane_info = dict(state.lane_info) if state.lane_info else {}
            lane_age = loop_start - state.lane_time if state.lane_time > 0 else 999.0

        with state.sign_lock:
            sign_age = loop_start - state.sign_time if state.sign_time > 0 else 999.0
            if sign_age <= sign_watchdog_timeout:
                active_signs = list(state.detected_signs)
            else:
                active_signs = []
                state.detected_signs = []

        esp_connected = bool(esp32.ser is not None and esp32.ser.is_open)
        esp_port = str(esp32.port or "")

        with state.ctrl_lock:
            max_spd = state.max_speed

        # WATCHDOG AN TOAN: Mat Cam Lan duong hoac du lieu lan qua han > 250ms -> PHANH XE NGAY
        if (not lane_cam_ok) or (lane_age > lane_watchdog_timeout):
            packet = esp32.stop()
            with state.ctrl_lock:
                state.last_packet = packet
                state.last_cmd = "STOP"
                state.last_speed = 0
                state.last_err = 0.0
                state.last_sign_action = "WATCHDOG STOP"
                state.esp32_connected = esp_connected
                state.esp32_port = esp_port
        else:
            speed_pct, err, cmd, sign_action = decide_pi_to_esp32_command(
                curve_val, active_signs, SIGN_INFO, lane_info=lane_info
            )
            if speed_pct > 0:
                speed_pct = min(speed_pct, max_spd)
            packet = esp32.send_command(speed_pct, err, cmd)
            with state.ctrl_lock:
                state.last_packet = packet
                state.last_cmd = cmd
                state.last_speed = speed_pct
                state.last_err = err
                state.last_sign_action = sign_action
                state.esp32_connected = esp_connected
                state.esp32_port = esp_port

        elapsed = time.time() - loop_start
        if elapsed < control_period:
            time.sleep(control_period - elapsed)


def _draw_sign_boxes(frame, signs):
    if frame is None or not signs:
        return frame
    for cls_name, x, y, w, h, score in signs:
        info = SIGN_INFO.get(cls_name, {"vn": cls_name, "color": (0, 215, 255)})
        cv2.rectangle(frame, (x, y), (x + w, y + h), info["color"], 2, cv2.LINE_AA)
        cv2.putText(
            frame, f"{info['vn']} ({score:.0%})", (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, info["color"], 1, cv2.LINE_AA
        )
    return frame


def thread_6_blackbox(state: RobotSharedState, cam_rig: cameraModule.DualCameraRig, recorder: BlackBoxRecorder):
    """Luong ghi hop den (Black Box) o che do nen 20 Hz, chi luu video khi xe thuc su di chuyen."""
    period = 0.05  # 20 Hz
    motion_det = MotionDetector()

    try:
        while state.running:
            t0 = time.time()

            with state.ctrl_lock:
                car_running = state.car_running
                drive_mode = state.drive_mode
                max_speed = state.max_speed
                bb_auto = state.blackbox_auto
                bb_manual = state.blackbox_manual
                spd = state.last_speed
                err = state.last_err
                cmd = state.last_cmd
                pkt = state.last_packet
                sign_action = state.last_sign_action

            raw_lane_for_motion = None
            if bb_auto and car_running and (not bb_manual) and motion_det.needs_frame(t0):
                with state.cam_lock:
                    if state.lane_latest_frame is not None:
                        raw_lane_for_motion = state.lane_latest_frame.copy()
                moving = motion_det.update(raw_lane_for_motion, spd, cmd, now=t0)
            elif not (bb_auto and car_running):
                motion_det.reset()
                moving = False
            else:
                moving = motion_det.is_moving

            should_rec = bool(bb_manual) or bool(bb_auto and car_running and moving)

            if should_rec and not recorder.is_recording():
                with state.cam_lock:
                    fov_pct = state.cam_fov_pct
                    roi_cfg = dict(state.roi_cfg)
                    shared_cam = state.shared_cam
                sid = recorder.start_session({
                    "trigger": "MANUAL_REC" if bb_manual else "AUTO_MOTION",
                    "drive_mode_initial": drive_mode,
                    "max_speed": int(max_speed),
                    "fov_pct": int(fov_pct),
                    "roi": roi_cfg,
                    "shared_cam": bool(shared_cam),
                    "cam1_src": cam_rig.get_lane_label(),
                    "cam2_src": cam_rig.get_sign_label(),
                })
                print(f"[BLACKBOX REC] Xe di chuyen -> Bat dau ghi hop den: {sid}")
            elif (not should_rec) and recorder.is_recording():
                stop_reason = "no_motion" if (bb_auto and car_running and not bb_manual) else "car_stopped"
                sid = recorder.stop_session(reason=stop_reason)
                print(f"[BLACKBOX SAVED] Xe dung di chuyen -> Da luu phien hop den: {sid}")

            if recorder.is_recording():
                with state.cam_lock:
                    raw_lane = state.lane_latest_frame.copy() if state.lane_latest_frame is not None else None
                    raw_sign = state.sign_latest_frame.copy() if state.sign_latest_frame is not None else None
                    c1_fps = round(float(state.lane_cam_fps), 1)
                    c2_fps = round(float(state.sign_cam_fps), 1)
                with state.lane_lock:
                    lane_vis = state.debug_lane_img.copy() if state.debug_lane_img is not None else raw_lane
                    lane_thres = state.lane_thres_img.copy() if state.lane_thres_img is not None else None
                    lane_warp = state.lane_warp_img.copy() if state.lane_warp_img is not None else None
                    curve_val = float(state.curve_val)
                    lane_info = dict(state.lane_info) if state.lane_info else {}
                    lane_age_ms = (t0 - state.lane_time) * 1000.0 if state.lane_time > 0 else 999.0
                    l_fps = round(float(state.lane_fps), 1)
                with state.sign_lock:
                    signs_raw = list(state.detected_signs)
                    s_fps = round(float(state.sign_fps), 1)

                sign_vis = _draw_sign_boxes(raw_sign, signs_raw)
                signs_payload = [
                    {
                        "name": SIGN_INFO.get(c, {}).get("vn", c),
                        "cls": c,
                        "conf": round(float(sc), 2),
                    }
                    for c, _, _, _, _, sc in signs_raw
                ]
                servo_deg = int(np.clip(round(90 + err * 60), 45, 135))

                recorder.record_step(
                    running=car_running,
                    mode=drive_mode,
                    cmd=cmd,
                    speed=spd,
                    err=err,
                    servo_deg=servo_deg,
                    packet=pkt,
                    sign_action=sign_action,
                    curve_val=curve_val,
                    lane_info=lane_info,
                    lane_age_ms=lane_age_ms,
                    signs=signs_payload,
                    fps_info={"cam1": c1_fps, "cam2": c2_fps, "lane": l_fps, "sign": s_fps},
                    lane_vis=lane_vis,
                    raw_lane=raw_lane,
                    sign_vis=sign_vis,
                    lane_thres=lane_thres,
                    lane_warp=lane_warp,
                )

            bb_stat = recorder.get_status()
            with state.ctrl_lock:
                state.blackbox_status = bb_stat

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        if recorder.is_recording():
            sid = recorder.stop_session(reason="shutdown")
            print(f"[BLACKBOX SAVED] Da dong & luu phien hop den: {sid}")


def run_robot():
    print("=" * 76)
    print("   KHOI DONG HE THONG XE TU HANH — 2 CAMERA DOC LAP (LAN DUONG + YOLO11n)")
    print("=" * 76)

    esp32 = Esp32Controller(baudrate=115200)
    sign_model = None
    sign_tracker = None
    if SIGN_AVAILABLE:
        try:
            sign_model = load_sign_model()
            sign_tracker = SignTracker(history_size=5, min_hits=2, max_age=3)
            print(f"[OK] Da nap thanh cong Model YOLO11n ({len(sign_model.class_names)} Bien bao)!")
        except Exception as e:
            print(f"[WARNING] Khong nap duoc model YOLO11n ({e}). Tiep tuc chay bam lan.")

    lane_cam_idx = _parse_int_arg(("--lane-cam", "--cam"))
    if lane_cam_idx is None:
        lane_cam_idx = 1
    sign_cam_idx = _parse_int_arg(("--sign-cam",))
    if sign_cam_idx is None:
        sign_cam_idx = lane_cam_idx
    force_shared_cam = "--dual-cam" not in sys.argv

    lane_size = (480, 240)
    sign_size = (480, 320)
    cam_rig = cameraModule.DualCameraRig()
    while not cam_rig.boot(
        lane_size=lane_size,
        sign_size=sign_size,
        lane_idx=lane_cam_idx,
        sign_idx=sign_cam_idx,
        max_retries=5,
        retry_delay=1.5,
        force_shared=force_shared_cam,
    ):
        esp32.stop()
        print("[AUTO-FIX] Dang cho Camera san sang (kiem tra cap CSI / USB)... Thu lai sau 3 giay.")
        time.sleep(3.0)

    display_mode = 1 if can_use_display() else 0
    if display_mode == 0:
        print("[INFO] Che do Headless -> Tat cua so GUI de toi uu FPS.")
    else:
        print("[INFO] Giao dien Quan sat 2 Camera: BAT ([C] Cam Lan | [V] Cam Bien | [X] Hoan doi | [D] Che do | [Q] Thoat).")

    state = RobotSharedState(shared_cam=cam_rig.shared_mode)
    recorder = BlackBoxRecorder()

    threads = [
        threading.Thread(target=thread_1a_lane_camera, args=(state, cam_rig), name="T1A-LaneCam", daemon=True),
        threading.Thread(target=thread_1b_sign_camera, args=(state, cam_rig), name="T1B-SignCam", daemon=True),
        threading.Thread(target=thread_2_lane_perception, args=(state, lane_size), name="T2-Lane", daemon=True),
        threading.Thread(target=thread_3_sign_perception, args=(state, sign_model, sign_tracker), name="T3-Sign", daemon=True),
        threading.Thread(target=thread_4_control_uart, args=(state, esp32), name="T4-Control", daemon=True),
        threading.Thread(target=thread_6_blackbox, args=(state, cam_rig, recorder), name="T6-BlackBox", daemon=True),
    ]
    for t in threads:
        t.start()
    print(f"[THREADS OK] Da khoi chay {len(threads)} luong song song: {[t.name for t in threads]}")

    last_log_time = time.time()
    ui = RobotDashboardUI() if display_mode != 0 else None
    if ui is not None and not ui.available:
        display_mode = 0

    try:
        while state.running:
            now = time.time()

            if now - last_log_time >= 0.50:
                last_log_time = now
                with state.cam_lock:
                    lc_fps = state.lane_cam_fps
                    sc_fps = state.sign_cam_fps
                with state.lane_lock:
                    l_fps = state.lane_fps
                    l_state = state.lane_state
                    light_mode = state.lighting_mode
                with state.sign_lock:
                    s_fps = state.sign_fps
                with state.ctrl_lock:
                    err = state.last_err
                    pkt = state.last_packet
                    s_act = state.last_sign_action or "Khong"

                print(
                    f"[DUAL-CAM] C1:{lc_fps:4.1f}fps C2:{sc_fps:4.1f}fps | Lane:{l_fps:4.1f}fps ({l_state:<9}, {light_mode:<6}) "
                    f"| YOLO:{s_fps:4.1f}fps ({s_act:<16}) | Lech:{err:+.2f} | UART->{pkt}"
                )

            if ui is not None and ui.available:
                if not ui.render_and_poll(state, cam_rig, SIGN_INFO):
                    break
            else:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Nhan lenh dung tu nguoi dung (Ctrl+C).")
    finally:
        state.running = False
        for t in threads:
            t.join(timeout=0.5)
        print("[SAFE STOP] Dang phanh xe va giai phong 2 Camera...")
        recorder.stop_session(reason="shutdown")
        esp32.stop()
        cam_rig.release_all()
        if ui is not None:
            ui.close()


if __name__ == "__main__":
    run_robot()
