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
from LaneDetectionModule import LaneDetector, PRESET_ROBOT_TOY
from MotorModule import Esp32Controller, decide_pi_to_esp32_command
from RobotUI import RobotDashboardUI

try:
    from TrafficSignDetectionModule import detect_traffic_signs, load_sign_model, SignTracker, SIGN_INFO
    SIGN_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Khong the nap module bien bao ({e}). Xe se chi chay bam lan duong.")
    SIGN_AVAILABLE = False
    SIGN_INFO = {}

# BUOC 2: BO NHO DUNG CHUNG AN TOAN DA LUONG (RobotSharedState)
class RobotSharedState:
    def __init__(self):
        self.running = True

        # 1. Du lieu Camera (Luong 1 -> Luong 2, 3)
        self.cam_lock = threading.Lock()
        self.latest_frame = None
        self.frame_id = 0
        self.frame_time = 0.0
        self.cam_ok = False
        self.cam_fps = 0.0
        self.cam_fov_pct = 100  # Mac dinh 100% = do rong toi da (Full-FOV) cua webcam

        # 2. Du lieu Lan duong (Luong 2 -> Luong 4 & GUI)
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

        # 3. Du lieu Bien bao (Luong 3 -> Luong 4 & GUI)
        self.sign_lock = threading.Lock()
        self.detected_signs = []
        self.sign_debug = {}
        self.sign_time = 0.0
        self.sign_fps = 0.0

        # 4. Trang thai Dieu khien UART (Luong 4 -> Terminal/GUI)
        self.ctrl_lock = threading.Lock()
        self.last_packet = "<0,+0.00,STOP>"
        self.last_cmd = "STOP"
        self.last_speed = 0
        self.last_err = 0.0
        self.last_sign_action = ""
        self.esp32_connected = False
        self.esp32_port = ""


def can_use_display():
    """Tu dong kiem tra Pi co cam man hinh (GUI) hay dang chay SSH khong man hinh (Headless)."""
    if "--headless" in sys.argv:
        return False
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))



def thread_1_camera_sense(state: RobotSharedState, frame_size=(480, 240)):
    fps_cnt = 0
    t0 = time.time()

    while state.running:
        with state.cam_lock:
            fov_pct = state.cam_fov_pct
        img = cameraModule.getImg(display=False, size=frame_size, fov_pct=fov_pct)
        now = time.time()
        if img is not None:
            fps_cnt += 1
            if fps_cnt >= 15:
                dt = now - t0
                cam_fps = fps_cnt / dt if dt > 0 else 0.0
                t0 = now
                fps_cnt = 0
            else:
                cam_fps = None

            with state.cam_lock:
                state.latest_frame = img
                state.frame_id += 1
                state.frame_time = now
                state.cam_ok = True
                if cam_fps is not None:
                    state.cam_fps = cam_fps
            time.sleep(0.002)
        else:
            with state.cam_lock:
                state.cam_ok = False
            time.sleep(0.03)


def thread_2_lane_perception(state: RobotSharedState, frame_size=(480, 240)):
    wT, hT = frame_size
    detector = LaneDetector(avg_window=6)
    points = np.float32([
        (PRESET_ROBOT_TOY["w_top"], PRESET_ROBOT_TOY["h_top"]),
        (wT - PRESET_ROBOT_TOY["w_top"], PRESET_ROBOT_TOY["h_top"]),
        (PRESET_ROBOT_TOY["w_bot"], PRESET_ROBOT_TOY["h_bot"]),
        (wT - PRESET_ROBOT_TOY["w_bot"], PRESET_ROBOT_TOY["h_bot"]),
    ])
    hsv_range = PRESET_ROBOT_TOY["hsv_range"]

    last_processed_id = -1
    fps_cnt = 0
    t0 = time.time()

    while state.running:
        with state.cam_lock:
            fid = state.frame_id
            frame = state.latest_frame.copy() if (state.latest_frame is not None and fid != last_processed_id) else None

        if frame is None:
            time.sleep(0.004)
            continue

        last_processed_id = fid
        try:
            curve_val, _, img_thres, img_warp_vis, img_hist_combined, img_result = detector.process_frame(
                frame, points, hsv_range=hsv_range, use_dual_edge=True, profile="robot", fresh_observation=True
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
    target_period = 1.0 / 15.0 

    while state.running:
        t_start = time.time()
        with state.cam_lock:
            fid = state.frame_id
            frame = state.latest_frame.copy() if (state.latest_frame is not None and fid != last_processed_id) else None

        if frame is None:
            time.sleep(0.01)
            continue

        last_processed_id = fid
        try:
            dets, sign_dbg = detect_traffic_signs(
                frame, model=sign_model, tracker=sign_tracker, conf_thresh=0.62,
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
            print(f"[WARNING] Loi xu ly - Bien bao: {e}")

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
            cam_ok = state.cam_ok

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

        # KIEM TRA WATCHDOG AN TOAN: Mat camera hoac du lieu lan duong qua han > 250ms -> PHANH XE NGAY
        if (not cam_ok) or (lane_age > lane_watchdog_timeout):
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

def run_robot():
    print("=" * 72)
    print("   KHOI DONG HE THONG XE TU HANH")
    print("=" * 72)
    esp32 = Esp32Controller(baudrate=115200)
    sign_model = None
    sign_tracker = None
    if SIGN_AVAILABLE:
        try:
            sign_model = load_sign_model()
            sign_tracker = SignTracker(history_size=5, min_hits=2, max_age=3)
            print("[OK] Da nap thanh cong Model 15 Bien bao giao thong!")
        except Exception as e:
            print(f"[WARNING] Khong nap duoc file model ONNX ({e}). Tiep tuc chay bam lan.")
    preferred_cam = None
    if "--cam" in sys.argv:
        try:
            ci = sys.argv.index("--cam")
            if ci + 1 < len(sys.argv):
                preferred_cam = int(sys.argv[ci + 1])
        except Exception:
            preferred_cam = None

    frame_size = (480, 240)
    while not cameraModule.boot_camera(size=frame_size, max_retries=5, retry_delay=1.5, preferred_idx=preferred_cam):
        esp32.stop()
        print("[AUTO-FIX] Dang cho Camera san sang (kiem tra cap CSI / USB)... Thu lai sau 3 giay.")
        time.sleep(3.0)

    display_mode = 1 if can_use_display() else 0
    if display_mode == 0:
        print("[INFO] Phat hien che do Headless (Khong man hinh) -> Tat cua so GUI de toi uu FPS.")
    else:
        print("[INFO] Che do hien thi GUI: BAT (Nhan 'q' de dung | 'c' hoac bam nut 'Doi Cam [C]' de chuyen camera).")

    state = RobotSharedState()

    # 4. Khoi chạy 4 Luong song song
    threads = [
        threading.Thread(target=thread_1_camera_sense, args=(state, frame_size), name="T1-Camera", daemon=True),
        threading.Thread(target=thread_2_lane_perception, args=(state, frame_size), name="T2-Lane", daemon=True),
        threading.Thread(target=thread_3_sign_perception, args=(state, sign_model, sign_tracker), name="T3-Sign", daemon=True),
        threading.Thread(target=thread_4_control_uart, args=(state, esp32), name="T4-Control", daemon=True),
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

            # In trang thai dinh ky moi 0.5 giay ra Terminal
            if now - last_log_time >= 0.50:
                last_log_time = now
                with state.cam_lock:
                    c_fps = state.cam_fps
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
                    f"[4-THREADS] Cam:{c_fps:4.1f}fps | Lane:{l_fps:4.1f}fps ({l_state:<9}, Light:{light_mode:<6}) "
                    f"| Sign:{s_fps:4.1f}fps ({s_act:<16}) | Lech:{err:+.2f} | UART->{pkt}"
                )

            # Hien thi cua so GUI qua RobotUI (neu co man hinh)
            if ui is not None and ui.available:
                if not ui.render_and_poll(state, cameraModule, frame_size, SIGN_INFO):
                    break
            else:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Nhan lenh dung tu nguoi dung (Ctrl+C).")
    finally:
        state.running = False
        for t in threads:
            t.join(timeout=0.5)
        print("[SAFE STOP] Dang phanh xe va giai phong Camera...")
        esp32.stop()
        cameraModule.release_camera()
        if ui is not None:
            ui.close()


if __name__ == "__main__":
    run_robot()
