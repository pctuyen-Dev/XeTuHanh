import time
from time import sleep
import numpy as np

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class Esp32Controller:
    def __init__(self, port=None, baudrate=115200):
        self.ser = None
        self.preferred_port = port
        self.port = port
        self.baudrate = baudrate
        self.last_packet = "<0,+0.00,STOP>"
        self._last_reconnect_try = 0.0
        self._connect(port)

    def _connect(self, preferred_port=None):
        if not SERIAL_AVAILABLE:
            print("[WARNING] Thu vien pyserial chua san sang. Chay o che do mo phong.")
            return False

        self._last_reconnect_try = time.time()
        candidates = []
        if preferred_port:
            candidates.append(preferred_port)
        candidates.extend(["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyAMA0", "/dev/serial0", "/dev/ttyS0"])
        try:
            for p in serial.tools.list_ports.comports():
                if p.device not in candidates:
                    candidates.append(p.device)
        except Exception:
            pass

        for dev in candidates:
            try:
                self.ser = serial.Serial(dev, self.baudrate, timeout=0.05, write_timeout=0.05)
                self.port = dev
                print(f"[ESP32 OK] Da ket noi Raspberry Pi -> ESP32 qua cong: {dev} ({self.baudrate} bps)")
                sleep(0.3)
                return True
            except Exception:
                continue

        print("[WARNING] Chua tim thay cong Serial ESP32. He thong se tu dong thu ket noi lai...")
        return False

    def send_command(self, speed_percent, lane_error, cmd="NORM"):
        spd = max(-100, min(100, int(round(speed_percent))))
        err = max(-1.0, min(1.0, float(lane_error)))
        packet = f"<{spd},{err:+.2f},{cmd}>\n"
        self.last_packet = packet.strip()

        # Neu chua ket noi ESP32, tu dong thu ket noi lai moi 2.5 giay ma khong lam treo xe
        if (self.ser is None or not self.ser.is_open) and SERIAL_AVAILABLE:
            if time.time() - self._last_reconnect_try >= 2.5:
                self._connect(self.preferred_port)

        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.write(packet.encode("ascii"))
            except Exception as e:
                print(f"[WARNING] Mat ket noi UART ESP32 ({e}). Dang dat lai de tu dong ket noi...")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

        return self.last_packet

    def stop(self):
        return self.send_command(0, 0.0, "STOP")


_last_known_turn_dir = "RIGHT"
_last_corner_dir_time = 0.0
_on_lane_since = None
_recent_recovery_times = []
_recovery_cooldown_until = 0.0

# =============================================================================
# CAU HINH TOC DO XE (PERCENT: 0 -> 100%) - TANG TOC VAO CUA THEO YEU CAU
# =============================================================================
SPEED_STRAIGHT = 90      # Toc do di thang NORM: chay rat nhanh (90%)
SPEED_CURVE_MAX = 75     # Toc do khi vao cua nhe (75%) - da tang toc
SPEED_CURVE_MIN = 58     # Toc do khi cua goc vua (58%) - da tang toc
SPEED_FAR_CORNER = 75    # Nhin thay goc cua o xa: van chay nhanh 75%
SPEED_APPROACH = 68      # DI THANG QUA DIEM MU 15CM voi toc do 68% (nhanh, khong bi cham)
SPEED_SHARP_TURN = 60    # Toc do khi be lai cua gat 90 do (60% - da tang toc so voi 45%)
SPEED_ONE_EDGE = 65      # Toc do khi xe bi lech chi con 1 vach (65%)

# =============================================================================
# CO CHE LUI PHUC HOI (K-TURN) KHI MAT VACH TAI GOC CUA:
#   PRE_REV  : Dung banh + xoay servo het co ve huong nguoc cua (vd: cua phai -> xoay trai het co 30 do)
#   REVERSE  : Lui MANH (-100%) dut khoat 1.0s - 1.8s (neu thay lai 2 vach ro rang sau 1.0s -> dung lui som)
#   PAUSE    : Dung ngan (0.15s) truoc khi chuyen sang tien
#   FWD_TURN : Tien + be lai vao cua toi da 0.65s; chi thoat khi thay vach on dinh >= 0.15s
#   Toi da MAX_RECOVERY_TRIES = 2 lan trong 1 khuc cua (chong tinh trang lui-tien-lui-tien nhieu lan)
# =============================================================================
PRE_REV_DURATION_S = 0.22
REV_MIN_DURATION_S = 1.00
REV_MAX_DURATION_S = 1.80
REV_PWM = -100           # LUI MANH TOI DA (-100%)
PAUSE_DURATION_S = 0.15
FWD_TURN_DURATION_S = 0.65
FWD_TURN_PWM = 60        # Tien be lai phuc hoi
MAX_RECOVERY_TRIES = 2
RECOVERY_MISSING_FRAMES_TRIGGER = 5
RECOVERY_COOLDOWN_S = 0.60
RECOVERY_WINDOW_S = 7.0
RECOVERY_OBS_CONFIRM_S = 0.15
ON_LANE_RESET_S = 2.5

_recovery = {
    "phase": "IDLE",
    "t0": 0.0,
    "dir": "RIGHT",
    "tries": 0,
    "obs_t0": None,
    "reacquired_in_rev": False,
}


def _set_recovery_phase(phase, now):
    _recovery["phase"] = phase
    _recovery["t0"] = now
    _recovery["obs_t0"] = None


def reset_recovery(cooldown_until=0.0):
    global _recovery_cooldown_until
    _recovery.update({
        "phase": "IDLE",
        "t0": 0.0,
        "tries": 0,
        "obs_t0": None,
        "reacquired_in_rev": False,
    })
    _recovery_cooldown_until = cooldown_until


def _update_obs_confirm(cond, now, required_s):
    if cond:
        if _recovery["obs_t0"] is None:
            _recovery["obs_t0"] = now
        return (now - _recovery["obs_t0"]) >= required_s
    _recovery["obs_t0"] = None
    return False


def _run_recovery(observed, lane_state, now):
    """Tra ve (speed, err, cmd, text) neu dang trong chu trinh lui phuc hoi, nguoc lai tra ve None."""
    phase = _recovery["phase"]
    rec_dir = _recovery["dir"]
    dt = now - _recovery["t0"]
    dir_vn = "PHAI" if rec_dir == "RIGHT" else "TRAI"
    rev_cmd = "REV_L" if rec_dir == "RIGHT" else "REV_R"
    rev_err = -1.0 if rec_dir == "RIGHT" else +1.0
    rev_wheel_vn = "TRAI" if rec_dir == "RIGHT" else "PHAI"

    if phase == "PRE_REV":
        if dt < PRE_REV_DURATION_S:
            return 0, rev_err, rev_cmd, f"CHUAN BI LUI (CUA {dir_vn}) -> XOAY BANH {rev_wheel_vn} HET CO"
        _set_recovery_phase("REVERSE", now)
        phase, dt = "REVERSE", 0.0

    if phase == "REVERSE":
        reacquired = False
        if dt >= REV_MIN_DURATION_S:
            cond = bool(observed and (lane_state == "TWO_EDGES" or dt >= 1.35))
            reacquired = _update_obs_confirm(cond, now, RECOVERY_OBS_CONFIRM_S)
        if not reacquired and dt < REV_MAX_DURATION_S:
            remain = REV_MAX_DURATION_S - dt
            return (
                REV_PWM,
                rev_err,
                rev_cmd,
                f"MAT VACH (CUA {dir_vn}) [Lan {_recovery['tries']}/{MAX_RECOVERY_TRIES}] -> XOAY {rev_wheel_vn} HET CO + LUI MANH ({remain:.1f}s)",
            )
        _recovery["reacquired_in_rev"] = reacquired
        _set_recovery_phase("PAUSE", now)
        phase, dt = "PAUSE", 0.0

    if phase == "PAUSE":
        if dt < PAUSE_DURATION_S:
            return 0, 0.0, "STOP", "DOI CHIEU: DUNG NGAN TRUOC KHI TIEN"
        if _recovery["reacquired_in_rev"] and observed:
            reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
            return None
        _set_recovery_phase("FWD_TURN", now)
        phase, dt = "FWD_TURN", 0.0

    if phase == "FWD_TURN":
        if _update_obs_confirm(bool(observed), now, RECOVERY_OBS_CONFIRM_S):
            # Da thay lai vach on dinh -> tra quyen dieu khien ve bam lan binh thuong
            reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
            return None
        if dt < FWD_TURN_DURATION_S:
            err = +1.0 if rec_dir == "RIGHT" else -1.0
            cmd = "SHARP_R" if rec_dir == "RIGHT" else "SHARP_L"
            return FWD_TURN_PWM, err, cmd, f"PHUC HOI (CUA {dir_vn}) -> TIEN + SERVO {dir_vn}"

        _recent_recovery_times[:] = [t for t in _recent_recovery_times if (now - t) <= RECOVERY_WINDOW_S]
        if len(_recent_recovery_times) < MAX_RECOVERY_TRIES:
            _recent_recovery_times.append(now)
            _recovery["tries"] += 1
            _recovery["reacquired_in_rev"] = False
            _set_recovery_phase("PRE_REV", now)
            return _run_recovery(observed, lane_state, now)

        _set_recovery_phase("GAVE_UP", now)
        phase = "GAVE_UP"

    if phase == "GAVE_UP":
        if _update_obs_confirm(bool(observed), now, 0.40):
            _recent_recovery_times.clear()
            reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
            return None
        return 0, 0.0, "STOP", f"MAT LAN DUONG: DA LUI MANH {MAX_RECOVERY_TRIES} LAN KHONG THANH CONG -> DUNG"

    return None


def decide_pi_to_esp32_command(curve_val, detected_signs=None, sign_info_map=None, lane_info=None):
    """
    Tong hop tin hieu Lan duong (curve_val, lane_info tu Luong 2) va Bien bao (detected_signs tu Luong 3)
    de tao lenh (speed_pct, error, cmd, sign_override_text) gui xuong ESP32.
    Ho tro:
      - Phan vung khoang cach goc cua (FAR / APPROACH / TURN_NOW)
      - Bo nho huong re bao quat (intent_direction) khi mat 1 vach o goc cua
      - Co che LUI NHE PHUC HOI (K-Turn: REV_R / REV_L -> TIEN) khi mat vach tai khuc cua
    """
    global _last_known_turn_dir, _last_corner_dir_time, _on_lane_since

    # Do nhay lai da duoc can chinh theo PD + goc huong BEV tu LaneDetector
    STEER_GAIN = 1.12
    c = float(np.clip(curve_val * STEER_GAIN, -1.0, 1.0))
    if abs(c) < 0.03:
        c = 0.0

    sign_override_text = ""

    # Xu ly thong tin trang thai lan duong & goc cua vuong 90 do tu Luong 2 (neu co)
    if lane_info:
        corner = lane_info.get("corner") or {}
        corner_dir = corner.get("direction", "UNKNOWN")
        corner_cand = corner.get("candidate", "UNKNOWN")
        intent_dir = corner.get("intent_direction", "UNKNOWN")
        turn_now = bool(corner.get("turn_now", False))
        dist_zone = corner.get("distance_zone", "NONE")
        lane_state = lane_info.get("lane_state", "UNKNOWN")
        observed = lane_info.get("observed", True)
        missing_frames = int(lane_info.get("missing_frames", 0))
        now = time.monotonic()

        # Cap nhat bo nho huong cua gan nhat tu tam nhin bao quat (uu tien thong tin goc cua, khong de nhieu 1 vach ghi de)
        if corner_dir in ("LEFT", "RIGHT"):
            _last_known_turn_dir = corner_dir
            _last_corner_dir_time = now
        elif corner_cand in ("LEFT", "RIGHT"):
            _last_known_turn_dir = corner_cand
            _last_corner_dir_time = now
        elif intent_dir in ("LEFT", "RIGHT"):
            _last_known_turn_dir = intent_dir
            _last_corner_dir_time = now
        elif observed and lane_state == "TWO_EDGES" and abs(c) >= 0.45 and (now - _last_corner_dir_time) >= 3.0:
            _last_known_turn_dir = "RIGHT" if c > 0 else "LEFT"

        # Chi xoa bo nho so lan lui khi xe da chay bam lan on dinh lien tuc >= ON_LANE_RESET_S
        if _recovery["phase"] == "IDLE" and observed:
            if _on_lane_since is None:
                _on_lane_since = now
            elif (now - _on_lane_since) >= ON_LANE_RESET_S:
                _recent_recovery_times.clear()
        else:
            _on_lane_since = None

        # 1. CO CHE LUI PHUC HOI (K-TURN) KHI MAT VACH TAI GOC CUA:
        # Khi duong re phai -> Banh xoay TRAI het co (REV_L, -1.00) va LUI MANH (-100%), toi da 1-2 lan.
        if (
            _recovery["phase"] == "IDLE"
            and not observed
            and missing_frames >= RECOVERY_MISSING_FRAMES_TRIGGER
            and not detected_signs
            and now >= _recovery_cooldown_until
        ):
            _recent_recovery_times[:] = [t for t in _recent_recovery_times if (now - t) <= RECOVERY_WINDOW_S]
            rec_dir = intent_dir if intent_dir in ("LEFT", "RIGHT") else _last_known_turn_dir
            if len(_recent_recovery_times) >= MAX_RECOVERY_TRIES:
                _recovery["dir"] = rec_dir
                _recovery["tries"] = MAX_RECOVERY_TRIES
                _set_recovery_phase("GAVE_UP", now)
            else:
                _recent_recovery_times.append(now)
                _recovery["dir"] = rec_dir
                _recovery["tries"] = 1
                _recovery["reacquired_in_rev"] = False
                _set_recovery_phase("PRE_REV", now)
        if _recovery["phase"] != "IDLE":
            rec_out = _run_recovery(observed, lane_state, now)
            if rec_out is not None:
                return rec_out

        # 2. BE LAI CUA GAT MANH TOI DA KHI XE TOI DIEM RE (turn_now == True)
        if turn_now and corner.get("confirmed") and corner_dir in ("LEFT", "RIGHT"):
            c = -1.0 if corner_dir == "LEFT" else +1.0
        elif turn_now and corner_cand in ("LEFT", "RIGHT") and (corner.get("consecutive_frames", 0) >= 2 or lane_state == "ONE_EDGE"):
            c = -1.0 if corner_cand == "LEFT" else +1.0
        elif turn_now and intent_dir in ("LEFT", "RIGHT") and lane_state == "ONE_EDGE":
            c = -1.0 if intent_dir == "LEFT" else +1.0
        elif dist_zone == "APPROACH" and corner_cand in ("LEFT", "RIGHT"):
            # Cho phep bam tam muot trong khi tien sat khuc cua, tranh khoa cung lam xe giat
            c = float(np.clip(c, -0.40, 0.40))
        elif lane_state == "ONE_EDGE" and not turn_now:
            # Khi xe di lech 1 ben tren duong thang (chi thay 1 vach): dieu chinh em ve giua lan
            c = float(np.clip(c * 1.15, -0.65, 0.65))

        if abs(c) >= 0.55:
            cmd = "SHARP_R" if c > 0 else "SHARP_L"
            throttle_pwm = SPEED_SHARP_TURN
        elif dist_zone == "APPROACH" and corner_cand in ("LEFT", "RIGHT"):
            cmd = "APPROACH"
            throttle_pwm = SPEED_APPROACH
        elif dist_zone == "FAR" and corner_cand in ("LEFT", "RIGHT"):
            cmd = "CURVE" if abs(c) >= 0.12 else "NORM"
            throttle_pwm = SPEED_FAR_CORNER
        elif lane_state == "ONE_EDGE":
            cmd = "CURVE" if abs(c) >= 0.08 else "NORM"
            throttle_pwm = SPEED_ONE_EDGE
        elif abs(c) >= 0.10:
            cmd = "CURVE"
            throttle_pwm = int(round(np.clip((0.82 - 0.28 * abs(c)) * 100, SPEED_CURVE_MIN, SPEED_CURVE_MAX)))
        else:
            cmd = "NORM"
            throttle_pwm = SPEED_STRAIGHT
    else:
        if abs(c) >= 0.55:
            cmd = "SHARP_R" if c > 0 else "SHARP_L"
            throttle_pwm = SPEED_SHARP_TURN
        elif abs(c) >= 0.10:
            cmd = "CURVE"
            throttle_pwm = int(round(np.clip((0.80 - 0.25 * abs(c)) * 100, SPEED_CURVE_MIN, SPEED_CURVE_MAX)))
        else:
            cmd = "NORM"
            throttle_pwm = SPEED_STRAIGHT

    if detected_signs and sign_info_map:
        primary = max(detected_signs, key=lambda d: d[3] * d[4])
        cls_name = primary[0]
        info = sign_info_map.get(cls_name)
        if info:
            sign_override_text = info["action"]
            spd_factor = info["speed_factor"]
            steer_bias = info["steer_bias"]

            if spd_factor == 0.0:
                return 0, 0.0, "STOP", f"BIEN BAO: {sign_override_text}"

            throttle_pwm = int(round(throttle_pwm * spd_factor))
            if abs(steer_bias) > 0.01:
                c = float(np.clip(c + steer_bias, -1.0, 1.0))
            if cls_name == "left":
                cmd = "SIGN_L"
            elif cls_name == "right":
                cmd = "SIGN_R"
            elif spd_factor < 0.75:
                cmd = "SLOW"

    return throttle_pwm, c, cmd, sign_override_text


def simulate_motors(curve_val, detected_signs=None, sign_info_map=None, lane_info=None):
    throttle_pwm, c, cmd, sign_override_text = decide_pi_to_esp32_command(
        curve_val, detected_signs, sign_info_map, lane_info=lane_info
    )
    if cmd == "STOP":
        return sign_override_text or "DUNG KHAN CAP", 0, 90, 0, 0, "<0,+0.00,STOP>"

    uart_packet = f"<{throttle_pwm},{c:+.2f},{cmd}>"
    servo_angle = int(np.clip(round(90 + c * 60), 30, 150))

    if cmd in ("REV_L", "REV_R"):
        left_speed = throttle_pwm
        right_speed = throttle_pwm
    elif cmd in ("SHARP_L", "SIGN_L") or c <= -0.55:
        left_speed = int(round(-0.30 * throttle_pwm))
        right_speed = int(np.clip(round(throttle_pwm * 1.35), 35, 100))
    elif cmd in ("SHARP_R", "SIGN_R") or c >= 0.55:
        left_speed = int(np.clip(round(throttle_pwm * 1.35), 35, 100))
        right_speed = int(round(-0.30 * throttle_pwm))
    else:
        turn_diff = int(round(c * 52.0))
        left_speed = int(np.clip(throttle_pwm + turn_diff, -100, 100))
        right_speed = int(np.clip(throttle_pwm - turn_diff, -100, 100))

    if sign_override_text:
        direction = sign_override_text
    elif cmd == "REV_L":
        direction = f"LUI + XOAY TRAI HET CO (Servo {servo_angle} deg)"
    elif cmd == "REV_R":
        direction = f"LUI + XOAY PHAI HET CO (Servo {servo_angle} deg)"
    elif cmd == "SHARP_R":
        direction = "GOC RE GAT PHAI (4WD Pivot)"
    elif cmd == "SHARP_L":
        direction = "GOC RE GAT TRAI (4WD Pivot)"
    elif cmd == "APPROACH":
        direction = "DI THANG QUA DIEM MU 15CM"
    elif c > 0.06:
        direction = f"GOC CUA PHAI (Servo {servo_angle} deg)"
    elif c < -0.06:
        direction = f"GOC CUA TRAI (Servo {servo_angle} deg)"
    else:
        direction = "DI THANG GIUA LAN (90 deg)"

    return direction, throttle_pwm, servo_angle, left_speed, right_speed, uart_packet


if __name__ == "__main__":
    esp = Esp32Controller()
    print("Gui thu lenh di thang:", esp.send_command(SPEED_STRAIGHT, 0.0, "NORM"))
    sleep(1)
    print("Gui thu lenh cua phai:", esp.send_command(70, 0.32, "CURVE"))
    sleep(1)
    print("Gui thu lenh dung:", esp.stop())
