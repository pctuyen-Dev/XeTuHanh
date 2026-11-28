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
# CAU HINH TOC DO XE (PERCENT: 0 -> 100%) - DUY TRI TOC DO >= 90%
# =============================================================================
SPEED_STRAIGHT = 100     # Toc do di thang NORM: toi da (100%)
SPEED_CURVE_MAX = 96     # Toc do khi vao cua nhe
SPEED_CURVE_MIN = 90     # Toc do khi cua goc vua (>= 90%)
SPEED_FAR_CORNER = 92    # Nhin thay goc cua o xa (>= 90%)
SPEED_SHARP_TURN = 90    # Toc do khi be lai cua gat 90 do (>= 90%)
SPEED_ONE_EDGE = 92      # Toc do khi xe chi con 1 vach (>= 90%)

# =============================================================================
# CO CHE LUI PHUC HOI (K-TURN) - LUI TOI DA 2 LAN DE CHINH GOC TOT NHAT:
#   PRE_REV  : Dung banh + xoay servo het co ve huong nguoc cua (vd: cua phai -> xoay trai het co)
#   REVERSE  : Lui MANH (-100%) toi da 2.60s de dua dau xe quay ve dung huong vao lan 2 vach:
#              - Dung lui som khi thay lai 2 vach (TWO_EDGES) va xe da tuong doi thang/can tam
#              - Hoac sau 2.00s neu 1 vach da can tam & thang goc (|e_norm| <= 0.35, |heading| <= 14 deg)
#   PAUSE    : Dung ngan (0.15s) truoc khi chuyen sang tien
#   FWD_TURN : Tien + be lai vao lan toi da 1.25s; thoat som khi thay TWO_EDGES can tam
#              hoac 1 vach da can giua & thang goc.
#              Neu chua vao duoc lan on dinh -> cho phep lui lai lan 2 de chinh góc tốt nhất;
#              sau toi da 2 lan neu van thay vach (observed=True) -> tra ve bam lan tien + cooldown.
# =============================================================================
PRE_REV_DURATION_S = 0.22
REV_MIN_DURATION_S = 1.05
REV_ONE_EDGE_EXIT_S = 2.00
REV_MAX_DURATION_S = 2.60
REV_PWM = -100           # LUI MANH TOI DA (-100%)
PAUSE_DURATION_S = 0.15
FWD_TURN_DURATION_S = 1.25
FWD_TURN_PWM = 90        # Tien phuc hoi duy tri >= 90%
MAX_RECOVERY_TRIES = 2
RECOVERY_MISSING_FRAMES_TRIGGER = 13
OFF_LANE_STUCK_E_THRESH = 0.90
OFF_LANE_STUCK_TRIGGER_S = 2.40
RECOVERY_COOLDOWN_S = 5.50
RECOVERY_WINDOW_S = 14.0
RECOVERY_OBS_CONFIRM_S = 0.15
ON_LANE_RESET_S = 4.0
SHARP_TURN_HOLD_S = 0.45

# Chu trinh dac biet khi thay dong thoi bien CAM DI THANG + RE PHAI o nga 3:
#   PRE_REV  : Xoay banh truoc sang TRAI het co (REV_L)
#   REVERSE  : Vua lui (-100%) vua giu banh trai de xoay mui xe ve huong RE PHAI
#   PAUSE    : Dung ngan doi chieu an toan
#   FWD_TURN : Tien (>= 90%) va be lai RE PHAI vao lan nga 3
SIGN_COMBO_HOLD_S = 0.45
SIGN_COMBO_APPROACH_MEM_S = 2.50
SIGN_COMBO_APPROACH_MIN_H_PX = 42.0
SIGN_COMBO_CONFIRM_S = 0.04
SIGN_COMBO_MIN_BOX_H_PX = 67.0   # Doi xe di thang sat vao nga 3 hon moi bat dau lui quay
SIGN_COMBO_MIN_Y_BOT_PX = 106.0  # Day bien khi ca 2 cung hien dien sat nga 3
SIGN_COMBO_RIGHT_FINAL_H_PX = 68.0
SIGN_COMBO_RIGHT_FINAL_YB_PX = 110.0  # Khi no_straight vua khuat mep trai nhung right da sat ngay dau xe
SIGN_COMBO_REV_MIN_S = 1.15
SIGN_COMBO_REV_MAX_S = 2.55
SIGN_COMBO_FWD_TURN_S = 1.30
SIGN_COMBO_COOLDOWN_S = 8.00
SIGN_COMBO_CLEAR_S = 1.80
SIGN_TURN_HOLD_S = 2.20

# Nguong kich thuoc toi thieu cua bien STOP/NO_ENTRY de dung dung khoang cach (tranh dung qua som khi bien con xa)
STOP_MIN_BOX_H_PX = 60.0
STOP_MIN_Y_BOT_PX = 88.0
STOP_WIDE_MIN_W_PX = 84.0
STOP_WIDE_MIN_YB_PX = 55.0
STOP_HOLD_S = 1.50

_stop_hold_until = 0.0
_straight_pass_hold = {
    "until": 0.0,
    "forbid_right": False,
    "forbid_left": False,
}

_sharp_turn_latch = {
    "dir": None,
    "until": 0.0,
}

_sign_turn_hold = {
    "dir": None,
    "until": 0.0,
}

_off_lane_stuck = {
    "t0": None,
    "min_abs_e": 1.0,
    "dir": None,
}

_sign_combo = {
    "last_no_straight_t": 0.0,
    "last_right_t": 0.0,
    "approach_no_straight_t": 0.0,
    "approach_right_t": 0.0,
    "both_t0": None,
    "cooldown_until": 0.0,
    "needs_clear": False,
    "absent_t0": None,
}

_recovery = {
    "phase": "IDLE",
    "mode": "LANE_LOST",
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


def reset_recovery(cooldown_until=0.0, clear_sign_combo=False):
    global _recovery_cooldown_until, _stop_hold_until
    _recovery.update({
        "phase": "IDLE",
        "mode": "LANE_LOST",
        "t0": 0.0,
        "tries": 0,
        "obs_t0": None,
        "reacquired_in_rev": False,
    })
    _sharp_turn_latch["dir"] = None
    _sharp_turn_latch["until"] = 0.0
    _off_lane_stuck["t0"] = None
    _off_lane_stuck["min_abs_e"] = 1.0
    _off_lane_stuck["dir"] = None
    _recovery_cooldown_until = cooldown_until
    if clear_sign_combo:
        _recent_recovery_times.clear()
        _sign_turn_hold["dir"] = None
        _sign_turn_hold["until"] = 0.0
        _stop_hold_until = 0.0
        _straight_pass_hold.update({"until": 0.0, "forbid_right": False, "forbid_left": False})
        _sign_combo.update({
            "last_no_straight_t": 0.0,
            "last_right_t": 0.0,
            "approach_no_straight_t": 0.0,
            "approach_right_t": 0.0,
            "both_t0": None,
            "cooldown_until": 0.0,
            "needs_clear": False,
            "absent_t0": None,
        })


def _update_obs_confirm(cond, now, required_s):
    if cond:
        if _recovery["obs_t0"] is None:
            _recovery["obs_t0"] = now
        return (now - _recovery["obs_t0"]) >= required_s
    _recovery["obs_t0"] = None
    return False


def _update_no_straight_right_trigger(detected_signs, now):
    """Phat hien khi dong thoi thay bien CAM DI THANG ('no_straight') va RE PHAI ('right')
    tai nga 3 de kich hoat chu trinh vua lui vua quay banh roi tien re phai.
    Ho tro truong hop no_straight vua khuat mep trai trong khi right sat ngay dau xe.
    """
    now_no_straight = False
    now_right = False
    now_right_final = False
    any_approach_ns = False
    any_approach_r = False
    has_stop_or_forbid_right = False

    if detected_signs:
        for det in detected_signs:
            cls_name = det[0]
            box_y = float(det[2]) if len(det) >= 5 else 0.0
            box_h = float(det[4]) if len(det) >= 5 else 0.0
            box_y_bot = box_y + box_h
            is_close = (box_h >= SIGN_COMBO_MIN_BOX_H_PX and box_y_bot >= SIGN_COMBO_MIN_Y_BOT_PX)
            if cls_name in ("stop", "no_entry", "no_right"):
                has_stop_or_forbid_right = True
            elif cls_name == "no_straight":
                if box_h >= SIGN_COMBO_APPROACH_MIN_H_PX:
                    any_approach_ns = True
                if is_close:
                    now_no_straight = True
            elif cls_name == "right":
                if box_h >= SIGN_COMBO_APPROACH_MIN_H_PX:
                    any_approach_r = True
                if is_close:
                    now_right = True
                if box_h >= SIGN_COMBO_RIGHT_FINAL_H_PX and box_y_bot >= SIGN_COMBO_RIGHT_FINAL_YB_PX:
                    now_right_final = True

    if any_approach_ns:
        _sign_combo["approach_no_straight_t"] = now
    if any_approach_r:
        _sign_combo["approach_right_t"] = now
    if now_no_straight:
        _sign_combo["last_no_straight_t"] = now
    if now_right:
        _sign_combo["last_right_t"] = now

    if _sign_combo["needs_clear"]:
        if not any_approach_ns and not any_approach_r:
            if _sign_combo["absent_t0"] is None:
                _sign_combo["absent_t0"] = now
            elif (now - _sign_combo["absent_t0"]) >= SIGN_COMBO_CLEAR_S:
                _sign_combo["needs_clear"] = False
                _sign_combo["absent_t0"] = None
        else:
            _sign_combo["absent_t0"] = None
        _sign_combo["both_t0"] = None
        return False

    if has_stop_or_forbid_right or now < _sign_combo["cooldown_until"]:
        _sign_combo["both_t0"] = None
        return False

    approached_both = (
        (_sign_combo["approach_no_straight_t"] > 0.0)
        and (_sign_combo["approach_right_t"] > 0.0)
        and (now - _sign_combo["approach_no_straight_t"]) <= SIGN_COMBO_APPROACH_MEM_S
        and (now - _sign_combo["approach_right_t"]) <= SIGN_COMBO_APPROACH_MEM_S
    )
    both_close_recent = (
        (_sign_combo["last_no_straight_t"] > 0.0)
        and (_sign_combo["last_right_t"] > 0.0)
        and (now - _sign_combo["last_no_straight_t"]) <= SIGN_COMBO_HOLD_S
        and (now - _sign_combo["last_right_t"]) <= SIGN_COMBO_HOLD_S
    )
    ready_now = both_close_recent or (approached_both and now_right_final)
    if not ready_now:
        _sign_combo["both_t0"] = None
        return False

    if _sign_combo["both_t0"] is None:
        _sign_combo["both_t0"] = now
        return False

    if (now - _sign_combo["both_t0"]) >= SIGN_COMBO_CONFIRM_S:
        _sign_combo["both_t0"] = None
        _sign_combo["needs_clear"] = True
        _sign_combo["absent_t0"] = None
        _sign_combo["cooldown_until"] = now + SIGN_COMBO_COOLDOWN_S
        return True
    return False


def _run_recovery(observed, lane_state, e_norm, heading_deg, now):
    """Tra ve (speed, err, cmd, text) neu dang trong chu trinh lui phuc hoi, nguoc lai tra ve None."""
    phase = _recovery["phase"]
    mode = _recovery.get("mode", "LANE_LOST")
    is_sign_combo = (mode == "SIGN_COMBO_R")
    rec_dir = _recovery["dir"]
    dt = now - _recovery["t0"]
    dir_vn = "PHAI" if rec_dir == "RIGHT" else "TRAI"
    rev_cmd = "REV_L" if rec_dir == "RIGHT" else "REV_R"
    rev_err = -1.0 if rec_dir == "RIGHT" else +1.0
    rev_wheel_vn = "TRAI" if rec_dir == "RIGHT" else "PHAI"
    abs_e = abs(float(e_norm))
    abs_hd = abs(float(heading_deg))

    rev_min_s = SIGN_COMBO_REV_MIN_S if is_sign_combo else REV_MIN_DURATION_S
    rev_max_s = SIGN_COMBO_REV_MAX_S if is_sign_combo else REV_MAX_DURATION_S
    fwd_max_s = SIGN_COMBO_FWD_TURN_S if is_sign_combo else FWD_TURN_DURATION_S

    if phase == "PRE_REV":
        if dt < PRE_REV_DURATION_S:
            msg = (
                "CAM DI THANG + RE PHAI: XOAY BANH TRAI CHUAN BI LUI"
                if is_sign_combo
                else f"CHUAN BI LUI (CUA {dir_vn}) -> XOAY BANH {rev_wheel_vn} HET CO"
            )
            return 0, rev_err, rev_cmd, msg
        _set_recovery_phase("REVERSE", now)
        phase, dt = "REVERSE", 0.0

    if phase == "REVERSE":
        two_edges_aligned = bool(
            observed and lane_state == "TWO_EDGES" and abs_e <= 0.50 and abs_hd <= 20.0
        )
        one_edge_aligned = bool(
            (not is_sign_combo) and observed and dt >= REV_ONE_EDGE_EXIT_S and abs_e <= 0.35 and abs_hd <= 14.0
        )
        reacquired = False
        if dt >= rev_min_s:
            reacquired = _update_obs_confirm(two_edges_aligned or one_edge_aligned, now, RECOVERY_OBS_CONFIRM_S)
        if not reacquired and dt < rev_max_s:
            remain = rev_max_s - dt
            msg = (
                f"NGA 3 (CAM THANG + RE PHAI) -> VUA LUI VUA XOAY TRAI ({remain:.1f}s)"
                if is_sign_combo
                else f"CHINH GOC (CUA {dir_vn}) [Lan {_recovery['tries']}/{MAX_RECOVERY_TRIES}] -> XOAY {rev_wheel_vn} + LUI ({remain:.1f}s)"
            )
            return REV_PWM, rev_err, rev_cmd, msg
        _recovery["reacquired_in_rev"] = bool(reacquired and two_edges_aligned)
        _set_recovery_phase("PAUSE", now)
        phase, dt = "PAUSE", 0.0

    if phase == "PAUSE":
        if dt < PAUSE_DURATION_S:
            return 0, 0.0, "STOP", "DOI CHIEU: DUNG NGAN TRUOC KHI TIEN"
        if _recovery["reacquired_in_rev"] and observed and lane_state == "TWO_EDGES" and abs_e <= 0.50:
            if is_sign_combo:
                _sign_combo["cooldown_until"] = now + SIGN_COMBO_COOLDOWN_S
            reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
            return None
        _set_recovery_phase("FWD_TURN", now)
        phase, dt = "FWD_TURN", 0.0

    if phase == "FWD_TURN":
        fwd_min_confirm_s = 0.45 if is_sign_combo else 0.0
        two_edges_ready = bool(
            dt >= fwd_min_confirm_s and observed and lane_state == "TWO_EDGES" and abs_e <= 0.48 and abs_hd <= 18.0
        )
        one_edge_ready = bool(
            observed and lane_state == "ONE_EDGE"
            and dt >= (0.75 if is_sign_combo else 0.55)
            and abs_e <= 0.25 and abs_hd <= 10.0
        )
        if _update_obs_confirm(two_edges_ready or one_edge_ready, now, RECOVERY_OBS_CONFIRM_S):
            # Da ve lai trong lan on dinh -> tra quyen dieu khien ve bam lan binh thuong
            if is_sign_combo:
                _sign_combo["cooldown_until"] = now + SIGN_COMBO_COOLDOWN_S
                _sign_turn_hold["dir"] = "RIGHT"
                _sign_turn_hold["until"] = now + SIGN_TURN_HOLD_S
            reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
            return None
        if dt < fwd_max_s:
            if observed and lane_state == "TWO_EDGES" and ((not is_sign_combo) or dt >= 0.30):
                err = float(np.clip(e_norm * 0.90, -0.85, 0.85))
                cmd = "CURVE" if abs(err) < 0.55 else ("SHARP_R" if err > 0 else "SHARP_L")
            elif (not is_sign_combo) and observed and ((rec_dir == "RIGHT" and e_norm < -0.25) or (rec_dir == "LEFT" and e_norm > 0.25)):
                err = float(np.clip(e_norm * 0.85, -0.52, 0.52))
                cmd = "CURVE"
            else:
                err = +1.0 if rec_dir == "RIGHT" else -1.0
                cmd = "SHARP_R" if rec_dir == "RIGHT" else "SHARP_L"
            msg = (
                "CHINH GOC XONG -> TIEN RE PHAI NGA 3"
                if is_sign_combo
                else f"PHUC HOI (CUA {dir_vn}) -> TIEN VAO LAN"
            )
            return FWD_TURN_PWM, err, cmd, msg

        if is_sign_combo:
            _sign_combo["cooldown_until"] = now + SIGN_COMBO_COOLDOWN_S
            _sign_turn_hold["dir"] = "RIGHT"
            _sign_turn_hold["until"] = now + SIGN_TURN_HOLD_S
            reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
            return None

        # Neu da thay lai lan ro (TWO_EDGES hoac |e| <= 0.55), tra ve bam lan tien truoc
        # thay vi lap tuc lui lan 2 tai cung 1 cho; chi lui ngay lan 2 neu van mu vach hoac lech sat mep (> 0.55).
        if observed and (lane_state == "TWO_EDGES" or abs_e <= 0.55):
            reset_recovery(cooldown_until=now + 1.50)
            return None

        _recent_recovery_times[:] = [t for t in _recent_recovery_times if (now - t) <= RECOVERY_WINDOW_S]
        if len(_recent_recovery_times) < MAX_RECOVERY_TRIES:
            _recent_recovery_times.append(now)
            _recovery["tries"] += 1
            _recovery["reacquired_in_rev"] = False
            _set_recovery_phase("PRE_REV", now)
            return _run_recovery(observed, lane_state, e_norm, heading_deg, now)

        # Chi co gang lui 1 lan: neu da thay vach (observed=True), tra ve bam lan tien voi cooldown dai
        # thay vi dung han hoac lui lai nhieu lan.
        if observed:
            reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
            return None

        _set_recovery_phase("GAVE_UP", now)
        phase = "GAVE_UP"

    if phase == "GAVE_UP":
        if _update_obs_confirm(bool(observed), now, 0.25):
            _recent_recovery_times.clear()
            reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
            return None
        return 0, 0.0, "STOP", f"MAT LAN DUONG: DA LUI CHINH GOC {MAX_RECOVERY_TRIES} LAN KHONG THANH CONG -> DUNG"

    return None


def resolve_sign_directives(detected_signs, sign_info_map=None):
    """Tong hop danh sach bien bao dong thoi (vd: DiThang + CamRePhai, CamDiThang + RePhai)
    thanh chi thi nhat quan cho ca Luong 2 (bam lan/nga 3) va Luong 4 (dieu khien dong co).
    """
    if not detected_signs:
        return {
            "junction_intent": "STRAIGHT",
            "primary_cls": None,
            "action_text": "",
            "speed_factor": 1.0,
            "steer_bias": 0.0,
            "force_turn_dir": None,
            "forbid_straight": False,
            "forbid_right": False,
            "forbid_left": False,
            "stop_now": False,
        }

    cls_best = {}
    for det in detected_signs:
        cls_name = det[0]
        area = float(det[3]) * float(det[4])
        if cls_name not in cls_best or area > cls_best[cls_name][0]:
            cls_best[cls_name] = (area, det)

    cls_set = set(cls_best.keys())

    # 1. Bien bao dung khan cap: chi dung han khi bien STOP/NO_ENTRY da vao du gan dau xe
    stop_far_cls = None
    if "stop" in cls_set or "no_entry" in cls_set:
        p_cls = "stop" if "stop" in cls_set else "no_entry"
        s_det = cls_best[p_cls][1]
        s_w = float(s_det[3]) if len(s_det) >= 5 else 0.0
        s_h = float(s_det[4]) if len(s_det) >= 5 else 0.0
        s_yb = (float(s_det[2]) + s_h) if len(s_det) >= 5 else 0.0
        stop_is_close = (
            (s_w <= 0.0 and s_h <= 0.0)
            or (s_h >= STOP_MIN_BOX_H_PX and s_yb >= STOP_MIN_Y_BOT_PX)
            or (s_w >= STOP_WIDE_MIN_W_PX and s_yb >= STOP_WIDE_MIN_YB_PX)
        )
        if stop_is_close:
            info = (sign_info_map or {}).get(p_cls, {})
            return {
                "junction_intent": None,
                "primary_cls": p_cls,
                "action_text": info.get("action", "DUNG - STOP"),
                "speed_factor": 0.0,
                "steer_bias": 0.0,
                "force_turn_dir": None,
                "forbid_straight": True,
                "forbid_right": False,
                "forbid_left": False,
                "stop_now": True,
            }
        stop_far_cls = p_cls

    has_straight = "straight" in cls_set
    has_right = "right" in cls_set
    has_left = "left" in cls_set
    has_no_straight = "no_straight" in cls_set
    has_no_right = "no_right" in cls_set
    has_no_left = "no_left" in cls_set

    # Tinh he so toc do thap nhat tu cac bien han che toc do / canh bao di kem (neu co)
    aux_speed_factor = 1.0
    if sign_info_map:
        for c_name in cls_set:
            if c_name in ("straight", "right", "left", "no_straight", "no_right", "no_left", "stop", "no_entry"):
                continue
            s_info = sign_info_map.get(c_name)
            if s_info and s_info.get("speed_factor", 1.0) > 0.0:
                aux_speed_factor = min(aux_speed_factor, float(s_info["speed_factor"]))

    # 2. Truong hop uu tien RE PHAI:
    #    - Co bien 'right' (ke ca di kem 'no_straight': Cam di thang + Re phai -> Re phai)
    #    - Hoac 'no_straight' + 'no_left' (khong co 'no_right')
    if (has_right and not has_no_right) or (has_no_straight and has_no_left and not has_no_right and not has_left):
        if has_no_straight and has_right:
            act_txt = "CAM DI THANG -> RE PHAI"
        elif has_no_straight and has_no_left:
            act_txt = "CAM DI THANG + CAM RE TRAI -> RE PHAI"
        else:
            act_txt = ((sign_info_map or {}).get("right") or {}).get("action", "RE PHAI")
        base_spd = 0.80 if has_no_straight else 0.85
        return {
            "junction_intent": "RIGHT",
            "primary_cls": "right" if has_right else "no_straight",
            "action_text": act_txt,
            "speed_factor": min(base_spd, aux_speed_factor),
            "steer_bias": 0.45 if has_right else 0.35,
            "force_turn_dir": "RIGHT",
            "forbid_straight": has_no_straight,
            "forbid_right": False,
            "forbid_left": has_no_left,
            "stop_now": False,
        }

    # 3. Truong hop uu tien RE TRAI:
    #    - Co bien 'left' (ke ca di kem 'no_straight': Cam di thang + Re trai -> Re trai)
    #    - Hoac 'no_straight' + 'no_right' (khong co 'no_left')
    if (has_left and not has_no_left) or (has_no_straight and has_no_right and not has_no_left and not has_right):
        if has_no_straight and has_left:
            act_txt = "CAM DI THANG -> RE TRAI"
        elif has_no_straight and has_no_right:
            act_txt = "CAM DI THANG + CAM RE PHAI -> RE TRAI"
        else:
            act_txt = ((sign_info_map or {}).get("left") or {}).get("action", "RE TRAI")
        base_spd = 0.80 if has_no_straight else 0.85
        return {
            "junction_intent": "LEFT",
            "primary_cls": "left" if has_left else "no_straight",
            "action_text": act_txt,
            "speed_factor": min(base_spd, aux_speed_factor),
            "steer_bias": -0.45 if has_left else -0.35,
            "force_turn_dir": "LEFT",
            "forbid_straight": has_no_straight,
            "forbid_right": has_no_right,
            "forbid_left": False,
            "stop_now": False,
        }

    # 4. Truong hop DI THANG & KHONG RE PHAI/TRAI:
    #    - Co 'straight' va/hoac 'no_right' / 'no_left' (khong co 'no_straight')
    if (has_straight or has_no_right or has_no_left) and not has_no_straight:
        if has_straight and has_no_right:
            act_txt = "DI THANG (CAM RE PHAI)"
        elif has_straight and has_no_left:
            act_txt = "DI THANG (CAM RE TRAI)"
        elif has_straight:
            act_txt = ((sign_info_map or {}).get("straight") or {}).get("action", "DI THANG")
        elif has_no_right:
            act_txt = "CAM RE PHAI (DI THANG)"
        else:
            act_txt = "CAM RE TRAI (DI THANG)"
        base_spd = 1.00 if has_straight else 0.90
        return {
            "junction_intent": "STRAIGHT",
            "primary_cls": "straight" if has_straight else ("no_right" if has_no_right else "no_left"),
            "action_text": act_txt,
            "speed_factor": min(base_spd, aux_speed_factor),
            "steer_bias": 0.0,  # Khong cong bias lech lan khi da bam duong thang qua nga 3
            "force_turn_dir": None,
            "forbid_straight": False,
            "forbid_right": has_no_right,
            "forbid_left": has_no_left,
            "stop_now": False,
        }

    # 5. Chi co 'no_straight' don le (khong thay bien re phai/trai di kem)
    if has_no_straight:
        info = (sign_info_map or {}).get("no_straight", {})
        return {
            "junction_intent": None,
            "primary_cls": "no_straight",
            "action_text": info.get("action", "CAM DI THANG"),
            "speed_factor": min(float(info.get("speed_factor", 0.50)), aux_speed_factor),
            "steer_bias": 0.0,
            "force_turn_dir": None,
            "forbid_straight": True,
            "forbid_right": has_no_right,
            "forbid_left": has_no_left,
            "stop_now": False,
        }

    # 6. Bien STOP/NO_ENTRY nhung con o xa -> tiep tuc bam lan tien gan them roi moi dung
    if stop_far_cls is not None and not (has_straight or has_right or has_left or has_no_straight or has_no_right or has_no_left):
        return {
            "junction_intent": "STRAIGHT",
            "primary_cls": stop_far_cls,
            "action_text": "THAY BIEN STOP (TIEN SAT ROI DUNG)",
            "speed_factor": min(0.92, aux_speed_factor),
            "steer_bias": 0.0,
            "force_turn_dir": None,
            "forbid_straight": False,
            "forbid_right": False,
            "forbid_left": False,
            "stop_now": False,
        }

    # 7. Cac bien bao khac (toc do, cong truong, canh bao...): chon bien co dien tich bbox lon nhat
    primary = max(detected_signs, key=lambda d: d[3] * d[4])
    cls_name = primary[0]
    info = (sign_info_map or {}).get(cls_name) or {}
    return {
        "junction_intent": "STRAIGHT",
        "primary_cls": cls_name,
        "action_text": info.get("action", cls_name),
        "speed_factor": float(info.get("speed_factor", 1.0)),
        "steer_bias": float(info.get("steer_bias", 0.0)),
        "force_turn_dir": None,
        "forbid_straight": False,
        "forbid_right": False,
        "forbid_left": False,
        "stop_now": False,
    }


def decide_pi_to_esp32_command(curve_val, detected_signs=None, sign_info_map=None, lane_info=None):
    """
    Tong hop tin hieu Lan duong (curve_val, lane_info tu Luong 2) va Bien bao (detected_signs tu Luong 3)
    de tao lenh (speed_pct, error, cmd, sign_override_text) gui xuong ESP32.
    Ho tro:
      - Phan vung khoang cach goc cua (FAR / APPROACH / TURN_NOW)
      - Bo nho huong re bao quat (intent_direction) khi mat 1 vach o goc cua
      - Co che LUI NHE PHUC HOI (K-Turn: REV_R / REV_L -> TIEN) khi mat vach tai khuc cua
    """
    global _last_known_turn_dir, _last_corner_dir_time, _on_lane_since, _stop_hold_until

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
        single_edge_side = lane_info.get("single_edge_side")
        observed = lane_info.get("observed", True)
        missing_frames = int(lane_info.get("missing_frames", 0))
        e_norm = float(lane_info.get("e_norm", 0.0))
        heading_deg = float(lane_info.get("heading_error_deg", 0.0))
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

        # Chi xoa bo nho so lan lui khi xe da chay bam lan on dinh (2 vach hoac can tam) >= ON_LANE_RESET_S
        if _recovery["phase"] == "IDLE" and observed and (lane_state == "TWO_EDGES" or abs(e_norm) < 0.45):
            if _on_lane_since is None:
                _on_lane_since = now
            elif (now - _on_lane_since) >= ON_LANE_RESET_S:
                _recent_recovery_times.clear()
        else:
            _on_lane_since = None

        # Phat hien ket lech lan (ONE_EDGE, |e_norm| lon keo dai) khi quay cua ma xe khong tu ve duoc lan 2 vach.
        # Chi kich hoat khi e_norm va goc lai c CUNG DAU (vd: xe dang o ben trai mep duong e_norm>0 va lai phai c>0),
        # tranh kich hoat nham nguoc chieu khi dang vao cua gat (vd: cua phai c>0 nhung tam du doan tam thoi am).
        off_lane_stuck_trigger = False
        off_lane_stuck_dir = None
        same_sign_off_lane = (
            (e_norm >= OFF_LANE_STUCK_E_THRESH and c >= 0.65)
            or (e_norm <= -OFF_LANE_STUCK_E_THRESH and c <= -0.65)
        )
        if (
            _recovery["phase"] == "IDLE"
            and observed
            and lane_state == "ONE_EDGE"
            and abs(e_norm) >= OFF_LANE_STUCK_E_THRESH
            and same_sign_off_lane
            and not detected_signs
            and now >= _recovery_cooldown_until
        ):
            stuck_dir = "RIGHT" if e_norm > 0 else "LEFT"
            if _off_lane_stuck["t0"] is None or _off_lane_stuck["dir"] != stuck_dir:
                _off_lane_stuck["t0"] = now
                _off_lane_stuck["min_abs_e"] = abs(e_norm)
                _off_lane_stuck["dir"] = stuck_dir
            else:
                _off_lane_stuck["min_abs_e"] = min(_off_lane_stuck["min_abs_e"], abs(e_norm))
                if (
                    (now - _off_lane_stuck["t0"]) >= OFF_LANE_STUCK_TRIGGER_S
                    and abs(e_norm) >= (_off_lane_stuck["min_abs_e"] - 0.05)
                ):
                    off_lane_stuck_trigger = True
                    off_lane_stuck_dir = stuck_dir
        else:
            _off_lane_stuck["t0"] = None
            _off_lane_stuck["min_abs_e"] = 1.0
            _off_lane_stuck["dir"] = None

        # 0. BIEN BAO DUNG KHAN CAP (STOP / CAM DI NGUOC CHIEU) KHI DA VAO GAN + GHI NHO DI THANG / CAM RE
        forbid_right_now = False
        forbid_left_now = False
        if detected_signs and sign_info_map:
            directives_now = resolve_sign_directives(detected_signs, sign_info_map)
            if directives_now["stop_now"] or directives_now["speed_factor"] == 0.0:
                _stop_hold_until = now + STOP_HOLD_S
                _straight_pass_hold["until"] = 0.0
                if _recovery["phase"] != "IDLE":
                    reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
                return 0, 0.0, "STOP", f"BIEN BAO: {directives_now['action_text']}"
            # Loc nhieu bien mong sat mep anh (vd w=14..16 px bi cat goc ben phai khi dang cua)
            # de khong kich hoat nham cam re giua khuc cua.
            valid_forbid_box = any(
                (s[0] in ("no_right", "no_left", "straight"))
                and (len(s) < 5 or (float(s[3]) >= 28.0 and float(s[4]) >= 40.0))
                for s in detected_signs
            )
            has_explicit_forbid = any(
                (s[0] in ("no_right", "no_left"))
                and (len(s) < 5 or (float(s[3]) >= 26.0 and float(s[4]) >= 36.0))
                for s in detected_signs
            )
            if valid_forbid_box and (has_explicit_forbid or (not turn_now and abs(c) < 0.70)):
                forbid_right_now = bool(directives_now.get("forbid_right", False))
                forbid_left_now = bool(directives_now.get("forbid_left", False))
            if directives_now.get("forbid_straight") or directives_now.get("force_turn_dir") in ("LEFT", "RIGHT"):
                _straight_pass_hold["until"] = 0.0
                _straight_pass_hold["forbid_right"] = False
                _straight_pass_hold["forbid_left"] = False
            elif valid_forbid_box and directives_now.get("junction_intent") == "STRAIGHT" and (
                has_explicit_forbid or (directives_now.get("primary_cls") == "straight" and not turn_now and abs(c) < 0.70)
            ):
                _straight_pass_hold["until"] = now + 1.60
                _straight_pass_hold["forbid_right"] = bool(forbid_right_now or directives_now.get("primary_cls") == "straight")
                _straight_pass_hold["forbid_left"] = bool(forbid_left_now)
        if now < _straight_pass_hold["until"]:
            forbid_right_now = forbid_right_now or bool(_straight_pass_hold["forbid_right"])
            forbid_left_now = forbid_left_now or bool(_straight_pass_hold["forbid_left"])
        if forbid_right_now:
            if _sharp_turn_latch["dir"] == "RIGHT":
                _sharp_turn_latch["dir"] = None
                _sharp_turn_latch["until"] = 0.0
            if _sign_turn_hold["dir"] == "RIGHT":
                _sign_turn_hold["dir"] = None
                _sign_turn_hold["until"] = 0.0
        if forbid_left_now:
            if _sharp_turn_latch["dir"] == "LEFT":
                _sharp_turn_latch["dir"] = None
                _sharp_turn_latch["until"] = 0.0
            if _sign_turn_hold["dir"] == "LEFT":
                _sign_turn_hold["dir"] = None
                _sign_turn_hold["until"] = 0.0
        if now < _stop_hold_until:
            return 0, 0.0, "STOP", "BIEN BAO: DUNG - STOP"

        # 0b. KICH HOAT CHU TRINH VUA LUI VUA QUAY BANH KHI THAY CAM DI THANG + RE PHAI O NGA 3
        sign_combo_trigger = _update_no_straight_right_trigger(detected_signs, now)
        if _recovery["phase"] == "IDLE" and sign_combo_trigger:
            _recovery["mode"] = "SIGN_COMBO_R"
            _recovery["dir"] = "RIGHT"
            _recovery["tries"] = 1
            _recovery["reacquired_in_rev"] = False
            _set_recovery_phase("PRE_REV", now)

        # 1. CO CHE LUI PHUC HOI (K-TURN) KHI MAT VACH HOAC KET LECH LAN TAI GOC CUA:
        # Khi duong re phai -> Banh xoay TRAI het co (REV_L, -1.00) va LUI MANH (-100%), co gang chi lui 1 lan.
        lost_lane_trigger = (
            not observed
            and missing_frames >= RECOVERY_MISSING_FRAMES_TRIGGER
            and not detected_signs
            and now >= _recovery_cooldown_until
        )
        if _recovery["phase"] == "IDLE" and (lost_lane_trigger or off_lane_stuck_trigger):
            _recent_recovery_times[:] = [t for t in _recent_recovery_times if (now - t) <= RECOVERY_WINDOW_S]
            if off_lane_stuck_trigger and off_lane_stuck_dir in ("LEFT", "RIGHT"):
                rec_dir = off_lane_stuck_dir
            else:
                rec_dir = intent_dir if intent_dir in ("LEFT", "RIGHT") else _last_known_turn_dir
            _off_lane_stuck["t0"] = None
            _off_lane_stuck["min_abs_e"] = 1.0
            _off_lane_stuck["dir"] = None
            if len(_recent_recovery_times) >= MAX_RECOVERY_TRIES:
                if observed:
                    reset_recovery(cooldown_until=now + RECOVERY_COOLDOWN_S)
                else:
                    _recovery["mode"] = "LANE_LOST"
                    _recovery["dir"] = rec_dir
                    _recovery["tries"] = MAX_RECOVERY_TRIES
                    _set_recovery_phase("GAVE_UP", now)
            else:
                _recent_recovery_times.append(now)
                _recovery["mode"] = "LANE_LOST"
                _recovery["dir"] = rec_dir
                _recovery["tries"] = 1
                _recovery["reacquired_in_rev"] = False
                _set_recovery_phase("PRE_REV", now)
        if _recovery["phase"] != "IDLE":
            rec_out = _run_recovery(observed, lane_state, e_norm, heading_deg, now)
            if rec_out is not None:
                return rec_out

        # 2. BE LAI CUA GAT MANH TOI DA KHI XE TOI DIEM RE (turn_now == True)
        # Chong om cua som lan vach trong: khi con thay ca 2 vach (TWO_EDGES) va xe da bat dau lech ve mep trong
        # (vd: cua phai nhung e_norm <= -0.18), KHONG ep khoa SHARP_R de LaneDetector tu giu xe khong lan vach.
        raw_c = c
        elbow_y_ratio_c = float(corner.get("elbow_y_ratio", 0.0) or 0.0)
        stem_span_y_c = float(corner.get("stem_span_y", 240.0) or 240.0)
        stem_still_far = (
            (not turn_now)
            and dist_zone in ("FAR", "APPROACH")
            and elbow_y_ratio_c < 0.50
            and stem_span_y_c > 75.0
        )
        sharp_dir_now = None
        if turn_now and corner.get("confirmed") and corner_dir in ("LEFT", "RIGHT"):
            sharp_dir_now = corner_dir
        elif turn_now and corner_cand in ("LEFT", "RIGHT") and corner.get("consecutive_frames", 0) >= 3:
            sharp_dir_now = corner_cand
        elif turn_now and intent_dir in ("LEFT", "RIGHT") and lane_state == "ONE_EDGE" and corner.get("consecutive_frames", 0) >= 2:
            sharp_dir_now = intent_dir
        elif lane_state == "ONE_EDGE" and abs(raw_c) >= 0.72:
            cand_s = "RIGHT" if raw_c > 0 else "LEFT"
            if not stem_still_far:
                sharp_dir_now = cand_s

        if (forbid_right_now and sharp_dir_now == "RIGHT") or (forbid_left_now and sharp_dir_now == "LEFT"):
            sharp_dir_now = None

        if sharp_dir_now in ("LEFT", "RIGHT"):
            encroaching_inside = observed and lane_state == "TWO_EDGES" and (
                (sharp_dir_now == "RIGHT" and -0.75 <= e_norm <= -0.20)
                or (sharp_dir_now == "LEFT" and 0.20 <= e_norm <= 0.75)
            )
            if encroaching_inside:
                sharp_dir_now = None
                _sharp_turn_latch["dir"] = None
                _sharp_turn_latch["until"] = 0.0

        if sharp_dir_now in ("LEFT", "RIGHT"):
            _sharp_turn_latch["dir"] = sharp_dir_now
            _sharp_turn_latch["until"] = now + SHARP_TURN_HOLD_S
            c = -1.0 if sharp_dir_now == "LEFT" else +1.0
        elif _sharp_turn_latch["dir"] in ("LEFT", "RIGHT"):
            latch_dir = _sharp_turn_latch["dir"]
            encroaching_inside = observed and lane_state == "TWO_EDGES" and (
                (latch_dir == "RIGHT" and -0.75 <= e_norm <= -0.20)
                or (latch_dir == "LEFT" and 0.20 <= e_norm <= 0.75)
            )
            latch_sign = 1.0 if latch_dir == "RIGHT" else -1.0
            turn_ang_cur = max(
                float(corner.get("turn_angle_deg", 0.0)),
                float(corner.get("ground_turn_angle_deg", 0.0)),
            )
            in_one_edge_corner = (
                observed
                and lane_state == "ONE_EDGE"
                and corner_cand == latch_dir
                and turn_ang_cur >= 35.0
                and (turn_now or (raw_c * latch_sign) >= 0.45)
            )
            # Khi da thay 2 vach on dinh (hoac cat qua mep trong trong TWO_EDGES / LaneDetector giam goc lai < 0.45) -> nha giu SHARP
            if encroaching_inside or (observed and not turn_now and (raw_c * latch_sign) < 0.45) or (observed and lane_state == "TWO_EDGES" and not turn_now and abs(e_norm) <= 0.65):
                _sharp_turn_latch["dir"] = None
                _sharp_turn_latch["until"] = 0.0
            elif now < _sharp_turn_latch["until"] or in_one_edge_corner:
                c = -1.0 if latch_dir == "LEFT" else +1.0
            else:
                _sharp_turn_latch["dir"] = None
                _sharp_turn_latch["until"] = 0.0
        elif lane_state == "ONE_EDGE" and not turn_now:
            # Khi chi thay 1 vach ngoai diem cua: neu con dang di tren than thang truoc khuyu cua (stem_still_far)
            # thi giu <= 0.50 de xe di thang len; neu da qua khuyu cua thi giu ty le muot toi da 0.70.
            max_c = 0.50 if stem_still_far else 0.70
            c = float(np.clip(c * 1.12, -max_c, max_c))

        # Chi khi da thay ca 2 vach (TWO_EDGES) va xe lech vao mep trong moi ha ve CURVE ty le de duoi thang vao lan moi
        if observed and lane_state == "TWO_EDGES" and not turn_now:
            if c >= 0.55 and -0.75 <= e_norm <= -0.20:
                c = 0.52
            elif c <= -0.55 and 0.20 <= e_norm <= 0.75:
                c = -0.52

        sharp_active = bool(_sharp_turn_latch["dir"] in ("LEFT", "RIGHT"))
        if observed and lane_state == "TWO_EDGES" and not turn_now and not sharp_active:
            sharp_thresh = 1.05
        elif observed and lane_state == "ONE_EDGE" and not turn_now and not sharp_active:
            sharp_thresh = 0.72
        else:
            sharp_thresh = 0.55

        if abs(c) >= sharp_thresh and not ((forbid_right_now and c > 0) or (forbid_left_now and c < 0)):
            cmd = "SHARP_R" if c > 0 else "SHARP_L"
            throttle_pwm = SPEED_SHARP_TURN
        elif dist_zone == "FAR" and corner_cand in ("LEFT", "RIGHT"):
            cmd = "CURVE" if abs(c) >= 0.12 else "NORM"
            throttle_pwm = SPEED_FAR_CORNER
        elif lane_state == "ONE_EDGE":
            cmd = "CURVE" if abs(c) >= 0.08 else "NORM"
            throttle_pwm = SPEED_ONE_EDGE
        elif abs(c) >= 0.10:
            cmd = "CURVE"
            throttle_pwm = int(round(np.clip((0.97 - 0.12 * abs(c)) * 100, SPEED_CURVE_MIN, SPEED_CURVE_MAX)))
        else:
            cmd = "NORM"
            throttle_pwm = SPEED_STRAIGHT

        if forbid_right_now and c > 0.22:
            c = 0.22
            cmd = "CURVE"
            throttle_pwm = max(90, throttle_pwm)
        elif forbid_left_now and c < -0.22:
            c = -0.22
            cmd = "CURVE"
            throttle_pwm = max(90, throttle_pwm)
    else:
        if abs(c) >= 0.55:
            cmd = "SHARP_R" if c > 0 else "SHARP_L"
            throttle_pwm = SPEED_SHARP_TURN
        elif abs(c) >= 0.10:
            cmd = "CURVE"
            throttle_pwm = int(round(np.clip((0.97 - 0.12 * abs(c)) * 100, SPEED_CURVE_MIN, SPEED_CURVE_MAX)))
        else:
            cmd = "NORM"
            throttle_pwm = SPEED_STRAIGHT

        if forbid_right_now and c > 0.22:
            c = 0.22
            cmd = "CURVE"
            throttle_pwm = max(90, throttle_pwm)
        elif forbid_left_now and c < -0.22:
            c = -0.22
            cmd = "CURVE"
            throttle_pwm = max(90, throttle_pwm)

    if detected_signs and sign_info_map:
        directives = resolve_sign_directives(detected_signs, sign_info_map)
        sign_override_text = directives["action_text"]
        spd_factor = directives["speed_factor"]
        steer_bias = directives["steer_bias"]
        force_dir = directives["force_turn_dir"]

        if directives["stop_now"] or spd_factor == 0.0:
            return 0, 0.0, "STOP", f"BIEN BAO: {sign_override_text}"

        # Giu toc do tien >= 90% cho cac bien chi dan / cam huong khi dang van hanh
        scaled_pwm = int(round(throttle_pwm * spd_factor))
        if directives["primary_cls"] in ("right", "left", "straight", "no_straight", "no_right", "no_left"):
            throttle_pwm = max(90, min(100, scaled_pwm if spd_factor >= 0.90 else throttle_pwm))
        else:
            throttle_pwm = max(90, scaled_pwm) if spd_factor >= 0.80 else scaled_pwm

        obs_now = bool(lane_info.get("observed", True)) if lane_info else True
        st_now = lane_info.get("lane_state", "UNKNOWN") if lane_info else "UNKNOWN"
        tn_now = bool((lane_info.get("corner") or {}).get("turn_now", False)) if lane_info else False
        # Do kich thuoc bien chi huong: khong khoa SIGN_R/SIGN_L (be lai +60/-60 deg) khi bien con o xa,
        # hoac dang tien toi nga 3 co 'no_straight' de xe bam lan di thang sat vao nga 3 roi moi lui/re.
        dir_box_h = 0.0
        dir_box_yb = 0.0
        has_ns_sign = False
        target_sign_cls = "right" if force_dir == "RIGHT" else ("left" if force_dir == "LEFT" else "")
        for s_det in detected_signs:
            s_cls = s_det[0]
            if s_cls == "no_straight":
                has_ns_sign = True
            elif s_cls == target_sign_cls and len(s_det) >= 5:
                s_h = float(s_det[4])
                s_yb = float(s_det[2]) + s_h
                if s_h > dir_box_h:
                    dir_box_h = s_h
                    dir_box_yb = s_yb
        ns_approach_recent = (
            force_dir == "RIGHT"
            and (
                has_ns_sign
                or (
                    _sign_combo["approach_no_straight_t"] > 0.0
                    and (time.monotonic() - _sign_combo["approach_no_straight_t"]) <= SIGN_COMBO_APPROACH_MEM_S
                )
            )
        )
        sign_is_close_for_lock = (
            (dir_box_h >= SIGN_COMBO_RIGHT_FINAL_H_PX and dir_box_yb >= SIGN_COMBO_RIGHT_FINAL_YB_PX)
            or (not obs_now)
            or (tn_now and not ns_approach_recent)
        )
        allow_sign_lock = (force_dir in ("LEFT", "RIGHT")) and (
            sign_is_close_for_lock or (not ns_approach_recent and dir_box_h >= 60.0)
        )

        if allow_sign_lock:
            _sign_turn_hold["dir"] = force_dir
            _sign_turn_hold["until"] = time.monotonic() + SIGN_TURN_HOLD_S

            if force_dir == "RIGHT":
                if (not obs_now) or c < 0.0:
                    c = 1.0
                elif st_now != "TWO_EDGES":
                    c = max(c, 0.55)
            elif force_dir == "LEFT":
                if (not obs_now) or c > 0.0:
                    c = -1.0
                elif st_now != "TWO_EDGES":
                    c = min(c, -0.55)

            if abs(steer_bias) > 0.01:
                c = float(np.clip(c + steer_bias, -1.0, 1.0))
            if force_dir == "RIGHT":
                c = max(0.0, c) if st_now == "TWO_EDGES" else max(0.60, c)
            elif force_dir == "LEFT":
                c = min(0.0, c) if st_now == "TWO_EDGES" else min(-0.60, c)

            if force_dir == "LEFT":
                cmd = "SIGN_L"
            elif force_dir == "RIGHT":
                cmd = "SIGN_R"
        elif spd_factor < 0.75:
            cmd = "SLOW"
    elif _sign_turn_hold["dir"] in ("LEFT", "RIGHT"):
        now_h = time.monotonic()
        hold_dir = _sign_turn_hold["dir"]
        obs_h = bool(lane_info.get("observed", True)) if lane_info else True
        st_h = lane_info.get("lane_state", "UNKNOWN") if lane_info else "UNKNOWN"
        e_h = abs(float(lane_info.get("e_norm", 0.0))) if lane_info else 0.0
        hd_h = abs(float(lane_info.get("heading_error_deg", 0.0))) if lane_info else 0.0
        if (obs_h and st_h == "TWO_EDGES" and e_h <= 0.45 and hd_h <= 18.0) or now_h >= _sign_turn_hold["until"]:
            _sign_turn_hold["dir"] = None
            _sign_turn_hold["until"] = 0.0
        elif st_h != "TWO_EDGES" and ((hold_dir == "RIGHT" and c < 0.0) or (hold_dir == "LEFT" and c > 0.0)):
            c = +1.0 if hold_dir == "RIGHT" else -1.0
            cmd = "SHARP_R" if hold_dir == "RIGHT" else "SHARP_L"
            throttle_pwm = max(90, SPEED_SHARP_TURN)

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
