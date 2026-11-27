import math
import time
import cv2
import numpy as np
import utils


def _fast_median_seq(vals):
    """Fast pure-Python median for small lists/sequences without NumPy dispatch overhead."""
    n = len(vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(vals[0])
    s = sorted(vals)
    mid = n >> 1
    if n & 1:
        return float(s[mid])
    return 0.5 * (float(s[mid - 1]) + float(s[mid]))


# --- Dieu khien goc lai: Pure Pursuit (feedforward hinh hoc) + Stanley (phan hoi bam tam) ---
# delta_cmd = delta_PP + W_STANLEY_FB * delta_Stanley  (giong controller JS Standalone)
W_STANLEY_FB = 0.65          # He so cong phan hoi Stanley (KHONG nhan giam PP)
STEER_TAU_NORMAL_S = 0.095   # Tang loc thoi gian giam nhay nhot o cam thuc te khi di thang / nha lai
STEER_TAU_ENTRY_S = 0.028    # Giam tre khi vao cua gap 90 deg nhung van loc nhieu 1 frame
STEER_ENTRY_PP_DEG = 7.0     # Chi coi la dang vao cua khi |pp| du lon (>= 7 deg), tranh nhieu nho kich hoat
E_CANCEL_GAIN = 0.0   # (v3: 0.75 -> 0.0; xem BAO_CAO muc 9)         # Stanley: giam e_lat nguoc huong cua theo (1 - E_CANCEL_GAIN * turn_prog_eff)
TURN_ZONE_PROG_FLOOR = 0.50  # turn_prog toi thieu khi dist_zone = APPROACH / TURN_NOW
LD_TURN_MIN = 0.85           # Lookahead (don vi lane-width) khi vao cua: 0.85 - 0.95
LD_TURN_MAX = 0.95
STRAIGHT_STEER_CAP = 0.50    # Tran |steer_norm| tren duong thang (< 0.55 de Motor khong gan SHARP)
# Cung tron lien tuc (vd R=0.55m, ~20 do Ackermann): heading chord near->look da chua san do cong cua duong,
# PP da bu phan do -> Stanley cong them 0.95*heading la TRUNG LAP, keo xe lech ve phia trong cung tron (bias CTE on dinh).
ARC_FF_DEG_START = 11.0      # |kappa_ff| (do) bat dau coi la cung tron chat (R lon / S-curve nhe nam duoi nguong nay)
ARC_FF_DEG_FULL = 17.0       # |kappa_ff| (do) dat toi da che do cung tron
ARC_HEADING_RELIEF = 0.65    # Giam trong so heading trong Stanley toi da 65% khi o cung tron (0 = tat tinh nang)
# BU SAI SO e THEO DO CONG: e do o HANG DAY BEV (~1.67 lw = 0.38m truoc truc sau). Tren cung tron ban kinh R, xe dung TREN tam
# van thay duong tam lech R-sqrt(R^2-y^2) (R=0.55m -> ~15cm > nua lan 11.5cm) => Stanley keo xe VAO TRONG ~15cm on dinh.
E_CURV_COMP_GAIN = 1.0       # 0 = tat; 1 = bu day du phan lech do hinh hoc cung tron
E_CURV_KAPPA_DEADBAND = 0.04 # (1/lw) bo qua curvature rat nho (duong thang) de khong khuech dai nhieu
E_CURV_KAPPA_MAX = 0.60      # (1/lw) tran curvature cua bo bu (R_min ~ 1.67 lw = 0.38m)
ARC_LD_RELIEF = 0.35           # Khi o cung tron chat: giam muc rut lookahead (ld_g ve ld_raw). 0 = tat (hanh vi cu), 1 = khong rut


def _clip_scalar(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _smoothstep(edge0, edge1, x):
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = (x - edge0) / (edge1 - edge0)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


PRESET_ROBOT_TOY = {
    "w_top": 123, "h_top": 105, "w_bot": 5, "h_bot": 222,
    "hsv_range": (np.array([0, 0, 170]), np.array([179, 85, 255])),
}

PRESET_HIGHWAY = {
    "w_top": 140, "h_top": 145, "w_bot": 25, "h_bot": 230,
    "hsv_range": (np.array([0, 0, 185]), np.array([179, 50, 255])),
}


class LaneDetector:
    """PC robot profile: tape ridge mask, observed boundary traces, corner evidence.

    Explicit highway profile preserves the existing vanishing-point algorithm.
    Geometry scores are heuristic evidence, not calibrated probabilities.
    The Pi convenience wrapper retains legacy behavior until it consumes status.
    """

    def __init__(self, avg_window=6, use_ai="auto", ai_model_path=None, ai_thresh=0.5):
        self.curve_list = []
        self.avg_window = avg_window
        self.use_ai = use_ai
        self._ai_segmenter = None
        if use_ai in ("auto", True):
            try:
                from lane_ai import AILaneSegmenter
                seg = AILaneSegmenter(model_path=ai_model_path, prob_thresh=ai_thresh)
                if seg.ready:
                    self._ai_segmenter = seg
                elif use_ai is True:
                    print("[WARNING] use_ai=True nhung chua co lane_unet.onnx hoac onnxruntime; dung fallback OpenCV.")
            except Exception as exc:
                if use_ai is True:
                    print(f"[WARNING] Khong the nap AILaneSegmenter ({exc}); dung fallback OpenCV.")
        self.reset()

    def reset(self):
        self.curve_list.clear()
        self.smooth_left_x = None
        self.smooth_right_x = None
        self.width_profile = None
        self.lane_width_ab = (0.95, 72.0)
        self._lane_width_calibrated = False
        self._anchor_lane_width_ab = None
        self._anchor_straight_hits = 0
        self._anchor_mismatch_hits = 0
        self._width_bad_hits = 0
        self._last_img_gray = None
        self._last_road_med_gray = None
        self.tape_width_ab = (0.045, 3.5)
        self.smooth_e_px = None
        self.smooth_e_norm = None
        self.left_segments = []
        self.right_segments = []
        self.single_edge_side = None
        self._left_is_tarp_virtual = False
        self._right_is_tarp_virtual = False
        self._last_left_bot_x = None
        self._last_right_bot_x = None
        self._active_turn_dir = "UNKNOWN"
        self._active_turn_hold = 0
        self._active_turn_hold_s = 0.0
        self._intent_turn_dir = "UNKNOWN"
        self._intent_turn_ttl = 0
        self._intent_turn_ttl_s = 0.0
        self._straight_clear_hits = 0
        self._prev_light_mode = None
        self._last_obs_time = None
        self._last_dt_s = 1.0 / 15.0
        self._last_corner_info = None
        self._last_junction_info = {
            "detected": False,
            "type": "NONE",
            "available_actions": ["STRAIGHT"],
            "branch_side": None,
            "straight_edge_side": None,
            "branch_elbow_y": None,
            "phase": "NONE",
            "confidence": 0.0,
        }
        self._junction_mem = {
            "type": "NONE",
            "branch_side": None,
            "straight_edge_side": None,
            "ttl_s": 0.0,
            "last_elbow_y": None,
        }
        self._smooth_center_pts = None
        self._smooth_left_segs = []
        self._smooth_right_segs = []
        self._smooth_turn_prog = None
        self._smooth_steer_norm = None
        self._smooth_elbow_y = None
        self._smooth_ground_turn_deg = None
        self.center_pts = np.empty((0, 2), dtype=np.float32)
        self.center_pred_flags = np.empty((0,), dtype=bool)
        self._last_valid_center_pts = np.empty((0, 2), dtype=np.float32)
        self._last_valid_pred_flags = np.empty((0,), dtype=bool)
        self.missing_frames = 0
        self._missing_s = 0.0
        self._corner_candidate = "UNKNOWN"
        self._corner_hits = 0
        self.boundary_traces = []
        self.observed_boundaries = []
        self._previous_boundaries = []
        self._next_boundary_id = 1
        self._corner_ids = ()
        self.front_wheel_angle_deg = 0.0
        self.servo_angle_deg = 90
        self.heading_error_deg = 0.0
        self.corner_memory = {
            "active": False,
            "direction": "UNKNOWN",
            "centerline": np.empty((0, 2), dtype=np.float32),
            "left_segments": [],
            "right_segments": [],
            "lane_width_ab": (0.95, 72.0),
            "stem_width_px": 0.0,
            "arm_width_px": 0.0,
            "elbow_pt": None,
            "turn_angle_deg": 0.0,
            "ttl": 0,
            "ttl_s": 0.0,
            "age_frames": 0,
            "age_s": 0.0,
        }
        self.last_detection = {
            "observed": False,
            "lane_state": "UNKNOWN",
            "center_offset_px": 0.0,
            "center_offset_norm": 0.0,
            "e_norm": 0.0,
            "offset_direction": "CAN GIUA",
            "vehicle_ref_x": 240.0,
            "car_ref_x": 240.0,
            "lowest_center_point": None,
            "lane_width_px": 0.0,
            "w_lane_bot": 0.0,
            "lane_width_calibrated": False,
            "tape_width_px": 0.0,
            "front_wheel_angle_deg": 0.0,
            "servo_angle_deg": 90,
            "heading_error_deg": 0.0,
            "psi_rad": 0.0,
            "psi_deg": 0.0,
            "left_segments": [],
            "right_segments": [],
            "centerline_predicted": False,
            "corner": {"direction": "UNKNOWN", "confidence": 0.0},
            "corner_memory": {"active": False, "direction": "UNKNOWN", "ttl": 0},
            "junction": {
                "detected": False,
                "type": "NONE",
                "available_actions": ["STRAIGHT"],
                "branch_side": None,
                "straight_edge_side": None,
                "branch_elbow_y": None,
                "phase": "NONE",
                "confidence": 0.0,
            },
            "lighting": {"mode": "NORMAL", "med_l": 110.0, "p98_l": 190.0},
        }

    def _resolve_dt_s(self, dt_s=None, fresh_observation=True):
        """Resolve per-frame delta time in seconds for FPS-independent memory/wait gating."""
        if not fresh_observation:
            return 0.0
        default_dt = 1.0 / 15.0
        if dt_s is not None:
            dt = _clip_scalar(float(dt_s), 1.0 / 120.0, 0.25)
            self._last_obs_time = time.monotonic()
            self._last_dt_s = dt
            return dt
        now = time.monotonic()
        prev = getattr(self, "_last_obs_time", None)
        self._last_obs_time = now
        if prev is None:
            dt = default_dt
        else:
            raw_dt = now - prev
            dt = raw_dt if 0.003 <= raw_dt <= 0.35 else default_dt
        self._last_dt_s = dt
        return dt

    @staticmethod
    def _extract_row_runs_with_dist(mask_row, dist_row, y, hT):
        idx = np.flatnonzero(mask_row)
        n = idx.size
        if n < 2:
            return []
        ratio = float(y) / float(hT if hT > 0 else 1)
        if ratio < 0.0:
            ratio = 0.0
        elif ratio > 1.0:
            ratio = 1.0
        max_slice_w = int(round(44 + 52 * ratio))
        max_perp_w = 14.0 + 18.0 * ratio
        runs = []
        diffs = np.flatnonzero(np.diff(idx) > 1)
        if diffs.size == 0:
            spans = ((int(idx[0]), int(idx[-1]) + 1),)
        else:
            spans = []
            prev = 0
            for d in diffs:
                di = int(d)
                spans.append((int(idx[prev]), int(idx[di]) + 1))
                prev = di + 1
            spans.append((int(idx[prev]), int(idx[-1]) + 1))
        for s, e in spans:
            rw = e - s
            if 2 <= rw <= max_slice_w:
                cx = 0.5 * (s + e - 1)
                icx = int(round(cx))
                lo = s if s > icx - 2 else icx - 2
                hi = e if e < icx + 3 else icx + 3
                d_sub = dist_row[lo:hi]
                local_r = float(d_sub.max()) if d_sub.size > 0 else 1.0
                perp_w = 2.0 * local_r
                if perp_w > float(rw):
                    perp_w = float(rw)
                if perp_w < 2.0:
                    perp_w = 2.0
                if perp_w <= max_perp_w:
                    runs.append((cx, perp_w, s, e, float(rw)))
        return runs

    @staticmethod
    def _build_row_runs_table(mask, dist):
        """Vectorized extraction of horizontal white tape spans for all rows in a single NumPy pass."""
        hT, wT = mask.shape[:2]
        table = [[] for _ in range(hT)]
        padded = np.zeros((hT, wT + 2), dtype=np.uint8)
        padded[:, 1:-1] = (mask > 0)
        sy, sx = np.nonzero((padded[:, :-1] == 0) & (padded[:, 1:] != 0))
        if sy.size == 0:
            return table
        _, ex = np.nonzero((padded[:, :-1] != 0) & (padded[:, 1:] == 0))
        rw = ex - sx
        hT_f = float(hT if hT > 0 else 1)
        ratio = np.clip(sy.astype(np.float32) / hT_f, 0.0, 1.0)
        max_sw = np.rint(44.0 + 52.0 * ratio).astype(np.int32)
        keep = (rw >= 2) & (rw <= max_sw)
        if not np.any(keep):
            return table

        sy = sy[keep]
        sx = sx[keep]
        ex = ex[keep]
        rw_f = rw[keep].astype(np.float32)
        max_pw = 14.0 + 18.0 * ratio[keep]

        cx_arr = 0.5 * (sx + ex - 1).astype(np.float32)
        icx = np.clip(np.rint(cx_arr).astype(np.int32), 0, wT - 1)
        ex_m1 = ex - 1
        d0 = dist[sy, icx]
        dm1 = dist[sy, np.maximum(sx, icx - 1)]
        dp1 = dist[sy, np.minimum(ex_m1, icx + 1)]
        dm2 = dist[sy, np.maximum(sx, icx - 2)]
        dp2 = dist[sy, np.minimum(ex_m1, icx + 2)]
        local_r = np.maximum(np.maximum(d0, np.maximum(dm1, dp1)), np.maximum(dm2, dp2))
        perp_w = np.maximum(2.0, np.minimum(rw_f, 2.0 * local_r))
        keep2 = perp_w <= max_pw
        if not np.any(keep2):
            return table

        for y_i, cx_i, pw_i, s_i, e_i, rw_i in zip(
            sy[keep2].tolist(),
            cx_arr[keep2].tolist(),
            perp_w[keep2].tolist(),
            sx[keep2].tolist(),
            ex[keep2].tolist(),
            rw_f[keep2].tolist(),
        ):
            table[y_i].append((cx_i, pw_i, s_i, e_i, rw_i))
        return table

    @classmethod
    def _trace_stem_and_turn(cls, mask, dist, gray, road_med_gray, seed_x, seed_y, side, top_limit_y=4, row_cache=None):
        hT, wT = mask.shape[:2]
        raw_obs = []  # (cx, y, perp_w, s, e, rw)
        is_table = isinstance(row_cache, list)
        if row_cache is None:
            row_cache = {}

        def _get_runs(y_idx):
            if is_table:
                return row_cache[y_idx]
            cached = row_cache.get(y_idx)
            if cached is None:
                cached = cls._extract_row_runs_with_dist(mask[y_idx], dist[y_idx], y_idx, hT)
                row_cache[y_idx] = cached
            return cached

        curr_x = float(seed_x)
        dx_per_y = 0.0
        miss_rows = 0
        min_obs_x = float(seed_x)
        max_obs_x = float(seed_x)
        s_rev_count = 0
        s_rev_peak_idx = 0
        wT_m1 = float(wT - 1)
        hT_f = float(hT if hT > 0 else 1)
        hT_38 = int(hT * 0.38)
        hT_45 = int(hT * 0.45)
        hT_68 = int(hT * 0.68)

        def _has_dash_support(cx_val, y_val):
            for y_chk in (y_val - 2, y_val - 4):
                if y_chk >= 4:
                    for it_chk in _get_runs(y_chk):
                        if abs(it_chk[0] - cx_val) <= 10.0:
                            return True
            return False

        y_stop = max(top_limit_y - 1, 3)
        for y in range(int(seed_y), y_stop, -2):
            # Huy som neu diem seed chi la dom nhieu don le (< 2 diem trong 8px dau)
            if (seed_y - y) > 8 and len(raw_obs) < 2:
                break

            pred_x = curr_x + dx_per_y * 2.0
            if miss_rows == 0:
                gate = 16.0
            else:
                slope_unc = 0.70 if len(raw_obs) < 3 else 0.45
                gate = 14.0 + min(22.0, miss_rows * slope_unc)

            med_rw_live = _fast_median_seq([it[5] for it in raw_obs[-6:]]) if (miss_rows >= 4 and raw_obs) else None
            best = None
            best_d = gate
            for y_scan in (y, y - 1):
                if y_scan < 0:
                    continue
                for item in _get_runs(y_scan):
                    d = abs(item[0] - pred_x)
                    if d < best_d:
                        if med_rw_live is not None and item[4] > max(20.0, med_rw_live * 2.2):
                            continue
                        best_d = d
                        best = item
                if best is not None:
                    break

            # Neu vua nhay qua khe ho net dut (>= 6px), yeu cau diem tiep theo co ho tro doc (tranh hat nhieu)
            if best is not None and miss_rows >= 6 and not _has_dash_support(best[0], y):
                best = None

            if best is not None:
                cx, perp_w, s, e, rw = best
                if gray is not None and road_med_gray is not None and y < hT_45:
                    inward_dir = 1 if side == "LEFT" else -1
                    ix1 = (e + 6) if inward_dir > 0 else (s - 28)
                    ix2 = (e + 28) if inward_dir > 0 else (s - 6)
                    if ix1 < 0:
                        ix1 = 0
                    elif ix1 > wT - 1:
                        ix1 = wT - 1
                    if ix2 < 0:
                        ix2 = 0
                    elif ix2 > wT - 1:
                        ix2 = wT - 1
                    if ix2 > ix1:
                        gray_y = gray[y]
                        inward_med = float(gray_y[ix1:ix2].sum()) / float(ix2 - ix1)
                        tape_med = (float(gray_y[s:e].sum()) / float(e - s)) if e > s else inward_med
                        min_inward_diff = 8.0 if tape_med >= 246.0 else 20.0
                        if tape_med - inward_med < min_inward_diff and inward_med > road_med_gray + 42.0:
                            break

                if raw_obs:
                    prev_x, prev_y = raw_obs[-1][0], raw_obs[-1][1]
                    # Stop tracing before crossing into the upper opposite-bending arm of an S-curve
                    if len(raw_obs) >= 6 and y < hT_38:
                        if (
                            side == "LEFT"
                            and (max_obs_x - float(seed_x)) >= 0.45 * float(wT)
                            and max_obs_x >= 0.62 * float(wT)
                        ):
                            if (max_obs_x - cx) >= 20.0 and cx < prev_x - 1.5:
                                s_rev_count += 1
                                if s_rev_count >= 4:
                                    raw_obs = raw_obs[: s_rev_peak_idx + 1]
                                    break
                            elif cx >= max_obs_x:
                                s_rev_count = 0
                        elif (
                            side == "RIGHT"
                            and (float(seed_x) - min_obs_x) >= 0.45 * float(wT)
                            and min_obs_x <= 0.38 * float(wT)
                        ):
                            if (cx - min_obs_x) >= 20.0 and cx > prev_x + 1.5:
                                s_rev_count += 1
                                if s_rev_count >= 4:
                                    raw_obs = raw_obs[: s_rev_peak_idx + 1]
                                    break
                            elif cx <= min_obs_x:
                                s_rev_count = 0
                    dy_step = float(prev_y - y)
                    if dy_step < 1.0:
                        dy_step = 1.0
                    max_slope = 2.5 if dy_step >= 10.0 else 3.5
                    inst_dx = _clip_scalar((cx - prev_x) / dy_step, -max_slope, max_slope)
                    if len(raw_obs) <= 2 and dy_step >= 8.0:
                        dx_per_y = 0.30 * dx_per_y + 0.70 * inst_dx
                    else:
                        dx_per_y = 0.58 * dx_per_y + 0.42 * inst_dx
                curr_x = cx
                pw_c = 2.0 if perp_w < 2.0 else (28.0 if perp_w > 28.0 else perp_w)
                raw_obs.append((float(cx), float(y), float(pw_c), s, e, rw))
                if cx <= min_obs_x:
                    min_obs_x = float(cx)
                    if side == "RIGHT":
                        s_rev_peak_idx = len(raw_obs) - 1
                if cx >= max_obs_x:
                    max_obs_x = float(cx)
                    if side == "LEFT":
                        s_rev_peak_idx = len(raw_obs) - 1
                miss_rows = 0
            else:
                curr_x = _clip_scalar(pred_x, 0.0, wT_m1)
                miss_rows += 2
                # Cho phep nhay qua khe ho net dut thich ung theo phoi canh khi da co than vach du dai (>= 6 diem)
                if len(raw_obs) >= 6:
                    max_miss = int(round(18 + 26 * (y / hT_f)))
                elif len(raw_obs) >= 3:
                    max_miss = 8
                else:
                    max_miss = 4 if y > hT_68 else 2
                if miss_rows > max_miss:
                    break

        if not raw_obs:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Correct border-clipped runs at the bottom (where s <= 1 or e >= wT - 1) using unclipped tape width
        unclipped_rws = [item[5] for item in raw_obs[:15] if item[3] > 2 and item[4] < wT - 2]
        ref_rw = _fast_median_seq(unclipped_rws) if unclipped_rws else None
        left_w_med = None
        right_w_med = None

        pts = []
        widths = []
        for cx, y, perp_w, s, e, rw in raw_obs:
            if ref_rw is not None and ref_rw <= 22.0 and rw < ref_rw * 0.88:
                if s <= 1:
                    true_cx = float(e) - 0.5 * ref_rw
                    if true_cx < -2.5:
                        continue
                    cx = max(0.0, true_cx)
                    if left_w_med is None:
                        lw = [it[2] for it in raw_obs[:12] if it[3] > 2]
                        left_w_med = _fast_median_seq(lw) if lw else 0.0
                    if left_w_med > perp_w:
                        perp_w = left_w_med
                elif e >= wT - 1:
                    true_cx = float(s) + 0.5 * ref_rw
                    if true_cx > wT + 1.5:
                        continue
                    cx = min(wT_m1, true_cx)
                    if right_w_med is None:
                        rw_l = [it[2] for it in raw_obs[:12] if it[4] < wT - 2]
                        right_w_med = _fast_median_seq(rw_l) if rw_l else 0.0
                    if right_w_med > perp_w:
                        perp_w = right_w_med
            pts.append((float(cx), float(y)))
            widths.append(float(perp_w))

        if len(pts) < 3 or ((pts[0][1] - pts[-1][1]) < 6.0 and abs(pts[-1][0] - pts[0][0]) < 14.0):
            return np.asarray(pts, dtype=np.float32), np.asarray(widths, dtype=np.float32)

        # Try horizontal/diagonal turn-arm continuation from near the top of pts
        cand_indices = [len(pts) - 1]
        if len(pts) >= 6:
            off_top = max(0, len(pts) - 5)
            top_window = pts[off_top:]
            if side == "RIGHT":
                eb_i = off_top + min(range(len(top_window)), key=lambda idx_w: top_window[idx_w][0])
            else:
                eb_i = off_top + max(range(len(top_window)), key=lambda idx_w: top_window[idx_w][0])
            if eb_i not in cand_indices:
                cand_indices.append(eb_i)

        best_full_pts = pts
        best_full_widths = widths
        best_reach_x = 0.0

        for start_idx in cand_indices:
            top_x, top_y = pts[start_idx]
            # Neu than vach da di ngang manh theo 1 huong (hoac da qua dinh < / >), khong quet nguoc 180 do
            stem_dx = float(top_x - pts[0][0])
            min_pt_x = min(p[0] for p in pts[: start_idx + 1])
            max_pt_x = max(p[0] for p in pts[: start_idx + 1])
            allowed_dirs = (1, -1)
            if abs(stem_dx) >= 0.32 * float(wT):
                # Da re ngang manh theo 1 huong tren khung hinh: khong quay nguoc dau tao moc chu Z / U o mep tren
                allowed_dirs = (1,) if stem_dx > 0.0 else (-1,)
            elif side == "RIGHT" and (top_x - min_pt_x) >= 6.0:
                allowed_dirs = (1,)
            elif side == "LEFT" and (max_pt_x - top_x) >= 6.0:
                allowed_dirs = (-1,)
            elif stem_dx > 28.0 and side == "RIGHT":
                allowed_dirs = (1,)
            elif stem_dx < -28.0 and side == "LEFT":
                allowed_dirs = (-1,)
            for dir_x in allowed_dirs:
                arm_pts = []
                arm_w = []
                curr_y = max(2.0, float(top_y))
                dy_per_x = -0.06
                miss_cols = 0
                start_col = max(2, min(wT - 3, int(round(top_x)) + dir_x * 2))
                end_col = wT - 2 if dir_x > 0 else 1
                for x in range(start_col, end_col, dir_x * 2):
                    pred_y = curr_y + dy_per_x * 2.0
                    if pred_y > top_y + 8.0 or pred_y < 1.0:
                        break
                    half_band = 8 if len(arm_pts) < 3 else 10
                    y_lo = max(1, int(round(pred_y - half_band)))
                    y_hi = min(hT - 2, int(round(min(top_y + 6.0, pred_y + half_band))))
                    if y_hi <= y_lo:
                        break
                    col_sub = mask[y_lo:y_hi, x]
                    best_cy = None
                    best_cw = None
                    best_dy = float(half_band) + 0.5
                    if np.any(col_sub):
                        max_wy = int(round(16 + 18 * (pred_y / hT_f)))
                        col_nz = np.flatnonzero(col_sub)
                        if col_nz.size == col_nz[-1] - col_nz[0] + 1:
                            sy = y_lo + int(col_nz[0])
                            ey = y_lo + int(col_nz[-1]) + 1
                            cw = ey - sy
                            cy = 0.5 * (sy + ey - 1)
                            d_cy = abs(cy - pred_y)
                            if 1 <= cw <= max_wy and d_cy < best_dy and cy <= top_y + 5.0 and cy <= curr_y + 3.0:
                                ok_bg = True
                                if gray is not None and road_med_gray is not None and cy > 25:
                                    by1 = min(hT - 1, ey + 4)
                                    by2 = min(hT, ey + 20)
                                    if by2 > by1:
                                        below_g = float(gray[by1:by2, x].sum()) / float(by2 - by1)
                                        if below_g > road_med_gray + 62.0:
                                            tape_g = float(gray[sy:ey, x].sum()) / float(max(1, ey - sy))
                                            if tape_g < 244.0 or (tape_g - below_g) < 12.0:
                                                ok_bg = False
                                if ok_bg:
                                    best_dy = d_cy
                                    ic_y = int(round(cy))
                                    perp_cw = max(2.0, min(float(cw), 2.0 * float(dist[ic_y, x])))
                                    best_cy = float(cy)
                                    best_cw = perp_cw
                        else:
                            c_diffs = np.flatnonzero(np.diff(col_nz) > 1)
                            cp = 0
                            c_spans = []
                            for cd in c_diffs:
                                cdi = int(cd)
                                c_spans.append((int(col_nz[cp]), int(col_nz[cdi]) + 1))
                                cp = cdi + 1
                            c_spans.append((int(col_nz[cp]), int(col_nz[-1]) + 1))
                            for g0, g1 in c_spans:
                                sy = y_lo + g0
                                ey = y_lo + g1
                                cw = ey - sy
                                cy = 0.5 * (sy + ey - 1)
                                d_cy = abs(cy - pred_y)
                                if 1 <= cw <= max_wy and d_cy < best_dy and cy <= top_y + 5.0 and cy <= curr_y + 3.0:
                                    ok_bg = True
                                    if gray is not None and road_med_gray is not None and cy > 25:
                                        by1 = min(hT - 1, ey + 4)
                                        by2 = min(hT, ey + 20)
                                        if by2 > by1:
                                            below_g = float(gray[by1:by2, x].sum()) / float(by2 - by1)
                                            if below_g > road_med_gray + 62.0:
                                                tape_g = float(gray[sy:ey, x].sum()) / float(max(1, ey - sy))
                                                if tape_g < 244.0 or (tape_g - below_g) < 12.0:
                                                    ok_bg = False
                                    if ok_bg:
                                        best_dy = d_cy
                                        ic_y = int(round(cy))
                                        perp_cw = max(2.0, min(float(cw), 2.0 * float(dist[ic_y, x])))
                                        best_cy = float(cy)
                                        best_cw = perp_cw
                    if best_cy is not None:
                        if arm_pts:
                            prev_ax, prev_ay = arm_pts[-1]
                            dx_step = abs(x - prev_ax)
                            if dx_step < 1.0:
                                dx_step = 1.0
                            inst_dy = _clip_scalar((best_cy - prev_ay) / dx_step, -0.85, 0.45)
                            dy_per_x = 0.62 * dy_per_x + 0.38 * inst_dy
                        curr_y = best_cy
                        arm_pts.append((float(x), float(best_cy)))
                        arm_w.append(_clip_scalar(best_cw, 2.0, 26.0))
                        miss_cols = 0
                    else:
                        if len(arm_pts) < 2:
                            break
                        curr_y = _clip_scalar(pred_y, 1.0, float(hT - 2))
                        miss_cols += 2
                        max_miss_c = 28 if len(arm_pts) >= 3 else 10
                        if miss_cols > max_miss_c:
                            break
                if len(arm_pts) >= 5:
                    span_x = abs(arm_pts[-1][0] - pts[start_idx][0])
                    if span_x >= 16.0 and span_x > best_reach_x:
                        best_reach_x = span_x
                        best_full_pts = pts[: start_idx + 1] + arm_pts
                        best_full_widths = widths[: start_idx + 1] + arm_w

        return np.asarray(best_full_pts, dtype=np.float32), np.asarray(best_full_widths, dtype=np.float32)

    @classmethod
    def _trace_horizontal_bar(cls, mask, dist, gray, road_med_gray, seed_x, seed_y, dir_x=0):
        """Trace a near-horizontal outer corner wall across columns when no vertical stem exists."""
        hT, wT = mask.shape[:2]
        hT_f = float(hT if hT > 0 else 1)

        def _scan_dir(sx, sy, dx_dir):
            pts_d = []
            w_d = []
            curr_y = float(sy)
            dy_per_x = 0.0
            miss_cols = 0
            end_col = (wT - 1) if dx_dir > 0 else 0
            start_col = max(0, min(wT - 1, int(round(sx))))
            for x in range(start_col, end_col + dx_dir, dx_dir * 2):
                pred_y = curr_y + dy_per_x * 2.0
                if pred_y < 2.0 or pred_y > float(hT - 12):
                    break
                half_band = 10
                y_lo = max(1, int(round(pred_y - half_band)))
                y_hi = min(hT - 2, int(round(pred_y + half_band)))
                if y_hi <= y_lo:
                    break
                col_sub = mask[y_lo:y_hi, x]
                best_cy = None
                best_cw = None
                best_dy = float(half_band) + 0.5
                if col_sub.any():
                    max_wy = int(round(18 + 20 * (pred_y / hT_f)))
                    col_nz = np.flatnonzero(col_sub)
                    if col_nz.size == col_nz[-1] - col_nz[0] + 1:
                        sy_r = y_lo + int(col_nz[0])
                        ey_r = y_lo + int(col_nz[-1]) + 1
                        cw = ey_r - sy_r
                        cy = 0.5 * (sy_r + ey_r - 1)
                        d_cy = abs(cy - pred_y)
                        if 2 <= cw <= max_wy and d_cy < best_dy:
                            ok_bg = True
                            # Neu dai trang cham tran y_lo/y_hi va van tiep tuc trang ra ngoai -> khoi san nha/dep trang, KHONG phai bang dinh!
                            if (sy_r == y_lo and y_lo > 2 and mask[y_lo - 1, x] > 0) or (
                                ey_r == y_hi and y_hi < hT - 2 and mask[y_hi, x] > 0
                            ):
                                ok_bg = False
                            if ok_bg and gray is not None and road_med_gray is not None and cy > 25:
                                by1 = min(hT - 1, ey_r + 4)
                                by2 = min(hT, ey_r + 20)
                                if by2 > by1:
                                    below_g = float(gray[by1:by2, x].sum()) / float(by2 - by1)
                                    if below_g > road_med_gray + 62.0:
                                        tape_g = float(gray[sy_r:ey_r, x].sum()) / float(max(1, ey_r - sy_r))
                                        if tape_g < 244.0 or (tape_g - below_g) < 12.0:
                                            ok_bg = False
                            if ok_bg:
                                best_dy = d_cy
                                ic_y = int(round(cy))
                                perp_cw = max(2.0, min(float(cw), 2.0 * float(dist[ic_y, x])))
                                best_cy = float(cy)
                                best_cw = perp_cw
                    else:
                        c_diffs = np.flatnonzero(np.diff(col_nz) > 1)
                        cp = 0
                        c_spans = []
                        for cd in c_diffs:
                            cdi = int(cd)
                            c_spans.append((int(col_nz[cp]), int(col_nz[cdi]) + 1))
                            cp = cdi + 1
                        c_spans.append((int(col_nz[cp]), int(col_nz[-1]) + 1))
                        for g0, g1 in c_spans:
                            sy_r = y_lo + g0
                            ey_r = y_lo + g1
                            cw = ey_r - sy_r
                            cy = 0.5 * (sy_r + ey_r - 1)
                            d_cy = abs(cy - pred_y)
                            if 2 <= cw <= max_wy and d_cy < best_dy:
                                ok_bg = True
                                if (sy_r == y_lo and y_lo > 2 and mask[y_lo - 1, x] > 0) or (
                                    ey_r == y_hi and y_hi < hT - 2 and mask[y_hi, x] > 0
                                ):
                                    ok_bg = False
                                if ok_bg and gray is not None and road_med_gray is not None and cy > 25:
                                    by1 = min(hT - 1, ey_r + 4)
                                    by2 = min(hT, ey_r + 20)
                                    if by2 > by1:
                                        below_g = float(gray[by1:by2, x].sum()) / float(by2 - by1)
                                        if below_g > road_med_gray + 62.0:
                                            tape_g = float(gray[sy_r:ey_r, x].sum()) / float(max(1, ey_r - sy_r))
                                            if tape_g < 244.0 or (tape_g - below_g) < 12.0:
                                                ok_bg = False
                                if ok_bg:
                                    best_dy = d_cy
                                    ic_y = int(round(cy))
                                    perp_cw = max(2.0, min(float(cw), 2.0 * float(dist[ic_y, x])))
                                    best_cy = float(cy)
                                    best_cw = perp_cw
                if best_cy is not None:
                    if pts_d:
                        prev_ax, prev_ay = pts_d[-1]
                        dx_step = max(1.0, abs(float(x) - prev_ax))
                        inst_dy = _clip_scalar((best_cy - prev_ay) / dx_step, -0.75, 0.75)
                        dy_per_x = 0.62 * dy_per_x + 0.38 * inst_dy
                    curr_y = best_cy
                    pts_d.append((float(x), float(best_cy)))
                    w_d.append(_clip_scalar(best_cw, 2.0, 26.0))
                    miss_cols = 0
                else:
                    if len(pts_d) < 2:
                        break
                    curr_y = _clip_scalar(pred_y, 2.0, float(hT - 12))
                    miss_cols += 2
                    if miss_cols > (24 if len(pts_d) >= 4 else 8):
                        break
            return pts_d, w_d

        if dir_x != 0:
            pts_res, w_res = _scan_dir(seed_x, seed_y, 1 if dir_x > 0 else -1)
        else:
            pts_r, w_r = _scan_dir(seed_x, seed_y, 1)
            pts_l, w_l = _scan_dir(seed_x - 2.0, seed_y, -1)
            pts_res = list(reversed(pts_l)) + pts_r
            w_res = list(reversed(w_l)) + w_r
        if len(pts_res) >= 2 and float(pts_res[0][1]) < float(pts_res[-1][1]) - 8.0:
            pts_res.reverse()
            w_res.reverse()
        return np.asarray(pts_res, dtype=np.float32), np.asarray(w_res, dtype=np.float32)

    @classmethod
    def _find_and_trace_boundaries(
        cls,
        mask,
        top_y_hint=105,
        gray=None,
        precomputed_cc=None,
        lane_width_ab=None,
        prev_left_segs=None,
        prev_right_segs=None,
    ):
        """
        Tim va bam vet CHINH XAC 2 vach trang 2 ben duong (ke ca khi ca 2 vach cung nam lech ve 1 phia
        so voi tam anh mid_x, hoac khi vao goc cua gat 90 do vach ngoai di vao tu mep doc Trai/Phai).
        Loai bo triet de bong loa anh den / nep gap bat den troi noi giua long duong.
        """
        hT, wT = mask.shape[:2]
        if gray is not None:
            gray_roi_sub = gray[int(hT * 0.45) : int(hT * 0.95) : 4, ::4]
            q_med, q_p30, q_p99 = utils._uint8_quantiles(gray_roi_sub, (0.50, 0.30, 0.995))
            road_med_gray = float(q_med)
            road_dark_p30 = float(q_p30)
            road_p99_gray = float(q_p99)
            if road_med_gray >= 25.0:
                bot_sub = gray[int(hT * 0.55) : int(hT * 0.96) : 3, int(wT * 0.08) : int(wT * 0.92) : 3]
                if bot_sub.size > 0:
                    dark_frac_bot = float(np.mean(bot_sub <= 155))
                    has_washout_tape_contrast = (
                        (road_p99_gray >= 246.0)
                        and ((road_p99_gray - road_dark_p30) >= 52.0)
                        and (road_dark_p30 <= 198.0)
                    )
                    if dark_frac_bot < 0.08 and road_dark_p30 > 162.0 and not has_washout_tape_contrast:
                        return None, None
        else:
            road_med_gray = None
            road_dark_p30 = None
            road_p99_gray = None
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        mid_x = wT * 0.5
        scan_top_y = max(int(hT * 0.25), min(int(top_y_hint), int(hT * 0.60)))
        bot_y = int(hT * 0.98)
        row_cache = cls._build_row_runs_table(mask, dist)
        seed_visited = np.zeros_like(mask)
        if (
            precomputed_cc is not None
            and len(precomputed_cc) == 2
            and precomputed_cc[0] is not None
            and precomputed_cc[1] is not None
            and precomputed_cc[0].shape == mask.shape
        ):
            cc_labels, cc_stats = precomputed_cc
        else:
            _, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        def _is_valid_trace(pts_try, w_try):
            if len(pts_try) < 5:
                return False
            min_y = float(pts_try[:, 1].min())
            max_y = float(pts_try[:, 1].max())
            span_y = max_y - min_y
            span_x = float(pts_try[:, 0].max() - pts_try[:, 0].min())
            diffs = np.diff(pts_try, axis=0)
            path_len = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
            if path_len < 24.0 or (span_y < 16.0 and span_x < 36.0) or max(span_x, span_y) < 16.0:
                return False
            if min_y >= hT * 0.60 and span_y < 24.0 and span_x < wT * 0.35:
                return False
            # Tren anh thuc te (road_med_gray >= 25): loai bo nep gap bat rat mong va bong loa troi noi giua duong
            if road_med_gray is not None and road_med_gray >= 25.0:
                min_x = float(pts_try[:, 0].min())
                max_x = float(pts_try[:, 0].max())
                touches_bottom = max_y >= hT * 0.67
                touches_border = (min_x <= 26.0) or (max_x >= wT - 27.0)
                is_wide_upper_wall = (touches_border or (span_y >= hT * 0.45 and path_len >= 130.0)) and (
                    span_x >= wT * 0.26
                )
                p_bot_pt = pts_try[int(np.argmax(pts_try[:, 1]))]
                p_top_pt = pts_try[int(np.argmin(pts_try[:, 1]))]
                bx_p, tx_p = float(p_bot_pt[0]), float(p_top_pt[0])
                is_persp_side_exit = (
                    max_y >= hT * 0.55
                    and span_y >= 42.0
                    and (
                        (bx_p <= 18.0 and 32.0 <= (tx_p - bx_p) <= 135.0 and tx_p >= max_x - 8.0)
                        or (bx_p >= wT - 19.0 and 32.0 <= (bx_p - tx_p) <= 135.0 and tx_p <= min_x + 8.0)
                    )
                )
                if not (touches_bottom or is_wide_upper_wall or is_persp_side_exit):
                    return False
                if (not touches_border) and path_len < 52.0 and span_x < 45.0 and span_y < 40.0 and max_y < hT * 0.86:
                    return False
                if len(w_try) > 0:
                    med_w = float(_fast_median_seq(w_try))
                    k_p85 = int(0.85 * (len(w_try) - 1))
                    p85_w = float(np.partition(w_try, k_p85)[k_p85])
                else:
                    med_w = 0.0
                    p85_w = 0.0
                if not touches_border and p85_w < 3.2:
                    return False
                # Loai bo vet loa / vien can xe mong o sat mep duoi khung hinh (y >= 0.76*hT)
                if min_y >= hT * 0.76 and span_y < 52.0 and med_w < 6.0 and not (min_x <= 15.0 or max_x >= wT - 16.0):
                    return False
            return True

        def _analyze_candidate_ground(pts_try):
            """Kiem tra dong thoi mep san nha va dieu kien co nen bat den (ke ca bat bong loa) ben canh vach."""
            if road_med_gray is None or road_med_gray < 25.0 or gray is None or len(pts_try) < 5:
                return False, True
            n_pts = len(pts_try)
            min_x = float(pts_try[:, 0].min())
            max_x = float(pts_try[:, 0].max())
            max_y_tr = float(pts_try[:, 1].max())
            step_idxs = np.linspace(0, n_pts - 1, min(10, n_pts), dtype=int)
            step_hits = 0
            plateau_hits = 0
            ridge_hits = 0
            black_hits = 0
            lower_samples = 0
            lower_black_hits = 0
            step_samples = 0
            road_low = road_dark_p30 if road_dark_p30 is not None else road_med_gray
            is_bright_washout = (
                (road_med_gray >= 150.0)
                and (road_low >= 115.0)
                and (road_p99_gray is not None)
                and (road_p99_gray >= 244.0)
            )
            if is_bright_washout:
                floor_thresh = max(162.0, min(242.0, road_low + 82.0))
                dark_thresh = min(160.0, road_low + 36.0)
                dark_surface_thresh = min(214.0, max(172.0, road_med_gray + 26.0))
                glossy_dark_cap = max(dark_surface_thresh, min(220.0, road_med_gray + 32.0))
                plateau_near_th = max(178.0, road_med_gray + 22.0)
                plateau_side_th = max(182.0, road_med_gray + 26.0)
                below_floor_th = max(178.0, min(232.0, road_med_gray + 42.0))
            else:
                floor_thresh = max(162.0, min(195.0, road_low + 82.0))
                dark_thresh = min(115.0, road_low + 36.0)
                # Thich ung voi bat den bong loa sang (road_low ~ 125-140) ma van loai bo vat troi tren san trang
                dark_surface_thresh = min(172.0, max(118.0, road_low + 42.0))
                glossy_dark_cap = max(dark_surface_thresh, min(168.0, road_med_gray + 58.0))
                plateau_near_th = 178.0
                plateau_side_th = 182.0
                below_floor_th = max(162.0, min(178.0, road_low + 95.0))
            is_border_hugging = (min_x <= 14.0 and max_x <= 72.0) or (max_x >= wT - 15.0 and min_x >= wT - 73.0)
            for idx_s in step_idxs:
                px = float(pts_try[idx_s, 0])
                py = float(pts_try[idx_s, 1])
                if py < 12.0 or py > float(hT - 8):
                    continue
                i0 = max(0, int(idx_s) - 2)
                i1 = min(n_pts - 1, int(idx_s) + 2)
                dx_t = float(pts_try[i1, 0] - pts_try[i0, 0])
                dy_t = float(pts_try[i1, 1] - pts_try[i0, 1])
                ln_t = math.hypot(dx_t, dy_t)
                if ln_t < 1e-3:
                    continue
                nx_t, ny_t = -dy_t / ln_t, dx_t / ln_t
                ixc = int(_clip_scalar(round(px), 0.0, float(wT - 1)))
                iyc = int(_clip_scalar(round(py), 0.0, float(hT - 1)))
                gc = float(gray[iyc, ixc])
                vals_p = []
                vals_n = []
                near_p = []
                near_n = []
                for d_off in (10.0, 14.0, 20.0, 26.0):
                    xp = int(round(px + d_off * nx_t))
                    yp = int(round(py + d_off * ny_t))
                    if 0 <= xp < wT and 0 <= yp < hT:
                        vp = float(gray[yp, xp])
                        vals_p.append(vp)
                        if d_off <= 14.5:
                            near_p.append(vp)
                    xn = int(round(px - d_off * nx_t))
                    yn = int(round(py - d_off * ny_t))
                    if 0 <= xn < wT and 0 <= yn < hT:
                        vn = float(gray[yn, xn])
                        vals_n.append(vn)
                        if d_off <= 14.5:
                            near_n.append(vn)
                if not vals_p and not vals_n:
                    continue
                if vals_p and vals_n:
                    mp = _fast_median_seq(vals_p)
                    mn = _fast_median_seq(vals_n)
                    min_side = mp if mp < mn else mn
                    max_side = mp if mp > mn else mn
                    m_np = _fast_median_seq(near_p) if near_p else mp
                    m_nn = _fast_median_seq(near_n) if near_n else mn
                    max_near = m_np if m_np > m_nn else m_nn
                else:
                    vals_in = vals_p if vals_p else vals_n
                    mp = _fast_median_seq(vals_in)
                    mn = mp
                    min_side = mp
                    max_side = mp
                    max_near = _fast_median_seq(near_p if near_p else (near_n if near_n else vals_in))
                step_samples += 1
                # Bang dinh trang that la 1 song sang cuc bo (ridge) hoac dai bang dinh trang sang ro (>=218)
                # nam tren bat den; TUYET DOI loai bo bac thang bat-san (white_floor_plateau: 1 ben san trang rong >=180,
                # khong co ranh toi giua vach va san nha, vd mep bat sat gach trang / dep trang).
                is_local_ridge = bool(vals_p and vals_n) and ((gc - max_near) >= 22.0)
                if is_local_ridge and min_side <= glossy_dark_cap:
                    ridge_hits += 1
                is_white_floor_plateau = (
                    bool(vals_p and vals_n)
                    and (not is_local_ridge)
                    and (max_near >= plateau_near_th)
                    and (max_side >= plateau_side_th)
                    and (min_side <= glossy_dark_cap)
                    and ((gc - max_near) < 18.0)
                )
                if is_white_floor_plateau:
                    plateau_hits += 1
                is_bright_ribbon = (
                    (gc >= 218.0)
                    and (not is_border_hugging)
                    and (not is_white_floor_plateau)
                    and (min_side <= glossy_dark_cap)
                    and ((gc - min_side) >= 48.0)
                )
                is_bare_step = is_white_floor_plateau or (
                    bool(vals_p and vals_n)
                    and (not is_local_ridge)
                    and (not is_bright_ribbon)
                    and (max_side >= 138.0)
                    and (max_near >= max_side - 20.0)
                    and (min_side <= dark_thresh)
                    and (gc <= max_side + 10.0)
                    and (gc < 218.0 or is_border_hugging)
                )
                has_dark = (not is_bare_step) and (
                    (min_side <= dark_surface_thresh and (gc - min_side) >= 24.0)
                    or (min_side <= 105.0 and gc >= 145.0)
                    or (is_local_ridge and min_side <= glossy_dark_cap and (gc - min_side) >= 52.0)
                    or (gc >= 220.0 and (not is_border_hugging) and min_side <= glossy_dark_cap and (gc - min_side) >= 72.0)
                )
                if has_dark:
                    black_hits += 1
                if py >= float(hT) * 0.52:
                    lower_samples += 1
                    if has_dark:
                        lower_black_hits += 1
                if (
                    vals_p
                    and vals_n
                    and (not is_local_ridge)
                    and (not is_bright_ribbon)
                    and py <= float(hT - 22)
                    and (
                        (mp >= floor_thresh and mp >= gc - 32.0 and mn <= dark_thresh and mn < gc - 35.0 and (mp - mn) >= 65.0)
                        or (mn >= floor_thresh and mn >= gc - 32.0 and mp <= dark_thresh and mp < gc - 35.0 and (mn - mp) >= 65.0)
                        or is_bare_step
                    )
                ):
                    step_hits += 1
            is_step = step_samples >= 3 and (step_hits * 2 >= step_samples or (step_hits >= 3 and min_x <= 24.0))
            if step_samples < 3:
                return is_step, False
            # Neu >= 30% mau la mep bac thang bat den - san trang (white_floor_plateau) va khong co than doc song sang (ridge)
            # -> loai han (khong cuu bang span_x!), van bao ve vach chu L that co than doc dai tren bat den!
            span_y_tr = max_y_tr - float(pts_try[:, 1].min())
            if (plateau_hits >= 3 or (plateau_hits >= 2 and plateau_hits * 3 >= step_samples)) and (
                span_y_tr < 65.0 or ridge_hits <= 2
            ):
                return True, False
            # Loai vat the ngoai duong (dep / chan ban / bao tai / o cam) nam troi o nua tren, phia duoi no la san gach trang
            if max_y_tr < float(hT) * 0.72:
                bx0 = max(0, int(round(min_x)))
                bx1 = min(wT, max(int(round(max_x)) + 1, bx0 + 48))
                by0 = min(190, int(round(max_y_tr)) + 8)
                by1 = min(hT - 2, max(int(round(max_y_tr)) + 44, 196))
                if bx1 > bx0 and by1 > by0 + 6:
                    below_patch = gray[by0:by1:2, bx0:bx1:2]
                    if below_patch.size > 0:
                        p70_below = float(np.percentile(below_patch, 70))
                        if p70_below >= below_floor_th:
                            return True, False
            has_black_support = (black_hits * 2 >= step_samples)
            if max_y_tr >= float(hT) * 0.58 and lower_samples >= 2 and (lower_black_hits * 2 < lower_samples):
                has_black_support = False
            return is_step, has_black_support

        def _stem_x(pts_arr):
            n_low = max(1, min(len(pts_arr), 12))
            return _fast_median_seq(pts_arr[:n_low, 0].tolist())

        def _path_len(pts_arr):
            if len(pts_arr) < 2:
                return 0.0
            df = np.diff(pts_arr, axis=0)
            return float(np.hypot(df[:, 0], df[:, 1]).sum())

        def _overlap_ratio(p1, p2):
            if float(p1[:, 0].min()) > float(p2[:, 0].max()) + 12.0 or float(p1[:, 0].max()) < float(p2[:, 0].min()) - 12.0:
                return 0.0
            if float(p1[:, 1].min()) > float(p2[:, 1].max()) + 12.0 or float(p1[:, 1].max()) < float(p2[:, 1].min()) - 12.0:
                return 0.0
            s1 = p1[:: max(1, len(p1) // 24)]
            s2 = p2[:: max(1, len(p2) // 24)]
            dx = s1[:, None, 0] - s2[None, :, 0]
            dy = s1[:, None, 1] - s2[None, :, 1]
            d2 = dx * dx + dy * dy
            ov1 = float(np.count_nonzero(d2.min(axis=1) < 144.0)) / float(len(s1))
            ov2 = float(np.count_nonzero(d2.min(axis=0) < 144.0)) / float(len(s2))
            return ov1 if ov1 >= ov2 else ov2

        def _add_candidate(pts_try, w_try, candidates, used_mask):
            if not _is_valid_trace(pts_try, w_try):
                if len(pts_try) >= 2:
                    cv2.polylines(seed_visited, [np.round(pts_try).astype(np.int32)], False, 255, 6)
                return False
            new_len = _path_len(pts_try)
            overlap_idxs = []
            for idx_c, (c_pts, _) in enumerate(candidates):
                if _overlap_ratio(pts_try, c_pts) >= 0.25:
                    overlap_idxs.append(idx_c)
            if overlap_idxs:
                if len(overlap_idxs) == 1:
                    c_old_pts, _ = candidates[overlap_idxs[0]]
                    old_len = _path_len(c_old_pts)
                    old_span_x = float(c_old_pts[:, 0].max() - c_old_pts[:, 0].min())
                    new_span_x = float(pts_try[:, 0].max() - pts_try[:, 0].min())
                    if new_len > old_len * 1.45 or (
                        new_span_x >= float(wT) * 0.62 and old_span_x < float(wT) * 0.58 and new_len > old_len * 1.15
                    ):
                        candidates[overlap_idxs[0]] = (pts_try, w_try)
                        cv2.polylines(used_mask, [np.round(pts_try).astype(np.int32)], False, 255, 22)
                        return True
                cv2.polylines(seed_visited, [np.round(pts_try).astype(np.int32)], False, 255, 6)
                return False
            candidates.append((pts_try, w_try))
            cv2.polylines(used_mask, [np.round(pts_try).astype(np.int32)], False, 255, 22)
            return True

        used_mask = np.zeros_like(mask)
        candidates = []
        bad_cids = set()

        # 1. Quet tu duoi len de bat cac than vach doc (vertical/diagonal stems)
        for y in range(bot_y, scan_top_y, -4):
            runs = row_cache[y]
            if not runs:
                continue
            runs_up2 = row_cache[y - 2] if y >= 2 else ()
            runs_up4 = row_cache[y - 4] if y >= 4 else ()
            for cx, perp_w, s, e, rw in sorted(runs, key=lambda r: abs(r[0] - mid_x)):
                icx = int(round(cx))
                if icx < 0:
                    icx = 0
                elif icx > wT - 1:
                    icx = wT - 1
                if used_mask[y, icx] > 0 or seed_visited[y, icx] > 0:
                    continue
                cid = int(cc_labels[y, icx])
                if cid == 0 or cid in bad_cids or (
                    int(cc_stats[cid, cv2.CC_STAT_HEIGHT]) < 8
                    and int(cc_stats[cid, cv2.CC_STAT_WIDTH]) < 14
                    and int(cc_stats[cid, cv2.CC_STAT_AREA]) < 18
                ):
                    seed_visited[y, icx] = 255
                    continue
                if rw < 12.0 and s > 2 and e < wT - 2:
                    has_up = False
                    for r_u in runs_up2:
                        if abs(r_u[0] - cx) <= 14.0:
                            has_up = True
                            break
                    if not has_up:
                        for r_u in runs_up4:
                            if abs(r_u[0] - cx) <= 16.0:
                                has_up = True
                                break
                    if not has_up:
                        seed_visited[y, icx] = 255
                        continue
                side_guess = "LEFT" if cx <= mid_x else "RIGHT"
                pts_try, w_try = cls._trace_stem_and_turn(
                    mask, dist, gray, road_med_gray, cx, y, side_guess, top_limit_y=4, row_cache=row_cache
                )
                if not _add_candidate(pts_try, w_try, candidates, used_mask):
                    seed_visited[max(0, y - 2) : min(hT, y + 3), max(0, icx - 4) : min(wT, icx + 5)] = 255
                    if int(cc_stats[cid, cv2.CC_STAT_WIDTH]) < int(wT * 0.38) and int(cc_stats[cid, cv2.CC_STAT_HEIGHT]) < int(hT * 0.42):
                        bad_cids.add(cid)
                if len(candidates) >= 6:
                    break
            if len(candidates) >= 6:
                break

        # 2. Quet doc theo 2 mep bien Trai (x=4..10) & Phai (x=wT-11..wT-5) tu duoi len de bat vach ngoai khi xe dang o giua goc cua
        has_l_full = any(_stem_x(c[0]) <= mid_x and (c[0][:, 1].max() - c[0][:, 1].min()) >= hT * 0.42 for c in candidates)
        has_r_full = any(_stem_x(c[0]) > mid_x and (c[0][:, 1].max() - c[0][:, 1].min()) >= hT * 0.42 for c in candidates)
        if not (has_l_full and has_r_full):
            border_y_lo = int(hT * 0.24)
            border_y_hi = int(hT * 0.80)
            for x_b, side_b in ((4, "LEFT"), (10, "LEFT"), (wT - 5, "RIGHT"), (wT - 11, "RIGHT")):
                col_idx = np.flatnonzero(mask[border_y_lo:border_y_hi, x_b])
                if col_idx.size == 0:
                    continue
                c_diffs = np.flatnonzero(np.diff(col_idx) > 1)
                if c_diffs.size == 0:
                    c_spans = [(int(col_idx[0]), int(col_idx[-1]) + 1)]
                else:
                    c_spans = []
                    cp = 0
                    for cd in c_diffs:
                        cdi = int(cd)
                        c_spans.append((int(col_idx[cp]), int(col_idx[cdi]) + 1))
                        cp = cdi + 1
                    c_spans.append((int(col_idx[cp]), int(col_idx[-1]) + 1))
                for g0, g1 in reversed(c_spans):
                    sy = border_y_lo + g0
                    ey = border_y_lo + g1
                    cw = ey - sy
                    cy = int(round(0.5 * (sy + ey - 1)))
                    if 2 <= cw <= 28 and used_mask[cy, x_b] == 0 and seed_visited[cy, x_b] == 0:
                        cid_b = int(cc_labels[cy, x_b])
                        if cid_b == 0 or cid_b in bad_cids:
                            continue
                        pts_try, w_try = cls._trace_stem_and_turn(
                            mask, dist, gray, road_med_gray, float(x_b), float(cy), side_b, top_limit_y=4, row_cache=row_cache
                        )
                        added = _add_candidate(pts_try, w_try, candidates, used_mask)
                        if not added:
                            # Thu quet ngang theo cot khi thanh vach ngoai nam ngang xuyen qua khung hinh (mid-corner)
                            pts_h, w_h = cls._trace_horizontal_bar(
                                mask, dist, gray, road_med_gray, float(x_b), float(cy), dir_x=(1 if x_b <= 15 else -1)
                            )
                            added = _add_candidate(pts_h, w_h, candidates, used_mask)
                        if not added:
                            seed_visited[max(0, cy - 2) : min(hT, cy + 3), max(0, x_b - 3) : min(wT, x_b + 4)] = 255
                            if int(cc_stats[cid_b, cv2.CC_STAT_WIDTH]) < int(wT * 0.38) and int(cc_stats[cid_b, cv2.CC_STAT_HEIGHT]) < int(hT * 0.42):
                                bad_cids.add(cid_b)

        # 3. Du phong neu chua du 2 vach hop le: quet cac cot giua anh de bat vach ngoai nam ngang giua cua (ke ca sat can xe)
        need_mid_scan = len(candidates) < 2
        if need_mid_scan:
            mid_y_lo = int(hT * 0.20) if (road_med_gray is not None and road_med_gray >= 25.0) else 12
            mid_y_hi = int(hT * 0.93)
            for x_m in (wT // 2, wT // 4, (3 * wT) // 4):
                col_idx = np.flatnonzero(mask[mid_y_lo:mid_y_hi, x_m])
                if col_idx.size == 0:
                    continue
                c_diffs = np.flatnonzero(np.diff(col_idx) > 1)
                if c_diffs.size == 0:
                    c_spans = [(int(col_idx[0]), int(col_idx[-1]) + 1)]
                else:
                    c_spans = []
                    cp = 0
                    for cd in c_diffs:
                        cdi = int(cd)
                        c_spans.append((int(col_idx[cp]), int(col_idx[cdi]) + 1))
                        cp = cdi + 1
                    c_spans.append((int(col_idx[cp]), int(col_idx[-1]) + 1))
                for g0, g1 in reversed(c_spans):
                    sy = mid_y_lo + g0
                    ey = mid_y_lo + g1
                    cw = ey - sy
                    cy = int(round(0.5 * (sy + ey - 1)))
                    if 3 <= cw <= 28 and used_mask[cy, x_m] == 0 and seed_visited[cy, x_m] == 0:
                        pts_h, w_h = cls._trace_horizontal_bar(
                            mask, dist, gray, road_med_gray, float(x_m), float(cy), dir_x=0
                        )
                        if len(pts_h) >= 8 and float(pts_h[:, 0].max() - pts_h[:, 0].min()) >= wT * 0.25:
                            _add_candidate(pts_h, w_h, candidates, used_mask)

        if not candidates:
            return None, None

        if road_med_gray is not None and road_med_gray >= 25.0 and gray is not None:
            ground_info = [_analyze_candidate_ground(p) for p, _ in candidates]
            candidates = [c for c, (_, has_blk) in zip(candidates, ground_info) if has_blk]
            step_flags = [st for st, has_blk in ground_info if has_blk]
            if any(step_flags):
                non_step = [c for c, st in zip(candidates, step_flags) if not st]
                kept = []
                for (p_c, w_c), st in zip(candidates, step_flags):
                    if not st:
                        kept.append((p_c, w_c))
                        continue
                    span_xc = float(p_c[:, 0].max() - p_c[:, 0].min())
                    if span_xc < float(wT) * 0.48:
                        continue
                    sx_c = _stem_x(p_c)
                    has_inner = any(
                        30.0 <= abs(_stem_x(pc) - sx_c) <= 165.0
                        and _path_len(pc) >= 110.0
                        and float(pc[:, 1].max()) >= hT * 0.55
                        for pc, _ in non_step
                    )
                    if not has_inner:
                        kept.append((p_c, w_c))
                candidates = kept

        if not candidates:
            return None, None

        def _classify_single_candidate(cand_pts, cand_w):
            # Dam bao chieu diem luon tu duoi len tren (y giam dan) de goc va phan loai vach luon dong nhat
            if len(cand_pts) >= 2 and float(cand_pts[0, 1]) < float(cand_pts[-1, 1]) - 6.0:
                cand_pts = cand_pts[::-1].copy()
                cand_w = cand_w[::-1].copy()
            dx_up = float(cand_pts[-1, 0] - cand_pts[0, 0])
            dy_up = float(cand_pts[0, 1] - cand_pts[-1, 1])
            # Vach ngoai quet cheo manh (vd: tu duoi-phai len tren-phai khi xe da sat tuong ngoai):
            # phan loai theo chieu nghieng (len-phai = vach TRAI cua cua PHAI; len-trai = vach PHAI cua cua TRAI)
            if abs(dx_up) >= 68.0 and dy_up >= 8.0 and abs(dx_up) >= 1.25 * max(1.0, dy_up):
                if dx_up > 0.0:
                    return (cand_pts, cand_w), None
                return None, (cand_pts, cand_w)
            if _stem_x(cand_pts) <= mid_x:
                return (cand_pts, cand_w), None
            return None, (cand_pts, cand_w)

        if len(candidates) == 1:
            return _classify_single_candidate(*candidates[0])

        def _pair_perp_gap(p1, p2):
            s1 = p1[:: max(1, len(p1) // 12)]
            s2 = p2[:: max(1, len(p2) // 12)]
            dx = s1[:, None, 0] - s2[None, :, 0]
            dy = s1[:, None, 1] - s2[None, :, 1]
            d_mat = np.sqrt(dx * dx + dy * dy)
            m1 = d_mat.min(axis=1)
            m0 = d_mat.min(axis=0)
            k1 = len(m1) >> 1
            k0 = len(m0) >> 1
            med1 = float(np.partition(m1, k1)[k1])
            med0 = float(np.partition(m0, k0)[k0])
            return med1 if med1 < med0 else med0

        # Sap xep cac candidate tu Trai sang Phai theo vi tri chan vach (stem_x)
        candidates.sort(key=lambda item: _stem_x(item[0]))

        best_pair = None
        best_score = -1e9
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                lp, lw = candidates[i]
                rp, rw = candidates[j]
                xl = _stem_x(lp)
                xr = _stem_x(rp)
                dx_stem = xr - xl
                straddles_center = (xl <= mid_x - 16.0) and (xr >= mid_x + 16.0)
                max_lane_w_limit = (
                    float(wT) * 0.98
                    if (lane_width_ab is not None or straddles_center)
                    else min(float(wT) * 0.60, 285.0)
                )
                if dx_stem < 24.0 or dx_stem > max_lane_w_limit:
                    continue
                if _overlap_ratio(lp, rp) >= 0.22:
                    continue
                perp_gap = _pair_perp_gap(lp, rp)
                if perp_gap < max(16.0, float(wT) * 0.045):
                    continue
                ll = _path_len(lp)
                rl = _path_len(rp)
                if road_med_gray is not None and road_med_gray >= 25.0:
                    if ll < 48.0 and rl < 48.0:
                        continue
                    y_bot_pair = max(float(lp[:, 1].max()), float(rp[:, 1].max()))
                    min_dx_plausible = max(96.0, float(wT) * 0.22) if y_bot_pair >= hT * 0.58 else max(64.0, float(wT) * 0.14)
                    if dx_stem < min_dx_plausible or perp_gap < max(46.0, float(wT) * 0.11):
                        continue
                    if lane_width_ab is not None:
                        y_ref_pair = float(min(lp[0, 1], rp[0, 1]))
                        exp_w_pair = max(80.0, float(lane_width_ab[0] * y_ref_pair + lane_width_ab[1]))
                        max_w_plausible = (
                            max_lane_w_limit
                            if (straddles_center and ll >= 65.0 and rl >= 65.0)
                            else min(max_lane_w_limit, max(140.0, 1.35 * exp_w_pair))
                        )
                        same_half_only = (
                            max(float(lp[:, 0].max()), float(rp[:, 0].max())) < mid_x - 16.0
                            or min(float(lp[:, 0].min()), float(rp[:, 0].min())) > mid_x + 16.0
                        )
                        if (
                            same_half_only
                            or (
                                dx_stem < 0.56 * exp_w_pair
                                and (max(xl, xr) < mid_x - 16.0 or min(xl, xr) > mid_x + 16.0)
                            )
                            or (
                                dx_stem < 0.42 * exp_w_pair
                                and (ll < 170.0 or rl < 170.0)
                            )
                            or dx_stem > max_w_plausible
                        ):
                            continue
                if road_med_gray is not None and road_med_gray >= 25.0 and len(lw) > 0 and len(rw) > 0:
                    med_lw = _fast_median_seq(lw.tolist())
                    med_rw = _fast_median_seq(rw.tolist())
                    if (
                        (med_lw < 0.42 * med_rw and ll < 0.65 * rl and rl >= 100.0)
                        or (med_rw < 0.42 * med_lw and rl < 0.65 * ll and ll >= 100.0)
                    ):
                        continue
                    l_bot_only = (float(lp[:, 1].min()) >= hT * 0.72) and (float(lp[:, 1].max() - lp[:, 1].min()) < 56.0)
                    r_bot_only = (float(rp[:, 1].min()) >= hT * 0.72) and (float(rp[:, 1].max() - rp[:, 1].min()) < 56.0)
                    if l_bot_only and rl >= 90.0 and med_rw >= 8.0 and (med_lw < 0.50 * med_rw or xl > mid_x + 24.0):
                        continue
                    if r_bot_only and ll >= 90.0 and med_lw >= 8.0 and (med_rw < 0.50 * med_lw or xr < mid_x - 24.0):
                        continue
                # Khong ghep 1 vach dai hoan chinh voi 1 doan dut ngan troi noi o mep tren (vd: khi 1 ben bi che)
                l_top_stub = (ll < 78.0) and (float(lp[:, 1].max()) < hT * 0.52) and (float(lp[:, 0].max() - lp[:, 0].min()) < wT * 0.18) and (float(lp[:, 1].max() - lp[:, 1].min()) < hT * 0.18)
                r_top_stub = (rl < 78.0) and (float(rp[:, 1].max()) < hT * 0.52) and (float(rp[:, 0].max() - rp[:, 0].min()) < wT * 0.18) and (float(rp[:, 1].max() - rp[:, 1].min()) < hT * 0.18)
                if (ll >= 140.0 and r_top_stub) or (rl >= 140.0 and l_top_stub):
                    continue
                # Khong ghep 1 vach dai voi 1 mau vach rat ngan vua cham goc duoi-trai / duoi-phai (gay chop tat TWO_EDGES <-> ONE_EDGE)
                l_bot_corner_stub = (
                    (ll < 78.0 and float(lp[:, 1].min()) >= hT * 0.74 and float(lp[:, 0].max()) <= 72.0)
                    or (
                        ll < 135.0
                        and float(lp[:, 1].min()) >= hT * 0.58
                        and float(lp[:, 0].max()) <= 108.0
                        and float(lp[-1, 0] - lp[0, 0]) < -15.0
                        and float(rp[-1, 0] - rp[0, 0]) < -140.0
                    )
                )
                r_bot_corner_stub = (
                    (rl < 78.0 and float(rp[:, 1].min()) >= hT * 0.74 and float(rp[:, 0].min()) >= wT - 72.0)
                    or (
                        rl < 135.0
                        and float(rp[:, 1].min()) >= hT * 0.58
                        and float(rp[:, 0].min()) >= wT - 108.0
                        and float(rp[-1, 0] - rp[0, 0]) > 15.0
                        and float(lp[-1, 0] - lp[0, 0]) > 140.0
                    )
                )
                if (ll >= 62.0 and r_bot_corner_stub) or (rl >= 62.0 and l_bot_corner_stub):
                    continue
                # Khong ghep 1 vach chinh dai cham day voi dom loa/nep nhan troi noi lung chung o nua tren-giua bat
                l_mid_float_glare = (ll < 150.0) and (float(lp[:, 1].max()) < hT * 0.79) and (float(lp[:, 1].min()) > 55.0)
                r_mid_float_glare = (rl < 150.0) and (float(rp[:, 1].max()) < hT * 0.79) and (float(rp[:, 1].min()) > 55.0)
                if (ll >= 165.0 and float(lp[:, 1].max()) >= hT * 0.82 and r_mid_float_glare) or (
                    rl >= 165.0 and float(rp[:, 1].max()) >= hT * 0.82 and l_mid_float_glare
                ):
                    continue
                len_score = ll + rl
                center_dist = abs(0.5 * (xl + xr) - mid_x)
                # Uu tien cap vach dai, ro net va nam bao quanh hoac gan khu vuc duong chay
                pair_score = len_score - 0.35 * center_dist
                if xl <= mid_x + 15.0 and xr >= mid_x - 15.0:
                    pair_score += 45.0
                # Uu tien vach cham day khung hinh (chong dom loa troi noi giua duong chiem mat vach that)
                l_max_y = float(lp[:, 1].max())
                r_max_y = float(rp[:, 1].max())
                if l_max_y >= hT * 0.85:
                    pair_score += 38.0
                elif l_max_y < hT * 0.74:
                    pair_score -= 32.0
                if r_max_y >= hT * 0.85:
                    pair_score += 38.0
                elif r_max_y < hT * 0.74:
                    pair_score -= 32.0
                # Khuyen khich do rong lan khop voi mo hinh da hieu chuan W_lane(y)
                if lane_width_ab is not None:
                    y_ref_sc = float(min(lp[0, 1], rp[0, 1]))
                    exp_w_sc = max(80.0, float(lane_width_ab[0] * y_ref_sc + lane_width_ab[1]))
                    w_err_frac = abs(dx_stem - exp_w_sc) / exp_w_sc
                    if w_err_frac <= 0.18:
                        pair_score += 65.0 * (1.0 - w_err_frac / 0.18)
                    elif w_err_frac >= 0.32:
                        pair_score -= min(110.0, 140.0 * (w_err_frac - 0.32))
                # Thuong tinh lien tuc theo thoi gian (Temporal continuity) voi cap vach frame truoc
                if prev_left_segs:
                    pl_y0 = float(lp[0, 1])
                    prev_lx = cls._eval_segments_x_at_y(prev_left_segs, pl_y0)
                    if prev_lx is not None:
                        dl_t = abs(xl - prev_lx)
                        if dl_t <= 42.0:
                            pair_score += 75.0 * (1.0 - dl_t / 42.0)
                        elif dl_t >= 80.0:
                            pair_score -= min(70.0, 0.7 * (dl_t - 80.0))
                if prev_right_segs:
                    pr_y0 = float(rp[0, 1])
                    prev_rx = cls._eval_segments_x_at_y(prev_right_segs, pr_y0)
                    if prev_rx is not None:
                        dr_t = abs(xr - prev_rx)
                        if dr_t <= 42.0:
                            pair_score += 75.0 * (1.0 - dr_t / 42.0)
                        elif dr_t >= 80.0:
                            pair_score -= min(70.0, 0.7 * (dr_t - 80.0))
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = ((lp, lw), (rp, rw))

        if best_pair is not None:
            return best_pair[0], best_pair[1]

        # Neu khong co cap nao thoa man, chon vach dai nhat
        best_single = max(candidates, key=lambda item: _path_len(item[0]))
        return _classify_single_candidate(*best_single)

    @classmethod
    def _detect_tarp_virtual_edge(cls, gray, side="LEFT", road_med_gray=None, x_exclude=None, lane_width_ab=None):
        """
        Phat hien Mep Bat Den (Virtual Tarp Edge) lam vach bien ao khi vach trang bi khuat hoac khong co:
        - Chi chap nhan ranh gioi ro rang giua bat den (<= 100) va san nha sang dong deu (>= 162, tuong phan >= 58).
        - Loai bo triet de bong loa anh den tren bat den bang kiem tra do sang dong deu ra ngoai mep va
          kiem tra huong hoi tu phoi canh + khoang cach hop ly so voi vach that doi dien.
        """
        if gray is None or gray.size == 0:
            return None, None
        hT, wT = gray.shape[:2]
        mid_x = wT * 0.5
        if road_med_gray is not None and road_med_gray > 125.0:
            # Bat den dang bi loa sang manh -> khong doan mep bat ao de tranh nham bong loa
            return None, None
        dark_th = 100.0
        bright_th = 162.0
        min_contrast = 58.0

        raw_row_cands = []
        y_start = int(hT * 0.90)
        y_end = int(hT * 0.35)

        for y in range(y_start, y_end, -6):
            row = gray[y, :].astype(np.float32)
            row_sm = cv2.GaussianBlur(row.reshape(1, -1), (15, 1), 3.0)[0]
            grad = np.diff(row_sm)

            cands_y = []
            if side == "LEFT":
                x_max_scan = int(min(mid_x - 18.0, (x_exclude[0] - 48.0) if x_exclude is not None else (mid_x - 18.0)))
                x_max_scan = min(wT - 24, max(20, x_max_scan))
                for x in range(18, x_max_scan):
                    xl = max(0, x - 14)
                    xl_far = max(0, x - 30)
                    xr = min(wT - 1, x + 14)
                    gl = row_sm[xl]
                    gl_far = row_sm[xl_far]
                    gr = row_sm[xr]
                    if gl >= bright_th and gl_far >= bright_th - 10.0 and gr <= dark_th and (gl - gr) >= min_contrast:
                        cands_y.append((grad[x], x))
                if cands_y:
                    cands_y.sort(key=lambda it: it[0])  # Diem giam sau nhat
                    raw_row_cands.append((y, cands_y[0][1]))
            elif side == "RIGHT":
                x_min_scan = int(max(mid_x + 18.0, (x_exclude[1] + 48.0) if x_exclude is not None else (mid_x + 18.0)))
                x_min_scan = max(24, min(wT - 20, x_min_scan))
                for x in range(x_min_scan, wT - 18):
                    xl = max(0, x - 14)
                    xr = min(wT - 1, x + 14)
                    xr_far = min(wT - 1, x + 30)
                    gl = row_sm[xl]
                    gr = row_sm[xr]
                    gr_far = row_sm[xr_far]
                    if gl <= dark_th and gr >= bright_th and gr_far >= bright_th - 10.0 and (gr - gl) >= min_contrast:
                        cands_y.append((grad[x], x))
                if cands_y:
                    cands_y.sort(key=lambda it: -it[0])  # Diem tang manh nhat
                    raw_row_cands.append((y, cands_y[0][1]))

        if len(raw_row_cands) < 5:
            return None, None

        raw_row_cands.sort(key=lambda item: -item[0])  # Duoi len tren
        best_inliers = []
        n_pts = len(raw_row_cands)
        rng = np.random.RandomState(42)
        for _ in range(35):
            idx1, idx2 = rng.choice(n_pts, 2, replace=False)
            y1, x1 = raw_row_cands[idx1]
            y2, x2 = raw_row_cands[idx2]
            if abs(y1 - y2) < 24:
                continue
            m = (x2 - x1) / float(y2 - y1)
            # Kiem tra huong nghieng phoi canh hop ly (trai nghieng phai khi len tren m < -0.10, phai nghieng trai m > +0.10)
            if side == "LEFT" and (m > -0.05 or m < -2.2):
                continue
            if side == "RIGHT" and (m < 0.05 or m > 2.2):
                continue
            c = x1 - m * y1
            inliers = []
            for y_i, x_i in raw_row_cands:
                x_pred = m * y_i + c
                if abs(x_i - x_pred) <= 18.0:
                    inliers.append((x_i, y_i))
            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        if len(best_inliers) < 5:
            return None, None

        best_inliers.sort(key=lambda p: -p[1])
        pts_arr = np.asarray(best_inliers, dtype=np.float32)
        if float(pts_arr[0, 1] - pts_arr[-1, 1]) < hT * 0.18:
            return None, None
        # Neu co mo hinh be rong lan + vach doi dien, kiem tra khoang cach khop trong [0.72, 1.35] * W_lane
        if lane_width_ab is not None and x_exclude is not None:
            y_mid_t = float(pts_arr[0, 1])
            exp_w = max(80.0, float(lane_width_ab[0] * y_mid_t + lane_width_ab[1]))
            if side == "LEFT":
                gap_t = float(x_exclude[0]) - float(pts_arr[0, 0])
            else:
                gap_t = float(pts_arr[0, 0]) - float(x_exclude[1])
            if gap_t < 0.72 * exp_w or gap_t > 1.38 * exp_w:
                return None, None
        w_arr = np.full(len(pts_arr), 8.0, dtype=np.float32)
        return pts_arr, w_arr

    @staticmethod
    def _fit_line_tls(pts):
        """Fit a straight line to (N, 2) points via closed-form 2x2 Total Least Squares."""
        n = len(pts)
        if n < 2:
            return pts[0].astype(np.float32), pts[-1].astype(np.float32), 0.0, np.zeros(n, dtype=np.float32)
        s = pts.sum(axis=0)
        mx = float(s[0]) / n
        my = float(s[1]) / n
        cx = pts[:, 0] - mx
        cy = pts[:, 1] - my
        cxx = float(np.dot(cx, cx))
        cyy = float(np.dot(cy, cy))
        cxy = float(np.dot(cx, cy))
        ang = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
        dx_d = math.cos(ang)
        dy_d = math.sin(ang)
        chord_x = float(pts[-1, 0] - pts[0, 0])
        chord_y = float(pts[-1, 1] - pts[0, 1])
        if chord_x * dx_d + chord_y * dy_d < 0.0:
            dx_d = -dx_d
            dy_d = -dy_d
        dists = np.abs(cx * (-dy_d) + cy * dx_d)
        t0 = float(cx[0] * dx_d + cy[0] * dy_d)
        t1 = float(cx[-1] * dx_d + cy[-1] * dy_d)
        p_start = np.array([mx + t0 * dx_d, my + t0 * dy_d], dtype=np.float32)
        p_end = np.array([mx + t1 * dx_d, my + t1 * dy_d], dtype=np.float32)
        return p_start, p_end, float(dists.max()), dists

    @classmethod
    def _fit_piecewise_segments(cls, pts, widths, min_seg_len=32.0, straight_tol=6.0, min_turn_deg=15.5, max_segments=4):
        """
        Approximate an observed white tape boundary with 1 to max_segments (4) connected straight segments:
          - Uses 1 single segment for a straight line.
          - Only adds segments when direction changes clearly or a corner appears (max 4 segments).
          - Never creates tiny segments (< min_seg_len) and never extends outside the observed span.
        """
        n = len(pts)
        if n < 2:
            return []

        ys = pts[:, 1]
        ws = np.asarray(widths, dtype=np.float32)
        if n >= 5 and float(ys.max() - ys.min()) >= 12.0:
            ym = float(ys.sum()) / n
            wm = float(ws.sum()) / n
            dy = ys - ym
            var_y = float(np.dot(dy, dy))
            slope_w = max(0.0, float(np.dot(dy, ws - wm) / var_y)) if var_y > 1e-6 else 0.0
            intercept_w = float(_fast_median_seq(ws - slope_w * ys))
            w_persp = np.clip(slope_w * ys + intercept_w, 2.0, 28.0)
            ws_smooth = 0.65 * w_persp + 0.35 * ws
        else:
            ws_smooth = np.clip(ws, 2.0, 28.0)

        mean_w = float(_fast_median_seq(ws_smooth))
        tol = max(straight_tol, 0.54 * mean_w)

        p0_fit, p1_fit, max_d, dists = cls._fit_line_tls(pts)
        total_len = math.hypot(float(p1_fit[0] - p0_fit[0]), float(p1_fit[1] - p0_fit[1]))
        k_p92 = int(0.92 * (len(dists) - 1))
        p92_d = float(np.partition(dists, k_p92)[k_p92])

        mid_idx = n // 2
        v1x, v1y = float(pts[mid_idx, 0] - pts[0, 0]), float(pts[mid_idx, 1] - pts[0, 1])
        v2x, v2y = float(pts[-1, 0] - pts[mid_idx, 0]), float(pts[-1, 1] - pts[mid_idx, 1])
        n1, n2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
        half_turn_deg = 0.0
        if n1 > 6.0 and n2 > 6.0:
            cos_a = _clip_scalar((v1x * v2x + v1y * v2y) / (n1 * n2), -1.0, 1.0)
            half_turn_deg = math.degrees(math.acos(cos_a))

        if total_len < 2.0 * min_seg_len or (p92_d <= tol and max_d <= tol * 1.35 and half_turn_deg < min_turn_deg):
            return [{
                "p1": (float(p0_fit[0]), float(p0_fit[1])),
                "p2": (float(p1_fit[0]), float(p1_fit[1])),
                "width_start": float(ws_smooth[0]),
                "width_end": float(ws_smooth[-1]),
                "width": mean_w,
                "length": total_len,
            }]

        knots = [0, n - 1]
        min_turn_req = max(11.0, min_turn_deg - 3.0)
        while len(knots) - 1 < max_segments:
            best_split = None
            best_score = -1.0

            for s_idx in range(len(knots) - 1):
                ia, ib = knots[s_idx], knots[s_idx + 1]
                if ib - ia < 6:
                    continue
                sub_pts = pts[ia : ib + 1]
                _, _, sub_max_d, sub_dists = cls._fit_line_tls(sub_pts)
                k_sub = int(0.92 * (len(sub_dists) - 1))
                if float(np.partition(sub_dists, k_sub)[k_sub]) <= tol and sub_max_d <= tol * 1.35:
                    continue
                ch_x = float(pts[ib, 0] - pts[ia, 0])
                ch_y = float(pts[ib, 1] - pts[ia, 1])
                chord_len = math.hypot(ch_x, ch_y)
                if chord_len < 2.0 * min_seg_len:
                    continue
                inv_ch = 1.0 / max(1e-5, chord_len)
                nx_ch = -ch_y * inv_ch
                ny_ch = ch_x * inv_ch

                sub_k = pts[ia + 3 : ib - 2]
                if len(sub_k) == 0:
                    continue
                v_l = sub_k - pts[ia]
                v_r = pts[ib] - sub_k
                l_l = np.hypot(v_l[:, 0], v_l[:, 1])
                l_r = np.hypot(v_r[:, 0], v_r[:, 1])
                d_perp = np.abs(v_l[:, 0] * nx_ch + v_l[:, 1] * ny_ch)
                cos_t = np.clip((v_l[:, 0] * v_r[:, 0] + v_l[:, 1] * v_r[:, 1]) / np.maximum(1e-6, l_l * l_r), -1.0, 1.0)
                turn_d = np.degrees(np.arccos(cos_t))
                valid = (l_l >= min_seg_len) & (l_r >= min_seg_len) & (d_perp > tol) & (turn_d >= min_turn_req)
                if np.any(valid):
                    scores = np.where(valid, d_perp * (1.0 + 0.02 * turn_d), -1.0)
                    loc_i = int(np.argmax(scores))
                    sc = float(scores[loc_i])
                    if sc > best_score:
                        best_score = sc
                        best_split = ia + 3 + loc_i

            if best_split is None:
                break
            knots.append(best_split)
            knots.sort()

        merged_changed = True
        while merged_changed and len(knots) > 2:
            merged_changed = False
            for j in range(1, len(knots) - 1):
                ia, im, ib = knots[j - 1], knots[j], knots[j + 1]
                pa, pm, _, _ = cls._fit_line_tls(pts[ia : im + 1])
                pm2, pb, _, _ = cls._fit_line_tls(pts[im : ib + 1])
                _, _, comb_max_d, comb_dists = cls._fit_line_tls(pts[ia : ib + 1])
                k_p92 = int(0.92 * (len(comb_dists) - 1))
                comb_p92 = float(np.partition(comb_dists, k_p92)[k_p92])
                v1 = pm - pa
                v2 = pb - pm2
                l1 = math.hypot(float(v1[0]), float(v1[1]))
                l2 = math.hypot(float(v2[0]), float(v2[1]))
                if l1 < 18.0 or l2 < 18.0:
                    knots.pop(j)
                    merged_changed = True
                    break
                dot12 = float(v1[0] * v2[0] + v1[1] * v2[1])
                cos_t = _clip_scalar(dot12 / (l1 * l2), -1.0, 1.0)
                ang_deg = math.degrees(math.acos(cos_t))
                if (ang_deg < min_turn_deg and comb_p92 <= tol * 1.15) or (comb_p92 <= tol and comb_max_d <= tol * 1.35):
                    knots.pop(j)
                    merged_changed = True
                    break

        raw_lines = []
        for j in range(len(knots) - 1):
            ia, ib = knots[j], knots[j + 1]
            pa, pb, _, _ = cls._fit_line_tls(pts[ia : ib + 1])
            raw_lines.append((pa, pb, ia, ib))

        vertices = [raw_lines[0][0].copy()]
        for j in range(len(raw_lines) - 1):
            p1, p2, _, ib = raw_lines[j]
            p3, p4, _, _ = raw_lines[j + 1]
            d1 = p2 - p1
            d2 = p4 - p3
            cross = float(d1[0] * d2[1] - d1[1] * d2[0])
            anchor = pts[ib]
            if abs(cross) > 1e-3:
                t = float(((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross)
                inter = p1 + t * d1
                if math.hypot(float(inter[0] - anchor[0]), float(inter[1] - anchor[1])) <= 10.0:
                    vertices.append(inter.astype(np.float32))
                else:
                    vertices.append((0.5 * (p2 + p3)).astype(np.float32))
            else:
                vertices.append((0.5 * (p2 + p3)).astype(np.float32))
        vertices.append(raw_lines[-1][1].copy())

        segments = []
        for j in range(len(raw_lines)):
            _, _, ia, ib = raw_lines[j]
            v_start = vertices[j]
            v_end = vertices[j + 1]
            seg_len = math.hypot(float(v_end[0] - v_start[0]), float(v_end[1] - v_start[1]))
            w_s = float(ws_smooth[ia])
            w_e = float(ws_smooth[ib])
            w_m = float(_fast_median_seq(ws_smooth[ia : ib + 1]))
            segments.append({
                "p1": (float(v_start[0]), float(v_start[1])),
                "p2": (float(v_end[0]), float(v_end[1])),
                "width_start": w_s,
                "width_end": w_e,
                "width": w_m,
                "length": seg_len,
            })
        return segments

    @staticmethod
    def _eval_segments_x_at_y(segments, y_val):
        best_x = None
        best_dist = 1e9
        for seg in segments:
            x1, y1 = seg["p1"]
            x2, y2 = seg["p2"]
            dy = y2 - y1
            if abs(dy) < 2.0:
                continue
            ymin, ymax = (y1, y2) if y1 < y2 else (y2, y1)
            if ymin - 1.5 <= y_val <= ymax + 1.5:
                t = (y_val - y1) / dy
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                return float(x1 + t * (x2 - x1))
            d = (ymin - y_val) if y_val < ymin else ((y_val - ymax) if y_val > ymax else 0.0)
            if d < best_dist:
                best_dist = d
                t = (y_val - y1) / dy
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                best_x = float(x1 + t * (x2 - x1))
        return best_x

    @staticmethod
    def _eval_segments_y_at_x(segments, x_val):
        best_y = None
        best_dist = 1e9
        for seg in segments:
            x1, y1 = seg["p1"]
            x2, y2 = seg["p2"]
            dx = x2 - x1
            if abs(dx) < 2.0:
                continue
            xmin, xmax = (x1, x2) if x1 < x2 else (x2, x1)
            if xmin - 1.5 <= x_val <= xmax + 1.5:
                t = (x_val - x1) / dx
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                return float(y1 + t * (y2 - y1))
            d = (xmin - x_val) if x_val < xmin else ((x_val - xmax) if x_val > xmax else 0.0)
            if d < best_dist:
                best_dist = d
                t = (x_val - x1) / dx
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                best_y = float(y1 + t * (y2 - y1))
        return best_y

    @staticmethod
    def _sample_polyline(segments, num_pts=25):
        if not segments:
            return np.empty((0, 2), dtype=np.float32)
        n_segs = len(segments)
        if n_segs == 1:
            p1 = np.asarray(segments[0]["p1"], dtype=np.float32)
            p2 = np.asarray(segments[0]["p2"], dtype=np.float32)
            t = np.linspace(0.0, 1.0, num_pts, dtype=np.float32)[:, None]
            return p1 + t * (p2 - p1)
        lengths = np.fromiter((max(1e-3, float(s["length"])) for s in segments), dtype=np.float64, count=n_segs)
        cum = np.r_[0.0, np.cumsum(lengths)]
        total = cum[-1]
        targets = np.linspace(0.0, total, num_pts, dtype=np.float64)
        idxs = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, n_segs - 1)
        seg_lens = lengths[idxs]
        t = np.clip((targets - cum[idxs]) / seg_lens, 0.0, 1.0)[:, None]
        p1 = np.array([s["p1"] for s in segments], dtype=np.float32)
        p2 = np.array([s["p2"] for s in segments], dtype=np.float32)
        return (p1[idxs] + t * (p2[idxs] - p1[idxs])).astype(np.float32)

    @staticmethod
    def _find_corner_elbow_seg_idx(segments):
        """
        Return the index (in 1..len(segments)-1) of the first segment that turns into a corner
        (including multi-segment chamfered 90-degree corners), or None if straight.
        """
        if len(segments) < 2:
            return None
        # Kiem tra xem phia tren co nhanh vat cheo / nam ngang cua goc cua khong
        has_strong_turn_arm = any(
            abs(float(s["p2"][0] - s["p1"][0])) >= abs(float(s["p2"][1] - s["p1"][1])) * 0.65
            and float(s["length"]) >= 12.0
            for s in segments[1:]
        )
        for idx in range(len(segments) - 1):
            s1, s2 = segments[idx], segments[idx + 1]
            v1x = float(s1["p2"][0] - s1["p1"][0])
            v1y = float(s1["p2"][1] - s1["p1"][1])
            v2x = float(s2["p2"][0] - s2["p1"][0])
            v2y = float(s2["p2"][1] - s2["p1"][1])
            l1 = math.hypot(v1x, v1y)
            l2 = math.hypot(v2x, v2y)
            if l1 < 10.0 or l2 < 10.0:
                continue
            cos_a = _clip_scalar((v1x * v2x + v1y * v2y) / (l1 * l2), -1.0, 1.0)
            ang_deg = math.degrees(math.acos(cos_a))
            # Bo qua zigzag loa ngang quay nguoc 180 do (vd: glare hook trai -> quay phai)
            if v1x * v2x < -100.0 and abs(v1x) > 2.0 * abs(v1y) and v2y > -12.0:
                continue
            # Bo qua vet gap nho sat can xe o day anh noi tiep len than vach thang phoi canh (bat ke goc gap cuc bo)
            if (
                idx == 0
                and float(s1["p2"][1]) > 172.0
                and v2y < -28.0
                and abs(v2x) < abs(v2y) * 1.15
                and float(s2["p2"][1]) < 150.0
            ):
                continue
            # Bo qua do cong phoi canh nhe cung chieu tren duong thang (vd: -11 deg -> -38 deg)
            if v1x * v2x > 0 and ang_deg < 48.0 and abs(v2x) < abs(v2y) * 1.25:
                if not any(abs(float(s["p2"][0] - s["p1"][0])) >= abs(float(s["p2"][1] - s["p1"][1])) * 1.25 for s in segments[idx + 1:]):
                    continue
            if ang_deg >= 14.0 and abs(v2x) >= abs(v2y) * 0.65:
                return idx + 1
            if has_strong_turn_arm and ang_deg >= 14.0 and abs(v2x) >= abs(v2y) * 0.32:
                return idx + 1
        return None

    @classmethod
    def _offset_segments_by_normal(cls, active_segs, side_sign, a_lane, b_lane, hT, wT, arm_width_hint=0.0, allow_thin_fallback=True, corner_memory_active=False):
        """
        Dinh huong tam duong tuong doi dua vao vector phap tuyen huong vao tam lan (Inward Normal Offset)
        ket hop Bo nho be rong lan truoc goc cua:
          1. Khi xe di lech lam mat 1 vach tren duong thang:
             - Hoi tam em ve giua lan theo do troi cua chan vach.
          2. Khi vao goc cua (co diem gap khuc hoac vach vat ngang):
             - Dung vector phap tuyen n_k = (-t_y * side_sign, t_x * side_sign) vuong goc voi tung doan
               de dat duong tam nam CHINH XAC giua long duong tren ca doan thang, doan cua va nhanh ngang!
        """
        if not active_segs:
            return []

        min_w_clamp = max(24.0, wT * 0.06)
        _w_list = sorted([float(s.get("width", 6.0)) for s in active_segs])
        med_tape_w = float(_w_list[len(_w_list) // 2]) if _w_list else 6.0
        pts_all_x = [float(active_segs[0]["p1"][0])] + [float(s["p2"][0]) for s in active_segs]
        span_x_all = max(pts_all_x) - min(pts_all_x)

        max_w_clamp = float(wT) * 1.05

        def w_lane_at_y(y_q):
            return float(np.clip(a_lane * float(y_q) + b_lane, min_w_clamp, max_w_clamp))

        b_verts = [np.array(active_segs[0]["p1"], dtype=np.float32)]
        for seg in active_segs:
            b_verts.append(np.array(seg["p2"], dtype=np.float32))

        m_segs = len(b_verts) - 1
        taus = []
        seg_normals = []
        use_persp_aspect = bool(a_lane >= 0.45)
        for k in range(m_segs):
            vec_k = b_verts[k + 1] - b_verts[k]
            if use_persp_aspect:
                y_mid_k = 0.5 * float(b_verts[k][1] + b_verts[k + 1][1])
                aspect_k = _clip_scalar(
                    (1.026 / max(1.0, float(wT))) * (w_lane_at_y(y_mid_k) / max(0.35, float(a_lane))),
                    0.34,
                    0.68,
                )
                vx_g = float(vec_k[0])
                vy_g = float(vec_k[1]) / aspect_k
                len_g = math.hypot(vx_g, vy_g)
                if len_g > 1e-3:
                    tx, ty = vx_g / len_g, vy_g / len_g
                else:
                    tx, ty = 0.0, -1.0
            else:
                len_k = float(np.linalg.norm(vec_k))
                if len_k > 1e-3:
                    tx, ty = float(vec_k[0] / len_k), float(vec_k[1] / len_k)
                else:
                    tx, ty = 0.0, -1.0
            # Vector phap tuyen huong vao giua lan trong he toa do mat dat (ground metric):
            nx = -ty * float(side_sign)
            ny = tx * float(side_sign)
            seg_normals.append(np.array([nx, ny], dtype=np.float32))

            dx = float(vec_k[0])
            dy_up = max(0.0, float(-vec_k[1]))
            # Do lech huong cua vach so voi do nghieng phoi canh tu nhien cua duong thang
            dx_dev = dx - side_sign * 0.52 * max(0.0, a_lane) * dy_up
            dx_turn = abs(dx_dev)
            tau = float(dx_turn / np.hypot(dx_turn, 1.65 * dy_up + 1e-3))
            taus.append(tau)

        max_tau = max(taus) if taus else 0.0
        x0, y0 = float(b_verts[0][0]), float(b_verts[0][1])
        mid_x = wT * 0.5
        edge_margin = wT * 0.055  # ~26px tren khung hinh 480px
        norm_base_x = edge_margin if side_sign > 0 else (wT - edge_margin)

        dx_span_wall = float(b_verts[-1][0] - b_verts[0][0])
        dy_span_wall = max(1.0, float(b_verts[0][1] - b_verts[-1][1]))
        dx_span_dev = dx_span_wall - side_sign * 0.52 * max(0.0, a_lane) * dy_span_wall
        is_corner_shape = (
            bool(corner_memory_active)
            or (arm_width_hint >= 16.0)
            or (m_segs >= 2 and max_tau >= 0.22)
            or (y0 < hT * 0.84 and max_tau >= 0.20)
            or (abs(dx_span_wall) > wT * 0.18)
            or (abs(dx_span_dev) > wT * 0.14)
        )

        if not is_corner_shape:
            # Straight / gently curved single edge: offset directly by calibrated 0.5 * W_lane(y)
            c_verts = []
            for vk in b_verts:
                xk, yk = float(vk[0]), float(vk[1])
                cx_k = float(np.clip(xk + side_sign * 0.5 * w_lane_at_y(yk), 4.0, wT - 5.0))
                c_verts.append(np.array([cx_k, yk], dtype=np.float32))
        else:
            # CHE DO DICH PHAP TUYEN CHINH XAC QUA GOC CUA 90 DO:
            dx_total = float(b_verts[-1][0] - b_verts[0][0])
            seg_dxs = [float(b_verts[i + 1][0] - b_verts[i][0]) for i in range(m_segs)]
            has_s_reversal = any(d > 18.0 for d in seg_dxs) and any(d < -18.0 for d in seg_dxs)
            c_verts = []
            for k, vk in enumerate(b_verts):
                xk, yk = float(vk[0]), float(vk[1])
                rx = 0.5 * w_lane_at_y(yk)
                if arm_width_hint >= 16.0:
                    ry = 0.5 * float(arm_width_hint)
                elif a_lane >= 0.45:
                    aspect_y = _clip_scalar(
                        (1.026 / max(1.0, float(wT))) * (w_lane_at_y(yk) / max(0.35, float(a_lane))),
                        0.34,
                        0.68,
                    )
                    ry = aspect_y * rx
                else:
                    ry = 0.58 * rx

                if k == 0:
                    n_vec = seg_normals[0]
                elif k == m_segs:
                    n_vec = seg_normals[-1]
                else:
                    n_sum = seg_normals[k - 1] + seg_normals[k]
                    n_norm = float(np.linalg.norm(n_sum))
                    if n_norm > 1e-3:
                        n_unit = n_sum / n_norm
                        miter = float(np.clip(1.0 / max(0.65, float(np.dot(n_unit, seg_normals[k]))), 1.0, 1.35))
                        n_vec = n_unit * miter
                    else:
                        n_vec = seg_normals[k]

                cx = float(xk + float(n_vec[0]) * rx)
                cy = float(max(4.0, yk + float(n_vec[1]) * ry))
                if is_corner_shape and len(c_verts) >= 1:
                    seg_dx_in = seg_dxs[k - 1]
                    if not has_s_reversal:
                        if dx_total > 20.0 and seg_dx_in >= -6.0:
                            cx = max(cx, float(c_verts[-1][0]))
                        elif dx_total < -20.0 and seg_dx_in <= 6.0:
                            cx = min(cx, float(c_verts[-1][0]))
                    cy = min(cy, float(c_verts[-1][1]))
                c_verts.append(np.array([cx, cy], dtype=np.float32))

        # Cat chinh xac doan duong tam voi mep duoi khung hinh (y = hT - 6) neu vector phap tuyen day diem dau xuong duoi day anh
        bot_anchor_y = float(hT - 6.0)
        if len(c_verts) >= 2 and float(c_verts[0][1]) > bot_anchor_y:
            idx_up = next((i for i, v in enumerate(c_verts) if float(v[1]) <= bot_anchor_y), None)
            if idx_up is not None and idx_up >= 1:
                v_low = c_verts[idx_up - 1]
                v_up = c_verts[idx_up]
                dy_seg = max(1e-3, float(v_low[1] - v_up[1]))
                alpha_bot = float(np.clip((float(v_low[1]) - bot_anchor_y) / dy_seg, 0.0, 1.0))
                x_bot = float(np.clip(float(v_low[0]) + alpha_bot * float(v_up[0] - v_low[0]), 4.0, wT - 5.0))
                c_verts = [np.array([x_bot, bot_anchor_y], dtype=np.float32)] + c_verts[idx_up:]
            else:
                for v in c_verts:
                    v[1] = min(float(v[1]), bot_anchor_y)

        for v in c_verts:
            v[0] = float(np.clip(v[0], 4.0, wT - 5.0))
            v[1] = float(np.clip(v[1], 4.0, bot_anchor_y))

        # When single outer wall is clipped at side border and bends sharply across the frame,
        # avoid dragging entry c_verts[0] deep into the wrong side behind the clipped corner stem.
        if is_corner_shape and len(c_verts) >= 2:
            clipped_outer = (side_sign > 0 and float(b_verts[0][0]) <= 14.0) or (
                side_sign < 0 and float(b_verts[0][0]) >= wT - 15.0
            )
            dx_wall = float(b_verts[-1][0] - b_verts[0][0])
            if clipped_outer and dx_wall * side_sign > wT * 0.35:
                # Do not clamp x coordinates to mid_x; preserve continuous natural curve
                pass

        # Trim trailing backward portion only when single outer wall is high near top of frame (< 0.50*hT)
        if (
            len(c_verts) >= 2
            and b_verts[0][1] < hT * 0.50
            and abs(float(b_verts[-1][0] - b_verts[0][0])) >= float(wT) * 0.45
            and abs(float(b_verts[0][1] - b_verts[-1][1])) <= 48.0
        ):
            dx_total_raw = float(c_verts[-1][0] - c_verts[0][0])
            x_trim_right = float(mid_x - 0.14 * wT)
            x_trim_left = float(mid_x + 0.14 * wT)
            trim_x = None
            if side_sign > 0 and dx_total_raw > 35.0 and c_verts[0][0] < x_trim_right - 12.0 and c_verts[-1][0] > x_trim_right + 12.0:
                trim_x = x_trim_right
            elif side_sign < 0 and dx_total_raw < -35.0 and c_verts[0][0] > x_trim_left + 12.0 and c_verts[-1][0] < x_trim_left - 12.0:
                trim_x = x_trim_left
            if trim_x is not None:
                trimmed = []
                for idx_v in range(len(c_verts) - 1):
                    va, vb = c_verts[idx_v], c_verts[idx_v + 1]
                    xa, xb = float(va[0]), float(vb[0])
                    if (side_sign > 0 and xb >= trim_x) or (side_sign < 0 and xb <= trim_x):
                        if not trimmed:
                            if abs(xb - xa) > 1e-3 and ((xa < trim_x <= xb) or (xb <= trim_x < xa)):
                                al = (trim_x - xa) / (xb - xa)
                                trimmed.append((va + al * (vb - va)).astype(np.float32))
                            else:
                                trimmed.append(va.copy())
                        trimmed.append(vb.copy())
                if len(trimmed) >= 2:
                    c_verts = trimmed

        # Neo diem dau tien xuong sat mep duoi man hinh (y = hT - 6) de mui ten luon nam o mep duoi
        if len(c_verts) >= 1 and c_verts[0][1] < bot_anchor_y - 2.0:
            dy_ext = float(bot_anchor_y - c_verts[0][1])
            dx_total_c = float(c_verts[-1][0] - c_verts[0][0]) if len(c_verts) >= 2 else 0.0
            if len(c_verts) >= 2 and (c_verts[0][1] - c_verts[1][1]) >= 3.0:
                slope_c = float(np.clip((c_verts[0][0] - c_verts[1][0]) / (c_verts[0][1] - c_verts[1][1]), -1.6, 1.6))
                bot_x = float(np.clip(c_verts[0][0] + slope_c * dy_ext, 4.0, wT - 5.0))
            elif len(b_verts) >= 2 and (b_verts[0][1] - b_verts[-1][1]) >= 10.0:
                ref_v = b_verts[1] if (b_verts[0][1] - b_verts[1][1]) >= 12.0 else b_verts[-1]
                slope_b = float(np.clip((b_verts[0][0] - ref_v[0]) / max(8.0, b_verts[0][1] - ref_v[1]), -1.6, 1.6))
                x_edge_bot = float(b_verts[0][0]) + slope_b * dy_ext
                bot_x = float(np.clip(x_edge_bot + side_sign * 0.5 * w_lane_at_y(bot_anchor_y), 4.0, wT - 5.0))
            else:
                bot_x = float(np.clip(c_verts[0][0], 4.0, wT - 5.0))
            c_verts.insert(0, np.array([bot_x, bot_anchor_y], dtype=np.float32))
        elif len(c_verts) >= 1:
            c_verts[0][1] = bot_anchor_y

        if len(c_verts) >= 2:
            return cls._resample_pts(c_verts, n_out=max(28, len(c_verts) * 8))

        center_pts = []
        for j in range(len(c_verts) - 1):
            v_start = c_verts[j]
            v_end = c_verts[j + 1]
            vec = v_end - v_start
            seg_len = float(np.linalg.norm(vec))
            if seg_len < 1.0:
                continue
            n_steps = max(5, int(round(seg_len / 8.0)))
            ts = np.linspace(0.0, 1.0, n_steps, endpoint=(j == len(c_verts) - 2))
            for t in ts:
                pt = v_start + t * vec
                raw_cx = float(pt[0])
                raw_cy = float(np.clip(pt[1], 2.0, hT - 2.0))
                if 2.0 <= raw_cx <= wT - 3.0:
                    center_pts.append((raw_cx, raw_cy))
                elif len(center_pts) > 0:
                    prev_cx, prev_cy = center_pts[-1]
                    bound_x = 2.0 if raw_cx < 2.0 else (wT - 3.0)
                    if abs(raw_cx - prev_cx) > 1e-3:
                        alpha = float(np.clip((bound_x - prev_cx) / (raw_cx - prev_cx), 0.0, 1.0))
                        center_pts.append((bound_x, float(prev_cy + alpha * (raw_cy - prev_cy))))
                    break
                else:
                    center_pts.append((float(np.clip(raw_cx, 2.0, wT - 3.0)), raw_cy))

        return center_pts

    @staticmethod
    def _trace_corridor_medial_axis(l_pts, r_pts, p_start, p_exit, hT, wT, scale=4):
        """
        Extract the scale-invariant medial axis (zero level-set of phi = (D_L - D_R)/(D_L + D_R))
        inside the corridor bounded by left polyline l_pts and right polyline r_pts.
        """
        gw = max(24, wT // scale)
        gh = max(16, hT // scale)
        m_l = np.full((gh, gw), 255, dtype=np.uint8)
        m_r = np.full((gh, gw), 255, dtype=np.uint8)
        l_g = np.round(l_pts / float(scale)).astype(np.int32)
        r_g = np.round(r_pts / float(scale)).astype(np.int32)
        cv2.polylines(m_l, [l_g], False, 0, 1)
        cv2.polylines(m_r, [r_g], False, 0, 1)

        d_l = cv2.GaussianBlur(cv2.distanceTransform(m_l, cv2.DIST_L2, 3), (3, 3), 0.75)
        d_r = cv2.GaussianBlur(cv2.distanceTransform(m_r, cv2.DIST_L2, 3), (3, 3), 0.75)
        phi_l = ((d_l - d_r) / (d_l + d_r + 1e-3)).tolist()

        px = float(p_start[0]) / float(scale)
        py = float(p_start[1]) / float(scale)
        ex = float(p_exit[0]) / float(scale)
        ey = float(p_exit[1]) / float(scale)
        gw_lim = gw - 1.001
        gh_lim = gh - 1.001

        def _sample_phi(x, y):
            xc = 0.0 if x < 0.0 else (gw_lim if x > gw_lim else x)
            yc = 0.0 if y < 0.0 else (gh_lim if y > gh_lim else y)
            x0, y0 = int(xc), int(yc)
            fx, fy = xc - x0, yc - y0
            omx, omy = 1.0 - fx, 1.0 - fy
            row0 = phi_l[y0]
            row1 = phi_l[y0 + 1]
            return (
                row0[x0] * omx * omy
                + row0[x0 + 1] * fx * omy
                + row1[x0] * omx * fy
                + row1[x0 + 1] * fx * fy
            )

        def _sample_grad(x, y):
            eps = 0.5
            gx = (_sample_phi(x + eps, y) - _sample_phi(x - eps, y))
            gy = (_sample_phi(x, y + eps) - _sample_phi(x, y - eps))
            return gx, gy

        # Refine starting point gently onto phi = 0 without drifting far from p_start
        sx, sy = px, py
        for _ in range(3):
            v = _sample_phi(sx, sy)
            gx, gy = _sample_grad(sx, sy)
            g2 = gx * gx + gy * gy + 1e-5
            step_x = _clip_scalar((v * gx) / g2, -1.2, 1.2)
            step_y = _clip_scalar((v * gy) / g2, -1.2, 1.2)
            sx -= step_x
            sy -= step_y

        pts_out = [(float(p_start[0]), float(p_start[1]))]
        prev_tx = ex - sx
        prev_ty = ey - sy
        n_init = math.hypot(prev_tx, prev_ty)
        if n_init < 1e-4:
            return pts_out
        prev_tx /= n_init
        prev_ty /= n_init

        cx, cy = sx, sy
        step_g = 1.6
        for _ in range(55):
            gx, gy = _sample_grad(cx, cy)
            tx, ty = -gy, gx
            nt = math.hypot(tx, ty)
            if nt < 1e-4:
                break
            tx /= nt
            ty /= nt
            if tx * prev_tx + ty * prev_ty < 0.0:
                tx, ty = -tx, -ty
            # Gentle alignment bias toward exit so flat saddles don't wander backward
            dx_e, dy_e = ex - cx, ey - cy
            d_exit = math.hypot(dx_e, dy_e)
            if d_exit < 2.2:
                break
            ux_e, uy_e = dx_e / d_exit, dy_e / d_exit
            if tx * ux_e + ty * uy_e < -0.35:
                tx = 0.65 * prev_tx + 0.35 * ux_e
                ty = 0.65 * prev_ty + 0.35 * uy_e
                nt2 = math.hypot(tx, ty)
                if nt2 > 1e-4:
                    tx /= nt2
                    ty /= nt2
            prev_tx, prev_ty = tx, ty
            nx = cx + step_g * tx
            ny = cy + step_g * ty
            for _ in range(2):
                v = _sample_phi(nx, ny)
                gx2, gy2 = _sample_grad(nx, ny)
                g2 = gx2 * gx2 + gy2 * gy2 + 1e-5
                nx -= _clip_scalar((v * gx2) / g2, -1.0, 1.0)
                ny -= _clip_scalar((v * gy2) / g2, -1.0, 1.0)
            if nx < 0.2 or nx > gw - 1.2 or ny < 0.2 or ny > gh - 1.2:
                break
            cx, cy = nx, ny
            fx_px = _clip_scalar(cx * float(scale), 2.0, float(wT - 3))
            fy_px = _clip_scalar(cy * float(scale), 2.0, float(hT - 2))
            if math.hypot(fx_px - pts_out[-1][0], fy_px - pts_out[-1][1]) >= 3.5:
                pts_out.append((fx_px, fy_px))

        if math.hypot(p_exit[0] - pts_out[-1][0], p_exit[1] - pts_out[-1][1]) >= 4.0:
            pts_out.append((_clip_scalar(float(p_exit[0]), 2.0, float(wT - 3)), _clip_scalar(float(p_exit[1]), 2.0, float(hT - 2))))
        return pts_out

    @classmethod
    def _compute_lane_and_centerline(cls, left_segs, right_segs, hT, wT, lane_width_ab=(1.15, 175.0), arm_width_hint=0.0, lane_width_calibrated=False, corner_memory_active=False):
        """
        Measure/calibrate perspective lane width W_lane(y) = a * y + b (strictly distinct from tape width!)
        and compute the exact medial centerline between the 2 white lines for straight lanes,
        chamfered/rounded corners, 90-degree corners (via scale-invariant distance-transform medial axis),
        and single-edge cases.
        """
        a_lane, b_lane = lane_width_ab
        min_w_clamp = max(24.0, wT * 0.06)
        max_w_clamp = float(wT) * 1.05

        def w_lane_at_y(y_q):
            return float(np.clip(a_lane * float(y_q) + b_lane, min_w_clamp, max_w_clamp))

        has_l = len(left_segs) > 0
        has_r = len(right_segs) > 0

        if has_l and has_r:
            l_elbow_idx = cls._find_corner_elbow_seg_idx(left_segs)
            r_elbow_idx = cls._find_corner_elbow_seg_idx(right_segs)

            l_stem = left_segs[:l_elbow_idx] if l_elbow_idx is not None else [
                s for s in left_segs if (s["p1"][1] - s["p2"][1]) >= max(8.0, 0.28 * abs(s["p2"][0] - s["p1"][0]))
            ]
            r_stem = right_segs[:r_elbow_idx] if r_elbow_idx is not None else [
                s for s in right_segs if (s["p1"][1] - s["p2"][1]) >= max(8.0, 0.28 * abs(s["p2"][0] - s["p1"][0]))
            ]
            if not l_stem:
                l_dx_only = float(left_segs[-1]["p2"][0] - left_segs[0]["p1"][0])
                l_dy_only = max(1.0, float(left_segs[0]["p1"][1] - left_segs[-1]["p2"][1]))
                if not (r_elbow_idx is not None and float(left_segs[0]["p1"][1]) < hT * 0.48 and abs(l_dx_only) >= 1.3 * l_dy_only):
                    l_stem = [left_segs[0]]
            if not r_stem:
                r_dx_only = float(right_segs[-1]["p2"][0] - right_segs[0]["p1"][0])
                r_dy_only = max(1.0, float(right_segs[0]["p1"][1] - right_segs[-1]["p2"][1]))
                if not (l_elbow_idx is not None and float(right_segs[0]["p1"][1]) < hT * 0.48 and abs(r_dx_only) >= 1.3 * r_dy_only):
                    r_stem = [right_segs[0]]

            if l_stem and r_stem:
                y_l_bot = max(s["p1"][1] for s in l_stem)
                y_l_top = min(s["p2"][1] for s in l_stem)
                y_r_bot = max(s["p1"][1] for s in r_stem)
                y_r_top = min(s["p2"][1] for s in r_stem)

                y_overlap_bot = min(y_l_bot, y_r_bot)
                y_overlap_top = max(y_l_top, y_r_top)
            else:
                y_l_bot = max(s["p1"][1] for s in left_segs)
                y_l_top = min(s["p2"][1] for s in left_segs)
                y_r_bot = max(s["p1"][1] for s in right_segs)
                y_r_top = min(s["p2"][1] for s in right_segs)
                y_overlap_bot = 0.0
                y_overlap_top = float(hT)

            if y_overlap_bot - y_overlap_top >= 12.0:
                ys_sample = np.linspace(y_overlap_top, y_overlap_bot, 14)
                widths_meas = []
                ys_valid = []
                for yq in ys_sample:
                    xl = cls._eval_segments_x_at_y(l_stem, yq)
                    xr = cls._eval_segments_x_at_y(r_stem, yq)
                    if xl is not None and xr is not None and (xr - xl) >= min_w_clamp:
                        widths_meas.append(xr - xl)
                        ys_valid.append(yq)
                if len(ys_valid) >= 4:
                    yv = np.asarray(ys_valid, dtype=np.float32)
                    wv = np.asarray(widths_meas, dtype=np.float32)
                    y_span_fit = float(np.ptp(yv))
                    if y_span_fit >= 18.0:
                        l_dx_s = float(l_stem[-1]["p2"][0] - l_stem[0]["p1"][0])
                        l_dy_s = max(1.0, float(l_stem[0]["p1"][1] - l_stem[-1]["p2"][1]))
                        r_dx_s = float(r_stem[-1]["p2"][0] - r_stem[0]["p1"][0])
                        r_dy_s = max(1.0, float(r_stem[0]["p1"][1] - r_stem[-1]["p2"][1]))
                        sl_l = l_dx_s / l_dy_s
                        sl_r = r_dx_s / r_dy_s
                        stems_straight_for_w = (
                            (l_elbow_idx is None)
                            and (r_elbow_idx is None)
                            and (abs(sl_l + sl_r) <= 0.65)
                            and (abs(sl_l) <= 1.35)
                            and (abs(sl_r) <= 1.35)
                        )
                        pred_mid_w = a_lane * float(_fast_median_seq(yv)) + b_lane
                        meas_mid_w = float(_fast_median_seq(wv))
                        can_calibrate_w = (
                            (y_span_fit >= 28.0)
                            and (abs(sl_l + sl_r) <= 0.85)
                            and (abs(sl_l) <= 1.35)
                            and (abs(sl_r) <= 1.35)
                            and (0.30 * float(wT) <= meas_mid_w <= 0.96 * float(wT))
                        )
                        if not lane_width_calibrated:
                            if can_calibrate_w:
                                ym_cal = float(yv.sum()) / len(yv)
                                wm_cal = float(wv.sum()) / len(wv)
                                dy_c = yv - ym_cal
                                var_yc = float(np.dot(dy_c, dy_c))
                                a_fit = float(np.clip(float(np.dot(dy_c, wv - wm_cal)) / var_yc, 0.35, 1.95)) if var_yc > 1e-5 else a_lane
                                b_fit = float(_fast_median_seq(wv - a_fit * yv))
                                if abs(meas_mid_w - pred_mid_w) > max(32.0, 0.25 * meas_mid_w):
                                    a_lane, b_lane = a_fit, b_fit
                                else:
                                    a_lane = 0.50 * a_lane + 0.50 * a_fit
                                    b_lane = 0.50 * b_lane + 0.50 * b_fit
                                lane_width_calibrated = True
                        elif stems_straight_for_w:
                            if y_span_fit >= 48.0 and len(ys_valid) >= 7:
                                ym_cal = float(yv.sum()) / len(yv)
                                wm_cal = float(wv.sum()) / len(wv)
                                dy_c = yv - ym_cal
                                var_yc = float(np.dot(dy_c, dy_c))
                                a_fit = float(np.clip(float(np.dot(dy_c, wv - wm_cal)) / var_yc, 0.35, 1.90)) if var_yc > 1e-5 else a_lane
                                b_fit = float(_fast_median_seq(wv - a_fit * yv))
                                if 0.65 * pred_mid_w <= meas_mid_w <= 1.45 * pred_mid_w:
                                    alpha_w = 0.28 if abs(meas_mid_w - pred_mid_w) > max(48.0, 0.32 * meas_mid_w) else 0.18
                                    a_lane = (1.0 - alpha_w) * a_lane + alpha_w * a_fit
                                    b_lane = (1.0 - alpha_w) * b_lane + alpha_w * b_fit
                                elif 0.30 * float(wT) <= meas_mid_w <= 0.96 * float(wT):
                                    # Recover cleanly from a mismatched initial calibration on a straight 2-edge corridor
                                    a_lane = 0.65 * a_lane + 0.35 * a_fit
                                    b_lane = 0.65 * b_lane + 0.35 * b_fit
                            elif 0.65 * pred_mid_w <= meas_mid_w <= 1.45 * pred_mid_w:
                                # Khong ngoai suy he so goc a_lane tu doan chong lan qua ngan (< 48px)
                                b_fit = float(_fast_median_seq(wv - a_lane * yv))
                                b_lane = 0.86 * b_lane + 0.14 * b_fit

            center_pts = []
            pred_flags = []

            y_max_bot = max(y_l_bot, y_r_bot)
            xl_ov = cls._eval_segments_x_at_y(l_stem, y_overlap_bot)
            xr_ov = cls._eval_segments_x_at_y(r_stem, y_overlap_bot)
            w_ov_actual = (
                (xr_ov - xl_ov)
                if (xl_ov is not None and xr_ov is not None and xr_ov > xl_ov)
                else w_lane_at_y(y_overlap_bot)
            )

            # Build stem centerline when stems overlap vertically
            if y_overlap_bot >= y_overlap_top + 8.0:
                if y_max_bot - y_overlap_bot >= 6.0:
                    # Perspective-symmetric width extension below clipped stem (fixes +25px border-clip bias)
                    n_tail = max(2, int(round((y_max_bot - y_overlap_bot) / 6.0)))
                    l0 = l_stem[0]
                    r0 = r_stem[0]
                    dy_l0 = float(l0["p1"][1] - l0["p2"][1])
                    dy_r0 = float(r0["p1"][1] - r0["p2"][1])
                    slope_l0 = float(l0["p1"][0] - l0["p2"][0]) / max(1.0, dy_l0) if dy_l0 >= 16.0 else None
                    slope_r0 = float(r0["p1"][0] - r0["p2"][0]) / max(1.0, dy_r0) if dy_r0 >= 16.0 else None

                    for yq in np.linspace(y_max_bot, y_overlap_bot, n_tail, endpoint=False):
                        dy_below = float(yq) - float(y_overlap_bot)
                        w_exp = _clip_scalar(w_ov_actual + max(0.0, a_lane) * dy_below, min_w_clamp, wT * 1.35)
                        if y_r_bot > y_l_bot:
                            xr = cls._eval_segments_x_at_y(r_stem, yq)
                            if xr is not None:
                                cx_w = xr - 0.5 * w_exp
                                max_sl_l = 1.35 if (l_elbow_idx is None) else 0.55
                                if slope_l0 is not None and xl_ov is not None and abs(slope_l0) <= max_sl_l:
                                    xl_ext = xl_ov + _clip_scalar(slope_l0, -max_sl_l, max_sl_l) * dy_below
                                    cx_t = 0.65 * (0.5 * (xl_ext + xr)) + 0.35 * cx_w if l_elbow_idx is None else 0.5 * (0.5 * (xl_ext + xr) + cx_w)
                                else:
                                    cx_t = cx_w
                                center_pts.append((_clip_scalar(cx_t, 4.0, wT - 5.0), float(yq)))
                                pred_flags.append(False)
                        else:
                            xl = cls._eval_segments_x_at_y(l_stem, yq)
                            if xl is not None:
                                cx_w = xl + 0.5 * w_exp
                                max_sl_r = 1.35 if (r_elbow_idx is None) else 0.55
                                if slope_r0 is not None and xr_ov is not None and abs(slope_r0) <= max_sl_r:
                                    xr_ext = xr_ov + _clip_scalar(slope_r0, -max_sl_r, max_sl_r) * dy_below
                                    cx_t = 0.65 * (0.5 * (xl + xr_ext)) + 0.35 * cx_w if r_elbow_idx is None else 0.5 * (0.5 * (xl + xr_ext) + cx_w)
                                else:
                                    cx_t = cx_w
                                center_pts.append((_clip_scalar(cx_t, 4.0, wT - 5.0), float(yq)))
                                pred_flags.append(False)

                l_dx_stem = float(l_stem[-1]["p2"][0] - l_stem[0]["p1"][0])
                l_dy_stem = max(1.0, float(l_stem[0]["p1"][1] - l_stem[-1]["p2"][1]))
                r_dx_stem = float(r_stem[-1]["p2"][0] - r_stem[0]["p1"][0])
                r_dy_stem = max(1.0, float(r_stem[0]["p1"][1] - r_stem[-1]["p2"][1]))
                curved_stems = (
                    (l_elbow_idx is not None or r_elbow_idx is not None)
                    and (abs(l_dx_stem) > l_dy_stem * 0.95 or abs(r_dx_stem) > r_dy_stem * 0.95)
                ) or (abs(l_dx_stem + r_dx_stem) > 0.25 * float(wT) and (y_overlap_bot - y_overlap_top) > 36.0)
                if curved_stems:
                    y_stem_end = max(float(y_overlap_top), float(y_overlap_bot) - min(42.0, 0.42 * float(y_overlap_bot - y_overlap_top)))
                elif (l_elbow_idx is not None or r_elbow_idx is not None) and (y_overlap_bot - y_overlap_top) > 32.0:
                    # Chuyen giao som sang truc giua Medial Axis truoc dinh cua 18-24px de tranh gap khuc zig-zag
                    y_stem_end = min(float(y_overlap_bot) - 16.0, float(y_overlap_top) + 22.0)
                else:
                    y_stem_end = float(y_overlap_top)
                n_stem = max(5, int(round((y_overlap_bot - y_stem_end) / 5.0)) + 1)
                for yq in np.linspace(y_overlap_bot, y_stem_end, n_stem):
                    xl = cls._eval_segments_x_at_y(l_stem, yq)
                    xr = cls._eval_segments_x_at_y(r_stem, yq)
                    if xl is not None and xr is not None:
                        center_pts.append((0.5 * (xl + xr), float(yq)))
                        pred_flags.append(False)

            # Neo diem thap nhat cua duong tam xuong mep duoi (hT - 6) khi chan duong tam nam gan day anh
            bot_anchor_y = float(hT - 6.0)
            if center_pts and center_pts[0][1] < bot_anchor_y - 2.0:
                if len(center_pts) >= 2:
                    p_low = center_pts[0]
                    ref_k = min(len(center_pts) - 1, 3)
                    p_up = center_pts[ref_k]
                    dy_ext = float(p_low[1] - p_up[1])
                    if dy_ext >= 3.0:
                        slope_xy = _clip_scalar((float(p_low[0]) - float(p_up[0])) / dy_ext, -1.6, 1.6)
                        x_ext = _clip_scalar(float(p_low[0]) + slope_xy * (bot_anchor_y - float(p_low[1])), 4.0, wT - 5.0)
                    else:
                        x_ext = center_pts[0][0]
                else:
                    x_ext = center_pts[0][0]
                center_pts.insert(0, (x_ext, bot_anchor_y))
                pred_flags.insert(0, False)

            # Check if there is a corner / chevron / upper turn / non-overlapping mid-corner
            l_dx_tot = float(left_segs[-1]["p2"][0] - left_segs[0]["p1"][0])
            l_dy_tot = max(1.0, float(left_segs[0]["p1"][1] - left_segs[-1]["p2"][1]))
            r_dx_tot = float(right_segs[-1]["p2"][0] - right_segs[0]["p1"][0])
            r_dy_tot = max(1.0, float(right_segs[0]["p1"][1] - right_segs[-1]["p2"][1]))
            has_corner_geom = (
                (l_elbow_idx is not None)
                or (r_elbow_idx is not None)
                or (y_overlap_bot < y_overlap_top + 8.0)
                or (abs(l_dx_tot) > l_dy_tot * 0.95 + 24.0)
                or (abs(r_dx_tot) > r_dy_tot * 0.95 + 24.0)
            )

            if not has_corner_geom:
                # Straight road: extend top stem smoothly if one line reaches higher
                y_min_top = min(y_l_top, y_r_top)
                if y_overlap_top - y_min_top >= 6.0 and center_pts:
                    xl_top = cls._eval_segments_x_at_y(l_stem, y_overlap_top)
                    xr_top = cls._eval_segments_x_at_y(r_stem, y_overlap_top)
                    w_top_act = (xr_top - xl_top) if (xl_top is not None and xr_top is not None and xr_top > xl_top) else w_lane_at_y(y_overlap_top)
                    n_top = max(2, int(round((y_overlap_top - y_min_top) / 6.0)))
                    for yq in np.linspace(y_overlap_top, y_min_top, n_top + 1)[1:]:
                        w_t = _clip_scalar(w_top_act + max(0.0, a_lane) * (float(yq) - float(y_overlap_top)), min_w_clamp, wT * 0.98)
                        if y_l_top < y_r_top:
                            xl = cls._eval_segments_x_at_y(l_stem, yq)
                            if xl is not None:
                                center_pts.append((_clip_scalar(xl + 0.5 * w_t, 4.0, wT - 5.0), float(yq)))
                                pred_flags.append(False)
                        else:
                            xr = cls._eval_segments_x_at_y(r_stem, yq)
                            if xr is not None:
                                center_pts.append((_clip_scalar(xr - 0.5 * w_t, 4.0, wT - 5.0), float(yq)))
                                pred_flags.append(False)
                if center_pts and (left_segs or right_segs):
                    for i in range(len(center_pts)):
                        px_c, py_c = float(center_pts[i][0]), float(center_pts[i][1])
                        if right_segs:
                            xr_bnd = cls._eval_segments_x_at_y(right_segs, py_c)
                            if xr_bnd is not None:
                                px_c = min(px_c, float(xr_bnd) - 14.0)
                        if left_segs:
                            xl_bnd = cls._eval_segments_x_at_y(left_segs, py_c)
                            if xl_bnd is not None:
                                px_c = max(px_c, float(xl_bnd) + 14.0)
                        center_pts[i] = (px_c, py_c)
                return np.asarray(center_pts, dtype=np.float32), np.asarray(pred_flags, dtype=bool), (a_lane, b_lane)

            # Corner / Chevron / Mid-corner: Corridor Medial Axis via Distance Transform (phi = 0)
            l_pts = np.array([left_segs[0]["p1"]] + [s["p2"] for s in left_segs], dtype=np.float32)
            r_pts = np.array([right_segs[0]["p1"]] + [s["p2"] for s in right_segs], dtype=np.float32)

            # Orient backwards-traced single top arm if needed (e.g. L traced right-to-left while R turns right)
            if len(l_pts) >= 2 and l_pts[-1][0] < l_pts[0][0] - 18.0 and abs(l_pts[-1][1] - l_pts[0][1]) < abs(l_pts[-1][0] - l_pts[0][0]) * 0.65:
                if float(np.mean(r_pts[:, 0])) > float(np.mean(l_pts[:, 0])) + 40.0 and r_pts[-1][0] >= r_pts[0][0] - 15.0:
                    l_pts = l_pts[::-1].copy()
            if len(r_pts) >= 2 and r_pts[-1][0] > r_pts[0][0] + 18.0 and abs(r_pts[-1][1] - r_pts[0][1]) < abs(r_pts[-1][0] - r_pts[0][0]) * 0.65:
                if float(np.mean(l_pts[:, 0])) + 40.0 < float(np.mean(r_pts[:, 0])) and l_pts[-1][0] <= l_pts[0][0] + 15.0:
                    r_pts = r_pts[::-1].copy()

            # Extend shorter exit arm along its final tangent when both walls move toward same exit
            l_ext = l_pts.copy()
            r_ext = r_pts.copy()
            if len(l_pts) >= 2 and len(r_pts) >= 2:
                vl = l_pts[-1] - l_pts[-2]
                vr = r_pts[-1] - r_pts[-2]
                nl = float(math.hypot(float(vl[0]), float(vl[1])))
                nr = float(math.hypot(float(vr[0]), float(vr[1])))
                if nl >= 10.0 and nr >= 10.0:
                    ul = vl / nl
                    ur = vr / nr
                    if float(np.dot(ul, ur)) > 0.35:
                        # Project each endpoint onto the other's exit axis
                        diff_end = l_pts[-1] - r_pts[-1]
                        proj_l_ahead = float(np.dot(diff_end, ur))
                        proj_r_ahead = float(np.dot(-diff_end, ul))
                        if proj_l_ahead > 14.0 and abs(ur[0]) >= abs(ur[1]) * 0.65:
                            ext_len = min(110.0, proj_l_ahead)
                            p_new = np.array([
                                _clip_scalar(float(r_pts[-1][0] + ext_len * ur[0]), 2.0, wT - 3.0),
                                _clip_scalar(float(r_pts[-1][1] + ext_len * ur[1]), 2.0, hT - 3.0),
                            ], dtype=np.float32)
                            r_ext = np.vstack([r_ext, p_new])
                        elif proj_r_ahead > 14.0 and abs(ul[0]) >= abs(ul[1]) * 0.65:
                            ext_len = min(110.0, proj_r_ahead)
                            p_new = np.array([
                                _clip_scalar(float(l_pts[-1][0] + ext_len * ul[0]), 2.0, wT - 3.0),
                                _clip_scalar(float(l_pts[-1][1] + ext_len * ul[1]), 2.0, hT - 3.0),
                            ], dtype=np.float32)
                            l_ext = np.vstack([l_ext, p_new])

            # Also extend entry stems slightly downward in the virtual distance mask to avoid endpoint pull at entry
            l_mask_pts = l_ext
            r_mask_pts = r_ext
            if y_overlap_bot >= y_overlap_top + 8.0:
                if len(l_ext) >= 2 and (l_ext[0][1] - l_ext[1][1]) > 8.0 and l_ext[0][1] < hT - 8.0:
                    v_dn = l_ext[0] - l_ext[1]
                    t_dn = min(60.0, (float(hT) - float(l_ext[0][1])) / max(0.2, float(v_dn[1]) / max(1.0, math.hypot(float(v_dn[0]), float(v_dn[1])))))
                    u_dn = v_dn / max(1.0, math.hypot(float(v_dn[0]), float(v_dn[1])))
                    l_mask_pts = np.vstack([l_ext[0] + t_dn * u_dn, l_ext])
                if len(r_ext) >= 2 and (r_ext[0][1] - r_ext[1][1]) > 8.0 and r_ext[0][1] < hT - 8.0:
                    v_dn = r_ext[0] - r_ext[1]
                    t_dn = min(60.0, (float(hT) - float(r_ext[0][1])) / max(0.2, float(v_dn[1]) / max(1.0, math.hypot(float(v_dn[0]), float(v_dn[1])))))
                    u_dn = v_dn / max(1.0, math.hypot(float(v_dn[0]), float(v_dn[1])))
                    r_mask_pts = np.vstack([r_ext[0] + t_dn * u_dn, r_ext])

            # If one wall terminated completely at the bottom overlap while the other rounds the corner,
            # extend smoothly past the shorter wall using normal offset calibrated to the measured overlap width
            if y_overlap_bot >= y_overlap_top + 8.0 and center_pts:
                l_rem_len = sum(float(s["length"]) for s in left_segs if min(s["p1"][1], s["p2"][1]) < y_overlap_top - 4.0)
                r_rem_len = sum(float(s["length"]) for s in right_segs if min(s["p1"][1], s["p2"][1]) < y_overlap_top - 4.0)
                if (l_rem_len > 55.0 and r_rem_len < 14.0) or (r_rem_len > 55.0 and l_rem_len < 14.0):
                    xl_top = cls._eval_segments_x_at_y(l_stem, y_overlap_top)
                    xr_top = cls._eval_segments_x_at_y(r_stem, y_overlap_top)
                    w_top_ov = (xr_top - xl_top) if (xl_top is not None and xr_top is not None and xr_top > xl_top) else w_lane_at_y(y_overlap_top)
                    dom_segs = left_segs if l_rem_len > r_rem_len else right_segs
                    dom_sign = +1.0 if l_rem_len > r_rem_len else -1.0
                    off_pts = cls._offset_segments_by_normal(
                        dom_segs,
                        dom_sign,
                        a_lane,
                        b_lane,
                        hT,
                        wT,
                        arm_width_hint=max(min_w_clamp, float(w_top_ov)),
                        allow_thin_fallback=(not lane_width_calibrated),
                        corner_memory_active=corner_memory_active,
                    )
                    upper_pts = [(float(px_o), float(py_o)) for (px_o, py_o) in off_pts if float(py_o) < float(center_pts[-1][1]) - 2.0]
                    if len(upper_pts) >= 2:
                        dx_join = float(center_pts[-1][0]) - float(upper_pts[0][0])
                        n_blend = min(len(upper_pts), 6)
                        for bi, (mx_raw, my_raw) in enumerate(upper_pts):
                            w_blend = max(0.0, 1.0 - bi / float(n_blend)) if bi < n_blend else 0.0
                            mx = mx_raw + dx_join * w_blend
                            if math.hypot(mx - center_pts[-1][0], my_raw - center_pts[-1][1]) >= 3.5:
                                if right_segs:
                                    xr_bnd = cls._eval_segments_x_at_y(right_segs, my_raw)
                                    if xr_bnd is not None:
                                        mx = min(mx, float(xr_bnd) - 14.0)
                                if left_segs:
                                    xl_bnd = cls._eval_segments_x_at_y(left_segs, my_raw)
                                    if xl_bnd is not None:
                                        mx = max(mx, float(xl_bnd) + 14.0)
                                center_pts.append((mx, my_raw))
                                pred_flags.append(False)
                        if has_corner_geom and len(center_pts) >= 5:
                            pts_sm = np.asarray(center_pts, dtype=np.float32)
                            for _ in range(3):
                                nxt = pts_sm.copy()
                                nxt[1:-1] = 0.25 * pts_sm[:-2] + 0.50 * pts_sm[1:-1] + 0.25 * pts_sm[2:]
                                pts_sm = nxt
                            center_pts = [(float(p[0]), float(p[1])) for p in pts_sm]
                        if center_pts and (left_segs or right_segs):
                            for i in range(len(center_pts)):
                                px_c, py_c = float(center_pts[i][0]), float(center_pts[i][1])
                                if right_segs:
                                    xr_bnd = cls._eval_segments_x_at_y(right_segs, py_c)
                                    if xr_bnd is not None:
                                        px_c = min(px_c, float(xr_bnd) - 14.0)
                                if left_segs:
                                    xl_bnd = cls._eval_segments_x_at_y(left_segs, py_c)
                                    if xl_bnd is not None:
                                        px_c = max(px_c, float(xl_bnd) + 14.0)
                                center_pts[i] = (px_c, py_c)
                        return np.asarray(center_pts, dtype=np.float32), np.asarray(pred_flags, dtype=bool), (a_lane, b_lane)

            if center_pts:
                p_join = center_pts[-1]
            elif abs(float(l_pts[0][1]) - float(r_pts[0][1])) > 32.0:
                if float(l_pts[0][1]) > float(r_pts[0][1]):
                    p0 = l_pts[0]
                    p1_ref = l_pts[1] if len(l_pts) >= 2 else (l_pts[0] + np.array([0.0, -10.0], dtype=np.float32))
                    j_sign = +1.0
                else:
                    p0 = r_pts[0]
                    p1_ref = r_pts[1] if len(r_pts) >= 2 else (r_pts[0] + np.array([0.0, -10.0], dtype=np.float32))
                    j_sign = -1.0
                vj = p1_ref - p0
                nj = max(1e-3, math.hypot(float(vj[0]), float(vj[1])))
                nx_j, ny_j = (-float(vj[1]) / nj) * j_sign, (float(vj[0]) / nj) * j_sign
                if ny_j > 0.35:
                    ny_j = 0.25
                    nx_j = math.copysign(math.sqrt(max(0.0, 1.0 - ny_j * ny_j)), nx_j if abs(nx_j) > 1e-3 else j_sign)
                hw_j = 0.5 * w_lane_at_y(float(p0[1]))
                p_join = (
                    _clip_scalar(float(p0[0]) + hw_j * nx_j, 4.0, wT - 5.0),
                    _clip_scalar(float(p0[1]) + hw_j * ny_j, 4.0, hT - 4.0),
                )
            else:
                p_join = (
                    _clip_scalar(0.5 * float(l_pts[0][0] + r_pts[0][0]), 4.0, wT - 5.0),
                    _clip_scalar(0.5 * float(l_pts[0][1] + r_pts[0][1]), 4.0, hT - 4.0),
                )

            l_arc = float(np.hypot(np.diff(l_ext[:, 0]), np.diff(l_ext[:, 1])).sum()) if len(l_ext) >= 2 else 0.0
            r_arc = float(np.hypot(np.diff(r_ext[:, 0]), np.diff(r_ext[:, 1])).sum()) if len(r_ext) >= 2 else 0.0
            end_gap = math.hypot(float(l_ext[-1][0] - r_ext[-1][0]), float(l_ext[-1][1] - r_ext[-1][1]))
            w_exit_est = max(min_w_clamp, float(arm_width_hint) if arm_width_hint > 20.0 else w_lane_at_y(float(l_ext[-1][1] if l_arc >= r_arc else r_ext[-1][1])))
            if l_arc > 1.8 * max(1.0, r_arc) and end_gap > 1.25 * w_exit_est and len(l_ext) >= 2:
                vt = l_ext[-1] - l_ext[-2]
                nt = max(1e-4, math.hypot(float(vt[0]), float(vt[1])))
                nx_in, ny_in = -float(vt[1]) / nt, float(vt[0]) / nt
                r_c = r_pts.mean(axis=0)
                if nx_in * (float(r_c[0]) - float(l_ext[-1][0])) + ny_in * (float(r_c[1]) - float(l_ext[-1][1])) < 0.0 and (
                    ny_in < 0.0 or float(r_c[1]) < float(l_ext[-1][1]) + 10.0
                ):
                    nx_in, ny_in = -nx_in, -ny_in
                p_exit = (
                    _clip_scalar(float(l_ext[-1][0]) + 0.5 * w_exit_est * nx_in, 4.0, wT - 5.0),
                    _clip_scalar(float(l_ext[-1][1]) + 0.5 * w_exit_est * ny_in, 4.0, hT - 4.0),
                )
            elif r_arc > 1.8 * max(1.0, l_arc) and end_gap > 1.25 * w_exit_est and len(r_ext) >= 2:
                vt = r_ext[-1] - r_ext[-2]
                nt = max(1e-4, math.hypot(float(vt[0]), float(vt[1])))
                nx_in, ny_in = float(vt[1]) / nt, -float(vt[0]) / nt
                l_c = l_pts.mean(axis=0)
                if nx_in * (float(l_c[0]) - float(r_ext[-1][0])) + ny_in * (float(l_c[1]) - float(r_ext[-1][1])) < 0.0 and (
                    ny_in < 0.0 or float(l_c[1]) < float(r_ext[-1][1]) + 10.0
                ):
                    nx_in, ny_in = -nx_in, -ny_in
                p_exit = (
                    _clip_scalar(float(r_ext[-1][0]) + 0.5 * w_exit_est * nx_in, 4.0, wT - 5.0),
                    _clip_scalar(float(r_ext[-1][1]) + 0.5 * w_exit_est * ny_in, 4.0, hT - 4.0),
                )
            else:
                p_exit = (
                    0.5 * float(l_ext[-1][0] + r_ext[-1][0]),
                    0.5 * float(l_ext[-1][1] + r_ext[-1][1]),
                )

            if float(p_exit[1]) > float(p_join[1]) - 6.0 and min(float(l_ext[-1][1]), float(r_ext[-1][1])) < float(p_join[1]) - 16.0:
                p_exit = (
                    _clip_scalar(0.5 * float(l_ext[-1][0] + r_ext[-1][0]), 4.0, wT - 5.0),
                    _clip_scalar(min(float(p_join[1]) - 14.0, 0.5 * float(l_ext[-1][1] + r_ext[-1][1])), 4.0, hT - 4.0),
                )

            l_arc_pre = float(np.hypot(np.diff(l_pts[:, 0]), np.diff(l_pts[:, 1])).sum()) if len(l_pts) >= 2 else 0.0
            r_arc_pre = float(np.hypot(np.diff(r_pts[:, 0]), np.diff(r_pts[:, 1])).sum()) if len(r_pts) >= 2 else 0.0
            unbalanced_corner_stub = (
                not center_pts
                and min(l_arc_pre, r_arc_pre) < 95.0
                and max(l_arc_pre, r_arc_pre) > 2.3 * max(1.0, min(l_arc_pre, r_arc_pre))
            )
            if unbalanced_corner_stub:
                medial_pts = []
                medial_ok = False
                outside_leak = True
                l_arc = l_arc_pre
                r_arc = r_arc_pre
            else:
                medial_pts = cls._trace_corridor_medial_axis(l_mask_pts, r_mask_pts, p_join, p_exit, hT, wT, scale=4)
                medial_ok = len(medial_pts) >= 3 and (bool(center_pts) or float(medial_pts[0][1]) >= float(medial_pts[-1][1]) - 4.0)
                outside_leak = False
            if medial_ok and float(l_pts[0][1]) - float(l_pts[-1][1]) >= hT * 0.42:
                out_hits, chk_n = 0, 0
                y_l_lo, y_l_hi = float(l_pts[-1][1]), float(l_pts[0][1])
                for mx_chk, my_chk in medial_pts:
                    if y_l_lo <= my_chk <= y_l_hi:
                        xl_chk = cls._eval_segments_x_at_y(left_segs, my_chk)
                        if xl_chk is not None:
                            chk_n += 1
                            if mx_chk < xl_chk - 12.0:
                                out_hits += 1
                if chk_n >= 3 and out_hits >= 2:
                    medial_ok = False
                    outside_leak = True
            if medial_ok and float(r_pts[0][1]) - float(r_pts[-1][1]) >= hT * 0.42:
                out_hits_r, chk_nr = 0, 0
                y_r_lo, y_r_hi = float(r_pts[-1][1]), float(r_pts[0][1])
                for mx_chk, my_chk in medial_pts:
                    if y_r_lo <= my_chk <= y_r_hi:
                        xr_chk = cls._eval_segments_x_at_y(right_segs, my_chk)
                        if xr_chk is not None:
                            chk_nr += 1
                            if mx_chk > xr_chk + 12.0:
                                out_hits_r += 1
                if chk_nr >= 3 and out_hits_r >= 2:
                    medial_ok = False
                    outside_leak = True
            if medial_ok:
                start_k = 1 if center_pts else 0
                for mx, my in medial_pts[start_k:]:
                    if not center_pts or math.hypot(mx - center_pts[-1][0], my - center_pts[-1][1]) >= 3.5:
                        center_pts.append((mx, my))
                        pred_flags.append(False)
            elif outside_leak or (not center_pts and abs(float(l_pts[0][1]) - float(r_pts[0][1])) > 24.0):
                use_left_dom = (l_arc >= r_arc) if outside_leak else (float(l_pts[0][1]) > float(r_pts[0][1]))
                dom_segs = left_segs if use_left_dom else right_segs
                dom_sign = +1.0 if use_left_dom else -1.0
                off_pts = cls._offset_segments_by_normal(
                    dom_segs,
                    dom_sign,
                    a_lane,
                    b_lane,
                    hT,
                    wT,
                    arm_width_hint=arm_width_hint,
                    allow_thin_fallback=(not lane_width_calibrated),
                )
                if center_pts:
                    y_top_cur = float(center_pts[-1][1])
                    valid_off = [(float(mx), float(my)) for mx, my in off_pts if float(my) < y_top_cur - 2.0]
                    if valid_off:
                        dx_join = float(center_pts[-1][0]) - float(valid_off[0][0])
                        n_blend = min(len(valid_off), 6)
                        for bi, (mx_raw, my_raw) in enumerate(valid_off):
                            w_blend = max(0.0, 1.0 - bi / float(n_blend)) if bi < n_blend else 0.0
                            mx = mx_raw + dx_join * w_blend
                            if math.hypot(mx - center_pts[-1][0], my_raw - center_pts[-1][1]) >= 3.5:
                                center_pts.append((float(mx), float(my_raw)))
                                pred_flags.append(False)
                else:
                    for mx, my in off_pts:
                        if not center_pts or math.hypot(mx - center_pts[-1][0], my - center_pts[-1][1]) >= 3.5:
                            center_pts.append((float(mx), float(my)))
                            pred_flags.append(False)
            else:
                l_full = cls._sample_polyline(left_segs, 18)
                r_full = cls._sample_polyline(right_segs, 18)
                if len(l_full) >= 2 and len(r_full) >= 2:
                    for pl, pr in zip(l_full, r_full):
                        cx_m = _clip_scalar(0.5 * float(pl[0] + pr[0]), 4.0, wT - 5.0)
                        cy_m = _clip_scalar(0.5 * float(pl[1] + pr[1]), 4.0, hT - 4.0)
                        if not center_pts or math.hypot(cx_m - center_pts[-1][0], cy_m - center_pts[-1][1]) >= 3.5:
                            center_pts.append((cx_m, cy_m))
                            pred_flags.append(False)

            if has_corner_geom and len(center_pts) >= 5:
                pts_sm = np.asarray(center_pts, dtype=np.float32)
                for _ in range(3):
                    nxt = pts_sm.copy()
                    nxt[1:-1] = 0.25 * pts_sm[:-2] + 0.50 * pts_sm[1:-1] + 0.25 * pts_sm[2:]
                    pts_sm = nxt
                center_pts = [(float(p[0]), float(p[1])) for p in pts_sm]

            if center_pts and (left_segs or right_segs):
                for i in range(len(center_pts)):
                    px_c, py_c = float(center_pts[i][0]), float(center_pts[i][1])
                    if right_segs:
                        xr_bnd = cls._eval_segments_x_at_y(right_segs, py_c)
                        if xr_bnd is not None:
                            px_c = min(px_c, float(xr_bnd) - 14.0)
                    if left_segs:
                        xl_bnd = cls._eval_segments_x_at_y(left_segs, py_c)
                        if xl_bnd is not None:
                            px_c = max(px_c, float(xl_bnd) + 14.0)
                    center_pts[i] = (px_c, py_c)

            return np.asarray(center_pts, dtype=np.float32), np.asarray(pred_flags, dtype=bool), (a_lane, b_lane)

        if not has_l and not has_r:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool), (a_lane, b_lane)

        # Single-Edge Mode: Chi con 1 bien -> Dung khoang cach phap tuyen 0.5 * W_lane ket hop Bo nho be rong goc cua
        active_segs = left_segs if has_l else right_segs
        side_sign = +1.0 if has_l else -1.0
        center_pts = cls._offset_segments_by_normal(
            active_segs,
            side_sign,
            a_lane,
            b_lane,
            hT,
            wT,
            arm_width_hint=arm_width_hint,
            allow_thin_fallback=(not lane_width_calibrated),
            corner_memory_active=corner_memory_active,
        )
        pred_flags = [True] * len(center_pts)

        return np.asarray(center_pts, dtype=np.float32), np.asarray(pred_flags, dtype=bool), (a_lane, b_lane)

    @staticmethod
    def _draw_blue_tapered_segments(img, segments, color=(255, 105, 0), border_color=(255, 190, 60)):
        """
        Draw each piecewise straight segment in blue along the center of the real white tape,
        with stroke thickness tapering linearly from width_start to width_end.
        """
        for seg in segments:
            x1, y1 = seg["p1"]
            x2, y2 = seg["p2"]
            vx = float(x2 - x1)
            vy = float(y2 - y1)
            length = math.hypot(vx, vy)
            if length < 1.0:
                continue
            inv_l = 1.0 / length
            nx = -vy * inv_l
            ny = vx * inv_l
            w1 = max(2.0, float(seg.get("width_start", seg["width"])))
            w2 = max(2.0, float(seg.get("width_end", seg["width"])))
            hw1 = 0.5 * w1
            hw2 = 0.5 * w2

            q1 = (round(x1 - hw1 * nx), round(y1 - hw1 * ny))
            q2 = (round(x1 + hw1 * nx), round(y1 + hw1 * ny))
            q3 = (round(x2 + hw2 * nx), round(y2 + hw2 * ny))
            q4 = (round(x2 - hw2 * nx), round(y2 - hw2 * ny))
            quad = np.array([q1, q2, q3, q4], dtype=np.int32)
            cv2.fillConvexPoly(img, quad, color, lineType=cv2.LINE_8)
            cv2.polylines(img, [quad], isClosed=True, color=border_color, thickness=1, lineType=cv2.LINE_8)

        for idx, seg in enumerate(segments):
            pt1 = (int(round(seg["p1"][0])), int(round(seg["p1"][1])))
            cv2.circle(img, pt1, 3, (255, 240, 150), cv2.FILLED, lineType=cv2.LINE_8)
            if idx == len(segments) - 1:
                pt2 = (int(round(seg["p2"][0])), int(round(seg["p2"][1])))
                cv2.circle(img, pt2, 3, (255, 240, 150), cv2.FILLED, lineType=cv2.LINE_8)

    def _associate_boundaries(self, candidates, h, w):
        previous = getattr(self, "_previous_boundaries", [])
        used = set()
        boundaries = []
        inv_w = 1.0 / float(max(1, w))
        inv_h = 1.0 / float(max(1, h))
        for candidate in candidates:
            pts = np.asarray(candidate["points"], float)
            step = max(1, len(pts) // 30)
            sample_x = pts[::step, 0] * inv_w
            sample_y = pts[::step, 1] * inv_h
            matches = []
            for old in previous:
                if old["id"] in used:
                    continue
                op = np.asarray(old["points"], float)
                op_step = max(1, len(op) // 30)
                op_x = op[::op_step, 0] * inv_w
                op_y = op[::op_step, 1] * inv_h
                dx = sample_x[:, None] - op_x[None, :]
                dy = sample_y[:, None] - op_y[None, :]
                d2 = dx * dx + dy * dy
                cost = float(np.sqrt(np.min(d2, axis=1)).mean())
                matches.append((cost, old["id"]))
            if matches and min(matches)[0] < 0.12:
                identity = min(matches)[1]
            else:
                identity = self._next_boundary_id
                self._next_boundary_id += 1
            used.add(identity)
            boundaries.append({
                "id": identity,
                "side": candidate.get("side", "UNKNOWN"),
                "observed": True,
                "points": candidate["points"],
                "segments": candidate.get("segments", []),
                "support_score": float(min(1.0, candidate["length"] / np.hypot(h, w))),
                "median_width": candidate["width"],
            })
        self._previous_boundaries = boundaries
        self.observed_boundaries = boundaries
        self.boundary_traces = [b["points"] for b in boundaries]

    def _classify_single_boundary_side(self, pts, segs, hT, wT):
        """
        Phan biet chinh xac vach don le (khi mat 1 vach) la VACH TRAI (LEFT) hay VACH PHAI (RIGHT)
        dua tren 5 yeu to:
          1. Vi tri du kien tu mo hinh chieu rong lan W_lane(y) + do lech e gan nhat.
          2. Bo nho bam vet (Temporal tracking) cua vach Trai/Phai o frame truoc.
          3. Bo nho lo trinh goc cua (Corner Trajectory Memory) neu xe dang trong khuc cua.
          4. Vi tri chan vach (bot_x) so voi truc tham chieu xe (wT * 0.5).
          5. Do nghieng phoi canh / huong uon cong cua vach.
        """
        if len(pts) == 0 or not segs:
            return "LEFT"

        bot_x = float(pts[0, 0])
        bot_y = float(pts[0, 1])
        mid_x = wT * 0.5
        span_x = float(pts[:, 0].max() - pts[:, 0].min())
        span_y = float(pts[:, 1].max() - pts[:, 1].min())

        # Thanh vach ngang xuyen khung hinh o giua goc cua (khong co than doc): xac dinh theo huong cua dang nho
        min_y_all = float(pts[:, 1].min())
        if (span_x >= wT * 0.35 and span_y < 48.0 and bot_y < hT * 0.80) or (
            span_x >= wT * 0.22 and span_y < 48.0 and min_y_all >= hT * 0.35 and bot_y < hT * 0.82
        ):
            turn_ctx = "UNKNOWN"
            if self.corner_memory.get("active") and self.corner_memory.get("ttl", 0) > 0:
                turn_ctx = self.corner_memory.get("direction", "UNKNOWN")
            elif self._active_turn_dir in ("LEFT", "RIGHT"):
                turn_ctx = self._active_turn_dir
            elif self._intent_turn_dir in ("LEFT", "RIGHT"):
                turn_ctx = self._intent_turn_dir
            if turn_ctx == "RIGHT":
                return "LEFT"
            if turn_ctx == "LEFT":
                return "RIGHT"
            p_left = pts[int(np.argmin(pts[:, 0]))]
            p_right = pts[int(np.argmax(pts[:, 0]))]
            if float(p_right[1]) < float(p_left[1]) - 6.0:
                return "LEFT"
            if float(p_left[1]) < float(p_right[1]) - 6.0:
                return "RIGHT"

        # Score < 0 -> LEFT, Score > 0 -> RIGHT
        score = 0.0

        # 1. Temporal polyline & endpoint continuity from previous frame (prevents L<->R flips across mid_x)
        if self._missing_s <= 0.45 and self.missing_frames <= 6:
            d_poly_l = 999.0
            d_poly_r = 999.0
            if self._smooth_left_segs:
                p_prev_l = self._sample_polyline(self._smooth_left_segs, 12)
                p_cur = pts[:: max(1, len(pts) // 12)]
                if len(p_prev_l) >= 2 and len(p_cur) >= 2:
                    d_mat = np.hypot(p_cur[:, None, 0] - p_prev_l[None, :, 0], p_cur[:, None, 1] - p_prev_l[None, :, 1])
                    d_poly_l = float(np.median(d_mat.min(axis=1)))
            if self._smooth_right_segs:
                p_prev_r = self._sample_polyline(self._smooth_right_segs, 12)
                p_cur = pts[:: max(1, len(pts) // 12)]
                if len(p_prev_r) >= 2 and len(p_cur) >= 2:
                    d_mat = np.hypot(p_cur[:, None, 0] - p_prev_r[None, :, 0], p_cur[:, None, 1] - p_prev_r[None, :, 1])
                    d_poly_r = float(np.median(d_mat.min(axis=1)))
            if d_poly_l < 65.0 and d_poly_l + 16.0 < d_poly_r:
                score -= 4.2
            elif d_poly_r < 65.0 and d_poly_r + 16.0 < d_poly_l:
                score += 4.2
            else:
                dl_prev = abs(bot_x - self._last_left_bot_x) if self._last_left_bot_x is not None else 999.0
                dr_prev = abs(bot_x - self._last_right_bot_x) if self._last_right_bot_x is not None else 999.0
                if dl_prev < 58.0 and dl_prev + 18.0 < dr_prev:
                    score -= 3.2
                elif dr_prev < 58.0 and dr_prev + 18.0 < dl_prev:
                    score += 3.2

        # 2. Expected lane width position (mild weight so off-center/yaw does not override temporal continuity)
        min_w = max(24.0, wT * 0.06)
        w_lane_bot = float(np.clip(self.lane_width_ab[0] * bot_y + self.lane_width_ab[1], min_w, wT * 1.15))
        recent_e = float(self.smooth_e_px) if (self.smooth_e_px is not None and self.missing_frames <= 6) else 0.0
        exp_center_x = mid_x + float(np.clip(recent_e, -wT * 0.35, wT * 0.35))
        exp_left_x = exp_center_x - 0.5 * w_lane_bot
        exp_right_x = exp_center_x + 0.5 * w_lane_bot
        d_left = abs(bot_x - exp_left_x)
        d_right = abs(bot_x - exp_right_x)
        score += float(np.clip((d_left - d_right) / max(32.0, 0.35 * w_lane_bot), -1.4, 1.4))

        # 3. Bo nho lo trinh goc cua: khi vao cua PHAI, vach bung phai thuong khuat truoc, con lai vach lung TRAI
        active_mem_dir = (
            self.corner_memory.get("direction", "UNKNOWN")
            if (self.corner_memory.get("active") and self.corner_memory.get("ttl", 0) > 0)
            else (self._active_turn_dir if self._active_turn_dir in ("LEFT", "RIGHT") else "UNKNOWN")
        )
        if active_mem_dir in ("LEFT", "RIGHT"):
            dx_total = float(segs[-1]["p2"][0] - segs[0]["p1"][0])
            span_x_m = float(pts[:, 0].max() - pts[:, 0].min())
            span_y_m = float(pts[:, 1].max() - pts[:, 1].min())
            min_y_m = float(pts[:, 1].min())
            if active_mem_dir == "RIGHT" and dx_total > 45.0:
                score -= 3.0
            elif active_mem_dir == "RIGHT" and dx_total > 20.0 and bot_x <= mid_x + 30.0:
                score -= 1.5
            elif active_mem_dir == "RIGHT" and span_x_m >= wT * 0.22 and span_y_m <= 34.0 and min_y_m >= hT * 0.35:
                score -= 3.5
            elif active_mem_dir == "LEFT" and dx_total < -45.0:
                score += 3.0
            elif active_mem_dir == "LEFT" and dx_total < -20.0 and bot_x >= mid_x - 30.0:
                score += 1.5
            elif active_mem_dir == "LEFT" and span_x_m >= wT * 0.22 and span_y_m <= 34.0 and min_y_m >= hT * 0.35:
                score += 3.5

        # 3b. Quy luat phoi canh & do doc don dieu: vach don le doc/cheo tu day anh nghieng sang Phai (dx >= +22,
        # khong co chan nghieng trai, dinh vach nam han ben phai tam anh) la VACH TRAI; nguoc lai nghieng sang Trai la VACH PHAI.
        dx_all = float(segs[-1]["p2"][0] - segs[0]["p1"][0])
        dy_all = float(segs[0]["p1"][1] - segs[-1]["p2"][1])
        dx_s0 = float(segs[0]["p2"][0] - segs[0]["p1"][0])
        top_x_all = float(segs[-1]["p2"][0])
        if dy_all >= 35.0 and dx_all >= 22.0 and dx_s0 >= 15.0 and top_x_all >= mid_x + 70.0:
            if active_mem_dir != "LEFT":
                score -= 6.5
        elif dy_all >= 35.0 and dx_all <= -22.0 and dx_s0 <= -15.0 and top_x_all <= mid_x - 70.0:
            if active_mem_dir != "RIGHT":
                score += 6.5
        elif dy_all >= 35.0 and dx_all >= 14.0 and dx_s0 >= 10.0 and max(bot_x, top_x_all) <= mid_x - 18.0:
            if active_mem_dir != "LEFT":
                score -= 5.5
        elif dy_all >= 35.0 and dx_all <= -14.0 and dx_s0 <= -10.0 and min(bot_x, top_x_all) >= mid_x + 18.0:
            if active_mem_dir != "RIGHT":
                score += 5.5

        # 4. Road-side context (floor/curb asymmetry on textured/color roads)
        img_g = getattr(self, "_last_img_gray", None)
        rmed = getattr(self, "_last_road_med_gray", None)
        if img_g is not None and rmed is not None and rmed >= 25.0 and len(pts) >= 4:
            p_b = pts[0]
            p_t = pts[-1]
            vx = float(p_t[0] - p_b[0])
            vy = float(p_t[1] - p_b[1])
            if vy > 4.0:
                vx, vy = -vx, -vy
            vn = math.hypot(vx, vy)
            if vn >= 12.0:
                # Image-right normal (-vy, vx)/vn, image-left normal (+vy, -vx)/vn
                nrx, nry = -vy / vn, vx / vn
                nlx, nly = vy / vn, -vx / vn
                vals_l, vals_r = [], []
                for idx_p in np.linspace(0, len(pts) - 1, min(6, len(pts)), dtype=int):
                    px_i, py_i = float(pts[idx_p, 0]), float(pts[idx_p, 1])
                    if py_i < 18.0 or py_i > hT - 12.0:
                        continue
                    for d_off in (24.0, 42.0, 62.0):
                        xr_s = int(round(px_i + d_off * nrx))
                        yr_s = int(round(py_i + d_off * nry))
                        if 0 <= xr_s < wT and 0 <= yr_s < hT:
                            vals_r.append(float(img_g[yr_s, xr_s]))
                        xl_s = int(round(px_i + d_off * nlx))
                        yl_s = int(round(py_i + d_off * nly))
                        if 0 <= xl_s < wT and 0 <= yl_s < hT:
                            vals_l.append(float(img_g[yl_s, xl_s]))
                if len(vals_l) >= 4 and len(vals_r) >= 4:
                    med_l = _fast_median_seq(vals_l)
                    med_r = _fast_median_seq(vals_r)
                    out_l = any(v >= rmed + 52.0 or v <= rmed - 18.0 for v in vals_l)
                    out_r = any(v >= rmed + 52.0 or v <= rmed - 18.0 for v in vals_r)
                    clean_l = abs(med_l - rmed) <= 28.0
                    clean_r = abs(med_r - rmed) <= 28.0
                    if med_l >= 158.0 and med_r <= min(132.0, rmed + 38.0) and (med_l - med_r) >= 52.0:
                        score -= 6.5
                    elif med_r >= 158.0 and med_l <= min(132.0, rmed + 38.0) and (med_r - med_l) >= 52.0:
                        score += 6.5
                    elif out_r and clean_l and not out_l:
                        score += 2.2
                    elif out_l and clean_r and not out_r:
                        score -= 2.2

        # 5. Mild bottom position tie-breaker (does not overpower continuity/slope/context)
        score += float(np.clip((bot_x - mid_x) / (wT * 0.22), -1.0, 1.0))

        # 6. Perspective convergence slope of the lower stem
        s0 = segs[0]
        dy_up = float(s0["p1"][1] - s0["p2"][1])
        dx_up = float(s0["p2"][0] - s0["p1"][0])
        if dy_up >= 16.0:
            slope_up = dx_up / dy_up
            if slope_up > 0.15:
                score -= 1.6
            elif slope_up < -0.15:
                score += 1.6

        return "RIGHT" if score > 0.0 else "LEFT"

    def inspect_corner(self, mask=None, fresh_observation=True, img_shape=None, dt_s=None):
        """
        Evaluate corner direction, elbow coordinates, turn angle, and distance-to-corner gating:
          - Supports both sharp 90-degree L-corners and multi-segment chamfered/rounded corners.
          - Distinguishes FAR / APPROACH corners from TURN_NOW corners.
          - Updates and maintains Pre-Corner Trajectory Memory (self.corner_memory).
        """
        if img_shape is not None:
            h, w = int(img_shape[0]), int(img_shape[1])
        elif mask is not None:
            h, w = mask.shape[:2]
        else:
            h, w = 240, 480
        dt_step = float(dt_s) if (dt_s is not None and fresh_observation) else (self._last_dt_s if fresh_observation else 0.0)
        evidence = {"LEFT": [], "RIGHT": []}
        stem_spans = {"LEFT": [], "RIGHT": []}
        start_ys = {"LEFT": [], "RIGHT": []}
        turn_angles = {"LEFT": [], "RIGHT": []}
        ground_turn_angles = {"LEFT": [], "RIGHT": []}
        votes = []
        single_straight_drifted = False

        a_lane_g, b_lane_g = self.lane_width_ab
        min_wg = max(24.0, w * 0.06)
        w_bot_g = max(min_wg, float(np.clip(a_lane_g * (h - 6.0) + b_lane_g, min_wg, w * 1.35)))
        f_px_g = 0.90 * float(w)
        h0_g = (float(h) * 0.5) + (b_lane_g / max(0.25, a_lane_g))
        cos_a_g = f_px_g / max(1.0, math.hypot(f_px_g, h0_g))
        k_persp_g = f_px_g / max(0.35, cos_a_g)

        def _to_ground_pt(px_v, py_v):
            wg = max(min_wg, float(np.clip(a_lane_g * py_v + b_lane_g, min_wg, w * 1.35)))
            xg = (float(px_v) - float(w) * 0.5) / wg
            yg = max(0.30, 1.67 + k_persp_g * (1.0 / wg - 1.0 / w_bot_g))
            return xg, yg

        def _ground_seg_angle(seg_st, seg_ex):
            g_st1, g_st2 = _to_ground_pt(*seg_st["p1"]), _to_ground_pt(*seg_st["p2"])
            g_ex1, g_ex2 = _to_ground_pt(*seg_ex["p1"]), _to_ground_pt(*seg_ex["p2"])
            vx_g1, vy_g1 = g_st2[0] - g_st1[0], g_st2[1] - g_st1[1]
            vx_g2, vy_g2 = g_ex2[0] - g_ex1[0], g_ex2[1] - g_ex1[1]
            n1 = math.hypot(vx_g1, vy_g1)
            n2 = math.hypot(vx_g2, vy_g2)
            if n1 < 1e-4 or n2 < 1e-4:
                return 0.0
            cos_g = _clip_scalar((vx_g1 * vx_g2 + vy_g1 * vy_g2) / (n1 * n2), -1.0, 1.0)
            return float(math.degrees(math.acos(cos_g)))

        for boundary in self.observed_boundaries:
            segs = boundary.get("segments", [])
            b_side = boundary.get("side", "UNKNOWN")
            if not segs:
                votes.append("UNKNOWN")
                continue
            best_turn = 0.0
            best_ground_turn = 0.0
            best_dir = "UNKNOWN"
            best_elbow = None
            best_stem_span = 0.0
            p_start = segs[0]["p1"]
            p_end = segs[-1]["p2"]
            dx_all = float(p_end[0] - p_start[0])
            dy_up_all = max(1.0, float(p_start[1] - p_end[1]))
            slope_ratio_all = abs(dx_all) / dy_up_all

            def _is_border_clip_diag(seg_st, seg_nx):
                x_lo = float(min(seg_st["p1"][0], seg_st["p2"][0]))
                x_hi = float(max(seg_st["p1"][0], seg_st["p2"][0]))
                vx = float(seg_nx["p2"][0] - seg_nx["p1"][0])
                vy = float(seg_nx["p2"][1] - seg_nx["p1"][1])
                if b_side == "LEFT" and x_hi <= 20.0 and vx > 0.0:
                    return (-vy >= h * 0.30) and (abs(vx) <= abs(vy) * 1.45) and float(seg_nx["p2"][1]) <= h * 0.32
                if b_side == "RIGHT" and x_lo >= float(w) - 21.0 and vx < 0.0:
                    return (-vy >= h * 0.30) and (abs(vx) <= abs(vy) * 1.45) and float(seg_nx["p2"][1]) <= h * 0.32
                return False

            if len(segs) >= 2:
                has_border_clip_main = _is_border_clip_diag(segs[0], segs[1])
                elbow_idx = self._find_corner_elbow_seg_idx(segs)
                if elbow_idx is not None and not (elbow_idx == 1 and has_border_clip_main):
                    # Dung vector than chinh tu chan vach segs[0]["p1"] den dinh khuyu (tranh bi nhieu do doan vat goc 45 deg)
                    s_stem = segs[elbow_idx - 1]
                    s_stem_macro = {"p1": segs[0]["p1"], "p2": segs[elbow_idx]["p1"]}
                    s_exit = segs[-1]
                    if float(s_exit["length"]) < 26.0 and elbow_idx < len(segs) - 1:
                        s_exit_macro = {"p1": segs[elbow_idx]["p1"], "p2": segs[-1]["p2"]}
                    else:
                        s_exit_macro = s_exit
                    vx_st = float(s_stem["p2"][0] - s_stem["p1"][0])
                    vy_st = float(s_stem["p2"][1] - s_stem["p1"][1])
                    vx_ex = float(s_exit["p2"][0] - s_exit["p1"][0])
                    vy_ex = float(s_exit["p2"][1] - s_exit["p1"][1])
                    l_st = math.hypot(vx_st, vy_st)
                    l_ex = math.hypot(vx_ex, vy_ex)
                    if l_st >= 10.0 and l_ex >= 12.0 and abs(vx_ex) >= abs(vy_ex) * 0.65:
                        cos_tot = _clip_scalar((vx_st * vx_ex + vy_st * vy_ex) / (l_st * l_ex), -1.0, 1.0)
                        tot_deg = math.degrees(math.acos(cos_tot))
                        cross_tot = vx_st * vy_ex - vy_st * vx_ex
                        is_straight_persp = (
                            (b_side == "RIGHT" and vx_st <= 2.0 and vx_ex < 0 and tot_deg < 48.0 and abs(vx_ex) < abs(vy_ex) * 1.25)
                            or (b_side == "LEFT" and vx_st >= -2.0 and vx_ex > 0 and tot_deg < 48.0 and abs(vx_ex) < abs(vy_ex) * 1.25)
                            or _is_border_clip_diag(s_stem, s_exit)
                        )
                        if tot_deg >= 20.0 and not is_straight_persp:
                            best_turn = tot_deg
                            best_ground_turn = _ground_seg_angle(s_stem_macro, s_exit_macro)
                            best_dir = "RIGHT" if (cross_tot > 0 or vx_ex > 18.0) else "LEFT"
                            best_elbow = (float(segs[elbow_idx]["p1"][0]), float(segs[elbow_idx]["p1"][1]))
                            best_stem_span = max(0.0, float(segs[0]["p1"][1] - segs[elbow_idx]["p1"][1]))

                for s1, s2 in zip(segs[:-1], segs[1:]):
                    if _is_border_clip_diag(s1, s2):
                        continue
                    if has_border_clip_main and float(s1["p1"][1]) <= h * 0.25 and float(s1["length"]) < 65.0:
                        continue
                    v1x = float(s1["p2"][0] - s1["p1"][0])
                    v1y = float(s1["p2"][1] - s1["p1"][1])
                    v2x = float(s2["p2"][0] - s2["p1"][0])
                    v2y = float(s2["p2"][1] - s2["p1"][1])
                    l1 = math.hypot(v1x, v1y)
                    l2 = math.hypot(v2x, v2y)
                    if l1 < 12.0 or l2 < 12.0:
                        continue
                    if v1x * v2x < -100.0 and abs(v1x) > 2.0 * abs(v1y) and v2y > -12.0:
                        continue
                    cos_a = _clip_scalar((v1x * v2x + v1y * v2y) / (l1 * l2), -1.0, 1.0)
                    ang_deg = math.degrees(math.acos(cos_a))
                    cross_z = v1x * v2y - v1y * v2x
                    if ang_deg >= 20.0 and abs(v2x) >= abs(v2y) * 0.75:
                        # Bo qua vet meo nho o sat can xe (y > 0.72*h) noi len than vach thang phoi canh binh thuong
                        is_bottom_bumper_kink = (
                            float(s1["p2"][1]) > h * 0.72
                            and v2y < -22.0
                            and abs(v2x) < abs(v2y) * 1.22
                        ) or (
                            float(s1["p1"][1]) >= h * 0.88
                            and float(s1["p2"][1]) > h * 0.75
                            and l1 < 48.0
                            and v2y < -18.0
                        )
                        if is_bottom_bumper_kink:
                            continue
                        # cross_z > 0: uon cong theo chieu kim dong ho (re PHAI); cross_z < 0: nguoc chieu kim dong ho (re TRAI)
                        if (b_side == "RIGHT" and v1x <= 2.0 and v2x < 0 and ang_deg < 48.0 and abs(v2x) < abs(v2y) * 1.25) or (
                            b_side == "LEFT" and v1x >= -2.0 and v2x > 0 and ang_deg < 48.0 and abs(v2x) < abs(v2y) * 1.25
                        ):
                            continue
                        direction = "RIGHT" if cross_z > 0 else "LEFT"
                        if best_dir == "UNKNOWN" or ang_deg > best_turn:
                            best_turn = max(best_turn, ang_deg)
                            if best_ground_turn <= 1.0:
                                best_ground_turn = _ground_seg_angle(s1, s2)
                            best_dir = direction
                            if best_elbow is None:
                                best_elbow = (float(s1["p2"][0]), float(s1["p2"][1]))
                                best_stem_span = max(0.0, float(segs[0]["p1"][1] - s1["p2"][1]))

            if best_dir == "UNKNOWN":
                if len(self.observed_boundaries) == 1 and (
                    (float(p_start[1]) >= h * 0.85 and slope_ratio_all < 1.20)
                    or (float(p_start[1]) >= h * 0.65 and slope_ratio_all < 1.15)
                ):
                    single_straight_drifted = True
                elif b_side == "LEFT" and (
                    (float(p_start[1]) < h * 0.85 and dx_all >= w * 0.26 and slope_ratio_all >= 1.40)
                    or (len(self.observed_boundaries) == 1 and dx_all >= w * 0.45 and slope_ratio_all >= 1.35)
                ):
                    best_dir = "RIGHT"
                    best_turn = 75.0
                    best_ground_turn = 82.0
                    best_elbow = (
                        float(0.5 * (p_start[0] + p_end[0])),
                        float(0.65 * p_start[1] + 0.35 * p_end[1]),
                    )
                    best_stem_span = 0.0
                elif b_side == "RIGHT" and (
                    (float(p_start[1]) < h * 0.85 and dx_all <= -w * 0.26 and slope_ratio_all >= 1.40)
                    or (len(self.observed_boundaries) == 1 and dx_all <= -w * 0.45 and slope_ratio_all >= 1.35)
                ):
                    best_dir = "LEFT"
                    best_turn = 75.0
                    best_ground_turn = 82.0
                    best_elbow = (
                        float(0.5 * (p_start[0] + p_end[0])),
                        float(0.65 * p_start[1] + 0.35 * p_end[1]),
                    )
                    best_stem_span = 0.0

            if best_dir != "UNKNOWN" and best_elbow is not None:
                votes.append(best_dir)
                evidence[best_dir].append(best_elbow)
                stem_spans[best_dir].append(best_stem_span)
                start_ys[best_dir].append(float(p_start[1]))
                turn_angles[best_dir].append(best_turn)
                ground_turn_angles[best_dir].append(best_ground_turn if best_ground_turn > 1.0 else best_turn)
            else:
                votes.append("UNKNOWN")

        valid_votes = [v for v in votes if v != "UNKNOWN"]
        # Khi nhin thay 2 vach: neu 1 ben chi co khuc khuyu thap do loa (elbow_y > 0.62*h)
        # trong khi vach ngoai phia doi dien chay thang lien tuc len sat mep tren (p2.y < 0.28*h), bo qua bao cua gia
        if len(valid_votes) == 1 and len(self.observed_boundaries) == 2:
            sole_dir = valid_votes[0]
            eb_list = evidence.get(sole_dir, [])
            if eb_list and float(eb_list[0][1]) > h * 0.62:
                opp_side = "LEFT" if sole_dir == "RIGHT" else "RIGHT"
                for b_opp in self.observed_boundaries:
                    if b_opp.get("side") == opp_side:
                        opp_segs = b_opp.get("segments", [])
                        if opp_segs:
                            op1, op2 = opp_segs[0]["p1"], opp_segs[-1]["p2"]
                            ody = max(1.0, float(op1[1] - op2[1]))
                            odx = abs(float(op2[0] - op1[0]))
                            opp_max_ang = 0.0
                            for s1_o, s2_o in zip(opp_segs[:-1], opp_segs[1:]):
                                ov1 = np.array(s1_o["p2"], dtype=np.float32) - np.array(s1_o["p1"], dtype=np.float32)
                                ov2 = np.array(s2_o["p2"], dtype=np.float32) - np.array(s2_o["p1"], dtype=np.float32)
                                ol1, ol2 = float(np.linalg.norm(ov1)), float(np.linalg.norm(ov2))
                                if ol1 >= 8.0 and ol2 >= 8.0:
                                    ocos = float(np.clip(np.dot(ov1, ov2) / (ol1 * ol2), -1.0, 1.0))
                                    opp_max_ang = max(opp_max_ang, float(np.rad2deg(np.arccos(ocos))))
                            if float(op2[1]) < h * 0.28 and ody >= h * 0.48 and opp_max_ang < 22.0 and odx < 1.15 * ody:
                                valid_votes = []
                                votes = ["UNKNOWN", "UNKNOWN"]
                                evidence[sole_dir] = []
                        break

        hypothesis = valid_votes[0] if len(valid_votes) == 1 else (valid_votes[0] if len(valid_votes) == 2 and valid_votes[0] == valid_votes[1] else "UNKNOWN")
        if len(valid_votes) == 2 and valid_votes[0] == valid_votes[1]:
            candidate = valid_votes[0]
        elif len(valid_votes) == 2 and self._intent_turn_dir in valid_votes:
            candidate = self._intent_turn_dir
        elif len(valid_votes) == 1 and len(self.observed_boundaries) >= 1:
            candidate = valid_votes[0]
        elif (
            len(self.observed_boundaries) == 1
            and not single_straight_drifted
            and self._intent_turn_dir in ("LEFT", "RIGHT")
        ):
            b_only = self.observed_boundaries[0]
            b_side = b_only.get("side", "UNKNOWN")
            if (self._intent_turn_dir == "RIGHT" and b_side == "LEFT") or (self._intent_turn_dir == "LEFT" and b_side == "RIGHT"):
                b_segs = b_only.get("segments", [])
                if b_segs:
                    p_st = b_segs[0]["p1"]
                    p_en = b_segs[-1]["p2"]
                    s_ratio = abs(float(p_en[0] - p_st[0])) / max(1.0, float(p_st[1] - p_en[1]))
                    if float(p_st[1]) < h * 0.84 or s_ratio >= 1.25:
                        candidate = self._intent_turn_dir
                        hypothesis = candidate
                        eb_pt = (float(b_segs[0]["p2"][0]), float(b_segs[0]["p2"][1]))
                        evidence[candidate].append(eb_pt)
                        stem_spans[candidate].append(max(0.0, float(b_segs[0]["p1"][1] - b_segs[0]["p2"][1])))
                        start_ys[candidate].append(float(p_st[1]))
                        turn_angles[candidate].append(float(self.corner_memory.get("turn_angle_deg", 85.0)))
                    else:
                        candidate = "UNKNOWN"
                else:
                    candidate = "UNKNOWN"
            else:
                candidate = "UNKNOWN"
        else:
            candidate = "UNKNOWN"

        ids = tuple(sorted(b["id"] for b in self.observed_boundaries))
        both_agree = len(valid_votes) == 2 and valid_votes[0] == valid_votes[1]
        if not fresh_observation:
            self._corner_hits = max(2 if both_agree else 1, self._corner_hits) if candidate != "UNKNOWN" else 0
        elif candidate != "UNKNOWN" and (candidate == self._corner_candidate or candidate == self._intent_turn_dir or both_agree):
            self._corner_hits += 1
        else:
            self._corner_hits = int(candidate != "UNKNOWN")
        self._corner_candidate, self._corner_ids = candidate, ids
        confirmed = candidate != "UNKNOWN" and (both_agree or self._corner_hits >= 2 or candidate == self._intent_turn_dir)
        conf_val = (
            0.95
            if (both_agree and candidate != "UNKNOWN")
            else (0.82 if confirmed else (0.74 if candidate != "UNKNOWN" else (0.45 if hypothesis != "UNKNOWN" else 0.0)))
        )

        # Cap nhat Bo nho tam nhin bao quat goc cua (_intent_turn_dir) & xoa ngay corner_memory khi doi chieu cua (S-bend)
        if candidate in ("LEFT", "RIGHT") and (confirmed or len(valid_votes) >= 1):
            mem_dir = self.corner_memory.get("direction", "UNKNOWN")
            if self.corner_memory.get("active") and mem_dir in ("LEFT", "RIGHT") and candidate != mem_dir and (
                confirmed or len(evidence.get(candidate, [])) >= 1
            ):
                self.corner_memory.update({
                    "active": False,
                    "direction": "UNKNOWN",
                    "centerline": np.empty((0, 2), dtype=np.float32),
                    "arm_width_px": 0.0,
                    "ttl": 0,
                    "ttl_s": 0.0,
                    "age_frames": 0,
                    "age_s": 0.0,
                })
                self._smooth_center_pts = None
            if self._intent_turn_dir in ("LEFT", "RIGHT") and candidate != self._intent_turn_dir:
                self._active_turn_hold_s = 0.0
                self._active_turn_hold = 0
            self._intent_turn_dir = candidate
            self._intent_turn_ttl_s = 4.0
            self._intent_turn_ttl = 60
            self._straight_clear_hits = 0
        elif len(self.observed_boundaries) == 2 and candidate == "UNKNOWN":
            if fresh_observation:
                self._straight_clear_hits += 1
                if self._straight_clear_hits >= 5:
                    self._intent_turn_dir = "UNKNOWN"
                    self._intent_turn_ttl_s = 0.0
                    self._intent_turn_ttl = 0
                    self.corner_memory["active"] = False
                    self.corner_memory["direction"] = "UNKNOWN"
                    self.corner_memory["ttl"] = 0
                    self.corner_memory["ttl_s"] = 0.0
        elif self._intent_turn_ttl_s > 0.0 and fresh_observation:
            self._intent_turn_ttl_s = max(0.0, self._intent_turn_ttl_s - dt_step)
            self._intent_turn_ttl = int(round(self._intent_turn_ttl_s * 15.0))
            if self._intent_turn_ttl_s <= 0.0:
                self._intent_turn_dir = "UNKNOWN"

        elbow_y = None
        elbow_x = None
        elbow_y_ratio = 0.0
        stem_span_y = float(h)
        min_start_y = float(h)
        mean_turn_deg = 0.0
        mean_ground_turn_deg = 0.0
        distance_zone = "NONE"
        turn_now = False

        if candidate in ("LEFT", "RIGHT") and evidence[candidate]:
            elbows = evidence[candidate]
            elbow_y = float(np.max([pt[1] for pt in elbows]))
            # Smooth float dt-aware EMA for elbow_y (prevents integer truncation drift & 1-frame flapping)
            if not hasattr(self, "_smooth_elbow_y") or self._smooth_elbow_y is None or not fresh_observation:
                self._smooth_elbow_y = elbow_y
            else:
                alpha_eb = 1.0 - math.exp(-max(0.015, float(dt_step)) / 0.085)
                elbow_y = float(alpha_eb * elbow_y + (1.0 - alpha_eb) * float(self._smooth_elbow_y))
                self._smooth_elbow_y = elbow_y
            elbow_x = float(np.mean([pt[0] for pt in elbows]))
            elbow_y_ratio = elbow_y / float(max(1, h))
            if stem_spans[candidate]:
                stem_span_y = float(np.min(stem_spans[candidate]))
            if start_ys[candidate]:
                min_start_y = float(np.min(start_ys[candidate]))
            if turn_angles[candidate]:
                mean_turn_deg = float(np.mean(turn_angles[candidate]))
            if ground_turn_angles[candidate]:
                raw_g_deg = float(np.mean(ground_turn_angles[candidate]))
                prev_g = getattr(self, "_smooth_ground_turn_deg", None)
                if prev_g is None or not fresh_observation:
                    mean_ground_turn_deg = raw_g_deg
                else:
                    mean_ground_turn_deg = float(0.35 * raw_g_deg + 0.65 * prev_g)
                self._smooth_ground_turn_deg = mean_ground_turn_deg
            else:
                mean_ground_turn_deg = mean_turn_deg

            is_single_edge = len(self.observed_boundaries) == 1
            corner_ready = (mean_turn_deg >= 38.0) and (both_agree or self._corner_hits >= 2 or not fresh_observation)
            single_edge_sharp_ready = (
                is_single_edge
                and elbow_y_ratio >= 0.56
                and mean_turn_deg >= 58.0
                and (self._corner_hits >= 2 or not fresh_observation)
            )

            if (
                (elbow_y_ratio >= 0.65 and corner_ready)
                or (elbow_y_ratio >= 0.58 and stem_span_y <= h * 0.12 and corner_ready)
                or (is_single_edge and min_start_y < h * 0.68 and corner_ready)
                or single_edge_sharp_ready
            ):
                distance_zone = "TURN_NOW"
                turn_now = True
                confirmed = True
                self._active_turn_dir = candidate
                self._active_turn_hold_s = 0.53
                self._active_turn_hold = 8
            elif (
                is_single_edge
                and self._active_turn_hold_s > 0.0
                and candidate == self._active_turn_dir
                and min_start_y < h * 0.76
            ):
                if fresh_observation:
                    self._active_turn_hold_s = max(0.0, self._active_turn_hold_s - dt_step)
                    self._active_turn_hold = int(round(self._active_turn_hold_s * 15.0))
                distance_zone = "TURN_NOW"
                turn_now = True
                confirmed = True
            elif elbow_y_ratio >= 0.35 and mean_turn_deg >= 30.0:
                distance_zone = "APPROACH"
                turn_now = False
            elif (
                (self._intent_turn_dir == candidate or self.corner_memory.get("active"))
                and min_start_y < h * 0.62
            ):
                # Giua goc cua: than vach doc da troi qua mep duoi -> khong duoc roi ve FAR
                if is_single_edge and (min_start_y < h * 0.50 or corner_ready):
                    distance_zone = "TURN_NOW"
                    turn_now = True
                    confirmed = True
                    self._active_turn_dir = candidate
                    self._active_turn_hold_s = 0.53
                    self._active_turn_hold = 8
                else:
                    distance_zone = "APPROACH"
                    turn_now = False
            else:
                distance_zone = "FAR"
                turn_now = False
        elif self._active_turn_hold_s > 0.0 and len(self.observed_boundaries) <= 1 and not single_straight_drifted:
            if fresh_observation:
                self._active_turn_hold_s = max(0.0, self._active_turn_hold_s - dt_step)
                self._active_turn_hold = int(round(self._active_turn_hold_s * 15.0))
            candidate = self._active_turn_dir
            confirmed = True
            distance_zone = "TURN_NOW"
            turn_now = True
        elif (
            self.corner_memory.get("active")
            and float(self.corner_memory.get("ttl_s", 0.0)) > 0.0
            and len(self.observed_boundaries) == 1
            and not single_straight_drifted
        ):
            candidate = self.corner_memory.get("direction", "UNKNOWN")
            confirmed = candidate in ("LEFT", "RIGHT")
            conf_val = max(conf_val, 0.82)
            mem_elbow = self.corner_memory.get("elbow_pt")
            if mem_elbow is not None:
                elbow_x = float(mem_elbow[0])
                elbow_y = float(min(h - 6.0, mem_elbow[1]))
                elbow_y_ratio = elbow_y / float(max(1, h))
            mean_turn_deg = float(self.corner_memory.get("turn_angle_deg", 90.0))
            distance_zone = "TURN_NOW" if (elbow_y_ratio is not None and elbow_y_ratio >= 0.62) else "APPROACH"
            turn_now = (distance_zone == "TURN_NOW")
        elif (
            self.corner_memory.get("active")
            and float(self.corner_memory.get("ttl_s", 0.0)) > 0.0
            and len(self.observed_boundaries) == 0
            and not single_straight_drifted
            and self._missing_s < 0.25
        ):
            # When zero boundaries are observed (MEMORY_HOLD), do not fabricate TURN_NOW
            candidate = self.corner_memory.get("direction", "UNKNOWN")
            confirmed = False
            conf_val = max(0.20, 0.55 - 1.4 * float(self._missing_s))
            mem_elbow = self.corner_memory.get("elbow_pt")
            if mem_elbow is not None:
                elbow_x = float(mem_elbow[0])
                elbow_y = float(min(h - 6.0, mem_elbow[1]))
                elbow_y_ratio = elbow_y / float(max(1, h))
            mean_turn_deg = float(self.corner_memory.get("turn_angle_deg", 90.0))
            mean_ground_turn_deg = float(getattr(self, "_smooth_ground_turn_deg", None) or mean_turn_deg)
            distance_zone = "APPROACH"
            turn_now = False
        else:
            self._smooth_elbow_y = None
            self._smooth_ground_turn_deg = None
            if fresh_observation:
                self._active_turn_hold_s = 0.0
                self._active_turn_hold = 0
                self._active_turn_dir = "UNKNOWN"

        return {
            "direction": candidate if confirmed else "UNKNOWN",
            "candidate": candidate,
            "intent_direction": self._intent_turn_dir,
            "single_edge_hypothesis": hypothesis,
            "confirmed": confirmed,
            "confidence": conf_val,
            "confidence_kind": "heuristic_geometry_score_not_probability",
            "support_left": len(evidence["LEFT"]),
            "support_right": len(evidence["RIGHT"]),
            "evidence": evidence,
            "consecutive_frames": self._corner_hits,
            "elbow_y": elbow_y,
            "elbow_x": elbow_x,
            "elbow_y_ratio": elbow_y_ratio,
            "stem_span_y": stem_span_y,
            "turn_angle_deg": mean_turn_deg,
            "ground_turn_angle_deg": float(mean_ground_turn_deg),
            "distance_zone": distance_zone,
            "turn_now": bool(turn_now),
        }

    @staticmethod
    def _smooth_segment_list(prev_segs, new_segs, alpha=0.35):
        if not new_segs:
            return []
        if not prev_segs:
            return new_segs

        def _interp_x_on_segs(segs, pt):
            px, py = float(pt[0]), float(pt[1])
            best_x = None
            best_dy = 1e9
            for sp in segs:
                x1, y1 = float(sp["p1"][0]), float(sp["p1"][1])
                x2, y2 = float(sp["p2"][0]), float(sp["p2"][1])
                if abs(y2 - y1) < 12.0:
                    continue
                y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
                if y_lo - 4.0 <= py <= y_hi + 4.0:
                    t = float(np.clip((py - y1) / (y2 - y1), 0.0, 1.0))
                    cand_x = x1 + t * (x2 - x1)
                    d_end = min(abs(py - y1), abs(py - y2))
                    if d_end < best_dy:
                        best_dy = d_end
                        best_x = cand_x
            if best_x is not None and abs(best_x - px) <= 22.0:
                return best_x
            return None

        if len(prev_segs) != len(new_segs):
            # Khi so doan thay doi (vd L:2<->3), van lam muot X theo cung cao do Y de chong giat vach
            a_cross = min(0.45, alpha * 1.15)
            blended = []
            for sn in new_segs:
                p1x, p1y = float(sn["p1"][0]), float(sn["p1"][1])
                p2x, p2y = float(sn["p2"][0]), float(sn["p2"][1])
                ox1 = _interp_x_on_segs(prev_segs, (p1x, p1y))
                ox2 = _interp_x_on_segs(prev_segs, (p2x, p2y))
                nx1 = p1x if ox1 is None else (a_cross * p1x + (1.0 - a_cross) * ox1)
                nx2 = p2x if ox2 is None else (a_cross * p2x + (1.0 - a_cross) * ox2)
                blended.append({
                    "p1": (float(nx1), p1y),
                    "p2": (float(nx2), p2y),
                    "width_start": float(sn["width_start"]),
                    "width_end": float(sn["width_end"]),
                    "width": float(sn["width"]),
                    "length": float(math.hypot(nx2 - nx1, p2y - p1y)),
                })
            return blended

        max_jump = 0.0
        for sp, sn in zip(prev_segs, new_segs):
            d1 = math.hypot(float(sn["p1"][0] - sp["p1"][0]), float(sn["p1"][1] - sp["p1"][1]))
            d2 = math.hypot(float(sn["p2"][0] - sp["p2"][0]), float(sn["p2"][1] - sp["p2"][1]))
            max_jump = max(max_jump, d1, d2)
        if max_jump > 95.0:
            return new_segs
        eff_alpha = alpha
        if max_jump < 2.0:
            eff_alpha = alpha * 0.45
        elif max_jump > 56.0:
            eff_alpha = min(0.40, alpha)

        out = []
        for sp, sn in zip(prev_segs, new_segs):
            p1_p, p2_p = np.array(sp["p1"], dtype=np.float32), np.array(sp["p2"], dtype=np.float32)
            p1_n, p2_n = np.array(sn["p1"], dtype=np.float32), np.array(sn["p2"], dtype=np.float32)
            p1_s = eff_alpha * p1_n + (1.0 - eff_alpha) * p1_p
            p2_s = eff_alpha * p2_n + (1.0 - eff_alpha) * p2_p
            out.append({
                "p1": (float(p1_s[0]), float(p1_s[1])),
                "p2": (float(p2_s[0]), float(p2_s[1])),
                "width_start": float(eff_alpha * sn["width_start"] + (1.0 - eff_alpha) * sp["width_start"]),
                "width_end": float(eff_alpha * sn["width_end"] + (1.0 - eff_alpha) * sp["width_end"]),
                "width": float(eff_alpha * sn["width"] + (1.0 - eff_alpha) * sp["width"]),
                "length": float(np.linalg.norm(p2_s - p1_s)),
            })
        return out

    @staticmethod
    def _resample_pts(pts, n_out=28):
        """
        Noi suy va lam muot quy dao tam duong thanh chuoi waypoint C1/C2 lien tuc
        dung Centripetal Catmull-Rom Spline (alpha=0.5) ket hop loc Laplacian bao toan diem dau/cuoi.
        Triet tieu hoan toan hien tuong gay khuc, goc nhon zic-zac o cho noi stem-lookahead.
        """
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 2:
            return pts
        if len(pts) == 2:
            t = np.linspace(0.0, 1.0, n_out, dtype=np.float32)[:, None]
            return (1.0 - t) * pts[0] + t * pts[1]

        # Loai bo cac diem trung nhau / qua gan nhau (< 1.5 px)
        diff = pts[1:] - pts[:-1]
        dist_sq = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
        keep = np.r_[True, dist_sq >= 2.25]
        pts = pts[keep]
        n_pts = len(pts)
        if n_pts < 2:
            return np.repeat(pts[:1], n_out, axis=0)
        if n_pts == 2:
            t = np.linspace(0.0, 1.0, n_out, dtype=np.float32)[:, None]
            return (1.0 - t) * pts[0] + t * pts[1]

        # Mo rong 2 diem ao o hai dau de dam bao Catmull-Rom di qua dung pts[0] va pts[-1]
        p_start = 2.0 * pts[0] - pts[1]
        p_end = 2.0 * pts[-1] - pts[-2]
        ext_pts = np.vstack([p_start, pts, p_end])

        n_segs = len(ext_pts) - 3
        pts_per_seg = max(6, int(np.ceil(n_out * 2.5 / max(1, n_segs))))

        # Vectorized Centripetal Catmull-Rom evaluation across all spline chords
        p0 = ext_pts[:n_segs, None, :]
        p1 = ext_pts[1:n_segs+1, None, :]
        p2 = ext_pts[2:n_segs+2, None, :]
        p3 = ext_pts[3:n_segs+3, None, :]

        d01 = np.sqrt(np.hypot(p1[..., 0] - p0[..., 0], p1[..., 1] - p0[..., 1]))[..., None]
        d12 = np.sqrt(np.hypot(p2[..., 0] - p1[..., 0], p2[..., 1] - p1[..., 1]))[..., None]
        d23 = np.sqrt(np.hypot(p3[..., 0] - p2[..., 0], p3[..., 1] - p2[..., 1]))[..., None]

        t0 = np.zeros_like(d01)
        t1 = d01
        t2 = t1 + d12
        t3 = t2 + d23

        dt21 = np.maximum(1e-4, t2 - t1)
        dt10 = np.maximum(1e-4, t1 - t0)
        dt32 = np.maximum(1e-4, t3 - t2)
        dt20 = np.maximum(1e-4, t2 - t0)
        dt31 = np.maximum(1e-4, t3 - t1)

        u_grid = np.linspace(0.0, 1.0, pts_per_seg, endpoint=False, dtype=np.float32)[None, :, None]
        t = t1 + u_grid * dt21

        a1 = ((t1 - t) / dt10) * p0 + ((t - t0) / dt10) * p1
        a2 = ((t2 - t) / dt21) * p1 + ((t - t1) / dt21) * p2
        a3 = ((t3 - t) / dt32) * p2 + ((t - t2) / dt32) * p3

        b1 = ((t2 - t) / dt20) * a1 + ((t - t0) / dt20) * a2
        b2 = ((t3 - t) / dt31) * a2 + ((t - t1) / dt31) * a3

        c = ((t2 - t) / dt21) * b1 + ((t - t1) / dt21) * b2
        dense_pts = np.vstack([c.reshape(-1, 2), ext_pts[-2]])

        diffs = np.diff(dense_pts, axis=0)
        cum = np.r_[0.0, np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))]
        tot = cum[-1]
        if tot < 1e-3:
            return np.repeat(pts[:1], n_out, axis=0)

        s_targets = np.linspace(0.0, tot, n_out, dtype=np.float32)
        idxs = np.clip(np.searchsorted(cum, s_targets, side="right") - 1, 0, len(dense_pts) - 2)
        ds = np.maximum(1e-4, cum[idxs + 1] - cum[idxs])
        u = np.clip((s_targets - cum[idxs]) / ds, 0.0, 1.0)[:, None]
        res = dense_pts[idxs] + u * (dense_pts[idxs + 1] - dense_pts[idxs])

        # 2 luot loc Laplacian nhe nhang giu nguyen toa do 2 dau (p[0] va p[-1])
        for _ in range(2):
            res[1:-1] = 0.25 * res[:-2] + 0.5 * res[1:-1] + 0.25 * res[2:]
        return res

    def _update_and_fuse_corner_memory(self, c_pts, p_flags, has_l, has_r, hT, wT, fresh_observation=True, corner_info=None, dt_s=None):
        """
        Bo nho lo trinh truoc goc cua (Pre-Corner Trajectory Memory):
          - Khi cam nhin thay 2 vach tai goc cua: ghi nho toan bo quy dao tam duong, be rong lan va be rong nhanh ngang.
          - Khi xe tien vao trong goc cua lam khuat 1 vach (ONE_EDGE) hoac khuat tam thoi ca 2 vach:
            dich chuyen quy dao da nho theo chuyen dong tien cua xe va ket hop voi quan sat hien tai
            de xe di chinh xac giua 2 line qua het khuc cua.
        """
        mem = self.corner_memory
        dt_step = float(dt_s) if (dt_s is not None and fresh_observation) else (self._last_dt_s if fresh_observation else 0.0)
        dt_ref = _clip_scalar(dt_step * 15.0, 0.25, 3.0) if fresh_observation else 0.0
        cand_dir = corner_info.get("candidate", "UNKNOWN") if corner_info is not None else "UNKNOWN"
        prev_mem_dir = mem.get("direction", "UNKNOWN")

        if mem.get("active") and prev_mem_dir in ("LEFT", "RIGHT") and cand_dir in ("LEFT", "RIGHT") and cand_dir != prev_mem_dir:
            mem.update({
                "active": False,
                "direction": "UNKNOWN",
                "centerline": np.empty((0, 2), dtype=np.float32),
                "arm_width_px": 0.0,
                "ttl": 0,
                "ttl_s": 0.0,
                "age_frames": 0,
                "age_s": 0.0,
            })

        if has_l and has_r and len(c_pts) >= 4:
            l_el = self._find_corner_elbow_seg_idx(self.left_segments)
            r_el = self._find_corner_elbow_seg_idx(self.right_segments)
            turn_deg_chk = float(corner_info.get("turn_angle_deg", 0.0)) if corner_info is not None else 0.0
            confirmed_chk = bool(corner_info.get("confirmed", False)) if corner_info is not None else False
            real_corner_trigger = (
                (l_el is not None and r_el is not None)
                or confirmed_chk
                or turn_deg_chk >= 26.0
            )
            if (l_el is not None or r_el is not None) and real_corner_trigger:
                # Tinh be rong nhanh ngang (arm_width_px) neu ca 2 vach deu co doan ngang
                arm_w = mem.get("arm_width_px", 0.0)
                l_h = [s for s in self.left_segments[l_el or 0 :] if abs(s["p2"][0] - s["p1"][0]) > abs(s["p2"][1] - s["p1"][1])]
                r_h = [s for s in self.right_segments[r_el or 0 :] if abs(s["p2"][0] - s["p1"][0]) > abs(s["p2"][1] - s["p1"][1])]
                if l_h and r_h:
                    yl_med = float(np.median([0.5 * (s["p1"][1] + s["p2"][1]) for s in l_h]))
                    yr_med = float(np.median([0.5 * (s["p1"][1] + s["p2"][1]) for s in r_h]))
                    if abs(yr_med - yl_med) >= 16.0:
                        arm_w = abs(yr_med - yl_med)
                if cand_dir in ("LEFT", "RIGHT"):
                    c_dir = cand_dir
                else:
                    dx_turn = float(c_pts[-1, 0] - c_pts[0, 0])
                    c_dir = "RIGHT" if dx_turn >= 0.0 else "LEFT"
                elbow_pt = (
                    (float(corner_info["elbow_x"]), float(corner_info["elbow_y"]))
                    if (corner_info is not None and corner_info.get("elbow_x") is not None and corner_info.get("elbow_y") is not None)
                    else mem.get("elbow_pt")
                )
                turn_deg = (
                    float(corner_info.get("turn_angle_deg", mem.get("turn_angle_deg", 0.0)))
                    if corner_info is not None
                    else float(mem.get("turn_angle_deg", 0.0))
                )
                mem.update({
                    "active": True,
                    "direction": c_dir,
                    "centerline": c_pts.copy(),
                    "left_segments": [dict(s) for s in self.left_segments],
                    "right_segments": [dict(s) for s in self.right_segments],
                    "lane_width_ab": self.lane_width_ab,
                    "arm_width_px": float(arm_w),
                    "elbow_pt": elbow_pt,
                    "turn_angle_deg": turn_deg,
                    "ttl": 27,
                    "ttl_s": 1.8,
                    "age_frames": 0,
                    "age_s": 0.0,
                    "straight_clear_cnt": 0,
                })
            elif mem.get("active"):
                # Decay and deactivate stale corner memory once BOTH boundaries are straight again on corner exit
                dx_span = abs(float(c_pts[-1, 0] - c_pts[0, 0]))
                if dx_span < float(wT) * 0.28:
                    mem["straight_clear_cnt"] = int(mem.get("straight_clear_cnt", 0)) + 1
                    mem["ttl_s"] = max(0.0, float(mem.get("ttl_s", 0.0)) - max(0.50, dt_step * 4.0))
                    mem["ttl"] = max(0, int(round(mem["ttl_s"] * 15.0)))
                    if mem["straight_clear_cnt"] >= 2 or mem["ttl"] <= 0:
                        mem["active"] = False
                        mem["direction"] = "UNKNOWN"
                        mem["arm_width_px"] = 0.0
                else:
                    mem["straight_clear_cnt"] = 0
            return c_pts, p_flags

        # Neu chi con 1 vach hoac mat vach khi dang o trong goc cua da nho:
        if mem.get("active") and float(mem.get("ttl_s", mem.get("ttl", 0) / 15.0)) > 0.0 and len(mem.get("centerline", ())) >= 4:
            mem_pts = self._resample_pts(mem["centerline"], n_out=len(c_pts) if len(c_pts) >= 4 else 28)
            if (has_l ^ has_r) and len(c_pts) == len(mem_pts):
                dx_cur = float(c_pts[-1, 0] - c_pts[0, 0])
                top_diff = float(np.mean(c_pts[-6:, 0] - mem_pts[-6:, 0]))
                if (
                    (prev_mem_dir == "RIGHT" and dx_cur < -float(wT) * 0.16 and top_diff < -float(wT) * 0.22)
                    or (prev_mem_dir == "LEFT" and dx_cur > float(wT) * 0.16 and top_diff > float(wT) * 0.22)
                ):
                    mem.update({
                        "active": False,
                        "direction": "UNKNOWN",
                        "centerline": np.empty((0, 2), dtype=np.float32),
                        "arm_width_px": 0.0,
                        "ttl": 0,
                        "ttl_s": 0.0,
                        "age_frames": 0,
                        "age_s": 0.0,
                    })
                    return c_pts, p_flags

            if fresh_observation:
                decay_rate = 3.0 if (not has_l and not has_r) else 1.0
                new_ttl_s = max(0.0, float(mem.get("ttl_s", mem.get("ttl", 0) / 15.0)) - decay_rate * dt_step)
                new_age_s = float(mem.get("age_s", mem.get("age_frames", 0) / 15.0)) + dt_step
                mem["ttl_s"] = new_ttl_s
                mem["ttl"] = max(0, int(round(new_ttl_s * 15.0)))
                mem["age_s"] = new_age_s
                mem["age_frames"] = int(round(new_age_s * 15.0))
                if (not has_l and not has_r) and (new_ttl_s <= 0.0 or self._missing_s > 0.35):
                    mem["active"] = False

            if (has_l ^ has_r) and len(c_pts) == len(mem_pts):
                # Gentle fusion preserving live W_lane-calibrated ONE_EDGE centering
                c_fused = (0.82 * c_pts + 0.18 * mem_pts).astype(np.float32)
                return c_fused, p_flags
            elif not has_l and not has_r and self._missing_s <= 0.35 and mem.get("active", False):
                return mem_pts.copy(), np.ones(len(mem_pts), dtype=bool)

        return c_pts, p_flags

    def _detect_junction_geometry(self, hT, wT, corner_info=None, dt_step=1.0 / 15.0, fresh_observation=True, junction_intent=None, top_y_hint=105):
        """
        Phat hien hinh hoc Nga 3 (3-way Junction) tu 2 bien lan duong:
          - RIGHT_BRANCH (|-): Vach TRAI di thang len dinh + vach PHAI re nhanh sang PHAI (hoac mo khoang trong ben phai).
            -> Cac huong hop le: ["STRAIGHT", "RIGHT"]
          - LEFT_BRANCH (-|):  Vach PHAI di thang len dinh + vach TRAI re nhanh sang TRAI (hoac mo khoang trong ben trai).
            -> Cac huong hop le: ["STRAIGHT", "LEFT"]
          - T_HEAD (chu T):    Vach ngang chan phia truoc khong cho di thang, 2 ben Trai/Phai deu mo.
            -> Cac huong hop le: ["LEFT", "RIGHT"]
        """
        top_reach_lim = min(float(hT) * 0.56, max(float(hT) * 0.34, float(top_y_hint) + 16.0))

        def _is_straight_up(segs, side):
            if not segs:
                return False
            tot_len = sum(float(s["length"]) for s in segs)
            if tot_len < 72.0:
                return False
            y_bot = float(segs[0]["p1"][1])
            y_top = float(segs[-1]["p2"][1])
            x_bot = float(segs[0]["p1"][0])
            x_top = float(segs[-1]["p2"][0])
            if y_bot < hT * 0.54 or y_top > top_reach_lim or (y_bot - y_top) < 58.0:
                return False
            if self._find_corner_elbow_seg_idx(segs) is not None:
                return False
            if abs(x_top - x_bot) > 1.28 * max(1.0, y_bot - y_top):
                return False
            if side == "LEFT" and x_top > wT * 0.74:
                return False
            if side == "RIGHT" and x_top < wT * 0.26:
                return False
            return True

        def _has_branch_bend(segs, branch_dir):
            if not segs or len(segs) < 2:
                return False, None, 0.0
            el_idx = self._find_corner_elbow_seg_idx(segs)
            if el_idx is None or el_idx <= 0 or el_idx >= len(segs):
                return False, None, 0.0
            s_prev = segs[el_idx - 1]
            s_arm = segs[el_idx]
            vx0 = float(s_prev["p2"][0] - s_prev["p1"][0])
            vy0 = float(s_prev["p2"][1] - s_prev["p1"][1])
            vx1 = float(s_arm["p2"][0] - s_arm["p1"][0])
            vy1 = float(s_arm["p2"][1] - s_arm["p1"][1])
            l0 = math.hypot(vx0, vy0)
            l1 = math.hypot(vx1, vy1)
            if l0 < 18.0 or l1 < 20.0:
                return False, None, 0.0
            cos_a = _clip_scalar((vx0 * vx1 + vy0 * vy1) / (l0 * l1), -1.0, 1.0)
            ang_deg = math.degrees(math.acos(cos_a))
            if ang_deg < 32.0:
                return False, None, 0.0
            arm_dx = float(segs[-1]["p2"][0] - s_arm["p1"][0])
            if branch_dir == "RIGHT" and arm_dx >= 22.0:
                return True, float(s_arm["p1"][1]), float(ang_deg)
            if branch_dir == "LEFT" and arm_dx <= -22.0:
                return True, float(s_arm["p1"][1]), float(ang_deg)
            return False, None, 0.0

        def _is_t_crossbar(segs):
            if not segs:
                return False, None
            # Khong nhan nham vach mep ngoai chu L cua khuc cua 90 do (co chan dung di tu day len khuyu) la thanh ngang chu T
            el_chk = self._find_corner_elbow_seg_idx(segs)
            if el_chk is not None and el_chk > 0:
                stem_dy = float(segs[0]["p1"][1]) - float(segs[el_chk - 1]["p2"][1])
                if stem_dy >= 24.0 and float(segs[0]["p1"][1]) >= hT * 0.60:
                    return False, None
            xs = [float(s["p1"][0]) for s in segs] + [float(segs[-1]["p2"][0])]
            ys = [float(s["p1"][1]) for s in segs] + [float(segs[-1]["p2"][1])]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            if span_x >= wT * 0.52 and span_y <= max(36.0, 0.24 * span_x):
                y_bar = float(np.median(ys))
                if 12.0 <= y_bar <= hT * 0.72 and min(xs) <= wT * 0.34 and max(xs) >= wT * 0.66:
                    return True, y_bar
            return False, None

        l_straight = _is_straight_up(self.left_segments, "LEFT")
        r_straight = _is_straight_up(self.right_segments, "RIGHT")
        r_bend_right, r_eb_y, r_ang = _has_branch_bend(self.right_segments, "RIGHT")
        l_bend_left, l_eb_y, l_ang = _has_branch_bend(self.left_segments, "LEFT")

        j_type = "NONE"
        avail = ["STRAIGHT"]
        branch_side = None
        straight_side = None
        eb_y = None
        phase = "NONE"
        conf = 0.0

        # 1. Nga 3 nhanh phai (|-) khi ca 2 bien deu thay ro
        if l_straight and r_bend_right:
            j_type = "RIGHT_BRANCH"
            avail = ["STRAIGHT", "RIGHT"]
            branch_side = "RIGHT"
            straight_side = "LEFT"
            eb_y = r_eb_y
            phase = "AT_JUNCTION" if (r_eb_y is not None and r_eb_y >= 88.0) else "APPROACH"
            conf = float(_clip_scalar(0.55 + (r_ang - 32.0) / 100.0, 0.55, 0.96))
        # 2. Nga 3 nhanh trai (-|) khi ca 2 bien deu thay ro
        elif r_straight and l_bend_left:
            j_type = "LEFT_BRANCH"
            avail = ["STRAIGHT", "LEFT"]
            branch_side = "LEFT"
            straight_side = "RIGHT"
            eb_y = l_eb_y
            phase = "AT_JUNCTION" if (l_eb_y is not None and l_eb_y >= 88.0) else "APPROACH"
            conf = float(_clip_scalar(0.55 + (l_ang - 32.0) / 100.0, 0.55, 0.96))
        # 3. Nga 3 chu T (T_HEAD): 2 ben mo nguoc chieu hoac thanh ngang chan truoc khong co duong thang
        elif (l_bend_left and r_bend_right) and not (l_straight or r_straight):
            j_type = "T_HEAD"
            avail = ["LEFT", "RIGHT"]
            branch_side = "BOTH"
            straight_side = None
            eb_y = float(np.mean([v for v in (l_eb_y, r_eb_y) if v is not None])) if (l_eb_y is not None or r_eb_y is not None) else None
            phase = "AT_JUNCTION" if (eb_y is not None and eb_y >= 82.0) else "APPROACH"
            conf = 0.88
        else:
            # Kiem tra thanh ngang chan truoc mat (T_HEAD) khi khong co vach di thang qua no
            t_bar_y = None
            if not (l_straight and r_straight):
                for seg_cand in (self.left_segments, self.right_segments):
                    ok_bar, y_b = _is_t_crossbar(seg_cand)
                    if ok_bar:
                        t_bar_y = y_b
                        break
            if t_bar_y is not None and (not l_straight) and (not r_straight):
                j_type = "T_HEAD"
                avail = ["LEFT", "RIGHT"]
                branch_side = "BOTH"
                straight_side = None
                eb_y = t_bar_y
                phase = "AT_JUNCTION" if t_bar_y >= 82.0 else "APPROACH"
                conf = 0.80

        mem = self._junction_mem
        if j_type != "NONE":
            mem["type"] = j_type
            mem["branch_side"] = branch_side
            mem["straight_edge_side"] = straight_side
            mem["ttl_s"] = 2.40
            mem["last_elbow_y"] = eb_y
        else:
            # Khi xe da vao giua mieng nga 3 (1 ben mo rong, ben doi dien van thang): giu trang thai AT_JUNCTION
            if fresh_observation and mem["ttl_s"] > 0.0:
                mem["ttl_s"] = max(0.0, float(mem["ttl_s"]) - float(dt_step))
            r_is_gap_or_high = (not self.right_segments) or (
                max(float(s["p1"][1]) for s in self.right_segments) < hT * 0.62
            )
            l_is_gap_or_high = (not self.left_segments) or (
                max(float(s["p1"][1]) for s in self.left_segments) < hT * 0.62
            )
            if l_straight and r_is_gap_or_high and (mem["type"] == "RIGHT_BRANCH" and mem["ttl_s"] > 0.0):
                j_type = "RIGHT_BRANCH"
                avail = ["STRAIGHT", "RIGHT"]
                branch_side = "RIGHT"
                straight_side = "LEFT"
                eb_y = mem.get("last_elbow_y")
                phase = "AT_JUNCTION"
                conf = 0.72
            elif r_straight and l_is_gap_or_high and (mem["type"] == "LEFT_BRANCH" and mem["ttl_s"] > 0.0):
                j_type = "LEFT_BRANCH"
                avail = ["STRAIGHT", "LEFT"]
                branch_side = "LEFT"
                straight_side = "RIGHT"
                eb_y = mem.get("last_elbow_y")
                phase = "AT_JUNCTION"
                conf = 0.72
            elif l_straight and r_straight:
                # Ca 2 vach da thang lai binh thuong -> da thoat khoi nga 3
                mem["ttl_s"] = max(0.0, float(mem["ttl_s"]) - 2.5 * float(dt_step))
                if mem["ttl_s"] <= 0.0:
                    mem["type"] = "NONE"
                    mem["branch_side"] = None
                    mem["straight_edge_side"] = None
                    mem["last_elbow_y"] = None
            elif (
                junction_intent in ("STRAIGHT", "RIGHT", "LEFT")
                and mem["ttl_s"] > 0.0
                and mem["type"] != "NONE"
            ):
                # Duy tri ngu canh nga 3 khi xe dang quay trong long nga 3
                j_type = mem["type"]
                branch_side = mem["branch_side"]
                straight_side = mem["straight_edge_side"]
                avail = (
                    ["STRAIGHT", "RIGHT"] if j_type == "RIGHT_BRANCH"
                    else (["STRAIGHT", "LEFT"] if j_type == "LEFT_BRANCH" else ["LEFT", "RIGHT"])
                )
                eb_y = mem.get("last_elbow_y")
                phase = "AT_JUNCTION"
                conf = 0.65

        return {
            "detected": bool(j_type != "NONE"),
            "type": j_type,
            "available_actions": avail,
            "branch_side": branch_side,
            "straight_edge_side": straight_side,
            "branch_elbow_y": eb_y,
            "phase": phase,
            "confidence": float(conf),
        }

    def detect_dual_lanes(self, mask, hT, wT, top_y_hint=105, img_gray=None, fresh_observation=True, precomputed_cc=None, dt_s=None, junction_intent=None):
        self._last_corner_info = None
        self._last_junction_info = {
            "detected": False,
            "type": "NONE",
            "available_actions": ["STRAIGHT"],
            "branch_side": None,
            "straight_edge_side": None,
            "branch_elbow_y": None,
            "phase": "NONE",
            "confidence": 0.0,
        }
        self._last_img_gray = img_gray
        self._left_is_tarp_virtual = False
        self._right_is_tarp_virtual = False
        if img_gray is not None:
            g_sub = img_gray[int(hT * 0.45) : int(hT * 0.95) : 4, ::4]
            self._last_road_med_gray = float(utils._uint8_quantiles(g_sub, (0.50,))[0]) if g_sub.size > 0 else None
        else:
            self._last_road_med_gray = None
        dt_step = self._resolve_dt_s(dt_s=dt_s, fresh_observation=fresh_observation)
        use_prev_segs = (self.missing_frames <= 3 and self._missing_s <= 0.30)
        l_res, r_res = self._find_and_trace_boundaries(
            mask,
            top_y_hint=top_y_hint,
            gray=img_gray,
            precomputed_cc=precomputed_cc,
            lane_width_ab=(self.lane_width_ab if self._lane_width_calibrated else None),
            prev_left_segs=(self._smooth_left_segs if use_prev_segs else None),
            prev_right_segs=(self._smooth_right_segs if use_prev_segs else None),
        )
        l_segs = self._fit_piecewise_segments(l_res[0], l_res[1]) if l_res is not None else []
        r_segs = self._fit_piecewise_segments(r_res[0], r_res[1]) if r_res is not None else []

        def _strip_bumper_hook(segs, res):
            if not segs or res is None:
                return segs, res
            trimmed = False
            while len(segs) >= 2:
                s0, s1 = segs[0], segs[1]
                x0_1, y0_1 = float(s0["p1"][0]), float(s0["p1"][1])
                x0_2, y0_2 = float(s0["p2"][0]), float(s0["p2"][1])
                x1_1 = float(s1["p1"][0])
                x1_2 = float(s1["p2"][0])
                is_bot_hook = (
                    y0_1 >= hT - 30.0
                    and y0_2 <= y0_1 - 3.0
                    and float(s0["length"]) <= 185.0
                    and y0_2 >= hT - 72.0
                    and abs(x0_2 - x0_1) <= wT * 0.42
                    and (
                        (x0_2 <= 24.0 and x0_1 > 24.0 and x0_2 < x0_1 - 8.0 and x1_2 > x1_1 + 10.0)
                        or (x0_2 >= wT - 25.0 and x0_1 < wT - 25.0 and x0_2 > x0_1 + 8.0 and x1_2 < x1_1 - 10.0)
                        or (x0_1 <= 18.0 and x0_2 <= 18.0 and x1_2 > x1_1 + 15.0)
                        or (x0_1 >= wT - 19.0 and x0_2 >= wT - 19.0 and x1_2 < x1_1 - 15.0)
                    )
                )
                is_bot_horiz = (
                    y0_1 >= hT - 20.0
                    and y0_2 >= hT - 22.0
                    and abs(y0_1 - y0_2) <= 12.0
                    and abs(x0_2 - x0_1) >= 18.0
                )
                if is_bot_hook or is_bot_horiz:
                    segs = segs[1:]
                    trimmed = True
                else:
                    break
            if trimmed and segs:
                pts_r, w_r = res
                y_cut = float(segs[0]["p1"][1]) + 2.0
                keep_m = pts_r[:, 1] <= y_cut
                if int(np.count_nonzero(keep_m)) >= 4:
                    res = (pts_r[keep_m], w_r[keep_m])
            return segs, res

        def _valid_segs(segs):
            if not segs:
                return []
            tot_len = sum(float(s["length"]) for s in segs)
            xs = [float(s["p1"][0]) for s in segs] + [float(segs[-1]["p2"][0])]
            ys = [float(s["p1"][1]) for s in segs] + [float(segs[-1]["p2"][1])]
            span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if tot_len < 18.0 or span < 16.0:
                return []
            if self._last_road_med_gray is not None and self._last_road_med_gray >= 25.0:
                touches_bot = max(ys) >= hT * 0.67
                span_x_s = max(xs) - min(xs)
                span_y_s = max(ys) - min(ys)
                is_wide_upper = (
                    (min(xs) <= 26.0) or (max(xs) >= wT - 27.0) or ((span_y_s >= hT * 0.45) and (tot_len >= 130.0))
                ) and (span_x_s >= wT * 0.26)
                bx_s, tx_s = float(segs[0]["p1"][0]), float(segs[-1]["p2"][0])
                is_persp_side = (
                    max(ys) >= hT * 0.55
                    and span_y_s >= 42.0
                    and (
                        (bx_s <= 18.0 and 32.0 <= (tx_s - bx_s) <= 135.0 and tx_s >= max(xs) - 8.0)
                        or (bx_s >= wT - 19.0 and 32.0 <= (bx_s - tx_s) <= 135.0 and tx_s <= min(xs) + 8.0)
                    )
                )
                if not (touches_bot or is_wide_upper or is_persp_side):
                    return []
            return segs

        l_segs, l_res = _strip_bumper_hook(l_segs, l_res)
        r_segs, r_res = _strip_bumper_hook(r_segs, r_res)
        l_segs = _valid_segs(l_segs)
        r_segs = _valid_segs(r_segs)
        if not l_segs:
            l_res = None
        if not r_segs:
            r_res = None

        # Cach 2: Chi doan Mep Bat Den (Virtual Tarp Edge) lam vach bien ao khi CHUA hieu chuan W_lane
        # va da co it nhat 1 vach trang that (KHONG tu bịa mep bat khi mat ca 2 vach trang!)
        allow_tarp_fallback = (
            img_gray is not None
            and (bool(l_segs) ^ bool(r_segs))
            and (not self._lane_width_calibrated)
            and not self.corner_memory.get("active", False)
            and self._active_turn_hold_s <= 0.0
        )
        if allow_tarp_fallback:
            ab_tarp = self.lane_width_ab if self._lane_width_calibrated else None
            if not l_segs:
                x_ex_r = (float(np.min(r_res[0][:, 0])), float(np.max(r_res[0][:, 0]))) if r_res is not None and len(r_res[0]) > 0 else None
                tarp_pts_l, tarp_ws_l = self._detect_tarp_virtual_edge(
                    img_gray, side="LEFT", road_med_gray=self._last_road_med_gray, x_exclude=x_ex_r, lane_width_ab=ab_tarp
                )
                if tarp_pts_l is not None:
                    l_segs_tarp = self._fit_piecewise_segments(tarp_pts_l, tarp_ws_l)
                    l_segs_tarp = _valid_segs(l_segs_tarp)
                    if l_segs_tarp:
                        l_segs = l_segs_tarp
                        l_res = (tarp_pts_l, tarp_ws_l)
                        self._left_is_tarp_virtual = True

            if not r_segs:
                x_ex_l = (float(np.min(l_res[0][:, 0])), float(np.max(l_res[0][:, 0]))) if l_res is not None and len(l_res[0]) > 0 else None
                tarp_pts_r, tarp_ws_r = self._detect_tarp_virtual_edge(
                    img_gray, side="RIGHT", road_med_gray=self._last_road_med_gray, x_exclude=x_ex_l, lane_width_ab=ab_tarp
                )
                if tarp_pts_r is not None:
                    r_segs_tarp = self._fit_piecewise_segments(tarp_pts_r, tarp_ws_r)
                    r_segs_tarp = _valid_segs(r_segs_tarp)
                    if r_segs_tarp:
                        r_segs = r_segs_tarp
                        r_res = (tarp_pts_r, tarp_ws_r)
                        self._right_is_tarp_virtual = True

        # Neu chi nhin thay 1 vach duy nhat -> Phan loai chinh xac do la VACH TRAI hay VACH PHAI
        alpha_seg_dt = 1.0 - math.exp(-max(0.015, float(dt_step)) / 0.062)
        prev_single_side = self.single_edge_side
        if bool(l_segs) ^ bool(r_segs):
            single_pts = l_res[0] if l_segs else r_res[0]
            single_w = l_res[1] if l_segs else r_res[1]
            single_segs = l_segs if l_segs else r_segs
            true_side = self._classify_single_boundary_side(single_pts, single_segs, hT, wT)
            span_x_s = float(single_pts[:, 0].max() - single_pts[:, 0].min())
            span_y_s = float(single_pts[:, 1].max() - single_pts[:, 1].min())
            min_y_s = float(single_pts[:, 1].min())
            if (span_x_s >= wT * 0.35 and span_y_s < 48.0) or (span_x_s >= wT * 0.22 and span_y_s < 48.0 and min_y_s >= hT * 0.35):
                if true_side == "LEFT" and float(single_pts[0, 0]) > float(single_pts[-1, 0]):
                    single_pts = single_pts[::-1].copy()
                    single_w = single_w[::-1].copy()
                    single_segs = self._fit_piecewise_segments(single_pts, single_w)
                elif true_side == "RIGHT" and float(single_pts[0, 0]) < float(single_pts[-1, 0]):
                    single_pts = single_pts[::-1].copy()
                    single_w = single_w[::-1].copy()
                    single_segs = self._fit_piecewise_segments(single_pts, single_w)
            if true_side == "LEFT":
                self.left_segments = self._smooth_segment_list(self._smooth_left_segs, single_segs, alpha=alpha_seg_dt)
                self.right_segments = []
                l_res = (single_pts, single_w)
                r_res = None
                self.single_edge_side = "LEFT"
                self._last_left_bot_x = float(self.left_segments[0]["p1"][0])
            else:
                self.right_segments = self._smooth_segment_list(self._smooth_right_segs, single_segs, alpha=alpha_seg_dt)
                self.left_segments = []
                r_res = (single_pts, single_w)
                l_res = None
                self.single_edge_side = "RIGHT"
                self._last_right_bot_x = float(self.right_segments[0]["p1"][0])
        else:
            self.left_segments = self._smooth_segment_list(self._smooth_left_segs, l_segs, alpha=alpha_seg_dt)
            self.right_segments = self._smooth_segment_list(self._smooth_right_segs, r_segs, alpha=alpha_seg_dt)
            if self.left_segments and self.right_segments:
                self.single_edge_side = "BOTH"
                self._last_left_bot_x = float(self.left_segments[0]["p1"][0])
                self._last_right_bot_x = float(self.right_segments[0]["p1"][0])
            else:
                self.single_edge_side = None

        if (
            prev_single_side in ("LEFT", "RIGHT")
            and self.single_edge_side in ("LEFT", "RIGHT")
            and prev_single_side != self.single_edge_side
        ):
            self.smooth_e_px = None
        self._smooth_left_segs = list(self.left_segments)
        self._smooth_right_segs = list(self.right_segments)

        cands = []
        if l_res is not None and self.left_segments:
            l_poly = [(float(s["p1"][0]), float(s["p1"][1])) for s in self.left_segments] + [
                (float(self.left_segments[-1]["p2"][0]), float(self.left_segments[-1]["p2"][1]))
            ]
            l_len = sum(float(s["length"]) for s in self.left_segments)
            l_w = _fast_median_seq([s["width"] for s in self.left_segments])
            cands.append({
                "side": "LEFT",
                "points": l_poly,
                "segments": self.left_segments,
                "quality": l_len,
                "length": l_len,
                "width": l_w,
                "turns": float(max(0, len(self.left_segments) - 1)),
            })
        if r_res is not None and self.right_segments:
            r_poly = [(float(s["p1"][0]), float(s["p1"][1])) for s in self.right_segments] + [
                (float(self.right_segments[-1]["p2"][0]), float(self.right_segments[-1]["p2"][1]))
            ]
            r_len = sum(float(s["length"]) for s in self.right_segments)
            r_w = _fast_median_seq([s["width"] for s in self.right_segments])
            cands.append({
                "side": "RIGHT",
                "points": r_poly,
                "segments": self.right_segments,
                "quality": r_len,
                "length": r_len,
                "width": r_w,
                "turns": float(max(0, len(self.right_segments) - 1)),
            })
        self._associate_boundaries(cands, hT, wT)
        corner_info = self.inspect_corner(None, fresh_observation=fresh_observation, img_shape=(hT, wT), dt_s=dt_step)
        junction_info = self._detect_junction_geometry(
            hT, wT, corner_info=corner_info, dt_step=dt_step,
            fresh_observation=fresh_observation, junction_intent=junction_intent,
            top_y_hint=top_y_hint,
        )
        self._last_junction_info = junction_info

        # Khi da co ket luan tu bien bao o Nga 3: dieu chinh hinh hoc theo huong da chon
        def _clip_segs_below_y(segs, y_cut):
            clipped = []
            for s in segs:
                x1, y1 = float(s["p1"][0]), float(s["p1"][1])
                x2, y2 = float(s["p2"][0]), float(s["p2"][1])
                if y1 <= y_cut + 2.0:
                    continue
                if y2 >= y_cut:
                    clipped.append(dict(s))
                else:
                    dy_s = y1 - y2
                    if dy_s < 1.0:
                        continue
                    t_c = _clip_scalar((y1 - y_cut) / dy_s, 0.0, 1.0)
                    xc = x1 + t_c * (x2 - x1)
                    yc = y1 + t_c * (y2 - y1)
                    new_len = math.hypot(xc - x1, yc - y1)
                    if new_len >= 16.0:
                        ns = dict(s)
                        ns["p2"] = (float(xc), float(yc))
                        ns["length"] = float(new_len)
                        clipped.append(ns)
            return clipped

        effective_intent = junction_intent
        if junction_info.get("detected") and effective_intent is None:
            # Cam di thang (vd: thay bien no_straight don le): tu dong chon nhanh re kha dung
            jt_auto = junction_info.get("type")
            if jt_auto == "RIGHT_BRANCH":
                effective_intent = "RIGHT"
            elif jt_auto == "LEFT_BRANCH":
                effective_intent = "LEFT"

        if effective_intent == "STRAIGHT":
            # Khi co bien DI THANG / CAM RE: xoa ngay bo nho cua cu tu khuc cua truoc
            if self.corner_memory.get("active"):
                self.corner_memory["active"] = False
                self.corner_memory["direction"] = "UNKNOWN"
            self._active_turn_hold_s = 0.0
            self._active_turn_hold = 0

        if junction_info.get("detected") and effective_intent == "STRAIGHT":
            j_type = junction_info.get("type")
            if j_type == "RIGHT_BRANCH" and self.left_segments:
                # Di thang qua nga 3 nhanh phai (|-): bo qua nhanh re phai (>) de khong bi keo lech phai
                r_el_j = self._find_corner_elbow_seg_idx(self.right_segments) if self.right_segments else None
                if r_el_j is not None and r_el_j > 0:
                    stem_r = list(self.right_segments[:r_el_j])
                    if sum(float(s["length"]) for s in stem_r) >= 32.0:
                        self.right_segments = stem_r
                    else:
                        self.right_segments = []
                        self.single_edge_side = "LEFT"
                elif self.right_segments and max(float(s["p1"][1]) for s in self.right_segments) < hT * 0.62:
                    self.right_segments = []
                    self.single_edge_side = "LEFT"
                if corner_info.get("direction") == "RIGHT" or corner_info.get("candidate") == "RIGHT":
                    corner_info = dict(corner_info)
                    corner_info["direction"] = "UNKNOWN"
                    corner_info["candidate"] = "UNKNOWN"
                    corner_info["turn_now"] = False
                    corner_info["distance_zone"] = "NONE"
                    corner_info["confirmed"] = False
                if self.corner_memory.get("active") and self.corner_memory.get("direction") == "RIGHT":
                    self.corner_memory["active"] = False
                    self.corner_memory["direction"] = "UNKNOWN"
            elif j_type == "LEFT_BRANCH" and self.right_segments:
                # Di thang qua nga 3 nhanh trai (-|): bo qua nhanh re trai (<) de khong bi keo lech trai
                l_el_j = self._find_corner_elbow_seg_idx(self.left_segments) if self.left_segments else None
                if l_el_j is not None and l_el_j > 0:
                    stem_l = list(self.left_segments[:l_el_j])
                    if sum(float(s["length"]) for s in stem_l) >= 32.0:
                        self.left_segments = stem_l
                    else:
                        self.left_segments = []
                        self.single_edge_side = "RIGHT"
                elif self.left_segments and max(float(s["p1"][1]) for s in self.left_segments) < hT * 0.62:
                    self.left_segments = []
                    self.single_edge_side = "RIGHT"
                if corner_info.get("direction") == "LEFT" or corner_info.get("candidate") == "LEFT":
                    corner_info = dict(corner_info)
                    corner_info["direction"] = "UNKNOWN"
                    corner_info["candidate"] = "UNKNOWN"
                    corner_info["turn_now"] = False
                    corner_info["distance_zone"] = "NONE"
                    corner_info["confirmed"] = False
                if self.corner_memory.get("active") and self.corner_memory.get("direction") == "LEFT":
                    self.corner_memory["active"] = False
                    self.corner_memory["direction"] = "UNKNOWN"
        elif (
            junction_info.get("detected")
            and effective_intent in ("RIGHT", "LEFT")
            and junction_info.get("phase") == "AT_JUNCTION"
        ):
            # Khi da den mieng nga 3 va ket luan la RE: cat bo phan vach di thang qua giao lo
            # va uu tien huong re vao nhanh da chon
            j_type = junction_info.get("type")
            eb_y_j = junction_info.get("branch_elbow_y")
            if effective_intent == "RIGHT" and j_type in ("RIGHT_BRANCH", "T_HEAD"):
                if j_type == "RIGHT_BRANCH" and self.left_segments:
                    if eb_y_j is not None and float(eb_y_j) < hT * 0.48:
                        stem_l_keep = _clip_segs_below_y(self.left_segments, float(eb_y_j) + 4.0)
                        if sum(float(s["length"]) for s in stem_l_keep) >= 24.0:
                            self.left_segments = stem_l_keep
                        elif self.right_segments:
                            self.left_segments = []
                            self.single_edge_side = "RIGHT"
                    elif self.right_segments:
                        self.left_segments = []
                        self.single_edge_side = "RIGHT"
                corner_info = dict(corner_info)
                corner_info["direction"] = "RIGHT"
                corner_info["candidate"] = "RIGHT"
                corner_info["intent_direction"] = "RIGHT"
                corner_info["confirmed"] = True
                corner_info["distance_zone"] = "TURN_NOW"
                corner_info["turn_now"] = True
            elif effective_intent == "LEFT" and j_type in ("LEFT_BRANCH", "T_HEAD"):
                if j_type == "LEFT_BRANCH" and self.right_segments:
                    if eb_y_j is not None and float(eb_y_j) < hT * 0.48:
                        stem_r_keep = _clip_segs_below_y(self.right_segments, float(eb_y_j) + 4.0)
                        if sum(float(s["length"]) for s in stem_r_keep) >= 24.0:
                            self.right_segments = stem_r_keep
                        elif self.left_segments:
                            self.right_segments = []
                            self.single_edge_side = "LEFT"
                    elif self.left_segments:
                        self.right_segments = []
                        self.single_edge_side = "LEFT"
                corner_info = dict(corner_info)
                corner_info["direction"] = "LEFT"
                corner_info["candidate"] = "LEFT"
                corner_info["intent_direction"] = "LEFT"
                corner_info["confirmed"] = True
                corner_info["distance_zone"] = "TURN_NOW"
                corner_info["turn_now"] = True

        self._last_corner_info = corner_info

        arm_w_hint = float(self.corner_memory.get("arm_width_px", 0.0)) if self.corner_memory.get("active") else 0.0
        active_ab = self.corner_memory.get("lane_width_ab", self.lane_width_ab) if (
            self.corner_memory.get("active") and not (self.left_segments and self.right_segments)
        ) else self.lane_width_ab

        has_l = len(self.left_segments) > 0
        has_r = len(self.right_segments) > 0

        c_pts, p_flags, new_ab = self._compute_lane_and_centerline(
            self.left_segments,
            self.right_segments,
            hT,
            wT,
            lane_width_ab=active_ab,
            arm_width_hint=arm_w_hint,
            lane_width_calibrated=self._lane_width_calibrated,
            corner_memory_active=bool(self.corner_memory.get("active", False)),
        )
        both_real_white = bool(has_l and has_r and not self._left_is_tarp_virtual and not self._right_is_tarp_virtual)
        if both_real_white:
            self.lane_width_ab = new_ab
        if both_real_white and len(c_pts) >= 2:
            y_chk_list = (float(hT) * 0.48, float(hT) * 0.72, float(hT) * 0.92)
            c_dx = abs(float(c_pts[-1, 0] - c_pts[0, 0]))
            c_dy = abs(float(c_pts[0, 1] - c_pts[-1, 1]))
            l_el = self._find_corner_elbow_seg_idx(self.left_segments)
            r_el = self._find_corner_elbow_seg_idx(self.right_segments)
            is_straight_corridor = (c_dx <= float(wT) * 0.22) and (c_dy >= float(hT) * 0.35) and (l_el is None) and (r_el is None)
            w_bot_new_chk = float(self.lane_width_ab[0] * (hT - 6.0) + self.lane_width_ab[1])
            l_bot_x_chk = float(self.left_segments[0]["p1"][0])
            r_bot_x_chk = float(self.right_segments[0]["p1"][0])
            plausible_boot_w = self._lane_width_calibrated or (
                w_bot_new_chk >= 210.0 and l_bot_x_chk < float(wT) * 0.48 and r_bot_x_chk > float(wT) * 0.52
            )
            if plausible_boot_w and (
                is_straight_corridor
                or (
                    new_ab != active_ab
                    and c_dy >= float(hT) * 0.35
                    and 0.35 <= float(self.lane_width_ab[0]) <= 1.95
                )
            ):
                self._lane_width_calibrated = True
            elif not plausible_boot_w:
                self.lane_width_ab = active_ab
            if self._anchor_lane_width_ab is None and is_straight_corridor and 0.35 <= float(self.lane_width_ab[0]) <= 1.95:
                self._anchor_straight_hits = getattr(self, "_anchor_straight_hits", 0) + 1
                if self._anchor_straight_hits >= 2:
                    self._anchor_lane_width_ab = self.lane_width_ab
            elif self._anchor_lane_width_ab is not None:
                ratios = [
                    float(self.lane_width_ab[0] * y_q + self.lane_width_ab[1])
                    / max(40.0, float(self._anchor_lane_width_ab[0] * y_q + self._anchor_lane_width_ab[1]))
                    for y_q in y_chk_list
                ]
                bad_slope = not (0.35 <= float(self.lane_width_ab[0]) <= 1.95)
                bad_ratio = any(r < 0.88 or r > 1.14 for r in ratios) or (
                    not is_straight_corridor and any(abs(r - 1.0) > 0.055 for r in ratios)
                )
                if is_straight_corridor and not bad_slope and not bad_ratio:
                    self._anchor_mismatch_hits = 0
                    self._anchor_lane_width_ab = (
                        0.88 * float(self._anchor_lane_width_ab[0]) + 0.12 * float(self.lane_width_ab[0]),
                        0.88 * float(self._anchor_lane_width_ab[1]) + 0.12 * float(self.lane_width_ab[1]),
                    )
                    self.lane_width_ab = (
                        0.72 * float(active_ab[0]) + 0.28 * float(self.lane_width_ab[0]),
                        0.72 * float(active_ab[1]) + 0.28 * float(self.lane_width_ab[1]),
                    )
                elif bad_slope or bad_ratio:
                    if is_straight_corridor and not bad_slope:
                        self._anchor_mismatch_hits = getattr(self, "_anchor_mismatch_hits", 0) + 1
                    else:
                        self._anchor_mismatch_hits = 0
                    if self._anchor_mismatch_hits >= 3:
                        self._anchor_lane_width_ab = self.lane_width_ab
                        self._anchor_mismatch_hits = 0
                    else:
                        self.lane_width_ab = self._anchor_lane_width_ab
                else:
                    self._anchor_mismatch_hits = 0
                    self.lane_width_ab = (
                        0.78 * float(active_ab[0]) + 0.22 * float(self.lane_width_ab[0]),
                        0.78 * float(active_ab[1]) + 0.22 * float(self.lane_width_ab[1]),
                    )

        # Lam muot duong tam va ket hop Bo nho lo trinh truoc goc cua (Corner Trajectory Memory)
        if len(c_pts) >= 2:
            c_resamp = self._resample_pts(c_pts, n_out=28)
            if len(p_flags) == len(c_resamp):
                p_resamp = p_flags.copy()
            elif len(p_flags) >= 2:
                idx_map = np.round(np.linspace(0, len(p_flags) - 1, len(c_resamp))).astype(int)
                p_resamp = np.asarray(p_flags[idx_map], dtype=bool)
            else:
                p_resamp = np.full(len(c_resamp), bool(np.all(p_flags)) if len(p_flags) else False, dtype=bool)
            c_resamp, p_resamp = self._update_and_fuse_corner_memory(
                c_resamp,
                p_resamp,
                has_l,
                has_r,
                hT,
                wT,
                fresh_observation=fresh_observation,
                corner_info=corner_info,
                dt_s=dt_step,
            )
            if (
                fresh_observation
                and self._smooth_center_pts is not None
                and len(self._smooth_center_pts) == len(c_resamp)
                and self.missing_frames == 0
            ):
                mean_diff = float(np.mean(np.linalg.norm(c_resamp - self._smooth_center_pts, axis=1)))
                if mean_diff < 110.0:
                    tau_c = 0.165 if mean_diff < 14.0 else 0.075
                    alpha_c = 1.0 - math.exp(-max(0.015, float(dt_step)) / tau_c)
                    if mean_diff < 3.2:
                        alpha_c *= 0.40  # Giam rung vi mo khi tam xe on dinh
                    elif mean_diff >= 52.0:
                        alpha_c = min(alpha_c, 0.34)  # Ham nhay vot 1 frame do nhieu loa sang
                    # Khi chuyen trang thai giua 2 vach va 1 vach: lam diu toc do bien thien de tranh giat toa do
                    is_dual_now = bool(has_l and has_r)
                    prev_had_dual = getattr(self, "_prev_had_dual", None)
                    if prev_had_dual is not None and prev_had_dual != is_dual_now:
                        alpha_c *= 0.55
                    c_resamp = (alpha_c * c_resamp + (1.0 - alpha_c) * self._smooth_center_pts).astype(np.float32)
            self._prev_had_dual = bool(has_l and has_r)
            self._smooth_center_pts = c_resamp.copy()
            c_pts = c_resamp
            p_flags = p_resamp
        else:
            c_pts, p_flags = self._update_and_fuse_corner_memory(
                c_pts,
                p_flags,
                has_l,
                has_r,
                hT,
                wT,
                fresh_observation=fresh_observation,
                corner_info=corner_info,
                dt_s=dt_step,
            )

        if has_l and has_r and len(c_pts) >= 2:
            self.center_pts = c_pts
            self.center_pred_flags = p_flags
            self._last_valid_center_pts = c_pts.copy()
            self._last_valid_pred_flags = p_flags.copy()
            self.missing_frames = 0
            self._missing_s = 0.0
            status = "2 BIEN QUAN SAT"
        elif (has_l or has_r) and len(c_pts) >= 2:
            self.center_pts = c_pts
            self.center_pred_flags = p_flags
            self._last_valid_center_pts = c_pts.copy()
            self._last_valid_pred_flags = p_flags.copy()
            self.missing_frames = 0
            self._missing_s = 0.0
            status = "ONE_EDGE (NHO LO TRINH)" if self.corner_memory.get("active") else "ONE_EDGE"
        else:
            self.missing_frames += 1
            if fresh_observation:
                self._missing_s += dt_step
            max_hold_s = 0.35 if self.corner_memory.get("active") else 0.25
            max_hold = max(3, int(round(max_hold_s / max(0.015, float(dt_step)))))
            if len(c_pts) >= 2 and self._missing_s <= max_hold_s and self.missing_frames <= max_hold:
                self.center_pts = c_pts.copy()
                self.center_pred_flags = np.ones(len(self.center_pts), dtype=bool)
                status = "NHO LO TRINH CUA" if self.corner_memory.get("active") else "GIU LAN"
            elif len(self._last_valid_center_pts) >= 2 and self._missing_s <= max_hold_s and self.missing_frames <= max_hold:
                self.center_pts = self._last_valid_center_pts.copy()
                self.center_pred_flags = np.ones(len(self.center_pts), dtype=bool)
                status = "NHO LO TRINH CUA" if self.corner_memory.get("active") else "GIU LAN"
            else:
                self.center_pts = np.empty((0, 2), dtype=np.float32)
                self.center_pred_flags = np.empty((0,), dtype=bool)
                self.corner_memory["active"] = False
                return None, None, None, None, "TIM LAN..."

        plot_y = self.center_pts[:, 1].copy()
        center_x = self.center_pts[:, 0].copy()
        min_w_clamp = max(24.0, wT * 0.06)
        half_w = 0.5 * np.clip(self.lane_width_ab[0] * plot_y + self.lane_width_ab[1], min_w_clamp, wT * 0.98)
        left_x = center_x - half_w
        right_x = center_x + half_w
        return plot_y, left_x, right_x, center_x, status

    def detect_highway_lanes(self, img_thres, hT, wT, top_y_hint=145):
        top_y = max(int(hT * 0.42), min(int(top_y_hint), int(hT * 0.65)))
        mid_y = int(hT * 0.78)
        bot_y = int(hT * 0.96)
        plot_y = np.linspace(top_y, bot_y, 28)
        yn = plot_y / float(hT)

        left_xs, left_ys, left_ws = [], [], []
        right_xs, right_ys, right_ws = [], [], []

        lines = cv2.HoughLinesP(img_thres, rho=1, theta=np.pi / 180, threshold=8, minLineLength=7, maxLineGap=30)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y1 == y2:
                    continue
                if y1 > y2:
                    x_bot, y_bot, x_top, y_top = x1, y1, x2, y2
                else:
                    x_bot, y_bot, x_top, y_top = x2, y2, x1, y1

                dy = float(y_bot - y_top)
                dx = float(x_bot - x_top)
                length = float(np.hypot(dx, dy))
                if dy < 4:
                    continue

                inv_slope = dx / dy
                x_at_bottom = x_bot + inv_slope * (bot_y - y_bot)
                x_at_mid = x_bot + inv_slope * (mid_y - y_bot)

                ts = np.linspace(0.0, 1.0, max(3, int(length // 2)))
                xs_seg = x_top + ts * dx
                ys_seg = y_top + ts * dy
                weight = length * length

                # Chon 2 bien cua lan duong chinh giua (Ego Lane) va loai bo lan phu 2 ben / xe phia truoc
                if (-2.95 <= inv_slope <= -0.35 and
                        (wT * 0.06 <= x_at_bottom <= wT * 0.47) and
                        (wT * 0.16 <= x_at_mid <= wT * 0.49) and
                        x_top <= wT * 0.51):
                    left_xs.extend(xs_seg)
                    left_ys.extend(ys_seg)
                    left_ws.extend([weight] * len(ts))
                elif (0.35 <= inv_slope <= 2.95 and
                        (wT * 0.53 <= x_at_bottom <= wT * 0.94) and
                        (wT * 0.51 <= x_at_mid <= wT * 0.84) and
                        x_top >= wT * 0.49):
                    right_xs.extend(xs_seg)
                    right_ys.extend(ys_seg)
                    right_ws.extend([weight] * len(ts))

        l_fit = None
        r_fit = None

        if len(left_ys) >= 4:
            ly, lx, lw = np.array(left_ys), np.array(left_xs), np.array(left_ws)
            deg = 2 if (np.ptp(ly) >= hT * 0.24 and len(ly) >= 25) else 1
            fit = np.polyfit(ly / float(hT), lx, deg, w=lw)
            l_fit = np.array([0.0, fit[0], fit[1]]) if deg == 1 else np.array([float(np.clip(fit[0], -wT * 0.4, wT * 0.4)), fit[1], fit[2]])

        if len(right_ys) >= 4:
            ry, rx, rw = np.array(right_ys), np.array(right_xs), np.array(right_ws)
            deg = 2 if (np.ptp(ry) >= hT * 0.24 and len(ry) >= 25) else 1
            fit = np.polyfit(ry / float(hT), rx, deg, w=rw)
            r_fit = np.array([0.0, fit[0], fit[1]]) if deg == 1 else np.array([float(np.clip(fit[0], -wT * 0.4, wT * 0.4)), fit[1], fit[2]])

        left_x = np.polyval(l_fit, yn) if l_fit is not None else None
        right_x = np.polyval(r_fit, yn) if r_fit is not None else None

        default_widths = np.linspace(wT * 0.14, wT * 0.56, len(plot_y))
        learned_widths = self.width_profile if (self.width_profile is not None and len(self.width_profile) == len(plot_y)) else default_widths

        if left_x is not None and right_x is not None:
            lane_w = right_x - left_x
            mid_idx = len(lane_w) // 2
            # Kiem tra be rong o day va giua lan (tranh bi loai sai khi diem tu nam thap hon top_y)
            if lane_w[-1] < wT * 0.25 or lane_w[-1] > wT * 0.92 or lane_w[mid_idx] < wT * 0.10:
                if len(left_ys) >= len(right_ys):
                    right_x = None
                else:
                    left_x = None

        if left_x is not None and right_x is not None:
            # Neu 2 duong hoi tu som (diem tu thap hon top_y), tu dong gioi han dinh lan duong duoi diem giao nhau
            valid_mask = (right_x - left_x) >= (wT * 0.08)
            if np.any(valid_mask) and not np.all(valid_mask):
                first_valid = int(np.argmax(valid_mask))
                eff_top_y = float(plot_y[first_valid])
                plot_y = np.linspace(eff_top_y, bot_y, 28)
                yn_eff = plot_y / float(hT)
                left_x = np.polyval(l_fit, yn_eff)
                right_x = np.polyval(r_fit, yn_eff)

            self.width_profile = 0.80 * learned_widths + 0.20 * np.maximum(wT * 0.08, right_x - left_x)
            status = "2 BIEN CHUAN (L + R)"
            self.missing_frames = 0
        elif left_x is not None:
            right_x = left_x + learned_widths
            status = "BAM BIEN TRAI (Noi suy Phai)"
            self.missing_frames = 0
        elif right_x is not None:
            left_x = right_x - learned_widths
            status = "BAM BIEN PHAI (Noi suy Trai)"
            self.missing_frames = 0
        else:
            self.missing_frames += 1
            if self.smooth_left_x is not None and self.smooth_right_x is not None and self.missing_frames <= 12:
                left_x = self.smooth_left_x.copy()
                right_x = self.smooth_right_x.copy()
                status = f"GIU LAN ({self.missing_frames}/12)"
            else:
                return None, None, None, None, "TIM LAN..."

        alpha = 0.50 if status.startswith("2 BIEN") else 0.30
        if self.smooth_left_x is None or len(self.smooth_left_x) != len(left_x):
            self.smooth_left_x = left_x.copy()
            self.smooth_right_x = right_x.copy()
        else:
            self.smooth_left_x = alpha * left_x + (1.0 - alpha) * self.smooth_left_x
            self.smooth_right_x = alpha * right_x + (1.0 - alpha) * self.smooth_right_x

        center_x = 0.5 * (self.smooth_left_x + self.smooth_right_x)
        return plot_y, self.smooth_left_x, self.smooth_right_x, center_x, status

    def _estimate_single_edge_curve(self, wT, hT, corner, hist_curve):
        """
        Xu ly du phong khi chi thay 1 vach bien (ONE_EDGE) tren san nha:
        Ket hop do nghieng cua vach bien duy nhat, gia thuyet goc cua (single_edge_hypothesis) va Histogram.
        """
        if not self.boundary_traces:
            return 0.0
        pts = np.asarray(self.boundary_traces[0], dtype=np.float32)
        if len(pts) < 4:
            return float(np.clip(hist_curve * 0.45, -45.0, 45.0))

        hyp = corner.get("single_edge_hypothesis", "UNKNOWN")
        if hyp == "LEFT":
            return -48.0
        if hyp == "RIGHT":
            return +48.0

        bot_x = float(pts[0, 0])
        top_x = float(pts[-1, 0])
        dx = top_x - bot_x
        half_w = wT * 0.5
        if bot_x < half_w:
            target_x = bot_x + wT * 0.28 + dx * 0.35
        else:
            target_x = bot_x - wT * 0.28 + dx * 0.35
        return float(np.clip(((target_x - half_w) / (wT * 0.28)) * 50.0, -55.0, 55.0))

    def process_frame(self, img, points, hsv_range=None, use_dual_edge=True,
                      profile="robot", fresh_observation=True, restrict_roi=False,
                      draw_aux_views=False, draw_overlay=True, dt_s=None, junction_intent=None):
        """Xu ly khung hinh va tra ve (norm_curve, img_roi, img_thres, img_warp_vis, img_hist_combined, img_result)."""
        hT, wT = img.shape[:2]
        img_result = img.copy()

        if profile not in ("robot", "highway"):
            raise ValueError("profile must be robot or highway")
        context = (profile, img.shape[:2], bool(use_dual_edge))
        if getattr(self, "_frame_context", context) != context:
            self.reset()
        self._frame_context = context
        if profile == "robot" and not use_dual_edge:
            self.boundary_traces, self.observed_boundaries, self._previous_boundaries = [], [], []
            self._corner_candidate, self._corner_hits, self._corner_ids = "UNKNOWN", 0, ()

        dt_step = self._resolve_dt_s(dt_s=dt_s, fresh_observation=fresh_observation)
        prev_mode = self._prev_light_mode if fresh_observation else None
        if profile == "robot":
            if self._ai_segmenter is not None and self._ai_segmenter.ready:
                img_thres, thres_aux = self._ai_segmenter.predict_mask(
                    img, roi_points=points, restrict_roi=restrict_roi, prev_mode=prev_mode
                )
            else:
                img_thres, thres_aux = utils.thresholding(
                    img, hsv_range, roi_points=points, restrict_roi=restrict_roi,
                    prev_mode=prev_mode, return_aux=True,
                )
        else:
            img_thres, thres_aux = utils.thresholding_highway(
                img, hsv_range, roi_points=points, restrict_roi=True,
                prev_mode=prev_mode, return_aux=True,
            )
        lighting_info = thres_aux["lighting_info"]
        if fresh_observation:
            self._prev_light_mode = lighting_info.get("mode")
        img_gray_aux = thres_aux.get("gray")
        cc_labels_aux = thres_aux.get("cc_labels")
        cc_stats_aux = thres_aux.get("cc_stats")
        pre_cc = (cc_labels_aux, cc_stats_aux) if (cc_labels_aux is not None and cc_stats_aux is not None) else None

        need_aux_init = bool(draw_aux_views) or (profile != "robot") or (not use_dual_edge)
        if need_aux_init:
            img_copy = img.copy()
            img_warp = utils.warpImg(img_thres, points, wT, hT)
            img_warp_vis = utils.warpImg(img, points, wT, hT)
            img_roi = utils.drawPoints(img_copy, points)
            mid_point, img_hist1 = utils.getHistogram(img_warp, display=True, minPer=0.5, region=4)
            avg_point, img_hist2 = utils.getHistogram(img_warp, display=True, minPer=0.7, region=1)
            hist_curve = float(avg_point - mid_point)
            img_hist_combined = cv2.addWeighted(img_hist1, 0.5, img_hist2, 0.5, 0)
        else:
            img_warp = None
            img_roi = img
            img_warp_vis = img_result
            mid_point = wT // 2
            avg_point = wT // 2
            hist_curve = 0.0
            img_hist_combined = None
        raw_curve = hist_curve

        corner = {"direction": "UNKNOWN", "candidate": "UNKNOWN", "intent_direction": self._intent_turn_dir,
                  "confirmed": False, "confidence": 0.0, "evidence": {"LEFT": [], "RIGHT": []},
                  "elbow_y": None, "elbow_x": None, "elbow_y_ratio": 0.0,
                  "stem_span_y": float(hT), "distance_zone": "NONE", "turn_now": False}
        c_x = None
        plot_y = None
        status = "HISTOGRAM_UNVERIFIED"
        car_ref_x = wT * 0.5
        e_px_out = 0.0
        e_norm_out = 0.0
        steer_norm_out = 0.0
        front_wheel_deg_out = 0.0
        servo_deg_out = 90
        heading_deg_out = 0.0
        psi_rad_out = 0.0
        offset_dir = "CAN GIUA"
        lowest_c_pt = None
        lookahead_c_pt = None
        path_curvature_out = 0.0
        lookahead_dist_out = 0.0
        ctrl_dbg = {}
        min_w_clamp = max(24.0, wT * 0.06)
        w_lane_ref = _clip_scalar(self.lane_width_ab[0] * (hT * 0.90) + self.lane_width_ab[1], min_w_clamp, wT * 1.15)
        w_lane_bot_out = _clip_scalar(self.lane_width_ab[0] * (hT - 6.0) + self.lane_width_ab[1], min_w_clamp, wT * 1.35)
        tape_w_ref = _fast_median_seq([s["width"] for s in (self.left_segments + self.right_segments)]) if (self.left_segments or self.right_segments) else 8.0

        tam_label_info = None
        if use_dual_edge:
            top_hint = int(points[0][1]) if points is not None else int(hT * 0.60)
            if profile == "robot":
                img_gray = (
                    img_gray_aux
                    if (img_gray_aux is not None and img_gray_aux.shape[:2] == (hT, wT))
                    else (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
                )
                plot_y, l_x, r_x, c_x, status = self.detect_dual_lanes(
                    img_thres,
                    hT,
                    wT,
                    top_y_hint=top_hint,
                    img_gray=img_gray,
                    fresh_observation=fresh_observation,
                    precomputed_cc=pre_cc,
                    dt_s=dt_step,
                    junction_intent=junction_intent,
                )
                corner = (
                    self._last_corner_info
                    if self._last_corner_info is not None
                    else self.inspect_corner(img_thres, fresh_observation=fresh_observation, dt_s=dt_step)
                )
                if corner.get("elbow_x") is not None and corner.get("elbow_y") is not None:
                    self.corner_memory["elbow_pt"] = (float(corner["elbow_x"]), float(corner["elbow_y"]))
                    cur_c_ang = float(corner.get("turn_angle_deg", 0.0))
                    prev_m_ang = float(self.corner_memory.get("turn_angle_deg", 0.0)) if self.corner_memory.get("active") else 0.0
                    self.corner_memory["turn_angle_deg"] = max(cur_c_ang, 0.85 * prev_m_ang)

                all_segs = self.left_segments + self.right_segments
                if all_segs:
                    tape_w_ref = _fast_median_seq([s["width"] for s in all_segs])

                if draw_overlay:
                    # 0. To mau nen hanh lang giua 2 vach trang (ban trong suot) de thay ro long duong
                    if self.left_segments and self.right_segments:
                        l_poly_vis = self._sample_polyline(self.left_segments, 12).astype(np.int32)
                        r_poly_vis = self._sample_polyline(self.right_segments, 12).astype(np.int32)
                        if len(l_poly_vis) >= 2 and len(r_poly_vis) >= 2:
                            corridor_poly = np.vstack((l_poly_vis, r_poly_vis[::-1]))
                            bx, by, bw, bh = cv2.boundingRect(corridor_poly)
                            bx1, by1 = max(0, bx), max(0, by)
                            bx2, by2 = min(wT, bx + bw), min(hT, by + bh)
                            if bx2 > bx1 and by2 > by1:
                                roi_res = img_result[by1:by2, bx1:bx2]
                                roi_ov = roi_res.copy()
                                corridor_shifted = corridor_poly.copy()
                                corridor_shifted[:, 0] -= bx1
                                corridor_shifted[:, 1] -= by1
                                cv2.fillPoly(roi_ov, [corridor_shifted], (0, 140, 55))
                                cv2.addWeighted(roi_res, 0.76, roi_ov, 0.24, 0, dst=roi_res)

                    # 1. Ve 2 vach: vach trang thuong ve mau xanh bien, neu la mep bat ao ve mau vang de phan biet
                    if getattr(self, "_left_is_tarp_virtual", False):
                        self._draw_blue_tapered_segments(img_result, self.left_segments, color=(0, 220, 255), border_color=(0, 160, 255))
                    else:
                        self._draw_blue_tapered_segments(img_result, self.left_segments)

                    if getattr(self, "_right_is_tarp_virtual", False):
                        self._draw_blue_tapered_segments(img_result, self.right_segments, color=(0, 220, 255), border_color=(0, 160, 255))
                    else:
                        self._draw_blue_tapered_segments(img_result, self.right_segments)

                # 2. Ve duong tam hinh hoc thuc cua lan (khong lech tam nhan tao, doi xung Trai/Phai)
                c_turn_dir = corner.get("direction") if corner.get("direction") in ("LEFT", "RIGHT") else corner.get("candidate", "UNKNOWN")
                elbow_y_val = corner.get("elbow_y")
                elbow_x_val = corner.get("elbow_x")
                turn_now_flag = bool(corner.get("turn_now", False))
                zone_now = corner.get("distance_zone")

                c_pts = self.center_pts
                p_flags = self.center_pred_flags
                if len(c_pts) >= 2:
                    if draw_overlay:
                        for i in range(len(c_pts) - 1):
                            pt1 = (int(round(c_pts[i, 0])), int(round(c_pts[i, 1])))
                            pt2 = (int(round(c_pts[i + 1, 0])), int(round(c_pts[i + 1, 1])))
                            is_pred = bool(p_flags[i] or p_flags[i + 1])
                            if not is_pred:
                                cv2.line(img_result, pt1, pt2, (255, 255, 255), 3, cv2.LINE_8)
                            else:
                                if i % 2 == 0:
                                    cv2.line(img_result, pt1, pt2, (255, 255, 255), 3, cv2.LINE_8)
                                else:
                                    cv2.line(img_result, pt1, pt2, (0, 215, 255), 2, cv2.LINE_8)

                        if np.all(p_flags) and not (self.left_segments and self.right_segments):
                            mid_c = c_pts[len(c_pts) // 2]
                            if self.corner_memory.get("active"):
                                side_tag = f"NHO LO TRINH CUA {self.corner_memory.get('direction', '')}"
                            else:
                                side_tag = "VACH TRAI" if self.single_edge_side == "LEFT" else ("VACH PHAI" if self.single_edge_side == "RIGHT" else "1 VACH")
                            tam_label_info = (f"TAM DU DOAN ({side_tag})", float(mid_c[0]), float(mid_c[1]))

                    # 3. Xac dinh diem thap nhat cua duong tam trang & tinh do lech tam e so voi vi tri tham chieu xe
                    bot_eval_y = float(hT - 6.0)
                    a_lane_cur, b_lane_cur = self.lane_width_ab
                    w_lane_bot_out = max(min_w_clamp, float(np.clip(a_lane_cur * bot_eval_y + b_lane_cur, min_w_clamp, wT * 1.35)))

                    # Chuan hoa tam lane o day ROI (cx_bot):
                    # Khi co DU 2 vach o day anh, do truc tiep trung diem (xl_b + xr_b)/2.
                    # Khi chi co 1 vach (ONE_EDGE) dang cua hoac cheo, tin dung c_pts[low_idx] tu phap tuyen _offset_segments_by_normal
                    # de tranh sai lech ngang lon giua mui ten do va duong tam trang.
                    cx_bot_meas = None
                    xl_b, xr_b = None, None
                    if self.left_segments and self.right_segments:
                        s_l = min(self.left_segments, key=lambda s: min(abs(s["p1"][1] - bot_eval_y), abs(s["p2"][1] - bot_eval_y)))
                        p1, p2 = s_l["p1"], s_l["p2"]
                        p_lo, p_hi = (p1, p2) if p1[1] > p2[1] else (p2, p1)
                        dy_l = float(p_lo[1] - p_hi[1])
                        if dy_l >= 8.0 and p_lo[1] >= float(hT) * 0.60:
                            slope_l = float(np.clip((p_lo[0] - p_hi[0]) / dy_l, -2.2, 0.35))
                            xl_b = float(p_lo[0] + slope_l * (bot_eval_y - p_lo[1]))
                        s_r = min(self.right_segments, key=lambda s: min(abs(s["p1"][1] - bot_eval_y), abs(s["p2"][1] - bot_eval_y)))
                        p1, p2 = s_r["p1"], s_r["p2"]
                        p_lo, p_hi = (p1, p2) if p1[1] > p2[1] else (p2, p1)
                        dy_r = float(p_lo[1] - p_hi[1])
                        if dy_r >= 8.0 and p_lo[1] >= float(hT) * 0.60:
                            slope_r = float(np.clip((p_lo[0] - p_hi[0]) / dy_r, -0.35, 2.2))
                            xr_b = float(p_lo[0] + slope_r * (bot_eval_y - p_lo[1]))
                        if xl_b is not None and xr_b is not None and (xr_b - xl_b) >= max(40.0, 0.45 * w_lane_bot_out):
                            cx_bot_meas = 0.5 * (xl_b + xr_b)
                    elif (bool(self.left_segments) ^ bool(self.right_segments)) and zone_now == "NONE":
                        only_segs = self.left_segments or self.right_segments
                        if len(only_segs) == 1:
                            p1, p2 = only_segs[0]["p1"], only_segs[0]["p2"]
                            p_lo, p_hi = (p1, p2) if p1[1] > p2[1] else (p2, p1)
                            dy_s = float(p_lo[1] - p_hi[1])
                            dx_s = float(p_lo[0] - p_hi[0])
                            if dy_s >= 18.0 and p_lo[1] >= float(hT) * 0.72 and abs(dx_s) <= 0.95 * dy_s:
                                x_edge_b = float(p_lo[0] + (dx_s / dy_s) * (bot_eval_y - p_lo[1]))
                                cx_bot_meas = (x_edge_b + 0.5 * w_lane_bot_out) if self.left_segments else (x_edge_b - 0.5 * w_lane_bot_out)

                    low_idx = int(np.argmax(c_pts[:, 1]))
                    cx_low_raw = float(c_pts[low_idx, 0])
                    cy_low = float(c_pts[low_idx, 1])

                    if cx_bot_meas is not None:
                        cx_low_raw = float(np.clip(cx_bot_meas, 4.0, wT - 5.0))
                    elif cy_low < bot_eval_y - 2.0 and len(c_pts) >= 2:
                        dy_bot = max(1.0, cy_low - float(c_pts[min(3, len(c_pts) - 1), 1]))
                        slope_c = float(np.clip((cx_low_raw - float(c_pts[min(3, len(c_pts) - 1), 0])) / dy_bot, -1.6, 1.6))
                        cx_low_raw = float(np.clip(cx_low_raw + slope_c * (bot_eval_y - cy_low), 4.0, wT - 5.0))

                    w_lane_ref = max(min_w_clamp, float(np.clip(a_lane_cur * cy_low + b_lane_cur, min_w_clamp, wT * 1.35)))

                    # car_ref_x MUST remain true vehicle/image center (no false schematic override)
                    car_ref_x = wT * 0.5
                    e_px_raw = cx_low_raw - car_ref_x

                    if self.smooth_e_px is None or not fresh_observation:
                        self.smooth_e_px = e_px_raw
                    else:
                        # Giam nhay vot 1 frame do nhieu loa / chuyen trang thai vach
                        de_signed = e_px_raw - self.smooth_e_px
                        de_raw = abs(de_signed)
                        if self.missing_frames == 0 and de_raw > 55.0:
                            de_signed = math.copysign(55.0, de_signed)
                            de_raw = 55.0
                        tau_e = 0.145 if de_raw < 18.0 else 0.090
                        alpha_e = 1.0 - math.exp(-max(0.015, float(dt_step)) / tau_e)
                        if de_raw < 2.8:
                            alpha_e *= 0.40  # Giam rung vi mo cho diem tam xe va chi so e
                        self.smooth_e_px = float(self.smooth_e_px + alpha_e * de_signed)

                    e_px_out = float(self.smooth_e_px)
                    e_norm_out = float(np.clip(e_px_out / (0.5 * w_lane_bot_out), -1.0, 1.0))
                    self.smooth_e_norm = e_norm_out
                    offset_dir = "CAN GIUA" if abs(e_px_out) <= 4.0 else ("LECH PHAI" if e_px_out < 0 else "LECH TRAI")

                    # 4. Lookahead & Perspective-Normalized BEV Stanley + Pure Pursuit + Curvature Feedforward:
                    elbow_y_val = corner.get("elbow_y")
                    elbow_x_val = corner.get("elbow_x")
                    turn_now_flag = bool(corner.get("turn_now", False))
                    c_turn_dir = corner.get("direction") if corner.get("direction") in ("LEFT", "RIGHT") else corner.get("candidate", "UNKNOWN")
                    if c_turn_dir not in ("LEFT", "RIGHT") and not (self.left_segments and self.right_segments):
                        for d_fb in (
                            corner.get("intent_direction"),
                            self.corner_memory.get("direction") if self.corner_memory.get("active") else None,
                        ):
                            if d_fb in ("LEFT", "RIGHT"):
                                c_turn_dir = d_fb
                                break
                        if c_turn_dir not in ("LEFT", "RIGHT"):
                            s_only_fb = self.left_segments or self.right_segments
                            if s_only_fb:
                                dx_fb = float(s_only_fb[-1]["p2"][0] - s_only_fb[0]["p1"][0])
                                if self.single_edge_side == "LEFT" and dx_fb >= 110.0:
                                    c_turn_dir = "RIGHT"
                                elif self.single_edge_side == "RIGHT" and dx_fb <= -110.0:
                                    c_turn_dir = "LEFT"

                    stem_c_pts = c_pts
                    diffs_c = np.diff(stem_c_pts, axis=0)
                    seg_lens_c = np.hypot(diffs_c[:, 0], diffs_c[:, 1])
                    cum_s = np.r_[0.0, np.cumsum(seg_lens_c)]
                    total_s = float(cum_s[-1])

                    def _interp_at_s(s_q):
                        sq = _clip_scalar(s_q, 0.0, total_s)
                        idx_s = min(len(stem_c_pts) - 2, max(0, int(np.searchsorted(cum_s, sq, side="right") - 1)))
                        ds_seg = float(seg_lens_c[idx_s])
                        if ds_seg < 1e-4:
                            return float(stem_c_pts[idx_s, 0]), float(stem_c_pts[idx_s, 1])
                        t_s = (sq - float(cum_s[idx_s])) / ds_seg
                        return (
                            float(stem_c_pts[idx_s, 0] + t_s * diffs_c[idx_s, 0]),
                            float(stem_c_pts[idx_s, 1] + t_s * diffs_c[idx_s, 1]),
                        )

                    s_near = min(total_s * 0.18, float(hT * 0.10))
                    cx_near, cy_near = _interp_at_s(s_near)

                    s_base = min(total_s * 0.62, float(hT * 0.44))
                    wide_blend = 0.0
                    is_high_bar_only = False
                    if turn_now_flag and not (self.left_segments and self.right_segments):
                        s_only = self.left_segments or self.right_segments
                        if (
                            s_only
                            and max(float(sg["p1"][1]) for sg in s_only) < float(hT) * 0.54
                            and (elbow_y_val is not None and float(elbow_y_val) < float(hT) * 0.52)
                        ):
                            is_high_bar_only = True
                    turn_ang = float(corner.get("turn_angle_deg", 0.0))
                    if self.corner_memory.get("active"):
                        turn_ang = max(turn_ang, float(self.corner_memory.get("turn_angle_deg", 0.0)))
                    if not (self.left_segments and self.right_segments) and c_turn_dir in ("LEFT", "RIGHT") and turn_ang >= 24.0:
                        single_segs_cur = self.left_segments or self.right_segments
                        dx_arm_cur = abs(float(single_segs_cur[-1]["p2"][0] - single_segs_cur[0]["p1"][0])) if single_segs_cur else 0.0
                        if self.corner_memory.get("active") and len(stem_c_pts) >= 2:
                            dx_arm_cur = max(dx_arm_cur, abs(float(stem_c_pts[-1, 0] - stem_c_pts[0, 0])))
                        arm_blend = _smoothstep(wT * 0.30, wT * 0.54, dx_arm_cur)
                        c_dir_w_sgn = 1.0 if c_turn_dir == "RIGHT" else -1.0
                        dx_c_turn = (
                            float(stem_c_pts[-1, 0] - stem_c_pts[0, 0]) * c_dir_w_sgn
                            if len(stem_c_pts) >= 2
                            else 0.0
                        )
                        s_blend = (
                            max(
                                _smoothstep(float(hT) * 1.15, float(hT) * 1.48, total_s),
                                _smoothstep(float(hT) * 0.78, float(hT) * 0.98, total_s)
                                * _smoothstep(wT * 0.22, wT * 0.32, dx_c_turn),
                            )
                            if (len(single_segs_cur) >= 2 or self.corner_memory.get("active"))
                            else 0.0
                        )
                        if turn_ang >= 36.0:
                            wide_blend = _smoothstep(45.0, 58.0, turn_ang) * arm_blend * s_blend
                        elif dx_arm_cur >= wT * 0.44:
                            wide_blend = 0.55 * arm_blend * s_blend
                    if (turn_now_flag or corner.get("distance_zone") == "TURN_NOW") and not is_high_bar_only:
                        turn_prog = 0.92 if not (self.left_segments and self.right_segments) else 1.0
                    elif elbow_y_val is not None and c_turn_dir in ("LEFT", "RIGHT"):
                        ang_scale = _smoothstep(24.0, 36.0, turn_ang)
                        ang_blend = _smoothstep(28.0, 58.0, turn_ang)
                        prog_steep = _smoothstep(0.24, 0.60, float(elbow_y_val) / float(max(1, hT)))
                        prog_gentle = _smoothstep(0.46, 0.72, float(elbow_y_val) / float(max(1, hT)))
                        turn_prog = ((1.0 - ang_blend) * prog_gentle + ang_blend * prog_steep) * ang_scale
                        if not (self.left_segments and self.right_segments) and turn_ang >= 24.0:
                            if turn_ang >= 36.0:
                                single_prog = 0.42 + 0.30 * _smoothstep(45.0, 56.0, turn_ang) * arm_blend + 0.18 * wide_blend
                                turn_prog = max(turn_prog, single_prog)
                            elif dx_arm_cur >= wT * 0.44:
                                # Goc cua vat 45 do nhin thay toan bo thanh ngang phia truoc (ONE_EDGE): giu turn_prog cao
                                turn_prog = max(turn_prog, 0.58 + 0.22 * arm_blend)
                    else:
                        turn_prog = 0.0

                    if fresh_observation and self._smooth_turn_prog is not None:
                        alpha_tp = 1.0 - math.exp(-max(0.015, float(dt_step)) / 0.085)
                        turn_prog = float(alpha_tp * turn_prog + (1.0 - alpha_tp) * self._smooth_turn_prog)
                    self._smooth_turn_prog = float(turn_prog) if fresh_observation else None

                    # Chi giu lookahead tren doan thang truoc khuyu cua khi xe CON o trong hanh lang 2 vach (TWO_EDGES)
                    # va dinh cua chua xuong sat mui xe (elbow_y < 166px); chuyen tiep muot sang nhanh cua khi tien gan,
                    # TUYET DOI khong khoa lookahead sau khuyu cua khi chi con 1 vach (ONE_EDGE) giua cua!
                    has_both_edges_now = bool(self.left_segments and self.right_segments)
                    approach_clamp_w = 0.0
                    if (
                        has_both_edges_now
                        and not turn_now_flag
                        and corner.get("distance_zone") != "TURN_NOW"
                        and elbow_y_val is not None
                        and float(elbow_y_val) < 172.0
                    ):
                        approach_clamp_w = 1.0 - _smoothstep(144.0, 172.0, float(elbow_y_val))

                    if turn_prog > 0.0 and (elbow_y_val is not None or not has_both_edges_now):
                        if elbow_x_val is not None and elbow_y_val is not None:
                            d2_eb = (stem_c_pts[:, 0] - float(elbow_x_val)) ** 2 + (stem_c_pts[:, 1] - float(elbow_y_val)) ** 2
                            s_elbow = float(cum_s[int(np.argmin(d2_eb))])
                        elif elbow_y_val is not None:
                            d2_eb = (stem_c_pts[:, 1] - float(elbow_y_val)) ** 2
                            s_elbow = float(cum_s[int(np.argmin(d2_eb))])
                        else:
                            s_elbow = float(total_s * 0.35)
                        if not has_both_edges_now and len(stem_c_pts) >= 3 and wide_blend > 0.05:
                            vx_ch = float(stem_c_pts[-1, 0] - stem_c_pts[0, 0])
                            vy_ch = float(stem_c_pts[-1, 1] - stem_c_pts[0, 1])
                            len_ch = math.hypot(vx_ch, vy_ch)
                            if len_ch > 24.0:
                                perp_d = np.abs(
                                    vy_ch * (stem_c_pts[:, 0] - stem_c_pts[0, 0])
                                    - vx_ch * (stem_c_pts[:, 1] - stem_c_pts[0, 1])
                                ) / len_ch
                                s_knee = float(cum_s[int(np.argmax(perp_d))])
                                s_elbow = max(s_elbow, s_knee)
                        c_dir_sgn = 1.0 if c_turn_dir == "RIGHT" else (-1.0 if c_turn_dir == "LEFT" else 0.0)
                        inside_drift = (-e_norm_out * c_dir_sgn) if c_dir_sgn != 0.0 else 0.0
                        inside_look_boost = (
                            0.28 * float(hT) * float(turn_prog) * _smoothstep(0.14, 0.34, inside_drift)
                            if not has_both_edges_now
                            else 0.0
                        )
                        if not has_both_edges_now and c_dir_sgn != 0.0 and (cx_low_raw - car_ref_x) * c_dir_sgn < -20.0:
                            inside_look_boost += float(turn_prog) * min(160.0, abs(car_ref_x - cx_low_raw) * 0.90)
                        s_min_turn = s_base + inside_look_boost
                        s_frac_cap = 0.84 if inside_look_boost > 12.0 else 0.72
                        s_corner_std = min(total_s * s_frac_cap, float(hT * 0.68) + inside_look_boost, max(s_min_turn, s_elbow + float(hT * 0.16)))
                        s_corner_wide = min(total_s * max(0.78, s_frac_cap), max(float(hT * 0.74) + inside_look_boost, s_elbow + float(hT * 0.28)))
                        s_corner = (1.0 - wide_blend) * s_corner_std + wide_blend * s_corner_wide
                        s_blend_turn = (1.0 - turn_prog) * s_base + turn_prog * s_corner
                        if approach_clamp_w > 0.0:
                            s_pre_eb = min(s_base, max(s_near + 8.0, s_elbow - 8.0))
                            target_s = approach_clamp_w * s_pre_eb + (1.0 - approach_clamp_w) * s_blend_turn
                        else:
                            target_s = s_blend_turn
                    else:
                        target_s = s_base
                    s_missing = max(0.0, float(hT - 6.0) - cy_low)
                    if s_missing > 6.0:
                        target_s = max(min(total_s * 0.35, 36.0), target_s - 0.55 * s_missing)
                    cx_look, cy_look = _interp_at_s(target_s)

                    lookahead_dist_out = float(target_s)
                    lookahead_c_pt = (cx_look, cy_look)
                    w_lane_look = max(min_w_clamp, float(np.clip(a_lane_cur * cy_look + b_lane_cur, min_w_clamp, wT * 1.35)))

                    # Signed local path curvature around lookahead (3-point Menger curvature in pixel^-1)
                    if total_s >= 24.0:
                        p_a = np.array(_interp_at_s(max(0.0, target_s - 0.18 * total_s)), dtype=np.float32)
                        p_b = np.array([cx_look, cy_look], dtype=np.float32)
                        p_c = np.array(_interp_at_s(min(total_s, target_s + 0.18 * total_s)), dtype=np.float32)
                        v_ab = p_b - p_a
                        v_bc = p_c - p_b
                        v_ac = p_c - p_a
                        dab = float(math.hypot(float(v_ab[0]), float(v_ab[1])))
                        dbc = float(math.hypot(float(v_bc[0]), float(v_bc[1])))
                        dac = float(math.hypot(float(v_ac[0]), float(v_ac[1])))
                        denom_k = dab * dbc * dac
                        if denom_k > 1e-2:
                            cross_k = float(v_ab[0] * v_bc[1] - v_ab[1] * v_bc[0])
                            path_curvature_out = (2.0 * cross_k) / denom_k

                    # Perspective-normalized ground coordinates (in lane-width units; 0.5 = half-lane)
                    k_persp = 1.45 * w_lane_bot_out

                    def _to_ground_lw(px_g, py_g):
                        w_g = max(min_w_clamp, float(np.clip(a_lane_cur * py_g + b_lane_cur, min_w_clamp, wT * 1.35)))
                        x_lat_g = (float(px_g) - car_ref_x) / w_g
                        # Rear-axle longitudinal distance (~1.67 lane widths to bottom ROI)
                        y_fwd_g = max(0.55, 1.67 + k_persp * (1.0 / w_g - 1.0 / w_lane_bot_out))
                        return x_lat_g, y_fwd_g

                    x_near_g, y_near_g = _to_ground_lw(cx_near, cy_near)
                    x_look_g, y_look_g = _to_ground_lw(cx_look, cy_look)

                    # Chuan hoa goc huong Stanley: tinh theo tiep tuyen tim duong tai vung day quanh truc truoc (y = 190 - 230 px),
                    # khong dung vector xien noi toi diem ngam nhanh re ngang
                    s_tan = min(total_s * 0.25, max(s_near, float(hT * 0.16)))
                    cx_tan, cy_tan = _interp_at_s(s_tan)
                    cx_bot, cy_bot = _interp_at_s(0.0)
                    x_bot_g, y_bot_g = _to_ground_lw(cx_bot, cy_bot)
                    x_tan_g, y_tan_g = _to_ground_lw(cx_tan, cy_tan)
                    dX_tan = x_tan_g - x_bot_g
                    dY_tan = max(0.18, y_tan_g - y_bot_g)
                    heading_rad = float(math.atan2(dX_tan, dY_tan))
                    psi_rad_out = heading_rad

                    # Ground-plane curvature feedforward
                    kappa_ff_rad = 0.0
                    wheelbase_lw = 0.87  # 0.20m / 0.23m
                    if total_s >= 24.0:
                        g_a = _to_ground_lw(float(p_a[0]), float(p_a[1]))
                        g_b = _to_ground_lw(float(p_b[0]), float(p_b[1]))
                        g_c = _to_ground_lw(float(p_c[0]), float(p_c[1]))
                        vg_ab = (g_b[0] - g_a[0], g_b[1] - g_a[1])
                        vg_bc = (g_c[0] - g_b[0], g_c[1] - g_b[1])
                        vg_ac = (g_c[0] - g_a[0], g_c[1] - g_a[1])
                        dg_ab = math.hypot(vg_ab[0], vg_ab[1])
                        dg_bc = math.hypot(vg_bc[0], vg_bc[1])
                        dg_ac = math.hypot(vg_ac[0], vg_ac[1])
                        denom_kg = dg_ab * dg_bc * dg_ac
                        if denom_kg > 1e-3:
                            # In (X right, Y forward), positive cross_right = RIGHT turn
                            cross_kg = vg_ab[1] * vg_bc[0] - vg_ab[0] * vg_bc[1]
                            kappa_g = _clip_scalar((2.0 * cross_kg) / denom_kg, -2.5, 2.5)
                            kappa_ff_rad = float(math.atan(kappa_g * wheelbase_lw))

                    # Pure Pursuit from rear axle (x_look_g, y_look_g) + Stanley (psi + k_lat * e_lat)
                    ld_raw = math.hypot(x_look_g, y_look_g)
                    ld_straight = max(1.80, ld_raw)
                    # Chi co rut lookahead khi thuc su dang vao cua (turn_prog) hoac duong cong/goc huong lon,
                    # KHONG co rut Ld tren duong thang chi vi lech tam ngang nho (tranh phong dai PP 4x-6x tren duong thang!)
                    curv_tp = _smoothstep(8.0, 18.0, abs(math.degrees(kappa_ff_rad))) if (kappa_ff_rad * x_look_g > 0.0) else 0.0
                    head_tp = _smoothstep(13.0, 26.0, abs(math.degrees(heading_rad))) if (heading_rad * x_look_g > 0.0) else 0.0
                    tp_look = max(
                        _smoothstep(0.10, 0.50, float(turn_prog)),
                        curv_tp,
                        head_tp,
                    )
                    # Cung tron chat: diem look o xa (y_look_g >= ~1.7 lw) nen rut ld ve 0.85-0.95 se phong dai kappa_pp (ld_raw/ld_g)^2 lan
                    # -> xe cat vao trong. Chi giam rut ld khi dung cung tron (khong o vung goc cua / turn_prog thap).  ARC_LD_RELIEF=0 de tat.
                    arc_ld_w = 0.0
                    if (ARC_LD_RELIEF > 0.0 and tp_look > 0.0 and kappa_ff_rad * x_look_g > 0.0
                            and corner.get("distance_zone") not in ("APPROACH", "TURN_NOW")
                            and not turn_now_flag):
                        arc_ld_w = _smoothstep(ARC_FF_DEG_START, ARC_FF_DEG_FULL, abs(math.degrees(kappa_ff_rad)))
                        arc_ld_w *= 1.0 - _smoothstep(0.05, 0.25, float(turn_prog))
                        tp_look *= (1.0 - ARC_LD_RELIEF * arc_ld_w)
                    # Giam muot tp_look khi dang tiep can thang truoc khuyu cua (TWO_EDGES, elbow_y < 138px)
                    if approach_clamp_w > 0.0:
                        tp_look *= (1.0 - approach_clamp_w)
                    ld_turn = _clip_scalar(ld_raw, LD_TURN_MIN, LD_TURN_MAX)
                    ld_g = (1.0 - tp_look) * ld_straight + tp_look * ld_turn
                    kappa_pp = (2.0 * x_look_g) / (ld_g * ld_g)
                    pp_rad = float(math.atan(kappa_pp * wheelbase_lw))

                    k_lat_stanley = 1.10 - 0.35 * _clip_scalar(turn_prog, 0.0, 1.0)
                    # Co khuc cua phia truoc: e_lat nguoc huong cua KHONG duoc trit tieu goc cua cua PP
                    zone_now = corner.get("distance_zone")
                    tp_eff = float(turn_prog)
                    if zone_now in ("APPROACH", "TURN_NOW"):
                        tp_eff = max(tp_eff, TURN_ZONE_PROG_FLOOR)
                    if c_turn_dir == "RIGHT":
                        dir_sign = 1.0
                    elif c_turn_dir == "LEFT":
                        dir_sign = -1.0
                    else:
                        dir_sign = 1.0 if heading_rad > 0.0 else (-1.0 if heading_rad < 0.0 else 0.0)

                    # Bu do tre nhan dien (Perception Latency Compensation) cho sai so ngang e_norm
                    # Tinh tien du bao sai so theo van toc va goc huong, suy giam em ai o goc cua de tranh overshoot
                    # Dung chu ky khung hinh on dinh tranh rung nhieu do CPU jitter giua cac frame
                    fps_val = float(self.fps) if (hasattr(self, "fps") and self.fps is not None and self.fps > 0) else 15.0
                    dt_lat = max(0.025, min(0.065, 1.0 / max(5.0, fps_val)))
                    k_pred = float(np.clip(3.8 * dt_lat, 0.0, 0.14)) * (1.0 - _smoothstep(0.15, 0.45, abs(heading_rad)))
                    e_norm_pred = float(_clip_scalar(e_norm_out + k_pred * math.sin(heading_rad), -1.0, 1.0))

                    # Bu phan lech do HINH HOC cung tron cua e do o hang day (chi ap dung tren cung tron TWO_EDGES thuc su)
                    e_stan_in = e_norm_pred
                    e_curv_comp = 0.0
                    kappa_near_g = 0.0
                    if (
                        E_CURV_COMP_GAIN > 0.0
                        and has_both_edges_now
                        and total_s >= 36.0
                        and zone_now not in ("APPROACH", "TURN_NOW")
                        and not turn_now_flag
                        and approach_clamp_w <= 0.0
                        and elbow_y_val is None
                    ):
                        s_m = min(total_s * 0.35, float(hT * 0.30))
                        gn_a = _to_ground_lw(*_interp_at_s(0.0))
                        gn_b = _to_ground_lw(*_interp_at_s(0.5 * s_m))
                        gn_c = _to_ground_lw(*_interp_at_s(s_m))
                        vn_ab = (gn_b[0] - gn_a[0], gn_b[1] - gn_a[1])
                        vn_bc = (gn_c[0] - gn_b[0], gn_c[1] - gn_b[1])
                        vn_ac = (gn_c[0] - gn_a[0], gn_c[1] - gn_a[1])
                        den_n = math.hypot(*vn_ab) * math.hypot(*vn_bc) * math.hypot(*vn_ac)
                        if den_n > 1e-3:
                            cross_n = vn_ab[1] * vn_bc[0] - vn_ab[0] * vn_bc[1]  # >0 = re PHAI
                            kappa_near_g = _clip_scalar((2.0 * cross_n) / den_n, -E_CURV_KAPPA_MAX, E_CURV_KAPPA_MAX)
                            if kappa_near_g * kappa_ff_rad > 0.0:
                                k_eff = math.copysign(max(0.0, abs(kappa_near_g) - E_CURV_KAPPA_DEADBAND), kappa_near_g)
                                if abs(k_eff) > 1e-6:
                                    u_k = min(0.90, abs(k_eff) * float(gn_a[1]))
                                    off_lw = (1.0 - math.sqrt(1.0 - u_k * u_k)) / abs(k_eff)
                                    gate_c = 1.0 - _smoothstep(0.05, 0.25, float(turn_prog))
                                    e_curv_comp = _clip_scalar(
                                        math.copysign(2.0 * off_lw, k_eff) * E_CURV_COMP_GAIN * gate_c,
                                        -0.65,
                                        0.65,
                                    )
                                    e_stan_in = float(_clip_scalar(e_norm_pred - e_curv_comp, -1.0, 1.0))
                    e_for_stanley = e_stan_in
                    if tp_eff > 0.0 and dir_sign != 0.0 and (e_stan_in * dir_sign) < 0.0:
                        e_for_stanley = e_stan_in * (1.0 - E_CANCEL_GAIN * _clip_scalar(tp_eff, 0.0, 1.0))
                        # Khi dang giua goc cua 1 vach (ONE_EDGE) va xe da di sat mep trong cua (|e| > 0.16),
                        # hoac dang TIEP CAN khuc cua gat 1 vach ngoai: khong de phan hoi Stanley nguoc chieu be lai nguoc han sang phia ngoai dam vao tuong ngoai
                        single_only_segs = (self.left_segments or self.right_segments) if not has_both_edges_now else None
                        single_dx_span = (
                            abs(float(single_only_segs[-1]["p2"][0] - single_only_segs[0]["p1"][0]))
                            if single_only_segs
                            else 0.0
                        )
                        is_single_sweeping_wall = (not has_both_edges_now) and (single_dx_span >= wT * 0.42)
                        if not has_both_edges_now and (zone_now in ("APPROACH", "TURN_NOW") or is_single_sweeping_wall):
                            tp_opp = max(float(turn_prog), 0.35 if zone_now == "APPROACH" else 0.70)
                            e_for_stanley *= (1.0 - 0.72 * _smoothstep(0.18, 0.55, tp_opp))
                            e_cap = 0.12 if (zone_now == "TURN_NOW" or (is_single_sweeping_wall and turn_prog >= 0.32)) else 0.18
                            if abs(e_for_stanley) > e_cap:
                                e_for_stanley = math.copysign(e_cap + 0.15 * (abs(e_for_stanley) - e_cap), e_for_stanley)
                    # Che do CUNG TRON: chi bat khi curvature ground-plane (kappa_ff) du lon, cung dau voi PP,
                    # khong o vung goc cua (APPROACH/TURN_NOW) va turn_prog thap -> khong anh huong goc 90 do / S-curve nhe.
                    arc_w = 0.0
                    if (
                        ARC_HEADING_RELIEF > 0.0
                        and kappa_ff_rad * pp_rad > 0.0
                        and zone_now not in ("APPROACH", "TURN_NOW")
                        and not turn_now_flag
                    ):
                        arc_w = _smoothstep(ARC_FF_DEG_START, ARC_FF_DEG_FULL, abs(math.degrees(kappa_ff_rad)))
                        arc_w *= 1.0 - _smoothstep(0.05, 0.25, float(turn_prog))
                    heading_w_stanley = 0.95 * (1.0 - ARC_HEADING_RELIEF * arc_w)
                    stanley_rad = heading_w_stanley * heading_rad + float(math.atan(k_lat_stanley * e_for_stanley))
                    ctrl_dbg = {
                        "e_norm_raw": float(e_norm_out), "e_curv_comp": float(e_curv_comp), "e_stanley": float(e_for_stanley),
                        "kappa_near_g": float(kappa_near_g), "kappa_ff_deg": float(math.degrees(kappa_ff_rad)),
                        "arc_w": float(arc_w), "heading_w": float(heading_w_stanley), "heading_deg": float(math.degrees(heading_rad)),
                        "pp_deg": float(math.degrees(pp_rad)), "ld_raw": float(ld_raw), "ld_g": float(ld_g),
                        "x_look_g": float(x_look_g), "y_look_g": float(y_look_g), "turn_prog": float(turn_prog), "arc_ld_w": float(arc_ld_w), "tp_look": float(tp_look),
                    }

                    # Kien truc cong huong: PP giu NGUYEN 100% (goc Ackermann arctan(L*kappa) khong bi nhan giam),
                    # Stanley chi la phan hoi hieu chinh CTE/heading cong them vao.
                    # Hoa tron 2 che do: duong thang (convex blend nhe, khong cong don) <-> cua (PP + 0.65*Stanley).
                    # mix theo turn_prog HOAC theo |pp| (cua tron R=0.55m co turn_prog = 0 nhung |pp| ~ 20 do)
                    mix_curve = max(
                        _smoothstep(0.10, 0.50, float(turn_prog)),
                        _smoothstep(4.0, 11.0, abs(math.degrees(pp_rad))),
                    )
                    raw_straight_rad = 0.72 * stanley_rad + 0.28 * pp_rad
                    raw_curve_rad = pp_rad + W_STANLEY_FB * stanley_rad
                    raw_steer_rad = (1.0 - mix_curve) * raw_straight_rad + mix_curve * raw_curve_rad
                    raw_wheel_deg = math.degrees(raw_steer_rad)
                    if (
                        not has_both_edges_now
                        and dir_sign != 0.0
                        and (raw_wheel_deg * dir_sign) < 0.0
                        and (zone_now in ("APPROACH", "TURN_NOW") or turn_now_flag)
                        and (turn_ang >= 40.0 or is_single_sweeping_wall)
                    ):
                        # Khong bao gio danh lai nguoc huong cua khi chi con 1 vach ngoai ngay truoc khuc cua gat
                        max_opp_deg = 0.0 if (zone_now == "TURN_NOW" or turn_now_flag) else 1.5
                        if abs(raw_wheel_deg) > max_opp_deg:
                            raw_wheel_deg = -dir_sign * max_opp_deg

                    heading_deg_out = math.degrees(heading_rad)
                    steer_cap_norm = STRAIGHT_STEER_CAP + (1.0 - STRAIGHT_STEER_CAP) * mix_curve
                    raw_steer_norm = _clip_scalar(raw_wheel_deg / 35.0, -steer_cap_norm, steer_cap_norm)
                    if fresh_observation and self._smooth_steer_norm is not None:
                        if raw_steer_norm * self._smooth_steer_norm < -0.25 and abs(raw_steer_norm - self._smooth_steer_norm) > 0.65:
                            steer_norm_out = float(0.70 * raw_steer_norm + 0.30 * self._smooth_steer_norm)
                        else:
                            # Vao cua (|lai| dang tang cung chieu, hoac PP da bao co duong cong): bo tre pha cua bo loc
                            entering_curve = (
                                raw_steer_norm * self._smooth_steer_norm >= 0.0
                                and (abs(raw_steer_norm) - abs(self._smooth_steer_norm)) >= 0.035
                                and (abs(math.degrees(pp_rad)) >= STEER_ENTRY_PP_DEG or turn_prog > 0.15)
                            )
                            tau_st = STEER_TAU_ENTRY_S if entering_curve else STEER_TAU_NORMAL_S
                            alpha_st = 1.0 - math.exp(-max(0.015, float(dt_step)) / tau_st)
                            steer_norm_out = float(alpha_st * raw_steer_norm + (1.0 - alpha_st) * self._smooth_steer_norm)
                    else:
                        steer_norm_out = raw_steer_norm
                    self._smooth_steer_norm = float(steer_norm_out) if fresh_observation else None

                    front_wheel_deg_out = _clip_scalar(steer_norm_out * 35.0, -35.0, 35.0)
                    servo_deg_out = int(_clip_scalar(round(90.0 + steer_norm_out * 60.0), 30, 150))
                    self.front_wheel_angle_deg = front_wheel_deg_out
                    self.servo_angle_deg = servo_deg_out
                    self.heading_error_deg = heading_deg_out

                    cx_low_disp = float(np.clip(car_ref_x + e_px_out, 10.0, wT - 11.0)) if fresh_observation else cx_low_raw
                    lowest_c_pt = (cx_low_disp, float(hT - 6))
                    raw_curve = steer_norm_out * 90.0

                    if draw_overlay:
                        # Ve diem ngam Lookahead (vong tron xanh ngoc) tren duong tam trang
                        pt_look = (int(round(np.clip(cx_look, 2.0, wT - 3.0))), int(round(np.clip(cy_look, 2.0, hT - 3.0))))
                        cv2.circle(img_result, pt_look, 5, (255, 255, 0), 2, lineType=cv2.LINE_AA)

                        # 5. VE TAM XE, VECTOR GOC QUAY BANH TRUOC & LO TRINH GOC CUA DA NHO
                        arrow_base_y = hT - 42
                        pt_arrow_base = (int(round(cx_low_disp)), arrow_base_y)
                        arrow_len = max(34, int(hT * 0.19))
                        pt_car_ref = (int(round(car_ref_x)), arrow_base_y)

                        # Duong ngam Pure Pursuit / Guidance spline tu mui xe toi Lookahead
                        # Ve theo dung spline c_pts tu pt_arrow_base toi pt_look de triet tieu hoan toan gay khuc / zig-zag
                        guide_pts = [pt_arrow_base]
                        if len(c_pts) >= 2:
                            for k_g in range(len(c_pts)):
                                if cy_look <= c_pts[k_g, 1] <= arrow_base_y:
                                    guide_pts.append((int(round(c_pts[k_g, 0])), int(round(c_pts[k_g, 1]))))
                        guide_pts.append(pt_look)
                        if len(guide_pts) >= 2:
                            cv2.polylines(img_result, [np.array(guide_pts, dtype=np.int32)], False, (0, 230, 180), 2, cv2.LINE_AA)

                        # Ve truc tham chieu xe (vach vang) va thanh ngang do lech e (giu phia tren dong chu HUD day)
                        cv2.line(
                            img_result,
                            (pt_car_ref[0], max(8, arrow_base_y - 12)),
                            (pt_car_ref[0], min(hT - 36, arrow_base_y + 3)),
                            (0, 255, 255), 2, cv2.LINE_AA
                        )
                        cv2.line(img_result, pt_car_ref, pt_arrow_base, (0, 180, 255), 2, cv2.LINE_AA)
                        cv2.circle(img_result, pt_car_ref, 4, (0, 255, 255), cv2.FILLED, lineType=cv2.LINE_AA)

                        # Ve hinh banh truoc xoay theo goc front_wheel_deg_out va mui ten huong banh truoc
                        wheel_rad = float(np.deg2rad(front_wheel_deg_out))
                        tip_x = int(round(cx_low_disp + arrow_len * np.sin(wheel_rad)))
                        tip_y = int(round(max(8.0, arrow_base_y - arrow_len * np.cos(wheel_rad))))
                        pt_arrow_tip = (int(np.clip(tip_x, 4, wT - 5)), int(np.clip(tip_y, 4, hT - 34)))

                        # Truc thang dung cua than xe (net manh mau trang xam de so sanh goc lech banh truoc)
                        cv2.line(
                            img_result,
                            pt_arrow_base,
                            (int(round(cx_low_disp)), max(8, arrow_base_y - int(arrow_len * 0.85))),
                            (180, 180, 180), 1, cv2.LINE_AA
                        )
                        # Mui ten DO bieu thi huong lai cua banh truoc & tam xe
                        cv2.arrowedLine(img_result, pt_arrow_base, pt_arrow_tip, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.28)
                        cv2.circle(img_result, pt_arrow_base, 5, (255, 255, 255), cv2.FILLED, lineType=cv2.LINE_AA)
                        cv2.circle(img_result, pt_arrow_base, 3, (0, 0, 255), cv2.FILLED, lineType=cv2.LINE_AA)

                        if tam_label_info is not None:
                            tam_text, mx_c, my_c = tam_label_info
                            (tw_t, th_t), _ = cv2.getTextSize(tam_text, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
                            tam_y = max(76, min(hT - 80, int(round(my_c)) + 16))
                            tam_x = max(8, min(wT - tw_t - 8, int(round(mx_c)) - (tw_t // 2)))
                            ax_lo = min(pt_arrow_base[0], pt_arrow_tip[0]) - 12
                            ax_hi = max(pt_arrow_base[0], pt_arrow_tip[0]) + 12
                            ay_top = min(pt_arrow_base[1], pt_arrow_tip[1])
                            if tam_y >= ay_top - 10 and (tam_x < ax_hi and tam_x + tw_t > ax_lo):
                                if pt_arrow_base[0] >= wT * 0.52 and ax_lo - tw_t - 6 >= 8:
                                    tam_x = ax_lo - tw_t - 6
                                elif pt_arrow_base[0] < wT * 0.48 and ax_hi + 6 + tw_t <= wT - 8:
                                    tam_x = ax_hi + 6
                                else:
                                    tam_y = max(76, min(tam_y, ay_top - 12))
                            cv2.putText(
                                img_result, tam_text,
                                (int(tam_x), int(tam_y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 230, 255), 1, cv2.LINE_AA
                            )

                        # Hien thi trang thai khoang cach goc cua, Goc cua thuc te (da bu phoi canh) & Ve chi dan huong cua
                        if corner.get("candidate") in ("LEFT", "RIGHT") and elbow_y_val is not None:
                            c_dir_vn = "PHAI" if corner["candidate"] == "RIGHT" else "TRAI"
                            g_turn_deg = float(corner.get("ground_turn_angle_deg") or corner.get("turn_angle_deg") or 0.0)
                            ang_tag = f" ~{int(round(g_turn_deg))} deg" if g_turn_deg >= 15.0 else ""
                            mem_tag = " [DA NHO LO TRINH]" if self.corner_memory.get("active") else ""
                            if turn_now_flag:
                                c_txt = f"CUA {c_dir_vn}{ang_tag} (y={int(elbow_y_val)}) -> RE NGAY!{mem_tag}"
                                c_col = (0, 165, 255)
                            elif elbow_y_val >= 165.0:
                                c_txt = f"CUA {c_dir_vn}{ang_tag} (y={int(elbow_y_val)}) -> VAO CUA{mem_tag}"
                                c_col = (0, 200, 255)
                            elif corner.get("distance_zone") == "APPROACH":
                                c_txt = f"CUA {c_dir_vn}{ang_tag} (y={int(elbow_y_val)}) -> TIEP CAN{mem_tag}"
                                c_col = (0, 230, 255)
                            else:
                                c_txt = f"CUA {c_dir_vn}{ang_tag} PHIA TRUOC (y={int(elbow_y_val)}){mem_tag}"
                                c_col = (255, 230, 0)
                            cv2.putText(img_result, c_txt, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.39, c_col, 1, cv2.LINE_AA)

                            # Ve diem dinh cua (elbow) kem nhan so do goc cua & mui ten chi dan thoat cua o phia truoc
                            if elbow_x_val is not None:
                                eb_pt = (
                                    int(_clip_scalar(round(float(elbow_x_val)), 6.0, float(wT - 7))),
                                    int(_clip_scalar(round(float(elbow_y_val)), 6.0, float(hT - 7))),
                                )
                                cv2.circle(img_result, eb_pt, 6, (0, 165, 255), 2, cv2.LINE_AA)
                                cv2.circle(img_result, eb_pt, 2, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)
                                if g_turn_deg >= 15.0:
                                    eb_lbl = f"{int(round(g_turn_deg))} deg"
                                    lbl_x = max(8, min(wT - 58, eb_pt[0] - 54 if corner["candidate"] == "RIGHT" else eb_pt[0] + 10))
                                    lbl_y = max(72, min(hT - 44, eb_pt[1] + 4))
                                    cv2.putText(img_result, eb_lbl, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 235, 255), 1, cv2.LINE_AA)

                            if len(c_pts) >= 4:
                                idx_tail0 = max(1, len(c_pts) - 4)
                                p_ex0 = (int(round(float(c_pts[idx_tail0, 0]))), int(round(float(c_pts[idx_tail0, 1]))))
                                p_ex1 = (int(round(float(c_pts[-1, 0]))), int(round(float(c_pts[-1, 1]))))
                                if math.hypot(p_ex1[0] - p_ex0[0], p_ex1[1] - p_ex0[1]) >= 12.0:
                                    cv2.arrowedLine(img_result, p_ex0, p_ex1, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)

                        wheel_dir_vn = (
                            "THANG" if abs(front_wheel_deg_out) < 1.5
                            else ("RE PHAI" if front_wheel_deg_out > 0 else "RE TRAI")
                        )
                        wheel_txt = f"Goc banh truoc: {front_wheel_deg_out:+.1f} deg ({wheel_dir_vn}) | Servo: {servo_deg_out} deg"
                        cv2.putText(img_result, wheel_txt, (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (80, 255, 160), 1, cv2.LINE_AA)

                        j_disp = getattr(self, "_last_junction_info", None) or {}
                        if j_disp.get("detected"):
                            jt = j_disp.get("type", "NONE")
                            j_lbl = (
                                "NGA 3 (|-): THANG / PHAI" if jt == "RIGHT_BRANCH"
                                else ("NGA 3 (-|): THANG / TRAI" if jt == "LEFT_BRANCH" else "NGA 3 CHU T: TRAI / PHAI")
                            )
                            if junction_intent in ("STRAIGHT", "LEFT", "RIGHT"):
                                act_vn = "DI THANG" if junction_intent == "STRAIGHT" else ("RE PHAI" if junction_intent == "RIGHT" else "RE TRAI")
                                j_lbl += f" => {act_vn}"
                            cv2.putText(img_result, j_lbl, (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

                    if draw_aux_views:
                        warp_lines = utils.warpImg(img_result, points, wT, hT)
                        img_warp_vis = cv2.addWeighted(img_warp_vis, 0.35, warp_lines, 0.65, 0)
                        cv2.putText(
                            img_hist_combined,
                            f"{status} | e:{e_px_out:+.1f}px ({offset_dir})",
                            (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 255, 255), 1
                        )
                else:
                    if draw_overlay:
                        # Khi mat vach hoan toan, van giu mui ten do co dinh o phia tren vung HUD day man hinh
                        arrow_base_y = hT - 42
                        arrow_len = max(34, int(hT * 0.19))
                        pt_arrow_base = (int(round(car_ref_x)), arrow_base_y)
                        pt_arrow_tip = (int(round(car_ref_x)), max(8, arrow_base_y - arrow_len))
                        cv2.arrowedLine(img_result, pt_arrow_base, pt_arrow_tip, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.28)
                        cv2.circle(img_result, pt_arrow_base, 5, (255, 255, 255), cv2.FILLED, lineType=cv2.LINE_AA)
                        cv2.circle(img_result, pt_arrow_base, 3, (0, 0, 255), cv2.FILLED, lineType=cv2.LINE_AA)
                    if img_hist_combined is not None:
                        cv2.putText(img_hist_combined, f"Mid: {mid_point} | Avg: {avg_point}", (10, 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            else:
                plot_y, l_x, r_x, c_x, status = self.detect_highway_lanes(img_thres, hT, wT, top_y_hint=top_hint)
                if c_x is not None:
                    far_x = float(np.mean(c_x[:6]))
                    lookahead_x = float(np.mean(c_x[6:14]))
                    near_x = float(np.mean(c_x[-6:]))
                    if abs(far_x - near_x) > 15.0:
                        target_x = 0.45 * far_x + 0.35 * lookahead_x + 0.20 * near_x
                    else:
                        target_x = 0.20 * far_x + 0.40 * lookahead_x + 0.40 * near_x
                    half_lane_bot = max(40.0, float(r_x[-1] - l_x[-1]) * 0.5)
                    raw_curve = ((target_x - (wT / 2.0)) / half_lane_bot) * 65.0
                    e_px_out = float(c_x[-1] - car_ref_x)
                    e_norm_out = float(np.clip(e_px_out / half_lane_bot, -1.0, 1.0))
                    offset_dir = "CAN GIUA" if abs(e_px_out) <= 4.0 else ("LECH PHAI" if e_px_out < 0 else "LECH TRAI")

                    if draw_overlay:
                        pts_l = np.column_stack((l_x, plot_y)).astype(np.int32)
                        pts_r = np.column_stack((r_x, plot_y)).astype(np.int32)
                        pts_c = np.column_stack((c_x, plot_y)).astype(np.int32)

                        overlay = img_result.copy()
                        cv2.fillPoly(overlay, [np.vstack((pts_l, pts_r[::-1]))], (0, 200, 60))
                        img_result = cv2.addWeighted(img_result, 0.62, overlay, 0.38, 0)
                        cv2.polylines(img_result, [pts_l], False, (255, 105, 0), 4, cv2.LINE_AA)
                        cv2.polylines(img_result, [pts_r], False, (255, 105, 0), 4, cv2.LINE_AA)
                        cv2.polylines(img_result, [pts_c], False, (255, 255, 255), 3, cv2.LINE_AA)

                    if draw_aux_views:
                        warp_lines = utils.warpImg(img_result, points, wT, hT)
                        img_warp_vis = cv2.addWeighted(img_warp_vis, 0.35, warp_lines, 0.65, 0)

                    if img_hist_combined is not None:
                        cv2.putText(img_hist_combined, f"{status} | Tam:{int(target_x)}", (8, 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)
                elif img_hist_combined is not None:
                    cv2.putText(img_hist_combined, f"Mid: {mid_point} | Avg: {avg_point}", (10, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        elif img_hist_combined is not None:
            cv2.putText(img_hist_combined, f"Mid: {mid_point} | Avg: {avg_point}", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        observed = (bool(self.boundary_traces) if profile == "robot" else
                    status.startswith(("2 BIEN", "BAM BIEN"))) if use_dual_edge else False
        edge_count = len(self.boundary_traces) if profile == "robot" and use_dual_edge else None
        self.last_detection = {
            "observed": observed,
            "lane_state": (
                "TWO_EDGES" if edge_count == 2
                else "ONE_EDGE" if edge_count == 1
                else ("MEMORY_HOLD" if c_x is not None else "UNKNOWN")
            ) if profile == "robot" else (status if observed else ("PREDICTED" if c_x is not None else "UNKNOWN")),
            "edge_count": edge_count,
            "center_available": c_x is not None,
            "center_offset_px": e_px_out,
            "center_offset_norm": e_norm_out,
            "e_norm": e_norm_out,
            "steer_offset_norm": steer_norm_out,
            "front_wheel_angle_deg": front_wheel_deg_out,
            "servo_angle_deg": servo_deg_out,
            "heading_error_deg": heading_deg_out,
            "psi_rad": psi_rad_out,
            "psi_deg": heading_deg_out,
            "offset_direction": offset_dir,
            "vehicle_ref_x": car_ref_x,
            "car_ref_x": car_ref_x,
            "lowest_center_point": lowest_c_pt,
            "lookahead_center_point": lookahead_c_pt,
            "lookahead_dist_px": lookahead_dist_out,
            "ctrl_debug": ctrl_dbg,
            "path_curvature": path_curvature_out,
            "single_edge_side": self.single_edge_side if profile == "robot" else None,
            "lane_width_px": w_lane_ref,
            "w_lane_bot": w_lane_bot_out,
            "lane_width_calibrated": bool(self._lane_width_calibrated),
            "lane_width_model_ab": self.lane_width_ab,
            "tape_width_px": tape_w_ref,
            "left_segments": self.left_segments if profile == "robot" else [],
            "right_segments": self.right_segments if profile == "robot" else [],
            "centerline_predicted": bool(np.any(self.center_pred_flags)) if (profile == "robot" and len(self.center_pred_flags) > 0) else (edge_count == 1),
            "boundaries": self.observed_boundaries if profile == "robot" and use_dual_edge else [],
            "centerline": (self.center_pts.tolist() if (profile == "robot" and len(self.center_pts) > 0) else (np.column_stack((c_x, plot_y)).tolist() if c_x is not None else None)),
            "confidence": (min(1.0, min(np.ptp(np.asarray(t)[:,1]) for t in self.boundary_traces)/(hT*.25))
                           if observed and profile == "robot" else (0.8 if observed else 0.0)),
            "confidence_kind": "heuristic_support_score_not_probability",
            "profile": profile,
            "missing_frames": self.missing_frames,
            "corner": corner,
            "corner_memory": {
                "active": bool(self.corner_memory.get("active", False)),
                "direction": self.corner_memory.get("direction", "UNKNOWN"),
                "ttl": int(self.corner_memory.get("ttl", 0)),
                "age_frames": int(self.corner_memory.get("age_frames", 0)),
                "arm_width_px": float(self.corner_memory.get("arm_width_px", 0.0)),
            },
            "junction": dict(getattr(self, "_last_junction_info", None) or {
                "detected": False,
                "type": "NONE",
                "available_actions": ["STRAIGHT"],
                "branch_side": None,
                "straight_edge_side": None,
                "branch_elbow_y": None,
                "phase": "NONE",
                "confidence": 0.0,
            }),
            "lighting": lighting_info,
            "restrict_roi": bool(restrict_roi or profile == "highway"),
        }
        if profile == "robot" and c_x is None:
            self._smooth_turn_prog = None
            self._smooth_steer_norm = None
            if edge_count == 1:
                if img_warp is None:
                    img_warp = utils.warpImg(img_thres, points, wT, hT)
                    mid_point, _ = utils.getHistogram(img_warp, display=False, minPer=0.5, region=4)
                    avg_point, _ = utils.getHistogram(img_warp, display=False, minPer=0.7, region=1)
                    hist_curve = float(avg_point - mid_point)
                raw_curve = self._estimate_single_edge_curve(wT, hT, corner, hist_curve)
            else:
                raw_curve = 0.0
                self.curve_list.clear()

        if profile == "robot" and c_x is not None:
            norm_curve = float(np.clip(steer_norm_out, -1.0, 1.0))
        else:
            self.curve_list.append(raw_curve)
            if len(self.curve_list) > self.avg_window:
                self.curve_list.pop(0)
            smooth_curve = sum(self.curve_list) / len(self.curve_list)
            norm_curve = float(np.clip(smooth_curve / 90.0, -1.0, 1.0))

        if draw_overlay and (not use_dual_edge or c_x is None) and profile == "highway":
            img_inv_warp = utils.warpImg(img_warp, points, wT, hT, inv=True)
            img_inv_warp = cv2.cvtColor(img_inv_warp, cv2.COLOR_GRAY2BGR)
            img_inv_warp[0:hT // 3, 0:wT] = 0, 0, 0
            lane_color = np.zeros_like(img)
            lane_color[:] = (0, 220, 0)
            lane_highlight = cv2.bitwise_and(img_inv_warp, lane_color)
            img_result = cv2.addWeighted(img_result, 1, lane_highlight, 0.7, 0)

        if draw_overlay and profile == "highway" and observed and c_x is not None:
            mid_y = int(hT * 0.80)
            center_x = wT // 2
            curve_offset = int(norm_curve * 100)
            cv2.arrowedLine(img_result, (center_x, hT - 26), (center_x, int(hT * 0.65)),
                            (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.25)
            cv2.line(img_result, (center_x, mid_y - 14), (center_x, mid_y + 14), (255, 255, 0), 2)
            cv2.line(img_result, (center_x, mid_y), (center_x + curve_offset, mid_y), (0, 0, 255), 4)
            cv2.circle(img_result, (center_x + curve_offset, mid_y), 6, (0, 255, 255), cv2.FILLED)

        light_mode = lighting_info["mode"]
        roi_scope = "ROI-ONLY" if (restrict_roi or profile == "highway") else "WIDE-CORNER"
        if draw_overlay:
            if points is not None:
                pts_roi = np.asarray(points, dtype=np.int32)
                if pts_roi.shape == (4, 2):
                    poly_roi = np.array([pts_roi[0], pts_roi[1], pts_roi[3], pts_roi[2]], dtype=np.int32)
                    cv2.polylines(img_result, [poly_roi], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    for pr in pts_roi:
                        cv2.circle(img_result, (int(pr[0]), int(pr[1])), 3, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
            if profile == "robot":
                cv2.putText(
                    img_result,
                    f"e = {e_px_out:+.1f}px ({e_norm_out:+.2f}) [{offset_dir}]",
                    (8, hT - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 2, cv2.LINE_AA
                )
                tarp_tag = ""
                if getattr(self, "_left_is_tarp_virtual", False) and getattr(self, "_right_is_tarp_virtual", False):
                    tarp_tag = " [MEP BAT L+R]"
                elif getattr(self, "_left_is_tarp_virtual", False):
                    tarp_tag = " [MEP BAT L]"
                elif getattr(self, "_right_is_tarp_virtual", False):
                    tarp_tag = " [MEP BAT R]"
                label = (
                    f"Lane:{self.last_detection['lane_state']}{tarp_tag} | "
                    f"Segs L:{len(self.left_segments)} R:{len(self.right_segments)} | "
                    f"W_lane:{int(w_lane_ref)}px W_tape:{tape_w_ref:.1f}px"
                )
                cv2.putText(img_result, label, (8, hT - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200, 255, 200), 1, cv2.LINE_AA)
            else:
                label = f"Lane:{self.last_detection['lane_state']} | Light:{light_mode} | {roi_scope}"
                cv2.putText(img_result, label, (6, hT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 255, 255), 1)
        if img_hist_combined is not None:
            cv2.putText(img_hist_combined,
                        f"Corner:{corner['candidate']} conf:{corner['confidence']:.2f} | Light:{light_mode}",
                        (6, 62), cv2.FONT_HERSHEY_SIMPLEX, .39, (0, 255, 255), 1)
        else:
            blank = getattr(self, "_blank_hist_img", None)
            if blank is None or blank.shape[:2] != (hT, wT):
                blank = np.zeros((hT, wT, 3), dtype=np.uint8)
                self._blank_hist_img = blank
            img_hist_combined = blank
        return norm_curve, img_roi, img_thres, img_warp_vis, img_hist_combined, img_result


_default_detector = LaneDetector(avg_window=6)


def getLaneCurve(img, display=2, profile="robot", return_details=False, restrict_roi=False):
    """Ham goi nhanh tren Raspberry Pi 5 (RobotMain.py)."""
    hT, wT = img.shape[:2]
    points = utils.valTrackbars(wT, hT, preset=profile)
    preset_cfg = PRESET_ROBOT_TOY if profile == "robot" else PRESET_HIGHWAY
    norm_curve, img_roi, img_thres, img_warp, img_hist, img_result = _default_detector.process_frame(
        img, points, hsv_range=preset_cfg["hsv_range"],
        use_dual_edge=True, profile=profile, restrict_roi=restrict_roi,
        draw_aux_views=(display == 2)
    )

    if display == 2:
        img_stacked = utils.imgHelper(0.7, ([img, img_roi, img_warp], [img_hist, img_thres, img_result]))
        cv2.imshow("ImageStack", img_stacked)
    elif display == 1:
        cv2.imshow("Result", img_result)

    if return_details:
        return norm_curve, dict(_default_detector.last_detection), img_result
    return norm_curve


if __name__ == "__main__":
    print("LaneDetectionModule ready.")
