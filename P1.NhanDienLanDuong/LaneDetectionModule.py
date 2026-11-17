import cv2
import numpy as np
import utils


PRESET_ROBOT_TOY = {
    "w_top": 105, "h_top": 105, "w_bot": 20, "h_bot": 230,
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

    def __init__(self, avg_window=6):
        self.curve_list = []
        self.avg_window = avg_window
        self.reset()

    def reset(self):
        self.curve_list.clear()
        self.smooth_left_x = None
        self.smooth_right_x = None
        self.width_profile = None
        self.lane_width_ab = (1.15, 175.0)
        self.tape_width_ab = (0.045, 3.5)
        self.smooth_e_px = None
        self.smooth_e_norm = None
        self.left_segments = []
        self.right_segments = []
        self.single_edge_side = None
        self._last_left_bot_x = None
        self._last_right_bot_x = None
        self._active_turn_dir = "UNKNOWN"
        self._active_turn_hold = 0
        self._intent_turn_dir = "UNKNOWN"
        self._intent_turn_ttl = 0
        self._straight_clear_hits = 0
        self._blind_spot_wait_frames = 0
        self._smooth_center_pts = None
        self._smooth_left_segs = []
        self._smooth_right_segs = []
        self.center_pts = np.empty((0, 2), dtype=np.float32)
        self.center_pred_flags = np.empty((0,), dtype=bool)
        self._last_valid_center_pts = np.empty((0, 2), dtype=np.float32)
        self._last_valid_pred_flags = np.empty((0,), dtype=bool)
        self.missing_frames = 0
        self._corner_candidate = "UNKNOWN"
        self._corner_hits = 0
        self.boundary_traces = []
        self.observed_boundaries = []
        self._previous_boundaries = []
        self._next_boundary_id = 1
        self._corner_ids = ()
        self.last_detection = {
            "observed": False,
            "lane_state": "UNKNOWN",
            "center_offset_px": 0.0,
            "center_offset_norm": 0.0,
            "offset_direction": "CAN GIUA",
            "vehicle_ref_x": 240.0,
            "lowest_center_point": None,
            "lane_width_px": 0.0,
            "tape_width_px": 0.0,
            "left_segments": [],
            "right_segments": [],
            "centerline_predicted": False,
            "corner": {"direction": "UNKNOWN", "confidence": 0.0},
            "lighting": {"mode": "NORMAL", "med_l": 110.0, "p98_l": 190.0},
        }

    @staticmethod
    def _extract_row_runs_with_dist(mask_row, dist_row, y, hT):
        idx = np.flatnonzero(mask_row)
        if idx.size == 0:
            return []
        ratio = float(np.clip(y / float(max(1, hT)), 0.0, 1.0))
        max_slice_w = int(round(28 + 38 * ratio))
        max_perp_w = 14.0 + 18.0 * ratio
        min_w = 2
        runs = []
        for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            s, e = int(g[0]), int(g[-1]) + 1
            rw = e - s
            if min_w <= rw <= max_slice_w:
                cx = 0.5 * (s + e - 1)
                icx = int(round(cx))
                local_r = float(np.max(dist_row[max(s, icx - 2) : min(e, icx + 3)]))
                perp_w = max(2.0, min(float(rw), 2.0 * local_r))
                if perp_w <= max_perp_w:
                    runs.append((cx, perp_w, s, e, float(rw)))
        return runs

    @classmethod
    def _trace_stem_and_turn(cls, mask, dist, gray, road_med_gray, seed_x, seed_y, side, top_limit_y=4):
        hT, wT = mask.shape[:2]
        raw_obs = []  # (cx, y, perp_w, s, e, rw)

        curr_x = float(seed_x)
        dx_per_y = 0.0
        miss_rows = 0

        def _has_dash_support(cx_val, y_val):
            for y_chk in (y_val - 2, y_val - 4):
                if y_chk >= 4:
                    for it_chk in cls._extract_row_runs_with_dist(mask[y_chk], dist[y_chk], y_chk, hT):
                        if abs(it_chk[0] - cx_val) <= 10.0:
                            return True
            return False

        for y in range(int(seed_y), max(top_limit_y - 1, 3), -2):
            # Huy som neu diem seed chi la dom nhieu don le (< 2 diem trong 8px dau)
            if (seed_y - y) > 8 and len(raw_obs) < 2:
                break

            pred_x = curr_x + dx_per_y * 2.0
            if miss_rows == 0:
                gate = 16.0
            else:
                slope_unc = 0.70 if len(raw_obs) < 3 else 0.45
                gate = 14.0 + min(22.0, miss_rows * slope_unc)

            med_rw_live = float(np.median([it[5] for it in raw_obs[-6:]])) if raw_obs else None
            best = None
            best_d = gate
            for y_scan in (y, y - 1):
                if y_scan < 0:
                    continue
                for item in cls._extract_row_runs_with_dist(mask[y_scan], dist[y_scan], y_scan, hT):
                    d = abs(item[0] - pred_x)
                    if d < best_d:
                        if miss_rows >= 4 and med_rw_live is not None and item[4] > max(20.0, med_rw_live * 2.2):
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
                if gray is not None and road_med_gray is not None:
                    inward_dir = +1 if side == "LEFT" else -1
                    ix1 = int(np.clip((e + 6 if inward_dir > 0 else s - 28), 0, wT - 1))
                    ix2 = int(np.clip((e + 28 if inward_dir > 0 else s - 6), 0, wT - 1))
                    if ix2 > ix1 and y < int(hT * 0.45):
                        inward_med = float(np.median(gray[y, ix1:ix2]))
                        tape_med = float(np.median(gray[y, s:e]))
                        if tape_med - inward_med < 20.0 and inward_med > road_med_gray + 42.0:
                            break

                if len(raw_obs) >= 1:
                    prev_x, prev_y = raw_obs[-1][0], raw_obs[-1][1]
                    dy_step = max(1.0, float(prev_y - y))
                    max_slope = 2.5 if dy_step >= 10.0 else 3.5
                    inst_dx = float(np.clip((cx - prev_x) / dy_step, -max_slope, max_slope))
                    if len(raw_obs) <= 2 and dy_step >= 8.0:
                        dx_per_y = 0.30 * dx_per_y + 0.70 * inst_dx
                    else:
                        dx_per_y = 0.58 * dx_per_y + 0.42 * inst_dx
                curr_x = cx
                raw_obs.append((float(cx), float(y), float(np.clip(perp_w, 2.0, 28.0)), s, e, rw))
                miss_rows = 0
            else:
                curr_x = float(np.clip(pred_x, 0.0, wT - 1.0))
                miss_rows += 2
                # Cho phep nhay qua khe ho net dut thich ung theo phoi canh (gan ~48px, xa ~24px)
                if len(raw_obs) >= 2:
                    max_miss = int(round(18 + 30 * (y / float(max(1, hT)))))
                else:
                    max_miss = 6 if y > int(hT * 0.68) else 2
                if miss_rows > max_miss:
                    break

        if not raw_obs:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Correct border-clipped runs at the bottom (where s <= 1 or e >= wT - 1) using unclipped tape width
        unclipped_rws = [item[5] for item in raw_obs[:15] if item[3] > 2 and item[4] < wT - 2]
        ref_rw = float(np.median(unclipped_rws)) if unclipped_rws else None

        pts = []
        widths = []
        for cx, y, perp_w, s, e, rw in raw_obs:
            if ref_rw is not None and rw < ref_rw * 0.88:
                if s <= 1:
                    true_cx = float(e) - 0.5 * ref_rw
                    if true_cx < -0.5:
                        continue
                    cx = max(0.0, true_cx)
                    perp_w = max(perp_w, float(np.median([it[2] for it in raw_obs[:12] if it[3] > 2] or [perp_w])))
                elif e >= wT - 1:
                    true_cx = float(s) + 0.5 * ref_rw
                    if true_cx > wT - 0.5:
                        continue
                    cx = min(float(wT - 1), true_cx)
                    perp_w = max(perp_w, float(np.median([it[2] for it in raw_obs[:12] if it[4] < wT - 2] or [perp_w])))
            pts.append((float(cx), float(y)))
            widths.append(float(perp_w))

        if len(pts) < 4:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Try horizontal/diagonal turn-arm continuation from either pts[-1] or the elbow near the top
        cand_indices = [len(pts) - 1]
        if len(pts) >= 8:
            top_window = pts[max(0, len(pts) - 7) :]
            if side == "RIGHT":
                min_i = int(np.argmin([p[0] for p in top_window]))
                eb_i = max(0, len(pts) - 7) + min_i
            else:
                max_i = int(np.argmax([p[0] for p in top_window]))
                eb_i = max(0, len(pts) - 7) + max_i
            if eb_i not in cand_indices:
                cand_indices.append(eb_i)

        best_full_pts = pts
        best_full_widths = widths
        best_reach_x = 0.0

        for start_idx in cand_indices:
            top_x, top_y = pts[start_idx]
            for dir_x in (+1, -1):
                arm_pts = []
                arm_w = []
                curr_y = float(max(2.0, top_y - 0.5))
                dy_per_x = -0.12
                miss_cols = 0
                start_col = max(2, min(wT - 3, int(round(top_x)) + dir_x * 2))
                end_col = wT - 2 if dir_x > 0 else 1
                for x in range(start_col, end_col, dir_x * 2):
                    pred_y = curr_y + dy_per_x * 2.0
                    if pred_y > top_y + 5.0 or pred_y < 1.0:
                        break
                    half_band = 5 if len(arm_pts) < 3 else 7
                    y_lo = max(1, int(round(pred_y - half_band)))
                    y_hi = min(int(round(top_y + 5)), int(round(pred_y + half_band)))
                    if y_hi <= y_lo:
                        break
                    col_slice = np.flatnonzero(mask[y_lo:y_hi, x])
                    best_cy = None
                    best_cw = None
                    best_dy = float(half_band) + 0.5
                    if col_slice.size > 0:
                        max_wy = int(round(12 + 15 * (pred_y / float(hT))))
                        for g in np.split(col_slice, np.flatnonzero(np.diff(col_slice) > 1) + 1):
                            sy = y_lo + int(g[0])
                            ey = y_lo + int(g[-1]) + 1
                            cw = ey - sy
                            cy = 0.5 * (sy + ey - 1)
                            if 1 <= cw <= max_wy and abs(cy - pred_y) < best_dy and cy <= top_y + 4.0:
                                if gray is not None and road_med_gray is not None:
                                    by1 = min(hT - 1, ey + 4)
                                    by2 = min(hT, ey + 20)
                                    if by2 > by1 and cy > 25:
                                        below_med = float(np.median(gray[by1:by2, x]))
                                        if below_med > road_med_gray + 52.0:
                                            continue
                                best_dy = abs(cy - pred_y)
                                ic_y = int(round(cy))
                                perp_cw = max(2.0, min(float(cw), 2.0 * float(dist[ic_y, x])))
                                best_cy = float(cy)
                                best_cw = perp_cw
                    if best_cy is not None:
                        if arm_pts:
                            prev_ax, prev_ay = arm_pts[-1]
                            dx_step = max(1.0, float(abs(x - prev_ax)))
                            inst_dy = float(np.clip((best_cy - prev_ay) / dx_step, -0.28, 0.12))
                            dy_per_x = 0.70 * dy_per_x + 0.30 * inst_dy
                        curr_y = best_cy
                        arm_pts.append((float(x), float(best_cy)))
                        arm_w.append(float(np.clip(best_cw, 2.0, 26.0)))
                        miss_cols = 0
                    else:
                        if len(arm_pts) < 2:
                            break
                        curr_y = float(np.clip(pred_y, 1.0, hT - 2.0))
                        miss_cols += 2
                        max_miss_c = 24 if len(arm_pts) >= 3 else 8
                        if miss_cols > max_miss_c:
                            break
                if len(arm_pts) >= 5:
                    span_x = abs(arm_pts[-1][0] - pts[cand_indices[-1]][0])
                    if span_x >= 18.0 and span_x > best_reach_x:
                        best_reach_x = span_x
                        best_full_pts = pts[: start_idx + 1] + arm_pts
                        best_full_widths = widths[: start_idx + 1] + arm_w

        return np.asarray(best_full_pts, dtype=np.float32), np.asarray(best_full_widths, dtype=np.float32)

    @classmethod
    def _find_and_trace_boundaries(cls, mask, top_y_hint=105, gray=None):
        hT, wT = mask.shape[:2]
        road_med_gray = float(np.median(gray[int(hT * 0.45) : int(hT * 0.95), :])) if gray is not None else None
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        mid_x = wT * 0.5
        scan_top_y = max(int(hT * 0.30), min(int(top_y_hint), int(hT * 0.58)))
        bot_y = int(hT * 0.97)

        def _is_valid_trace(pts_try):
            if len(pts_try) < 5:
                return False
            span_y = float(pts_try[0, 1] - np.min(pts_try[:, 1]))
            path_len = float(np.linalg.norm(np.diff(pts_try, axis=0), axis=1).sum())
            return span_y >= 16.0 and path_len >= 24.0

        left_seed = None
        right_seed = None

        for y in range(bot_y, scan_top_y, -2):
            runs = cls._extract_row_runs_with_dist(mask[y], dist[y], y, hT)
            if not runs:
                continue
            if left_seed is None:
                l_runs = [r for r in runs if r[0] <= mid_x - 16]
                for cx, perp_w, s, e, rw in sorted(l_runs, key=lambda r: -r[0]):
                    pts_try, w_try = cls._trace_stem_and_turn(mask, dist, gray, road_med_gray, cx, y, "LEFT", top_limit_y=4)
                    if _is_valid_trace(pts_try):
                        left_seed = (pts_try, w_try)
                        break
            if right_seed is None:
                r_runs = [r for r in runs if r[0] >= mid_x + 16]
                for cx, perp_w, s, e, rw in sorted(r_runs, key=lambda r: r[0]):
                    pts_try, w_try = cls._trace_stem_and_turn(mask, dist, gray, road_med_gray, cx, y, "RIGHT", top_limit_y=4)
                    if _is_valid_trace(pts_try):
                        right_seed = (pts_try, w_try)
                        break
            if left_seed is not None and right_seed is not None:
                break

        # Fallback: Neu xe bi lech sat vao vach ke duong (vach nam ngay giua anh mid_x +- 16)
        if left_seed is None and right_seed is None:
            for y in range(bot_y, scan_top_y, -2):
                runs = cls._extract_row_runs_with_dist(mask[y], dist[y], y, hT)
                c_runs = [r for r in runs if (mid_x - 20) < r[0] < (mid_x + 20)]
                for cx, perp_w, s, e, rw in c_runs:
                    side_guess = "LEFT" if cx <= mid_x else "RIGHT"
                    pts_try, w_try = cls._trace_stem_and_turn(mask, dist, gray, road_med_gray, cx, y, side_guess, top_limit_y=4)
                    if _is_valid_trace(pts_try):
                        if side_guess == "LEFT":
                            left_seed = (pts_try, w_try)
                        else:
                            right_seed = (pts_try, w_try)
                        break
                if left_seed is not None or right_seed is not None:
                    break

        # Deduplicate if a single diagonal boundary crossing mid_x was picked by both sides
        if left_seed is not None and right_seed is not None:
            lp, rp = left_seed[0], right_seed[0]
            dists = np.linalg.norm(lp[:, None, :] - rp[None, :, :], axis=2)
            overlap_l = float(np.mean(np.min(dists, axis=1) < 14.0))
            overlap_r = float(np.mean(np.min(dists, axis=0) < 14.0))
            if max(overlap_l, overlap_r) >= 0.25:
                if len(lp) >= len(rp):
                    right_seed = None
                else:
                    left_seed = None

        return left_seed, right_seed

    @staticmethod
    def _fit_line_tls(pts):
        """Fit a straight line to (N, 2) points via Total Least Squares and project endpoints onto it."""
        if len(pts) < 2:
            return pts[0].copy(), pts[-1].copy(), 0.0, np.zeros(len(pts), dtype=np.float32)
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.dot(centered.T, centered) / max(1, len(pts))
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        if np.dot(pts[-1] - pts[0], direction) < 0:
            direction = -direction
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        dists = np.abs(np.dot(centered, normal))
        t0 = float(np.dot(pts[0] - mean, direction))
        t1 = float(np.dot(pts[-1] - mean, direction))
        p_start = mean + t0 * direction
        p_end = mean + t1 * direction
        return p_start.astype(np.float32), p_end.astype(np.float32), float(np.max(dists)), dists

    @classmethod
    def _fit_piecewise_segments(cls, pts, widths, min_seg_len=34.0, straight_tol=5.8, min_turn_deg=16.0, max_segments=3):
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
        if n >= 5 and np.ptp(ys) >= 12.0:
            A = np.column_stack((ys, np.ones_like(ys)))
            coef, _, _, _ = np.linalg.lstsq(A, ws, rcond=None)
            slope_w = max(0.0, float(coef[0]))
            intercept_w = float(np.median(ws - slope_w * ys))
            w_persp = np.clip(slope_w * ys + intercept_w, 2.0, 28.0)
            ws_smooth = 0.65 * w_persp + 0.35 * ws
        else:
            ws_smooth = np.clip(ws, 2.0, 28.0)

        mean_w = float(np.median(ws_smooth))
        tol = max(straight_tol, 0.48 * mean_w)

        p0_fit, p1_fit, max_d, dists = cls._fit_line_tls(pts)
        total_len = float(np.linalg.norm(p1_fit - p0_fit))
        p92_d = float(np.percentile(dists, 92))

        mid_idx = n // 2
        v1 = pts[mid_idx] - pts[0]
        v2 = pts[-1] - pts[mid_idx]
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        half_turn_deg = 0.0
        if n1 > 6.0 and n2 > 6.0:
            cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            half_turn_deg = float(np.rad2deg(np.arccos(cos_a)))

        if total_len < 2.0 * min_seg_len or (p92_d <= tol and max_d <= tol * 1.35 and half_turn_deg < min_turn_deg):
            return [{
                "p1": (float(p0_fit[0]), float(p0_fit[1])),
                "p2": (float(p1_fit[0]), float(p1_fit[1])),
                "width_start": float(ws_smooth[0]),
                "width_end": float(ws_smooth[-1]),
                "width": float(np.median(ws_smooth)),
                "length": total_len,
            }]

        knots = [0, n - 1]
        while len(knots) - 1 < max_segments:
            best_split = None
            best_score = -1.0

            for s_idx in range(len(knots) - 1):
                ia, ib = knots[s_idx], knots[s_idx + 1]
                if ib - ia < 6:
                    continue
                sub_pts = pts[ia : ib + 1]
                _, _, sub_max_d, sub_dists = cls._fit_line_tls(sub_pts)
                if float(np.percentile(sub_dists, 92)) <= tol and sub_max_d <= tol * 1.35:
                    continue
                chord_vec = pts[ib] - pts[ia]
                chord_len = float(np.linalg.norm(chord_vec))
                if chord_len < 2.0 * min_seg_len:
                    continue
                unit_chord = chord_vec / max(1e-5, chord_len)
                normal_chord = np.array([-unit_chord[1], unit_chord[0]], dtype=np.float32)

                for k in range(ia + 3, ib - 2):
                    v_left = pts[k] - pts[ia]
                    v_right = pts[ib] - pts[k]
                    l_left = float(np.linalg.norm(v_left))
                    l_right = float(np.linalg.norm(v_right))
                    if l_left < min_seg_len or l_right < min_seg_len:
                        continue
                    d_perp = abs(float(np.dot(pts[k] - pts[ia], normal_chord)))
                    if d_perp <= tol:
                        continue
                    cos_turn = float(np.clip(np.dot(v_left, v_right) / (l_left * l_right), -1.0, 1.0))
                    turn_deg = float(np.rad2deg(np.arccos(cos_turn)))
                    if turn_deg < max(9.0, min_turn_deg - 3.0):
                        continue
                    score = d_perp * (1.0 + 0.02 * turn_deg)
                    if score > best_score:
                        best_score = score
                        best_split = k

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
                comb_p92 = float(np.percentile(comb_dists, 92))
                v1 = pm - pa
                v2 = pb - pm2
                l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if l1 < 18.0 or l2 < 18.0:
                    knots.pop(j)
                    merged_changed = True
                    break
                cos_t = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
                ang_deg = float(np.rad2deg(np.arccos(cos_t)))
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
                if float(np.linalg.norm(inter - anchor)) <= 10.0:
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
            seg_len = float(np.linalg.norm(v_end - v_start))
            w_s = float(ws_smooth[ia])
            w_e = float(ws_smooth[ib])
            w_m = float(np.median(ws_smooth[ia : ib + 1]))
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
            ymin, ymax = min(y1, y2), max(y1, y2)
            if ymin - 1.5 <= y_val <= ymax + 1.5:
                t = float(np.clip((y_val - y1) / dy, 0.0, 1.0))
                return float(x1 + t * (x2 - x1))
            d = min(abs(y_val - ymin), abs(y_val - ymax))
            if d < best_dist:
                best_dist = d
                t = float(np.clip((y_val - y1) / dy, 0.0, 1.0))
                best_x = float(x1 + t * (x2 - x1))
        return best_x

    @staticmethod
    def _sample_polyline(segments, num_pts=25):
        if not segments:
            return np.empty((0, 2), dtype=np.float32)
        lengths = [max(1e-3, float(s["length"])) for s in segments]
        cum = np.r_[0.0, np.cumsum(lengths)]
        total = cum[-1]
        targets = np.linspace(0.0, total, num_pts)
        out = []
        for s_val in targets:
            idx = min(len(segments) - 1, max(0, int(np.searchsorted(cum, s_val, side="right") - 1)))
            seg = segments[idx]
            t = float(np.clip((s_val - cum[idx]) / lengths[idx], 0.0, 1.0))
            p1 = np.array(seg["p1"], dtype=np.float32)
            p2 = np.array(seg["p2"], dtype=np.float32)
            out.append(p1 + t * (p2 - p1))
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _find_corner_elbow_seg_idx(segments):
        """
        Return the index (in 1..len(segments)-1) of the first segment that turns into a corner,
        or None if the boundary is a single straight/converging stem.
        """
        if len(segments) < 2:
            return None
        for idx in range(len(segments) - 1):
            s1, s2 = segments[idx], segments[idx + 1]
            v1 = np.array(s1["p2"], dtype=np.float32) - np.array(s1["p1"], dtype=np.float32)
            v2 = np.array(s2["p2"], dtype=np.float32) - np.array(s2["p1"], dtype=np.float32)
            l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if l1 < 10.0 or l2 < 10.0:
                continue
            cos_a = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
            ang_deg = float(np.rad2deg(np.arccos(cos_a)))
            if ang_deg >= 14.0 and abs(v2[0]) >= abs(v2[1]) * 0.65:
                return idx + 1
        return None

    @classmethod
    def _offset_segments_by_normal(cls, active_segs, side_sign, a_lane, b_lane, hT, wT):
        """
        Dinh huong tam duong tuong doi dua vao vi tri diem bat dau so voi mep canh & mep duoi:
          1. Khi xe di lech lam mat 1 vach tren duong thang (chan vach nam o mep duoi va dich vao giua):
             - So sanh vi tri chan vach x0 voi vi tri chuan sat mep (X_L0 ~ 26px hoac X_R0 ~ wT - 26px).
             - Tinh tam duong lech ty le muot theo do troi cua chan vach (khong cong cung +0.5*W_lane lam vach nhay ra mep anh),
               giup xe danh lai nhe nhang quay ve giua lan cu!
          2. Khi vao goc cua (diem bat dau nam cao hon tren mep canh va co doan vat cheo/vuong goc):
             - Dich chuyen duong tam xuong duoi theo truc Y mot khoang tau * 0.46 * 0.5 * W_lane(y)
               de duong tam nam giua long duong cua nhanh re.
        """
        if not active_segs:
            return []

        def w_lane_at_y(y_q):
            return float(np.clip(a_lane * float(y_q) + b_lane, wT * 0.22, wT * 0.98))

        persp_y_scale = 0.46
        b_verts = [np.array(active_segs[0]["p1"], dtype=np.float32)]
        for seg in active_segs:
            b_verts.append(np.array(seg["p2"], dtype=np.float32))

        m_segs = len(b_verts) - 1
        taus = []
        for k in range(m_segs):
            dx = float(b_verts[k + 1][0] - b_verts[k][0])
            dy_up = max(0.0, float(b_verts[k][1] - b_verts[k + 1][1]))
            dx_turn = max(0.0, side_sign * dx - 0.52 * a_lane * dy_up)
            tau = float(dx_turn / np.hypot(dx_turn, 1.65 * dy_up + 1e-3))
            taus.append(tau)

        max_tau = max(taus) if taus else 0.0
        x0, y0 = float(b_verts[0][0]), float(b_verts[0][1])
        mid_x = wT * 0.5
        edge_margin = wT * 0.055  # ~26px tren khung hinh 480px
        norm_base_x = edge_margin if side_sign > 0 else (wT - edge_margin)

        # Kiem tra xem day la vach thang khi xe bi lech (chan vach nam thap o mep duoi, khong co nhanh vat ngang)
        # hay la vach goc cua (diem mep nam cao hon hoac co doan vat cheo gan nhu vuong goc)
        is_corner_shape = (m_segs >= 2 and max_tau >= 0.42) or (y0 < hT * 0.84 and max_tau >= 0.35)

        if not is_corner_shape:
            # CHE DO HOI TAM EM KHI XE DI LECH MAT 1 VACH:
            # Dua vao vi tri tuong doi cua diem bat dau x0 so voi mep canh chuan norm_base_x:
            # - Neu di deu (x0 ~ norm_base_x): tam duong o khoang giua (mid_x).
            # - Neu xe lech trai (vach trai troi vao giua, x0 > norm_base_x): tam lech phai vua phai -> banh re phai ve duong cu.
            # - Neu xe lech phai (vach phai troi vao giua, x0 < norm_base_x): tam lech trai vua phai -> banh re trai ve duong cu.
            drift_from_edge = x0 - norm_base_x
            rel_offset = float(np.clip(0.45 * drift_from_edge, -wT * 0.22, wT * 0.22))
            cx_bot = float(np.clip(mid_x + rel_offset, wT * 0.26, wT * 0.74))
            dx_base = cx_bot - x0

            c_verts = []
            for k, vk in enumerate(b_verts):
                xk, yk = float(vk[0]), float(vk[1])
                # Giam dan khoang cach ngang theo phoi canh khi di len dinh anh de duong tam huong ve giua lan
                y_ratio = float(np.clip((yk - hT * 0.15) / max(1.0, y0 - hT * 0.15), 0.35, 1.0))
                cx_k = float(np.clip(xk + dx_base * y_ratio, wT * 0.18, wT * 0.82))
                c_verts.append(np.array([cx_k, yk], dtype=np.float32))
        else:
            # CHE DO GOC CUA VUONG 90 DO (co diem gap khuc hoac vach vat ngang):
            c_verts = []
            for k, vk in enumerate(b_verts):
                xk, yk = float(vk[0]), float(vk[1])
                rx = 0.5 * w_lane_at_y(yk)
                ry = persp_y_scale * rx

                if k == 0:
                    drift_from_edge = xk - norm_base_x
                    rel_offset = float(np.clip(0.45 * drift_from_edge, -wT * 0.22, wT * 0.22))
                    cx = float(np.clip(mid_x + rel_offset, wT * 0.25, wT * 0.75))
                    tau_v = taus[0] if (taus and yk < hT * 0.82) else 0.0
                    cy = min(float(hT - 6.0), yk + 0.35 * tau_v * ry)
                else:
                    tau_prev = taus[k - 1]
                    tau_next = taus[k] if k < m_segs else taus[k - 1]
                    tau_v = max(tau_prev, tau_next)
                    cx = xk + side_sign * (1.0 - 0.52 * tau_v) * rx
                    cy = min(float(hT - 6.0), yk + tau_v * ry)

                c_verts.append(np.array([cx, cy], dtype=np.float32))

        # Neo diem dau tien xuong sat mep duoi man hinh (y = hT - 6) de mui ten do luon nam o mep duoi
        bot_anchor_y = float(hT - 6.0)
        if len(c_verts) >= 1 and c_verts[0][1] < bot_anchor_y - 2.0:
            bot_x = float(np.clip(c_verts[0][0], wT * 0.22, wT * 0.78))
            c_verts.insert(0, np.array([bot_x, bot_anchor_y], dtype=np.float32))
        elif len(c_verts) >= 1:
            c_verts[0][1] = bot_anchor_y

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

    @classmethod
    def _compute_lane_and_centerline(cls, left_segs, right_segs, hT, wT, lane_width_ab=(1.15, 175.0)):
        """
        Measure/calibrate perspective lane width W_lane(y) = a * y + b (strictly distinct from tape width!)
        and compute the white centerline for straight lanes, curves, 90-degree corners, and single-edge cases.
        """
        a_lane, b_lane = lane_width_ab

        def w_lane_at_y(y_q):
            return float(np.clip(a_lane * float(y_q) + b_lane, wT * 0.22, wT * 0.98))

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
                l_stem = [left_segs[0]]
            if not r_stem:
                r_stem = [right_segs[0]]

            y_l_bot = max(s["p1"][1] for s in l_stem)
            y_l_top = min(s["p2"][1] for s in l_stem)
            y_r_bot = max(s["p1"][1] for s in r_stem)
            y_r_top = min(s["p2"][1] for s in r_stem)

            y_overlap_bot = min(y_l_bot, y_r_bot)
            y_overlap_top = max(y_l_top, y_r_top)

            if y_overlap_bot - y_overlap_top >= 12.0:
                ys_sample = np.linspace(y_overlap_top, y_overlap_bot, 16)
                widths_meas = []
                ys_valid = []
                for yq in ys_sample:
                    xl = cls._eval_segments_x_at_y(l_stem, yq)
                    xr = cls._eval_segments_x_at_y(r_stem, yq)
                    if xl is not None and xr is not None and (xr - xl) >= wT * 0.18:
                        widths_meas.append(xr - xl)
                        ys_valid.append(yq)
                if len(ys_valid) >= 4:
                    yv = np.asarray(ys_valid, dtype=np.float32)
                    wv = np.asarray(widths_meas, dtype=np.float32)
                    if np.ptp(yv) >= 18.0:
                        A = np.column_stack((yv, np.ones_like(yv)))
                        coef, _, _, _ = np.linalg.lstsq(A, wv, rcond=None)
                        a_fit = float(np.clip(coef[0], 0.35, 2.20))
                        b_fit = float(np.median(wv - a_fit * yv))
                        a_lane = 0.70 * a_lane + 0.30 * a_fit
                        b_lane = 0.70 * b_lane + 0.30 * b_fit

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

            if y_max_bot - y_overlap_bot >= 6.0:
                n_tail = max(2, int(round((y_max_bot - y_overlap_bot) / 6.0)))
                for yq in np.linspace(y_max_bot, y_overlap_bot, n_tail, endpoint=False):
                    wl = max(wT * 0.22, w_ov_actual + a_lane * float(yq - y_overlap_bot))
                    if y_r_bot > y_l_bot:
                        xr = cls._eval_segments_x_at_y(r_stem, yq)
                        if xr is not None:
                            center_pts.append((float(xr - 0.5 * wl), float(yq)))
                            pred_flags.append(True)
                    else:
                        xl = cls._eval_segments_x_at_y(l_stem, yq)
                        if xl is not None:
                            center_pts.append((float(xl + 0.5 * wl), float(yq)))
                            pred_flags.append(True)

            if y_overlap_bot >= y_overlap_top:
                n_stem = max(6, int(round((y_overlap_bot - y_overlap_top) / 5.0)) + 1)
                for yq in np.linspace(y_overlap_bot, y_overlap_top, n_stem):
                    xl = cls._eval_segments_x_at_y(l_stem, yq)
                    xr = cls._eval_segments_x_at_y(r_stem, yq)
                    if xl is not None and xr is not None:
                        center_pts.append((0.5 * (xl + xr), float(yq)))
                        pred_flags.append(False)

            # Neo diem thap nhat cua duong tam xuong mep duoi (hT - 6) theo do doc doan duoi (chinh xac khi net dut o day anh)
            bot_anchor_y = float(hT - 6.0)
            if center_pts and center_pts[0][1] < bot_anchor_y - 2.0:
                if len(center_pts) >= 2:
                    p_low = center_pts[0]
                    ref_k = min(len(center_pts) - 1, 3)
                    p_up = center_pts[ref_k]
                    dy_ext = float(p_low[1] - p_up[1])
                    if dy_ext >= 4.0:
                        slope_xy = float(np.clip((float(p_low[0]) - float(p_up[0])) / dy_ext, -1.8, 1.8))
                        x_ext = float(np.clip(float(p_low[0]) + slope_xy * (bot_anchor_y - float(p_low[1])), 4.0, wT - 5.0))
                    else:
                        x_ext = center_pts[0][0]
                else:
                    x_ext = center_pts[0][0]
                center_pts.insert(0, (x_ext, bot_anchor_y))
                pred_flags.insert(0, True)

            # Xu ly doan cua vuong 90 do phia tren than lan (KHONG bao gio nhay nguoc xuong day anh!):
            if l_elbow_idx is not None and r_elbow_idx is not None:
                l_arm = left_segs[l_elbow_idx:]
                r_arm = right_segs[r_elbow_idx:]
                l_arm_poly = cls._sample_polyline(l_arm, 10)
                r_arm_poly = cls._sample_polyline(r_arm, 10)
                for pl, pr in zip(l_arm_poly, r_arm_poly):
                    cx_m = float(np.clip(0.5 * (pl[0] + pr[0]), 2.0, wT - 3.0))
                    cy_m = float(np.clip(0.5 * (pl[1] + pr[1]), 2.0, hT - 2.0))
                    if not center_pts or np.hypot(cx_m - center_pts[-1][0], cy_m - center_pts[-1][1]) >= 4.0:
                        center_pts.append((cx_m, cy_m))
                        pred_flags.append(False)
            elif (l_elbow_idx is not None or r_elbow_idx is not None) and center_pts:
                # Chi lay nhanh re PHIA TREN (tu elbow tro di) va noi tiep muot tu dinh cua center_pts[-1]
                if l_elbow_idx is not None:
                    arm_segs = left_segs[l_elbow_idx:]
                else:
                    arm_segs = right_segs[r_elbow_idx:]
                arm_poly = cls._sample_polyline(arm_segs, 10)
                if len(arm_poly) >= 2:
                    anchor_top = np.array(center_pts[-1], dtype=np.float32)
                    arm_start = arm_poly[0]
                    for pt_a in arm_poly[1:]:
                        disp = pt_a - arm_start
                        cx_a = float(anchor_top[0] + disp[0])
                        cy_a = float(np.clip(anchor_top[1] + 0.55 * disp[1], 2.0, hT - 2.0))
                        if 2.0 <= cx_a <= wT - 3.0:
                            center_pts.append((cx_a, cy_a))
                            pred_flags.append(True)
                        else:
                            break
            else:
                l_poly = cls._sample_polyline(left_segs, 32)
                r_poly = cls._sample_polyline(right_segs, 32)
                l_upper = l_poly[l_poly[:, 1] <= y_overlap_top + 2.0]
                r_upper = r_poly[r_poly[:, 1] <= y_overlap_top + 2.0]
                if len(l_upper) >= 2 and len(r_upper) >= 2:
                    idx_l = np.linspace(0, len(l_upper) - 1, 10)
                    idx_r = np.linspace(0, len(r_upper) - 1, 10)
                    for il, ir in zip(idx_l[1:], idx_r[1:]):
                        pl = l_upper[int(round(il))]
                        pr = r_upper[int(round(ir))]
                        center_pts.append((0.5 * float(pl[0] + pr[0]), 0.5 * float(pl[1] + pr[1])))
                        pred_flags.append(False)

            return np.asarray(center_pts, dtype=np.float32), np.asarray(pred_flags, dtype=bool), (a_lane, b_lane)

        if not has_l and not has_r:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool), (a_lane, b_lane)

        # Single-Edge Mode: Chi con 1 bien -> Dung khoang cach tuong doi 0.5 * W_lane muot ma
        active_segs = left_segs if has_l else right_segs
        side_sign = +1.0 if has_l else -1.0
        center_pts = cls._offset_segments_by_normal(active_segs, side_sign, a_lane, b_lane, hT, wT)
        pred_flags = [True] * len(center_pts)

        return np.asarray(center_pts, dtype=np.float32), np.asarray(pred_flags, dtype=bool), (a_lane, b_lane)

    @staticmethod
    def _draw_blue_tapered_segments(img, segments, color=(255, 105, 0), border_color=(255, 190, 60)):
        """
        Draw each piecewise straight segment in blue along the center of the real white tape,
        with stroke thickness tapering linearly from width_start to width_end.
        """
        for seg in segments:
            p1 = np.array(seg["p1"], dtype=np.float32)
            p2 = np.array(seg["p2"], dtype=np.float32)
            vec = p2 - p1
            length = float(np.linalg.norm(vec))
            if length < 1.0:
                continue
            unit = vec / length
            normal = np.array([-unit[1], unit[0]], dtype=np.float32)
            w1 = max(2.0, float(seg.get("width_start", seg["width"])))
            w2 = max(2.0, float(seg.get("width_end", seg["width"])))

            q1 = p1 - 0.5 * w1 * normal
            q2 = p1 + 0.5 * w1 * normal
            q3 = p2 + 0.5 * w2 * normal
            q4 = p2 - 0.5 * w2 * normal
            quad = np.round(np.array([q1, q2, q3, q4])).astype(np.int32)
            cv2.fillConvexPoly(img, quad, color, lineType=cv2.LINE_AA)
            cv2.polylines(img, [quad], isClosed=True, color=border_color, thickness=1, lineType=cv2.LINE_AA)

        for idx, seg in enumerate(segments):
            pt1 = (int(round(seg["p1"][0])), int(round(seg["p1"][1])))
            cv2.circle(img, pt1, 3, (255, 240, 150), cv2.FILLED, lineType=cv2.LINE_AA)
            if idx == len(segments) - 1:
                pt2 = (int(round(seg["p2"][0])), int(round(seg["p2"][1])))
                cv2.circle(img, pt2, 3, (255, 240, 150), cv2.FILLED, lineType=cv2.LINE_AA)

    def _associate_boundaries(self, candidates, h, w):
        previous = getattr(self, "_previous_boundaries", [])
        used = set()
        boundaries = []
        for candidate in candidates:
            pts = np.asarray(candidate["points"], float)
            sample = pts[:: max(1, len(pts) // 30)] / np.array([w, h])
            matches = []
            for old in previous:
                if old["id"] in used:
                    continue
                op = np.asarray(old["points"], float)[:: max(1, len(old["points"]) // 30)] / np.array([w, h])
                cost = float(np.linalg.norm(sample[:, None, :] - op[None, :, :], axis=2).min(axis=1).mean())
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
        dua tren 4 yeu to:
          1. Vi tri du kien tu mo hinh chieu rong lan W_lane(y) + do lech e gan nhat.
          2. Bo nho bam vet (Temporal tracking) cua vach Trai/Phai o frame truoc.
          3. Vi tri chan vach (bot_x) so voi truc tham chieu xe (wT * 0.5).
          4. Do nghieng phoi canh (Perspective slope dx/dy) cua doan than vach.
        """
        if len(pts) == 0 or not segs:
            return "LEFT"

        bot_x = float(pts[0, 0])
        bot_y = float(pts[0, 1])
        mid_x = wT * 0.5

        # Score < 0 -> LEFT, Score > 0 -> RIGHT
        score = 0.0

        # 1. So sanh voi vi tri du kien cua vach Trai & Phai dua tren chieu rong lan da hieu chuan
        w_lane_bot = float(np.clip(self.lane_width_ab[0] * bot_y + self.lane_width_ab[1], wT * 0.25, wT * 0.98))
        recent_e = float(self.smooth_e_px) if (self.smooth_e_px is not None and self.missing_frames <= 6) else 0.0
        exp_center_x = mid_x + float(np.clip(recent_e, -wT * 0.35, wT * 0.35))
        exp_left_x = exp_center_x - 0.5 * w_lane_bot
        exp_right_x = exp_center_x + 0.5 * w_lane_bot
        d_left = abs(bot_x - exp_left_x)
        d_right = abs(bot_x - exp_right_x)
        score += float(np.clip((d_left - d_right) / max(40.0, 0.35 * w_lane_bot), -2.5, 2.5))

        # 2. Bo nho bam vet (Temporal continuity) tu cac frame lien ke truoc do
        if self.missing_frames <= 5:
            dl_prev = abs(bot_x - self._last_left_bot_x) if self._last_left_bot_x is not None else 999.0
            dr_prev = abs(bot_x - self._last_right_bot_x) if self._last_right_bot_x is not None else 999.0
            if dl_prev < 52.0 and dl_prev + 24.0 < dr_prev:
                score -= 2.2
            elif dr_prev < 52.0 and dr_prev + 24.0 < dl_prev:
                score += 2.2

        # 3. Vi tri truc tiep cua chan vach so voi tam xe
        score += float(np.clip((bot_x - mid_x) / (wT * 0.18), -2.0, 2.0))

        # 4. Do nghieng phoi canh (Perspective convergence slope) cua doan than duoi
        s0 = segs[0]
        dy_up = float(s0["p1"][1] - s0["p2"][1])
        dx_up = float(s0["p2"][0] - s0["p1"][0])
        if dy_up >= 16.0:
            slope_up = dx_up / dy_up
            # Vach trai nghieng sang phai khi di len (slope_up > 0), vach phai nghieng sang trai (slope_up < 0)
            if slope_up > 0.15:
                score -= 1.2
            elif slope_up < -0.15:
                score += 1.2

        return "RIGHT" if score > 0.0 else "LEFT"

    def inspect_corner(self, mask, fresh_observation=True):
        """
        Evaluate corner direction, elbow coordinates, and distance-to-corner gating:
          - Distinguishes FAR / APPROACH corners (where car must keep driving straight on the stem)
            from TURN_NOW corners (where the corner apex has reached the turn-in zone in front of the car).
        """
        h, w = mask.shape
        evidence = {"LEFT": [], "RIGHT": []}
        stem_spans = {"LEFT": [], "RIGHT": []}
        start_ys = {"LEFT": [], "RIGHT": []}
        votes = []
        single_straight_drifted = False

        for boundary in self.observed_boundaries:
            segs = boundary.get("segments", [])
            b_side = boundary.get("side", "UNKNOWN")
            if not segs:
                votes.append("UNKNOWN")
                continue
            best_turn = 0.0
            best_dir = "UNKNOWN"
            best_elbow = None
            best_stem_span = 0.0
            p_start = segs[0]["p1"]
            p_end = segs[-1]["p2"]
            dx_all = float(p_end[0] - p_start[0])
            dy_up_all = max(1.0, float(p_start[1] - p_end[1]))
            slope_ratio_all = abs(dx_all) / dy_up_all

            if len(segs) >= 2:
                for s1, s2 in zip(segs[:-1], segs[1:]):
                    v1 = np.array(s1["p2"], dtype=np.float32) - np.array(s1["p1"], dtype=np.float32)
                    v2 = np.array(s2["p2"], dtype=np.float32) - np.array(s2["p1"], dtype=np.float32)
                    l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                    if l1 < 12.0 or l2 < 12.0:
                        continue
                    cos_a = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
                    ang_deg = float(np.rad2deg(np.arccos(cos_a)))
                    # Nhanh tren cua goc cua phai vat cheo/ngang ro ret (abs(v2[0]) >= abs(v2[1]) * 0.90)
                    if ang_deg >= 18.0 and abs(v2[0]) >= abs(v2[1]) * 0.90:
                        direction = "RIGHT" if v2[0] > 0 else "LEFT"
                        if ang_deg > best_turn:
                            best_turn = ang_deg
                            best_dir = direction
                            best_elbow = (float(s1["p2"][0]), float(s1["p2"][1]))
                            best_stem_span = max(0.0, float(segs[0]["p1"][1] - s1["p2"][1]))

            # Phan biet ro:
            # - Vach thang khi xe di lech 1 ben: xuat phat o mep duoi (p_start[1] >= 0.85*h) va co phuong gan nhu thang dung (slope_ratio_all < 1.20)
            # - Vach goc cua khi xe tien sat khuc cua: diem bat dau nam cao hon tren mep canh (p_start[1] < 0.85*h) va vat cheo gan nhu vuong goc (slope_ratio_all >= 1.45)
            if best_dir == "UNKNOWN" and len(self.observed_boundaries) == 1:
                if (float(p_start[1]) >= h * 0.85 and slope_ratio_all < 1.20) or (float(p_start[1]) >= h * 0.65 and slope_ratio_all < 1.15):
                    single_straight_drifted = True
                elif b_side == "LEFT" and float(p_start[1]) < h * 0.85 and dx_all >= w * 0.28 and slope_ratio_all >= 1.45:
                    best_dir = "RIGHT"
                    best_elbow = (
                        float(0.5 * (p_start[0] + p_end[0])),
                        float(0.65 * p_start[1] + 0.35 * p_end[1]),
                    )
                    best_stem_span = 0.0
                elif b_side == "RIGHT" and float(p_start[1]) < h * 0.85 and dx_all <= -w * 0.28 and slope_ratio_all >= 1.45:
                    best_dir = "LEFT"
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
            else:
                votes.append("UNKNOWN")

        valid_votes = [v for v in votes if v != "UNKNOWN"]
        hypothesis = valid_votes[0] if len(valid_votes) == 1 else "UNKNOWN"
        if len(valid_votes) == 2 and valid_votes[0] == valid_votes[1]:
            candidate = valid_votes[0]
        elif len(valid_votes) == 1 and len(self.observed_boundaries) >= 1:
            candidate = valid_votes[0]
        elif (
            len(self.observed_boundaries) == 1
            and not single_straight_drifted
            and self._intent_turn_dir in ("LEFT", "RIGHT")
        ):
            # Chi nho huong re (_intent_turn_dir) khi vach don le thuc su nam cao tren mep canh hoac nghieng cheo vao cua
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
                    else:
                        candidate = "UNKNOWN"
                else:
                    candidate = "UNKNOWN"
            else:
                candidate = "UNKNOWN"
        else:
            candidate = "UNKNOWN"

        ids = tuple(sorted(b["id"] for b in self.observed_boundaries))
        if not fresh_observation:
            self._corner_hits = max(1, self._corner_hits) if candidate != "UNKNOWN" else 0
        elif candidate != "UNKNOWN" and (candidate == self._corner_candidate or candidate == self._intent_turn_dir):
            self._corner_hits += 1
        else:
            self._corner_hits = int(candidate != "UNKNOWN")
        self._corner_candidate, self._corner_ids = candidate, ids
        confirmed = candidate != "UNKNOWN" and (self._corner_hits >= 2 or candidate == self._intent_turn_dir)
        conf_val = (
            0.90
            if (len(valid_votes) == 2 and candidate != "UNKNOWN")
            else (0.78 if confirmed else (0.72 if candidate != "UNKNOWN" else (0.45 if hypothesis != "UNKNOWN" else 0.0)))
        )

        # Cap nhat Bo nho tam nhin bao quat goc cua (_intent_turn_dir) de xe biet truoc huong re khi vao cua hoac khi mat vach
        if candidate in ("LEFT", "RIGHT") and (confirmed or len(valid_votes) >= 1):
            self._intent_turn_dir = candidate
            self._intent_turn_ttl = 45
            self._straight_clear_hits = 0
        elif len(self.observed_boundaries) == 2 and candidate == "UNKNOWN":
            if fresh_observation:
                self._straight_clear_hits += 1
                if self._straight_clear_hits >= 5:
                    self._intent_turn_dir = "UNKNOWN"
                    self._intent_turn_ttl = 0
        elif self._intent_turn_ttl > 0 and fresh_observation:
            self._intent_turn_ttl -= 1
            if self._intent_turn_ttl == 0:
                self._intent_turn_dir = "UNKNOWN"

        # Tinh khoang cach toi goc cua & BU DIEM MU 15CM TRUOC MUI XE:
        # Khi camera thay goc cua o y ~ 85..135px (0.36*h .. 0.56*h), thuc te mui xe con cach goc cua ~15cm diem mu.
        # Do do:
        #   - FAR (elbow_y < 0.36*h): Goc cua con o xa -> DI THANG bam lan, ghi nho huong re.
        #   - APPROACH (0.36*h <= elbow_y < 0.58*h): Dang di qua 15cm diem mu -> GIAM TOC DO (SLOW), KHONG RE SOM!
        #   - TURN_NOW (elbow_y >= 0.58*h hoac diem bat dau vach da leo cao len mep canh p_start_y < 0.75*h
        #               hoac da di het chu ky dem diem mu): Be lai re ngay!
        elbow_y = None
        elbow_x = None
        elbow_y_ratio = 0.0
        stem_span_y = float(h)
        min_start_y = float(h)
        distance_zone = "NONE"
        turn_now = False

        if candidate in ("LEFT", "RIGHT") and evidence[candidate]:
            elbows = evidence[candidate]
            elbow_y = float(np.max([pt[1] for pt in elbows]))
            elbow_x = float(np.mean([pt[0] for pt in elbows]))
            elbow_y_ratio = elbow_y / float(max(1, h))
            if stem_spans[candidate]:
                stem_span_y = float(np.min(stem_spans[candidate]))
            if start_ys[candidate]:
                min_start_y = float(np.min(start_ys[candidate]))

            is_single_edge = len(self.observed_boundaries) == 1
            if fresh_observation:
                # Tinh doan chay them de bu 15cm diem mu truoc dau xe:
                # Khi khuc cua bat dau troi xuong vung duoi anh (elbow_y_ratio >= 0.55),
                # bat dau dem frame de cho xe tiep tuc di thang them 1 doan qua diem mu 15cm roi moi be lai.
                if elbow_y_ratio >= 0.55:
                    self._blind_spot_wait_frames += 1
                elif elbow_y_ratio < 0.38:
                    self._blind_spot_wait_frames = 0

            # KHI DA DI THANG QUA HET 15CM DIEM MU:
            # - Hoac da dem du >= 6 frames (~0.25 - 0.35s o toc do 65%) ke tu khi goc cua vao vung tiep can
            # - Hoac goc cua da troi xuong rat sau (elbow_y_ratio >= 0.72)
            # - Hoac diem bat dau vach da tut xuong mep duoi
            blind_spot_cleared = (self._blind_spot_wait_frames >= 6) or (elbow_y_ratio >= 0.72)

            if (
                (elbow_y_ratio >= 0.65 and blind_spot_cleared)
                or (elbow_y_ratio >= 0.55 and stem_span_y <= h * 0.12 and blind_spot_cleared)
                or (is_single_edge and min_start_y < h * 0.68 and blind_spot_cleared)
            ):
                distance_zone = "TURN_NOW"
                turn_now = True
                confirmed = True
                self._active_turn_dir = candidate
                self._active_turn_hold = 6
            elif elbow_y_ratio >= 0.35:
                distance_zone = "APPROACH"
                turn_now = False
            else:
                distance_zone = "FAR"
                turn_now = False
        elif self._active_turn_hold > 0 and len(self.observed_boundaries) <= 1 and not single_straight_drifted:
            # Duy tri nhip cua muot khi xe dang o giua goc cua 90 do va 1 vach tam khuat
            if fresh_observation:
                self._active_turn_hold -= 1
            candidate = self._active_turn_dir
            confirmed = True
            distance_zone = "TURN_NOW"
            turn_now = True
        else:
            if fresh_observation:
                self._active_turn_hold = 0
                self._active_turn_dir = "UNKNOWN"
                self._blind_spot_wait_frames = 0

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
            "distance_zone": distance_zone,
            "turn_now": bool(turn_now),
        }

    @staticmethod
    def _smooth_segment_list(prev_segs, new_segs, alpha=0.58):
        if not new_segs:
            return []
        if not prev_segs or len(prev_segs) != len(new_segs):
            return new_segs
        out = []
        for sp, sn in zip(prev_segs, new_segs):
            p1_p, p2_p = np.array(sp["p1"], dtype=np.float32), np.array(sp["p2"], dtype=np.float32)
            p1_n, p2_n = np.array(sn["p1"], dtype=np.float32), np.array(sn["p2"], dtype=np.float32)
            if float(np.linalg.norm(p1_n - p1_p)) > 42.0 or float(np.linalg.norm(p2_n - p2_p)) > 42.0:
                return new_segs
            p1_s = alpha * p1_n + (1.0 - alpha) * p1_p
            p2_s = alpha * p2_n + (1.0 - alpha) * p2_p
            out.append({
                "p1": (float(p1_s[0]), float(p1_s[1])),
                "p2": (float(p2_s[0]), float(p2_s[1])),
                "width_start": float(alpha * sn["width_start"] + (1.0 - alpha) * sp["width_start"]),
                "width_end": float(alpha * sn["width_end"] + (1.0 - alpha) * sp["width_end"]),
                "width": float(alpha * sn["width"] + (1.0 - alpha) * sp["width"]),
                "length": float(np.linalg.norm(p2_s - p1_s)),
            })
        return out

    @staticmethod
    def _resample_pts(pts, n_out=20):
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 2:
            return pts
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.r_[0.0, np.cumsum(seg_lens)]
        total = float(cum[-1])
        if total < 1e-3:
            return np.repeat(pts[:1], n_out, axis=0)
        targets = np.linspace(0.0, total, n_out)
        out = []
        for s_val in targets:
            idx = min(len(pts) - 2, max(0, int(np.searchsorted(cum, s_val, side="right") - 1)))
            ds = max(1e-4, float(seg_lens[idx]))
            t = float(np.clip((s_val - cum[idx]) / ds, 0.0, 1.0))
            out.append(pts[idx] + t * diffs[idx])
        return np.asarray(out, dtype=np.float32)

    def detect_dual_lanes(self, mask, hT, wT, top_y_hint=105, img_gray=None):
        l_res, r_res = self._find_and_trace_boundaries(mask, top_y_hint=top_y_hint, gray=img_gray)
        l_segs = self._fit_piecewise_segments(l_res[0], l_res[1]) if l_res is not None else []
        r_segs = self._fit_piecewise_segments(r_res[0], r_res[1]) if r_res is not None else []

        # Neu chi nhin thay 1 vach duy nhat -> Phan loai chinh xac do la VACH TRAI hay VACH PHAI
        if bool(l_segs) ^ bool(r_segs):
            single_pts = l_res[0] if l_segs else r_res[0]
            single_segs = l_segs if l_segs else r_segs
            true_side = self._classify_single_boundary_side(single_pts, single_segs, hT, wT)
            if true_side == "LEFT":
                self.left_segments = self._smooth_segment_list(self._smooth_left_segs, single_segs)
                self.right_segments = []
                l_res = (single_pts, l_res[1] if l_segs else r_res[1])
                r_res = None
                self.single_edge_side = "LEFT"
                self._last_left_bot_x = float(self.left_segments[0]["p1"][0])
            else:
                self.right_segments = self._smooth_segment_list(self._smooth_right_segs, single_segs)
                self.left_segments = []
                r_res = (single_pts, r_res[1] if r_segs else l_res[1])
                l_res = None
                self.single_edge_side = "RIGHT"
                self._last_right_bot_x = float(self.right_segments[0]["p1"][0])
        else:
            self.left_segments = self._smooth_segment_list(self._smooth_left_segs, l_segs)
            self.right_segments = self._smooth_segment_list(self._smooth_right_segs, r_segs)
            if self.left_segments and self.right_segments:
                self.single_edge_side = "BOTH"
                self._last_left_bot_x = float(self.left_segments[0]["p1"][0])
                self._last_right_bot_x = float(self.right_segments[0]["p1"][0])
            else:
                self.single_edge_side = None

        self._smooth_left_segs = list(self.left_segments)
        self._smooth_right_segs = list(self.right_segments)

        cands = []
        if l_res is not None and self.left_segments:
            l_poly = [(float(s["p1"][0]), float(s["p1"][1])) for s in self.left_segments] + [
                (float(self.left_segments[-1]["p2"][0]), float(self.left_segments[-1]["p2"][1]))
            ]
            l_len = sum(float(s["length"]) for s in self.left_segments)
            l_w = float(np.median([s["width"] for s in self.left_segments]))
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
            r_w = float(np.median([s["width"] for s in self.right_segments]))
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

        c_pts, p_flags, new_ab = self._compute_lane_and_centerline(
            self.left_segments, self.right_segments, hT, wT, lane_width_ab=self.lane_width_ab
        )
        self.lane_width_ab = new_ab

        # Lam muot duong tam qua cac frame lien tiep de vach khong nhay lung tung
        if len(c_pts) >= 2:
            c_resamp = self._resample_pts(c_pts, n_out=20)
            p_resamp = np.full(len(c_resamp), bool(np.any(p_flags)), dtype=bool)
            if self._smooth_center_pts is not None and len(self._smooth_center_pts) == len(c_resamp) and self.missing_frames == 0:
                mean_diff = float(np.mean(np.linalg.norm(c_resamp - self._smooth_center_pts, axis=1)))
                if mean_diff < 52.0:
                    c_resamp = (0.60 * c_resamp + 0.40 * self._smooth_center_pts).astype(np.float32)
            self._smooth_center_pts = c_resamp.copy()
            c_pts = c_resamp
            p_flags = p_resamp

        has_l = len(self.left_segments) > 0
        has_r = len(self.right_segments) > 0
        if has_l and has_r and len(c_pts) >= 2:
            self.center_pts = c_pts
            self.center_pred_flags = p_flags
            self._last_valid_center_pts = c_pts.copy()
            self._last_valid_pred_flags = p_flags.copy()
            self.missing_frames = 0
            status = "2 BIEN QUAN SAT"
        elif (has_l or has_r) and len(c_pts) >= 2:
            self.center_pts = c_pts
            self.center_pred_flags = p_flags
            self._last_valid_center_pts = c_pts.copy()
            self._last_valid_pred_flags = p_flags.copy()
            self.missing_frames = 0
            status = "ONE_EDGE"
        else:
            self.missing_frames += 1
            if len(self._last_valid_center_pts) >= 2 and self.missing_frames <= 8:
                self.center_pts = self._last_valid_center_pts.copy()
                self.center_pred_flags = np.ones(len(self.center_pts), dtype=bool)
                status = "GIU LAN"
            else:
                self.center_pts = np.empty((0, 2), dtype=np.float32)
                self.center_pred_flags = np.empty((0,), dtype=bool)
                return None, None, None, None, "TIM LAN..."

        plot_y = self.center_pts[:, 1].copy()
        center_x = self.center_pts[:, 0].copy()
        half_w = 0.5 * np.clip(self.lane_width_ab[0] * plot_y + self.lane_width_ab[1], wT * 0.22, wT * 0.98)
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
                      profile="robot", fresh_observation=True, restrict_roi=False):
        """Xu ly khung hinh va tra ve (norm_curve, img_roi, img_thres, img_warp_vis, img_hist_combined, img_result)."""
        hT, wT = img.shape[:2]
        img_copy = img.copy()
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

        lighting_info = utils.analyze_lighting(img, roi_points=points)
        if profile == "robot":
            img_thres = utils.thresholding(img, hsv_range, roi_points=points, restrict_roi=restrict_roi)
        else:
            img_thres = utils.thresholding_highway(img, hsv_range, roi_points=points, restrict_roi=True)

        img_warp = utils.warpImg(img_thres, points, wT, hT)
        img_warp_vis = utils.warpImg(img, points, wT, hT)
        img_roi = utils.drawPoints(img_copy, points)

        mid_point, img_hist1 = utils.getHistogram(img_warp, display=True, minPer=0.5, region=4)
        avg_point, img_hist2 = utils.getHistogram(img_warp, display=True, minPer=0.7, region=1)
        hist_curve = float(avg_point - mid_point)
        raw_curve = hist_curve
        img_hist_combined = cv2.addWeighted(img_hist1, 0.5, img_hist2, 0.5, 0)

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
        offset_dir = "CAN GIUA"
        lowest_c_pt = None
        lookahead_c_pt = None
        w_lane_ref = float(self.lane_width_ab[0] * (hT * 0.90) + self.lane_width_ab[1])
        tape_w_ref = float(np.median([s["width"] for s in (self.left_segments + self.right_segments)]) if (self.left_segments or self.right_segments) else 8.0)

        if use_dual_edge:
            top_hint = int(points[0][1]) if points is not None else int(hT * 0.60)
            if profile == "robot":
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                plot_y, l_x, r_x, c_x, status = self.detect_dual_lanes(img_thres, hT, wT, top_y_hint=top_hint, img_gray=img_gray)
                corner = self.inspect_corner(img_thres, fresh_observation=fresh_observation)
                all_segs = self.left_segments + self.right_segments
                if all_segs:
                    tape_w_ref = float(np.median([s["width"] for s in all_segs]))

                # 1. Ve 2 vach mau xanh bien bam chinh xac vao giua 2 vach trang thuc te voi do day theo tung vi tri
                self._draw_blue_tapered_segments(img_result, self.left_segments)
                self._draw_blue_tapered_segments(img_result, self.right_segments)

                # 2. Ve duong tam mau trang (net lien khi 2 vach quan sat, net dut danh dau khi du doan tu 1 vach)
                c_pts = self.center_pts
                p_flags = self.center_pred_flags
                if len(c_pts) >= 2:
                    for i in range(len(c_pts) - 1):
                        pt1 = (int(round(c_pts[i, 0])), int(round(c_pts[i, 1])))
                        pt2 = (int(round(c_pts[i + 1, 0])), int(round(c_pts[i + 1, 1])))
                        is_pred = bool(p_flags[i] or p_flags[i + 1])
                        if not is_pred:
                            cv2.line(img_result, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)
                        else:
                            if i % 2 == 0:
                                cv2.line(img_result, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)
                            else:
                                cv2.line(img_result, pt1, pt2, (0, 215, 255), 2, cv2.LINE_AA)

                    if np.all(p_flags):
                        mid_c = c_pts[len(c_pts) // 2]
                        side_tag = "VACH TRAI" if self.single_edge_side == "LEFT" else ("VACH PHAI" if self.single_edge_side == "RIGHT" else "1 VACH")
                        cv2.putText(
                            img_result, f"TAM DU DOAN ({side_tag})",
                            (max(6, min(wT - 190, int(mid_c[0]) - 65)), max(32, int(mid_c[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA
                        )

                    # 3. Xac dinh diem thap nhat cua duong tam trang & tinh do lech tam e so voi vi tri tham chieu xe
                    low_idx = int(np.argmax(c_pts[:, 1]))
                    cx_low_raw = float(c_pts[low_idx, 0])
                    cy_low = float(c_pts[low_idx, 1])
                    w_lane_ref = max(60.0, float(self.lane_width_ab[0] * cy_low + self.lane_width_ab[1]))
                    e_px_raw = cx_low_raw - car_ref_x

                    if self.smooth_e_px is None or not fresh_observation:
                        self.smooth_e_px = e_px_raw
                    else:
                        alpha_e = 0.48 if (len(self.left_segments) > 0 and len(self.right_segments) > 0) else 0.35
                        delta_e = float(np.clip(e_px_raw - self.smooth_e_px, -18.0, 18.0))
                        self.smooth_e_px = float(self.smooth_e_px + alpha_e * delta_e)

                    e_px_out = float(self.smooth_e_px)
                    e_norm_out = float(np.clip(e_px_out / (0.5 * w_lane_ref), -1.0, 1.0))
                    self.smooth_e_norm = e_norm_out
                    offset_dir = "CAN GIUA" if abs(e_px_out) <= 4.0 else ("LECH PHAI" if e_px_out > 0 else "LECH TRAI")

                    # 4. Tinh diem ngam lai Lookahead tren duong tam:
                    # Neu phat hien goc cua o xa (turn_now == False), GIU DIEM NGAM O DOAN THANG TRUOC GOC CUA
                    # de xe khong bao gio re som khi chua toi khuc cua!
                    elbow_y_val = corner.get("elbow_y")
                    turn_now_flag = bool(corner.get("turn_now", False))
                    c_turn_dir = corner.get("direction") if corner.get("direction") in ("LEFT", "RIGHT") else corner.get("candidate", "UNKNOWN")

                    if not turn_now_flag and elbow_y_val is not None:
                        # Chi lay cac diem nam tren doan thang truoc khuc cua (y > elbow_y + 10)
                        valid_mask = c_pts[:, 1] >= (float(elbow_y_val) + 10.0)
                        stem_c_pts = c_pts[valid_mask] if np.any(valid_mask) else c_pts
                        y_look_target = max(float(hT * 0.68), float(elbow_y_val) + 22.0)
                        look_idx = int(np.argmin(np.abs(stem_c_pts[:, 1] - y_look_target)))
                    elif turn_now_flag:
                        # Khi da toi diem re (TURN_NOW), ngam vao nhanh re cua duong tam de be lai ngay!
                        stem_c_pts = c_pts
                        look_idx = min(len(stem_c_pts) - 1, max(0, int(round((len(stem_c_pts) - 1) * 0.78))))
                    else:
                        stem_c_pts = c_pts
                        y_look_target = float(hT * 0.68)
                        look_idx = int(np.argmin(np.abs(stem_c_pts[:, 1] - y_look_target)))

                    cx_look = float(stem_c_pts[look_idx, 0])
                    cy_look = float(stem_c_pts[look_idx, 1])
                    lookahead_c_pt = (cx_look, cy_look)
                    w_lane_look = max(60.0, float(self.lane_width_ab[0] * cy_look + self.lane_width_ab[1]))
                    e_look_norm = float(np.clip((cx_look - car_ref_x) / (0.5 * w_lane_look), -1.0, 1.0))

                    if turn_now_flag and c_turn_dir in ("LEFT", "RIGHT"):
                        turn_sign = +1.0 if c_turn_dir == "RIGHT" else -1.0
                        steer_norm_out = float(np.clip(0.15 * e_norm_out + 0.85 * e_look_norm, -1.0, 1.0))
                        if turn_sign > 0:
                            steer_norm_out = max(steer_norm_out, +0.95)
                        else:
                            steer_norm_out = min(steer_norm_out, -0.95)
                    else:
                        if status == "ONE_EDGE" and c_turn_dir not in ("LEFT", "RIGHT"):
                            # Khi xe di lech sang 1 ben tren doan thang (chi con 1 vach):
                            # Dieu huong banh re ve giua duong dua vao do lech tuong doi cua diem bat dau
                            steer_norm_out = float(np.clip(0.60 * e_norm_out + 0.40 * e_look_norm, -0.55, 0.55))
                        else:
                            steer_norm_out = float(np.clip(0.35 * e_norm_out + 0.65 * e_look_norm, -1.0, 1.0))
                            if corner.get("distance_zone") == "APPROACH" and c_turn_dir in ("LEFT", "RIGHT"):
                                # DANG DI THANG QUA 15CM DIEM MU: Giu thang lai tren than duong, khong be lai som!
                                steer_norm_out = float(np.clip(steer_norm_out, -0.12, 0.12))

                    # Ve diem ngam Lookahead (vong tron xanh ngoc) tren duong tam trang
                    pt_look = (int(round(np.clip(cx_look, 2.0, wT - 3.0))), int(round(np.clip(cy_look, 2.0, hT - 3.0))))
                    cv2.circle(img_result, pt_look, 5, (255, 255, 0), 2, lineType=cv2.LINE_AA)

                    # Hien thi trang thai khoang cach goc cua tren HUD
                    if corner.get("candidate") in ("LEFT", "RIGHT") and elbow_y_val is not None:
                        c_dir_vn = "PHAI" if corner["candidate"] == "RIGHT" else "TRAI"
                        if turn_now_flag:
                            c_txt = f"CUA {c_dir_vn} (y={int(elbow_y_val)}) -> RE NGAY!"
                            c_col = (0, 165, 255)
                        elif corner.get("distance_zone") == "APPROACH":
                            c_txt = f"CUA {c_dir_vn} (y={int(elbow_y_val)}) -> DI THANG BU DIEM MU 15CM"
                            c_col = (0, 230, 255)
                        else:
                            c_txt = f"CUA {c_dir_vn} PHIA TRUOC (y={int(elbow_y_val)}) -> DI THANG"
                            c_col = (255, 230, 0)
                        cv2.putText(img_result, c_txt, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, c_col, 1, cv2.LINE_AA)

                    # 5. MUI TEN DO CO DINH O MEP DUOI (y = hT - 6), DI CHUYEN NGANG CUNG TAM LAN (x = cx_low_disp)
                    #    DE TINH DO LECH TAM e SO VOI VACH THAM CHIEU VANG O GIUA XE (car_ref_x = wT / 2)
                    arrow_base_y = hT - 6
                    cx_low_disp = float(np.clip(car_ref_x + e_px_out, 10.0, wT - 11.0)) if fresh_observation else cx_low_raw
                    lowest_c_pt = (cx_low_disp, float(arrow_base_y))
                    pt_arrow_base = (int(round(cx_low_disp)), arrow_base_y)
                    arrow_len = max(38, int(hT * 0.22))
                    pt_arrow_tip = (int(round(cx_low_disp)), max(8, arrow_base_y - arrow_len))
                    pt_car_ref = (int(round(car_ref_x)), arrow_base_y)

                    # Ve vi tri tham chieu thuc cua xe (vach vang co dinh o giua mep duoi) va thanh ngang do lech e
                    cv2.line(
                        img_result,
                        (pt_car_ref[0], max(8, arrow_base_y - 14)),
                        (pt_car_ref[0], min(hT - 2, arrow_base_y + 4)),
                        (0, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.line(img_result, pt_car_ref, pt_arrow_base, (0, 180, 255), 2, cv2.LINE_AA)
                    cv2.circle(img_result, pt_car_ref, 4, (0, 255, 255), cv2.FILLED, lineType=cv2.LINE_AA)

                    # Ve mui ten DO co dinh mep duoi (y = hT - 6), di cung tam lan (x = cx_low_disp)
                    cv2.arrowedLine(img_result, pt_arrow_base, pt_arrow_tip, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.28)
                    cv2.circle(img_result, pt_arrow_base, 5, (255, 255, 255), cv2.FILLED, lineType=cv2.LINE_AA)
                    cv2.circle(img_result, pt_arrow_base, 3, (0, 0, 255), cv2.FILLED, lineType=cv2.LINE_AA)

                    raw_curve = steer_norm_out * 90.0

                    warp_lines = utils.warpImg(img_result, points, wT, hT)
                    img_warp_vis = cv2.addWeighted(img_warp_vis, 0.35, warp_lines, 0.65, 0)
                    cv2.putText(
                        img_hist_combined,
                        f"{status} | e:{e_px_out:+.1f}px ({offset_dir})",
                        (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 255, 255), 1
                    )
                else:
                    # Khi mat vach hoan toan, van giu mui ten do co dinh o goc duoi giua man hinh
                    arrow_base_y = hT - 6
                    arrow_len = max(38, int(hT * 0.22))
                    pt_arrow_base = (int(round(car_ref_x)), arrow_base_y)
                    pt_arrow_tip = (int(round(car_ref_x)), max(8, arrow_base_y - arrow_len))
                    cv2.arrowedLine(img_result, pt_arrow_base, pt_arrow_tip, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.28)
                    cv2.circle(img_result, pt_arrow_base, 5, (255, 255, 255), cv2.FILLED, lineType=cv2.LINE_AA)
                    cv2.circle(img_result, pt_arrow_base, 3, (0, 0, 255), cv2.FILLED, lineType=cv2.LINE_AA)
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
                    offset_dir = "CAN GIUA" if abs(e_px_out) <= 4.0 else ("LECH PHAI" if e_px_out > 0 else "LECH TRAI")

                    pts_l = np.column_stack((l_x, plot_y)).astype(np.int32)
                    pts_r = np.column_stack((r_x, plot_y)).astype(np.int32)
                    pts_c = np.column_stack((c_x, plot_y)).astype(np.int32)

                    overlay = img_result.copy()
                    cv2.fillPoly(overlay, [np.vstack((pts_l, pts_r[::-1]))], (0, 200, 60))
                    img_result = cv2.addWeighted(img_result, 0.62, overlay, 0.38, 0)
                    cv2.polylines(img_result, [pts_l], False, (255, 105, 0), 4, cv2.LINE_AA)
                    cv2.polylines(img_result, [pts_r], False, (255, 105, 0), 4, cv2.LINE_AA)
                    cv2.polylines(img_result, [pts_c], False, (255, 255, 255), 3, cv2.LINE_AA)

                    warp_lines = utils.warpImg(img_result, points, wT, hT)
                    img_warp_vis = cv2.addWeighted(img_warp_vis, 0.35, warp_lines, 0.65, 0)

                    cv2.putText(img_hist_combined, f"{status} | Tam:{int(target_x)}", (8, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)
                else:
                    cv2.putText(img_hist_combined, f"Mid: {mid_point} | Avg: {avg_point}", (10, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        else:
            cv2.putText(img_hist_combined, f"Mid: {mid_point} | Avg: {avg_point}", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        observed = (bool(self.boundary_traces) if profile == "robot" else
                    status.startswith(("2 BIEN", "BAM BIEN"))) if use_dual_edge else False
        edge_count = len(self.boundary_traces) if profile == "robot" and use_dual_edge else None
        self.last_detection = {
            "observed": observed,
            "lane_state": ("TWO_EDGES" if edge_count == 2 else "ONE_EDGE" if edge_count == 1 else "UNKNOWN")
                          if profile == "robot" else (status if observed else ("PREDICTED" if c_x is not None else "UNKNOWN")),
            "edge_count": edge_count,
            "center_available": c_x is not None,
            "center_offset_px": e_px_out,
            "center_offset_norm": e_norm_out,
            "steer_offset_norm": steer_norm_out,
            "offset_direction": offset_dir,
            "vehicle_ref_x": car_ref_x,
            "lowest_center_point": lowest_c_pt,
            "lookahead_center_point": lookahead_c_pt,
            "single_edge_side": self.single_edge_side if profile == "robot" else None,
            "lane_width_px": w_lane_ref,
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
            "lighting": lighting_info,
            "restrict_roi": bool(restrict_roi or profile == "highway"),
        }
        if profile == "robot" and c_x is None:
            if edge_count == 1:
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

        if (not use_dual_edge or c_x is None) and profile == "highway":
            img_inv_warp = utils.warpImg(img_warp, points, wT, hT, inv=True)
            img_inv_warp = cv2.cvtColor(img_inv_warp, cv2.COLOR_GRAY2BGR)
            img_inv_warp[0:hT // 3, 0:wT] = 0, 0, 0
            lane_color = np.zeros_like(img)
            lane_color[:] = (0, 220, 0)
            lane_highlight = cv2.bitwise_and(img_inv_warp, lane_color)
            img_result = cv2.addWeighted(img_result, 1, lane_highlight, 0.7, 0)

        if profile == "highway" and observed and c_x is not None:
            mid_y = int(hT * 0.88)
            center_x = wT // 2
            curve_offset = int(norm_curve * 100)
            cv2.arrowedLine(img_result, (center_x, hT - 4), (center_x, int(hT * 0.70)),
                            (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.25)
            cv2.line(img_result, (center_x, mid_y - 14), (center_x, mid_y + 14), (255, 255, 0), 2)
            cv2.line(img_result, (center_x, mid_y), (center_x + curve_offset, mid_y), (0, 0, 255), 4)
            cv2.circle(img_result, (center_x + curve_offset, mid_y), 6, (0, 255, 255), cv2.FILLED)

        light_mode = lighting_info["mode"]
        roi_scope = "ROI-ONLY" if (restrict_roi or profile == "highway") else "WIDE-CORNER"
        if profile == "robot":
            cv2.putText(
                img_result,
                f"e = {e_px_out:+.1f}px ({e_norm_out:+.2f}) [{offset_dir}]",
                (8, hT - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 2, cv2.LINE_AA
            )
            label = (
                f"Lane:{self.last_detection['lane_state']} | "
                f"Segs L:{len(self.left_segments)} R:{len(self.right_segments)} | "
                f"W_lane:{int(w_lane_ref)}px W_tape:{tape_w_ref:.1f}px"
            )
            cv2.putText(img_result, label, (8, hT - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200, 255, 200), 1, cv2.LINE_AA)
        else:
            label = f"Lane:{self.last_detection['lane_state']} | Light:{light_mode} | {roi_scope}"
            cv2.putText(img_result, label, (6, hT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 255, 255), 1)
        cv2.putText(img_hist_combined,
                    f"Corner:{corner['candidate']} conf:{corner['confidence']:.2f} | Light:{light_mode}",
                    (6, 62), cv2.FONT_HERSHEY_SIMPLEX, .39, (0, 255, 255), 1)
        return norm_curve, img_roi, img_thres, img_warp_vis, img_hist_combined, img_result


_default_detector = LaneDetector(avg_window=6)


def getLaneCurve(img, display=2, profile="robot", return_details=False, restrict_roi=False):
    """Ham goi nhanh tren Raspberry Pi 5 (RobotMain.py)."""
    hT, wT = img.shape[:2]
    points = utils.valTrackbars(wT, hT, preset=profile)
    preset_cfg = PRESET_ROBOT_TOY if profile == "robot" else PRESET_HIGHWAY
    norm_curve, img_roi, img_thres, img_warp, img_hist, img_result = _default_detector.process_frame(
        img, points, hsv_range=preset_cfg["hsv_range"],
        use_dual_edge=True, profile=profile, restrict_roi=restrict_roi
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
