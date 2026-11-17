import time
import cv2
import numpy as np


class RobotDashboardUI:
    """Quan ly cua so GUI giam sat da chieu va tuong tac chuot/phim cho RobotMain."""

    TOP_H = 40
    BOT_H = 36

    # Kich thuoc che do Chi tiet (Full Dashboard: 1020 x 556)
    LEFT_W = 560
    RIGHT_W = 460
    TOTAL_W = LEFT_W + RIGHT_W  # 1020
    MAIN_H = 300
    PIPE_H = 180
    SIGN_H = 260
    CTRL_H = 220
    BODY_H = MAIN_H + PIPE_H    # 480

    # Kich thuoc che do Thu gon (Compact: 640 x 396)
    COMPACT_W = 640
    COMPACT_H = 320

    def __init__(self, win_name="Xe Tu Hanh 4 Luong Song Song (Pi 5)", detail_mode=True):
        self.win_name = win_name
        self.available = False
        self.detail_mode = detail_mode
        self._pending_action = None
        self._buttons = {}
        try:
            cv2.namedWindow(self.win_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.win_name, self._on_mouse)
            self.available = True
        except cv2.error:
            self.available = False

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for action, (x1, y1, x2, y2) in self._buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._pending_action = action
                break

    @staticmethod
    def _draw_button(canvas, rect, text, bg=(48, 54, 66), border=(115, 125, 140), text_color=(245, 245, 245)):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, cv2.FILLED)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        tx = x1 + max(4, (x2 - x1 - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2 - 1
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_badge(canvas, x, y, text, bg=(34, 40, 50), border=(90, 100, 115), text_color=(220, 230, 240)):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        pad_x = 8
        w = tw + pad_x * 2
        h = 22
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, cv2.FILLED)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (x + pad_x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)
        return x + w + 6

    @staticmethod
    def _draw_section_header(panel, title, right_text="", accent=(0, 200, 255)):
        w = panel.shape[1]
        cv2.rectangle(panel, (0, 0), (w, 26), (30, 35, 44), cv2.FILLED)
        cv2.line(panel, (0, 26), (w, 26), (62, 70, 84), 1)
        cv2.rectangle(panel, (0, 0), (4, 26), accent, cv2.FILLED)
        cv2.putText(panel, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 240, 248), 1, cv2.LINE_AA)
        if right_text:
            (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
            cv2.putText(panel, right_text, (max(180, w - tw - 10), 17), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (165, 185, 205), 1, cv2.LINE_AA)

    @staticmethod
    def _fit_panel(img, w, h, title=None, placeholder="Dang cho du lieu..."):
        out = np.full((h, w, 3), (20, 23, 29), dtype=np.uint8)
        if img is not None and img.size > 0:
            if len(img.shape) == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img
            resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            out[:] = resized
        else:
            (tw, th), _ = cv2.getTextSize(placeholder, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(out, placeholder, ((w - tw) // 2, (h + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 120, 135), 1, cv2.LINE_AA)

        if title:
            cv2.rectangle(out, (0, 0), (w, 18), (16, 19, 24), cv2.FILLED)
            cv2.putText(out, title, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (205, 220, 235), 1, cv2.LINE_AA)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (58, 65, 78), 1)
        return out

    @staticmethod
    def _draw_progress_bar(canvas, x, y, w, h, ratio, fill_color, label_left="", label_right=""):
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if label_left:
            cv2.putText(canvas, label_left, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 225), 1, cv2.LINE_AA)
        if label_right:
            (tw, _), _ = cv2.getTextSize(label_right, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(canvas, label_right, (x + w - tw, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 245, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 40, 49), cv2.FILLED)
        fill_w = int(round(w * ratio))
        if fill_w > 0:
            cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), fill_color, cv2.FILLED)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (75, 84, 98), 1)

    @staticmethod
    def _draw_bipolar_bar(canvas, x, y, w, h, val, label_left="", label_right=""):
        val = float(np.clip(val, -1.0, 1.0))
        if label_left:
            cv2.putText(canvas, label_left, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 225), 1, cv2.LINE_AA)
        if label_right:
            (tw, _), _ = cv2.getTextSize(label_right, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(canvas, label_right, (x + w - tw, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 245, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 40, 49), cv2.FILLED)
        mid_x = x + w // 2
        fill_x = mid_x + int(round((w // 2) * val))
        col = (0, 220, 140) if abs(val) < 0.25 else ((0, 190, 255) if abs(val) < 0.6 else (70, 130, 255))
        if fill_x >= mid_x:
            cv2.rectangle(canvas, (mid_x, y + 1), (fill_x, y + h - 1), col, cv2.FILLED)
        else:
            cv2.rectangle(canvas, (fill_x, y + 1), (mid_x, y + h - 1), col, cv2.FILLED)
        cv2.line(canvas, (mid_x, y - 2), (mid_x, y + h + 2), (220, 225, 235), 1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (75, 84, 98), 1)

    def _build_top_bar(self, width, cam_label, fov_pct, cam_ok, esp_ok, c_fps, l_fps, s_fps):
        bar = np.full((self.TOP_H, width, 3), (24, 27, 34), dtype=np.uint8)
        cv2.line(bar, (0, self.TOP_H - 1), (width, self.TOP_H - 1), (65, 72, 85), 1)

        self._buttons = {
            "switch_cam": (8, 6, 128, 34),
            "fov_minus": (134, 6, 164, 34),
            "fov_reset": (168, 6, 260, 34),
            "fov_plus": (264, 6, 294, 34),
            "toggle_mode": (302, 6, 412, 34),
        }

        self._draw_button(bar, self._buttons["switch_cam"], f"{cam_label} [C]", bg=(35, 62, 95), border=(0, 200, 255))
        self._draw_button(bar, self._buttons["fov_minus"], "-")
        is_full = fov_pct == 100
        self._draw_button(
            bar,
            self._buttons["fov_reset"],
            f"Rong:{fov_pct}%",
            bg=(0, 125, 75) if is_full else (48, 54, 66),
            border=(0, 235, 150) if is_full else (115, 125, 140),
        )
        self._draw_button(bar, self._buttons["fov_plus"], "+")
        mode_lbl = "Thu gon [D]" if self.detail_mode else "Chi tiet [D]"
        self._draw_button(
            bar,
            self._buttons["toggle_mode"],
            mode_lbl,
            bg=(56, 48, 92) if self.detail_mode else (48, 54, 66),
            border=(170, 140, 255) if self.detail_mode else (115, 125, 140),
        )

        if width >= 900:
            nx = 424
            nx = self._draw_badge(
                bar, nx, 9,
                "CAM OK" if cam_ok else "CAM ERR",
                bg=(20, 75, 45) if cam_ok else (35, 35, 95),
                border=(0, 220, 130) if cam_ok else (80, 90, 255),
            )
            self._draw_badge(
                bar, nx, 9,
                "ESP32 OK" if esp_ok else "ESP32 SIM",
                bg=(20, 70, 85) if esp_ok else (50, 50, 58),
                border=(0, 200, 240) if esp_ok else (130, 135, 145),
            )

        fps_txt = f"FPS  Cam:{c_fps:.0f} | Lane:{l_fps:.0f} | Sign:{s_fps:.0f}"
        (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(
            bar,
            fps_txt,
            (width - tw - 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (110, 245, 160),
            1,
            cv2.LINE_AA,
        )
        return bar

    def _build_bottom_bar(self, width, pkt, lane_state, sign_action):
        bar = np.full((self.BOT_H, width, 3), (24, 27, 34), dtype=np.uint8)
        cv2.line(bar, (0, 0), (width, 0), (65, 72, 85), 1)

        is_stop = "STOP" in pkt
        pkt_col = (90, 140, 255) if is_stop else (90, 255, 140)
        cv2.putText(bar, f"UART: {pkt}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, pkt_col, 1, cv2.LINE_AA)

        act_txt = sign_action if sign_action else "Khong"
        hint = "  |  [C] Cam  [+/-] FOV  [D] Che do  [Q] Thoat" if width >= 900 else ""
        right_txt = f"Lan: {lane_state}  |  Bien: {act_txt}{hint}"
        (tw, _), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.39, 1)
        cv2.putText(
            bar,
            right_txt,
            (max(230, width - tw - 12), 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (215, 222, 235),
            1,
            cv2.LINE_AA,
        )
        return bar

    def _build_lane_pipeline_panel(self, lane_thres, lane_warp, lane_hist, lane_info):
        panel = np.full((self.PIPE_H, self.LEFT_W, 3), (22, 25, 32), dtype=np.uint8)
        edge_cnt = lane_info.get("edge_count")
        edge_str = str(edge_cnt) if edge_cnt is not None else "-"
        lane_w = lane_info.get("lane_width_px")
        w_str = f"{lane_w:.0f}px" if isinstance(lane_w, (int, float)) else "-"
        miss = lane_info.get("missing_frames", 0)
        self._draw_section_header(
            panel,
            "XU LY LAN DUONG (LANE PIPELINE)",
            f"Bien: {edge_str} | Rong lan: {w_str} | Mat: {miss}",
            accent=(0, 225, 140),
        )

        thres_vis = None
        if lane_thres is not None and lane_thres.size > 0:
            thres_vis = cv2.cvtColor(lane_thres, cv2.COLOR_GRAY2BGR)
            thres_vis[lane_thres > 0] = (140, 255, 160)

        cell_w, cell_h = 178, 142
        y0 = 32
        p1 = self._fit_panel(thres_vis, cell_w, cell_h, "1. Nguong (Threshold)")
        p2 = self._fit_panel(lane_warp, cell_w, cell_h, "2. Bird-Eye Warp")
        for gx in range(cell_w // 4, cell_w, cell_w // 4):
            cv2.line(p2, (gx, 18), (gx, cell_h - 1), (55, 65, 75), 1)
        p3 = self._fit_panel(lane_hist, cell_w, cell_h, "3. Histogram & Bien")

        panel[y0:y0 + cell_h, 8:8 + cell_w] = p1
        panel[y0:y0 + cell_h, 191:191 + cell_w] = p2
        panel[y0:y0 + cell_h, 374:374 + cell_w] = p3
        cv2.rectangle(panel, (0, 0), (self.LEFT_W - 1, self.PIPE_H - 1), (55, 62, 75), 1)
        return panel

    def _build_sign_panel(self, signs, sign_dbg, sign_info, sign_action):
        panel = np.full((self.SIGN_H, self.RIGHT_W, 3), (22, 25, 32), dtype=np.uint8)
        n_det = len(signs)
        cands = sign_dbg.get("candidate_boxes", []) if sign_dbg else []
        self._draw_section_header(
            panel,
            "NHAN DIEN BIEN BAO (TRAFFIC SIGNS)",
            f"Xac nhan: {n_det} | Vung nghi: {len(cands)}",
            accent=(0, 190, 255),
        )

        # 1. Trai-tren: Mat na mau HSV (Do / Xanh) + khung ung vien
        mask_w, mask_h = 200, 128
        mask_vis = None
        if sign_dbg and "mask_r" in sign_dbg and "mask_b" in sign_dbg:
            mr = sign_dbg["mask_r"]
            mb = sign_dbg["mask_b"]
            sh, sw = mr.shape[:2]
            mask_vis = np.zeros((sh, sw, 3), dtype=np.uint8)
            mask_vis[mr > 0] = (40, 60, 255)
            mask_vis[mb > 0] = (255, 160, 30)
            for bx, by, bw, bh in cands:
                cv2.rectangle(mask_vis, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 130), 2)
        p_mask = self._fit_panel(mask_vis, mask_w, mask_h, "Loc mau HSV (Do/Xanh)", "Chua bat bien bao")
        panel[32:32 + mask_h, 8:8 + mask_w] = p_mask

        # 2. Phai-tren: ONNX Crop Previews (ke ca ung vien bi loai de giam sat)
        prev_w, prev_h = 236, 128
        p_prev = np.full((prev_h, prev_w, 3), (18, 21, 27), dtype=np.uint8)
        cv2.rectangle(p_prev, (0, 0), (prev_w, 18), (16, 19, 24), cv2.FILLED)
        cv2.putText(p_prev, "Phan loai ONNX (Ung vien)", (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (205, 220, 235), 1, cv2.LINE_AA)
        previews = sign_dbg.get("onnx_previews", []) if sign_dbg else []
        if previews:
            for idx, prev in enumerate(previews[:2]):
                ry = 24 + idx * 50
                crop = prev.get("crop_bgr")
                cls_name = prev.get("cls_name", "?")
                sc = float(prev.get("score", 0.0))
                ok = bool(prev.get("accepted", False))
                if crop is not None and crop.size > 0:
                    thumb = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_AREA)
                    p_prev[ry:ry + 40, 8:48] = thumb
                b_col = (0, 235, 130) if ok else (0, 150, 255)
                cv2.rectangle(p_prev, (8, ry), (48, ry + 40), b_col, 1)
                vn_name = sign_info.get(cls_name, {"vn": cls_name})["vn"]
                status_txt = "NHAN" if ok else "LOAI"
                cv2.putText(p_prev, vn_name[:20], (56, ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 240, 248), 1, cv2.LINE_AA)
                cv2.putText(p_prev, f"{sc:.0%} [{status_txt}]", (56, ry + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.36, b_col, 1, cv2.LINE_AA)
        else:
            cv2.putText(p_prev, "Khong co ung vien", (46, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 120, 135), 1, cv2.LINE_AA)
        cv2.rectangle(p_prev, (0, 0), (prev_w - 1, prev_h - 1), (58, 65, 78), 1)
        panel[32:32 + prev_h, 216:216 + prev_w] = p_prev

        # 3. Duoi: Ket qua xac nhan (Tracker) + Hanh dong dieu khien
        box_x, box_y, box_w, box_h = 8, 166, self.RIGHT_W - 16, 86
        cv2.rectangle(panel, (box_x, box_y), (box_x + box_w, box_y + box_h), (26, 30, 38), cv2.FILLED)
        cv2.rectangle(panel, (box_x, box_y), (box_x + box_w, box_y + box_h), (58, 65, 78), 1)

        act_lbl = sign_action if sign_action else "Khong co hanh dong"
        act_col = (80, 220, 255) if sign_action else (150, 160, 175)
        cv2.putText(panel, f"Hanh dong: {act_lbl}", (box_x + 10, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, act_col, 1, cv2.LINE_AA)

        if signs:
            for idx, (cls_name, sx, sy, sw, sh, score) in enumerate(signs[:2]):
                info = sign_info.get(cls_name, {"vn": cls_name, "color": (0, 255, 0)})
                ly = box_y + 44 + idx * 20
                cv2.circle(panel, (box_x + 15, ly - 4), 5, info["color"], cv2.FILLED)
                txt = f"{info['vn']} ({score:.0%})  |  Kich thuoc: {sw}x{sh}px"
                cv2.putText(panel, txt, (box_x + 28, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (235, 242, 250), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Chua phat hien bien bao on dinh", (box_x + 10, box_y + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (120, 130, 145), 1, cv2.LINE_AA)

        cv2.rectangle(panel, (0, 0), (self.RIGHT_W - 1, self.SIGN_H - 1), (55, 62, 75), 1)
        return panel

    def _build_control_panel(self, lane_state, lane_info, corner_info, light_mode,
                             curve_val, speed_pct, err_val, cmd_str,
                             lane_age_ms, sign_age_ms, esp_ok, esp_port):
        panel = np.full((self.CTRL_H, self.RIGHT_W, 3), (22, 25, 32), dtype=np.uint8)
        self._draw_section_header(
            panel,
            "DIEU KHIEN & TRANG THAI XU LY",
            f"CMD: {cmd_str}",
            accent=(255, 175, 55),
        )

        # Thanh Toc do & Goc lai
        spd_col = (70, 120, 255) if speed_pct == 0 else ((0, 215, 255) if speed_pct < 60 else (0, 225, 130))
        self._draw_progress_bar(
            panel, 14, 52, self.RIGHT_W - 28, 14,
            abs(speed_pct) / 100.0, spd_col,
            "Toc do dong co (Speed)", f"{speed_pct}%"
        )
        self._draw_bipolar_bar(
            panel, 14, 92, self.RIGHT_W - 28, 14,
            err_val,
            "Goc lai / Lech tam (Err)", f"Err:{err_val:+.2f} | Curve:{curve_val:+.2f}"
        )

        # Thong so chi tiet
        conf = float(lane_info.get("confidence", 0.0))
        c_state = corner_info.get("state", "NONE") if isinstance(corner_info, dict) else "NONE"
        c_dir = corner_info.get("direction", "-") if isinstance(corner_info, dict) else "-"
        esp_txt = f"Ket noi ({esp_port})" if esp_ok else "Mo phong (Chua co cong)"

        rows = [
            ("Trang thai lan:", f"{lane_state} (Tin cay: {conf:.0%})", (120, 240, 175)),
            ("Goc cua / Huong:", f"{c_state} ({c_dir})", (220, 228, 240)),
            ("Anh sang / Watchdog:", f"{light_mode}  |  Lane:{lane_age_ms:.0f}ms  Sign:{sign_age_ms:.0f}ms", (205, 215, 230)),
            ("ESP32 UART:", esp_txt, (85, 225, 255) if esp_ok else (165, 175, 190)),
        ]
        y_start = 132
        for idx, (k, v, v_col) in enumerate(rows):
            ry = y_start + idx * 21
            cv2.putText(panel, k, (14, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (155, 165, 180), 1, cv2.LINE_AA)
            cv2.putText(panel, v, (162, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.39, v_col, 1, cv2.LINE_AA)

        cv2.rectangle(panel, (0, 0), (self.RIGHT_W - 1, self.CTRL_H - 1), (55, 62, 75), 1)
        return panel

    def render_and_poll(self, state, camera_module, frame_size, sign_info):
        """
        Ve giao dien giam sat (Top Bar + Main View + Cac panel giam sat + Bottom Bar)
        va xu ly su kien chuot/phim. Tra ve False neu nguoi dung bam thoat ('q' / ESC).
        """
        if not self.available:
            return True

        now = time.time()
        with state.cam_lock:
            c_fps = state.cam_fps
            fov_pct = state.cam_fov_pct
            cam_ok = state.cam_ok

        with state.lane_lock:
            vis = state.debug_lane_img.copy() if state.debug_lane_img is not None else None
            lane_thres = state.lane_thres_img.copy() if getattr(state, "lane_thres_img", None) is not None else None
            lane_warp = state.lane_warp_img.copy() if getattr(state, "lane_warp_img", None) is not None else None
            lane_hist = state.lane_hist_img.copy() if getattr(state, "lane_hist_img", None) is not None else None
            l_fps = state.lane_fps
            l_state = state.lane_state
            l_info = dict(state.lane_info) if state.lane_info else {}
            c_info = dict(state.corner_info) if state.corner_info else {}
            light_mode = state.lighting_mode
            curve_val = state.curve_val
            lane_age_ms = (now - state.lane_time) * 1000.0 if state.lane_time > 0 else 999.0

        with state.sign_lock:
            signs = list(state.detected_signs)
            sign_dbg = dict(getattr(state, "sign_debug", {}) or {})
            s_fps = state.sign_fps
            sign_age_ms = (now - state.sign_time) * 1000.0 if state.sign_time > 0 else 999.0

        with state.ctrl_lock:
            pkt = state.last_packet
            cmd_str = state.last_cmd
            speed_pct = state.last_speed
            err_val = state.last_err
            sign_action = state.last_sign_action
            esp_ok = bool(getattr(state, "esp32_connected", False))
            esp_port = str(getattr(state, "esp32_port", ""))

        if vis is None:
            return True

        for cls_name, x, y, w, h, score in signs:
            info = sign_info.get(cls_name, {"vn": cls_name, "color": (0, 255, 0)})
            cv2.rectangle(vis, (x, y), (x + w, y + h), info["color"], 2)
            cv2.putText(
                vis,
                f"{info['vn']} ({score:.0%})",
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cam_label = camera_module.get_active_camera_label()
        if self.detail_mode:
            main_cam = cv2.resize(vis, (self.LEFT_W, self.MAIN_H))
            conf = float(l_info.get("confidence", 0.0))
            hud_txt = f"Lan: {l_state} | Sang: {light_mode} | Tin cay: {conf:.0%}"
            cv2.rectangle(main_cam, (6, 6), (345, 28), (18, 22, 28), cv2.FILLED)
            cv2.putText(main_cam, hud_txt, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (215, 235, 245), 1, cv2.LINE_AA)
            cv2.rectangle(main_cam, (0, 0), (self.LEFT_W - 1, self.MAIN_H - 1), (55, 62, 75), 1)

            pipe_panel = self._build_lane_pipeline_panel(lane_thres, lane_warp, lane_hist, l_info)
            left_col = np.vstack([main_cam, pipe_panel])

            sign_panel = self._build_sign_panel(signs, sign_dbg, sign_info, sign_action)
            ctrl_panel = self._build_control_panel(
                l_state, l_info, c_info, light_mode,
                curve_val, speed_pct, err_val, cmd_str,
                lane_age_ms, sign_age_ms, esp_ok, esp_port
            )
            right_col = np.vstack([sign_panel, ctrl_panel])

            body = np.hstack([left_col, right_col])
            top_bar = self._build_top_bar(self.TOTAL_W, cam_label, fov_pct, cam_ok, esp_ok, c_fps, l_fps, s_fps)
            bot_bar = self._build_bottom_bar(self.TOTAL_W, pkt, l_state, sign_action)
            dashboard = np.vstack([top_bar, body, bot_bar])
        else:
            video_panel = cv2.resize(vis, (self.COMPACT_W, self.COMPACT_H))
            top_bar = self._build_top_bar(self.COMPACT_W, cam_label, fov_pct, cam_ok, esp_ok, c_fps, l_fps, s_fps)
            bot_bar = self._build_bottom_bar(self.COMPACT_W, pkt, l_state, sign_action)
            dashboard = np.vstack([top_bar, video_panel, bot_bar])

        act = self._pending_action
        if act is not None:
            self._pending_action = None
            if act == "switch_cam":
                camera_module.switch_camera(size=frame_size)
            elif act == "fov_minus":
                with state.cam_lock:
                    state.cam_fov_pct = max(40, state.cam_fov_pct - 10)
            elif act == "fov_plus":
                with state.cam_lock:
                    state.cam_fov_pct = min(100, state.cam_fov_pct + 10)
            elif act == "fov_reset":
                with state.cam_lock:
                    state.cam_fov_pct = 100
            elif act == "toggle_mode":
                self.detail_mode = not self.detail_mode

        try:
            cv2.imshow(self.win_name, dashboard)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                state.running = False
                return False
            if key == ord("c"):
                camera_module.switch_camera(size=frame_size)
            elif key == ord("d"):
                self.detail_mode = not self.detail_mode
            elif key in (ord("-"), ord("_")):
                with state.cam_lock:
                    state.cam_fov_pct = max(40, state.cam_fov_pct - 10)
            elif key in (ord("+"), ord("=")):
                with state.cam_lock:
                    state.cam_fov_pct = min(100, state.cam_fov_pct + 10)
        except cv2.error:
            self.available = False
        return True

    def close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
