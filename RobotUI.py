import time
import cv2
import numpy as np


class RobotDashboardUI:
    """
    Man hinh quan sat 2 Camera doc lap cho Xe Tu Hanh (Dual-Cam Observation Cockpit):
      - CAM 1 (Trai): Camera Lan duong (nhin xuong mat duong)
      - CAM 2 (Phai): Camera Bien bao YOLO11n (nhin thang phia truoc)
      - Hang duoi (che do Chi tiet [D]): Pipeline Lan duong | Ung vien YOLO11n & Dieu khien Xe
    """

    TOP_H = 42
    BOT_H = 38

    CAM_W = 600
    CAM_H = 340
    TOTAL_W = CAM_W * 2          # 1200
    LOWER_H = 236                # Hang pipeline & dieu khien ben duoi

    OBS_CAM_H = 370              # Chieu cao camera o che do Quan sat thuan (Thu gon)
    OBS_HUD_H = 44               # Dai thanh toc do / goc lai o che do Quan sat thuan

    def __init__(self, win_name="Xe Tu Hanh - Man Hinh Quan Sat 2 Camera", detail_mode=True):
        self.win_name = win_name
        self.available = False
        self.detail_mode = detail_mode
        self._pending_action = None
        self._buttons = {}
        try:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.win_name, self.TOTAL_W, self.TOP_H + self.CAM_H + self.LOWER_H + self.BOT_H)
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
    def _draw_button(canvas, rect, text, bg=(42, 48, 60), border=(105, 118, 138), text_color=(242, 246, 252)):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, cv2.FILLED)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.39, 1)
        tx = x1 + max(4, (x2 - x1 - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2 - 1
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.39, text_color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_badge(canvas, x, y, text, bg=(30, 38, 48), border=(90, 100, 115), text_color=(225, 235, 245)):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        pad_x = 7
        w = tw + pad_x * 2
        h = 22
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, cv2.FILLED)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (x + pad_x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, text_color, 1, cv2.LINE_AA)
        return x + w + 6

    @staticmethod
    def _draw_section_header(panel, title, right_text="", accent=(0, 200, 255)):
        w = panel.shape[1]
        cv2.rectangle(panel, (0, 0), (w, 26), (26, 31, 40), cv2.FILLED)
        cv2.line(panel, (0, 26), (w, 26), (58, 66, 80), 1)
        cv2.rectangle(panel, (0, 0), (4, 26), accent, cv2.FILLED)
        cv2.putText(panel, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 240, 248), 1, cv2.LINE_AA)
        if right_text:
            (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            cv2.putText(panel, right_text, (max(180, w - tw - 10), 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (165, 185, 205), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_viewfinder_corners(img, x0, y0, x1, y1, color=(80, 105, 130), length=14):
        """Ve 4 goc ngam kieu man hinh giam sat camera."""
        t = 1
        cv2.line(img, (x0, y0), (x0 + length, y0), color, t)
        cv2.line(img, (x0, y0), (x0, y0 + length), color, t)
        cv2.line(img, (x1, y0), (x1 - length, y0), color, t)
        cv2.line(img, (x1, y0), (x1, y0 + length), color, t)
        cv2.line(img, (x0, y1), (x0 + length, y1), color, t)
        cv2.line(img, (x0, y1), (x0, y1 - length), color, t)
        cv2.line(img, (x1, y1), (x1 - length, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 - length), color, t)

    @staticmethod
    def _fit_panel(img, w, h, title=None, placeholder="Dang cho du lieu..."):
        out = np.full((h, w, 3), (18, 21, 27), dtype=np.uint8)
        if img is not None and img.size > 0:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            out[:] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        else:
            (tw, th), _ = cv2.getTextSize(placeholder, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(out, placeholder, ((w - tw) // 2, (h + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 120, 135), 1, cv2.LINE_AA)

        if title:
            cv2.rectangle(out, (0, 0), (w, 19), (15, 18, 24), cv2.FILLED)
            cv2.putText(out, title, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (205, 220, 235), 1, cv2.LINE_AA)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (55, 62, 76), 1)
        return out

    @staticmethod
    def _draw_progress_bar(canvas, x, y, w, h, ratio, fill_color, label_left="", label_right=""):
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if label_left:
            cv2.putText(canvas, label_left, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200, 210, 225), 1, cv2.LINE_AA)
        if label_right:
            (tw, _), _ = cv2.getTextSize(label_right, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
            cv2.putText(canvas, label_right, (x + w - tw, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (240, 245, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (33, 38, 48), cv2.FILLED)
        fill_w = int(round(w * ratio))
        if fill_w > 0:
            cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), fill_color, cv2.FILLED)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (75, 84, 98), 1)

    @staticmethod
    def _draw_bipolar_bar(canvas, x, y, w, h, val, label_left="", label_right=""):
        val = float(np.clip(val, -1.0, 1.0))
        if label_left:
            cv2.putText(canvas, label_left, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200, 210, 225), 1, cv2.LINE_AA)
        if label_right:
            (tw, _), _ = cv2.getTextSize(label_right, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
            cv2.putText(canvas, label_right, (x + w - tw, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (240, 245, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (33, 38, 48), cv2.FILLED)
        mid_x = x + w // 2
        fill_x = mid_x + int(round((w // 2) * val))
        col = (0, 220, 140) if abs(val) < 0.25 else ((0, 190, 255) if abs(val) < 0.6 else (70, 130, 255))
        if fill_x >= mid_x:
            cv2.rectangle(canvas, (mid_x, y + 1), (fill_x, y + h - 1), col, cv2.FILLED)
        else:
            cv2.rectangle(canvas, (fill_x, y + 1), (mid_x, y + h - 1), col, cv2.FILLED)
        cv2.line(canvas, (mid_x, y - 2), (mid_x, y + h + 2), (220, 225, 235), 1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (75, 84, 98), 1)

    def _build_top_bar(self, width, lane_cam_lbl, sign_cam_lbl, fov_pct,
                       lane_cam_ok, sign_cam_ok, shared_cam, esp_ok,
                       l_cam_fps, s_cam_fps, l_fps, s_fps):
        bar = np.full((self.TOP_H, width, 3), (20, 24, 31), dtype=np.uint8)
        cv2.line(bar, (0, self.TOP_H - 1), (width, self.TOP_H - 1), (62, 70, 84), 1)

        self._buttons = {
            "switch_lane_cam": (8, 7, 152, 35),
            "switch_sign_cam": (158, 7, 326, 35),
            "swap_cams":       (332, 7, 424, 35),
            "fov_minus":       (432, 7, 460, 35),
            "fov_reset":       (464, 7, 542, 35),
            "fov_plus":        (546, 7, 574, 35),
            "toggle_mode":     (582, 7, 702, 35),
        }

        self._draw_button(bar, self._buttons["switch_lane_cam"], f"Lan:{lane_cam_lbl} [C]", bg=(22, 68, 52), border=(0, 220, 140))
        self._draw_button(bar, self._buttons["switch_sign_cam"], f"Bien:{sign_cam_lbl[:11]} [V]", bg=(26, 56, 92), border=(0, 195, 255))
        self._draw_button(bar, self._buttons["swap_cams"], "Doi cho [X]", bg=(64, 52, 30), border=(255, 190, 60))
        self._draw_button(bar, self._buttons["fov_minus"], "-")
        self._draw_button(
            bar, self._buttons["fov_reset"], f"FOV:{fov_pct}%",
            bg=(0, 115, 72) if fov_pct == 100 else (42, 48, 60),
            border=(0, 230, 145) if fov_pct == 100 else (105, 118, 138),
        )
        self._draw_button(bar, self._buttons["fov_plus"], "+")
        mode_lbl = "Quan sat [D]" if self.detail_mode else "Chi tiet [D]"
        self._draw_button(
            bar, self._buttons["toggle_mode"], mode_lbl,
            bg=(56, 44, 92) if self.detail_mode else (42, 48, 60),
            border=(175, 140, 255) if self.detail_mode else (105, 118, 138),
        )

        nx = 714
        nx = self._draw_badge(
            bar, nx, 10,
            "CAM1 OK" if lane_cam_ok else "CAM1 ERR",
            bg=(18, 68, 42) if lane_cam_ok else (35, 35, 95),
            border=(0, 220, 130) if lane_cam_ok else (80, 90, 255),
        )
        sign_badge = "CAM2 CHUNG" if shared_cam else ("CAM2 OK" if sign_cam_ok else "CAM2 ERR")
        nx = self._draw_badge(
            bar, nx, 10, sign_badge,
            bg=(72, 58, 22) if shared_cam else ((18, 60, 88) if sign_cam_ok else (35, 35, 95)),
            border=(255, 200, 60) if shared_cam else ((0, 200, 255) if sign_cam_ok else (80, 90, 255)),
        )
        self._draw_badge(
            bar, nx, 10,
            "ESP32" if esp_ok else "SIM",
            bg=(20, 70, 85) if esp_ok else (48, 48, 56),
            border=(0, 200, 240) if esp_ok else (125, 130, 140),
        )

        fps_txt = f"FPS  C1:{l_cam_fps:.0f} C2:{s_cam_fps:.0f} | Lane:{l_fps:.0f} YOLO:{s_fps:.0f}"
        (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
        cv2.putText(bar, fps_txt, (width - tw - 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (115, 245, 165), 1, cv2.LINE_AA)
        return bar

    def _build_camera_viewport(self, frame, w, h, title, cam_label, accent, status_dot_col,
                               status_text, hud_left="", hud_right="", placeholder="Dang cho tin hieu Camera..."):
        panel = np.full((h, w, 3), (14, 17, 23), dtype=np.uint8)
        hdr_h = 28
        body_h = h - hdr_h

        # Header camera kieu man hinh giam sat
        cv2.rectangle(panel, (0, 0), (w, hdr_h), (24, 29, 38), cv2.FILLED)
        cv2.rectangle(panel, (0, 0), (4, hdr_h), accent, cv2.FILLED)
        cv2.circle(panel, (16, 14), 5, status_dot_col, cv2.FILLED)
        cv2.putText(panel, title, (28, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (238, 244, 252), 1, cv2.LINE_AA)

        right_hdr = f"{cam_label}  |  {status_text}"
        (tw, _), _ = cv2.getTextSize(right_hdr, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
        cv2.putText(panel, right_hdr, (max(260, w - tw - 10), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.37, accent, 1, cv2.LINE_AA)
        cv2.line(panel, (0, hdr_h), (w, hdr_h), (58, 66, 80), 1)

        # Vung hien thi hinh anh camera
        if frame is not None and frame.size > 0:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
            panel[hdr_h:h, 0:w] = cv2.resize(bgr, (w, body_h), interpolation=cv2.INTER_AREA)
        else:
            (ptw, pth), _ = cv2.getTextSize(placeholder, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.putText(panel, placeholder, ((w - ptw) // 2, hdr_h + (body_h + pth) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (115, 125, 140), 1, cv2.LINE_AA)

        # 4 goc ngam quan sat + HUD ban trong suot o day man hinh camera
        self._draw_viewfinder_corners(panel, 8, hdr_h + 8, w - 9, h - 9, color=(95, 115, 140), length=16)

        if hud_left or hud_right:
            ov_y0 = h - 26
            overlay = panel[ov_y0:h - 2, 2:w - 2].copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (14, 18, 24), cv2.FILLED)
            cv2.addWeighted(overlay, 0.68, panel[ov_y0:h - 2, 2:w - 2], 0.32, 0, panel[ov_y0:h - 2, 2:w - 2])
            if hud_left:
                cv2.putText(panel, hud_left, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (225, 238, 248), 1, cv2.LINE_AA)
            if hud_right:
                (rtw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.39, 1)
                cv2.putText(panel, hud_right, (max(260, w - rtw - 12), h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 235, 255), 1, cv2.LINE_AA)

        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (62, 70, 86), 1)
        return panel

    def _build_lane_pipeline_panel(self, lane_thres, lane_warp, lane_hist, lane_info):
        panel = np.full((self.LOWER_H, self.CAM_W, 3), (20, 24, 31), dtype=np.uint8)
        edge_cnt = lane_info.get("edge_count")
        edge_str = str(edge_cnt) if edge_cnt is not None else "-"
        lane_w = lane_info.get("lane_width_px")
        w_str = f"{lane_w:.0f}px" if isinstance(lane_w, (int, float)) else "-"
        miss = lane_info.get("missing_frames", 0)
        self._draw_section_header(
            panel,
            "XU LY LAN DUONG (LANE PIPELINE - CAM 1)",
            f"Bien: {edge_str} | Rong lan: {w_str} | Mat: {miss}",
            accent=(0, 225, 140),
        )

        thres_vis = None
        if lane_thres is not None and lane_thres.size > 0:
            thres_vis = cv2.cvtColor(lane_thres, cv2.COLOR_GRAY2BGR)
            thres_vis[lane_thres > 0] = (140, 255, 160)

        cell_w, cell_h = 190, 194
        y0 = 34
        p1 = self._fit_panel(thres_vis, cell_w, cell_h, "1. Nguong (Threshold)")
        p2 = self._fit_panel(lane_warp, cell_w, cell_h, "2. Bird-Eye Warp")
        for gx in range(cell_w // 4, cell_w, cell_w // 4):
            cv2.line(p2, (gx, 19), (gx, cell_h - 1), (55, 65, 75), 1)
        p3 = self._fit_panel(lane_hist, cell_w, cell_h, "3. Histogram & Bien")

        panel[y0:y0 + cell_h, 8:8 + cell_w] = p1
        panel[y0:y0 + cell_h, 205:205 + cell_w] = p2
        panel[y0:y0 + cell_h, 402:402 + cell_w] = p3
        cv2.rectangle(panel, (0, 0), (self.CAM_W - 1, self.LOWER_H - 1), (55, 62, 75), 1)
        return panel

    def _build_sign_and_control_panel(self, signs, sign_dbg, sign_info, sign_action,
                                      lane_state, lane_info, corner_info, light_mode,
                                      curve_val, speed_pct, err_val, cmd_str,
                                      lane_age_ms, sign_age_ms, esp_ok, esp_port):
        panel = np.full((self.LOWER_H, self.CAM_W, 3), (20, 24, 31), dtype=np.uint8)
        cands = sign_dbg.get("candidate_boxes", []) if sign_dbg else []
        self._draw_section_header(
            panel,
            "BIEN BAO YOLO11n (CAM 2) & DIEU KHIEN XE",
            f"Xac nhan: {len(signs)} | Ung vien: {len(cands)} | CMD: {cmd_str}",
            accent=(0, 195, 255),
        )

        # Cot trai (250px): Danh sach Ung vien / Crop YOLO11n + Hanh dong uu tien
        left_w = 248
        box_y, box_h = 34, 194
        cv2.rectangle(panel, (8, box_y), (8 + left_w, box_y + box_h), (16, 20, 27), cv2.FILLED)
        cv2.rectangle(panel, (8, box_y), (8 + left_w, box_y + box_h), (55, 62, 76), 1)
        cv2.rectangle(panel, (8, box_y), (8 + left_w, box_y + 20), (24, 30, 40), cv2.FILLED)
        cv2.putText(panel, "Ung vien YOLO11n (Crop)", (14, box_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (205, 222, 240), 1, cv2.LINE_AA)

        previews = sign_dbg.get("yolo_previews", sign_dbg.get("onnx_previews", [])) if sign_dbg else []
        if previews:
            for idx, prev in enumerate(previews[:2]):
                ry = box_y + 28 + idx * 54
                crop = prev.get("crop_bgr")
                cls_name = prev.get("cls_name", "?")
                sc = float(prev.get("score", 0.0))
                ok = bool(prev.get("accepted", False))
                if crop is not None and crop.size > 0:
                    panel[ry:ry + 44, 16:60] = cv2.resize(crop, (44, 44), interpolation=cv2.INTER_AREA)
                b_col = (0, 235, 130) if ok else (0, 155, 255)
                cv2.rectangle(panel, (16, ry), (60, ry + 44), b_col, 1)
                vn_name = sign_info.get(cls_name, {"vn": cls_name})["vn"]
                st_txt = "NHAN" if ok else "THAP"
                cv2.putText(panel, vn_name[:18], (68, ry + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (235, 242, 250), 1, cv2.LINE_AA)
                cv2.putText(panel, f"{sc:.0%} [{st_txt}]", (68, ry + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.37, b_col, 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Chua phat hien bien bao", (38, box_y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 122, 138), 1, cv2.LINE_AA)

        # Hop hanh dong uu tien bien bao o day cot trai
        cv2.line(panel, (12, box_y + 140), (left_w + 4, box_y + 140), (50, 58, 70), 1)
        act_lbl = sign_action if sign_action else "Khong co hanh dong"
        act_col = (80, 225, 255) if sign_action else (140, 150, 165)
        cv2.putText(panel, "Uu tien bien bao:", (16, box_y + 158), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 162, 178), 1, cv2.LINE_AA)
        cv2.putText(panel, act_lbl[:26], (16, box_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.42, act_col, 1, cv2.LINE_AA)

        # Cot phai (330px): Dieu khien dong co + Goc lai + Thong so Watchdog/ESP32
        rx = 266
        rw = self.CAM_W - rx - 8
        cv2.rectangle(panel, (rx, box_y), (rx + rw, box_y + box_h), (16, 20, 27), cv2.FILLED)
        cv2.rectangle(panel, (rx, box_y), (rx + rw, box_y + box_h), (55, 62, 76), 1)

        spd_col = (70, 120, 255) if speed_pct == 0 else ((0, 215, 255) if speed_pct < 60 else (0, 225, 130))
        self._draw_progress_bar(
            panel, rx + 12, box_y + 26, rw - 24, 14,
            abs(speed_pct) / 100.0, spd_col,
            "Toc do dong co", f"{speed_pct}%",
        )
        self._draw_bipolar_bar(
            panel, rx + 12, box_y + 66, rw - 24, 14,
            err_val,
            "Goc lai / Lech tam", f"Err:{err_val:+.2f} | Curve:{curve_val:+.2f}",
        )

        conf = float(lane_info.get("confidence", 0.0))
        c_state = corner_info.get("state", "NONE") if isinstance(corner_info, dict) else "NONE"
        c_dir = corner_info.get("direction", "-") if isinstance(corner_info, dict) else "-"
        esp_txt = f"Ket noi ({esp_port})" if esp_ok else "Mo phong (Chua co cong)"

        rows = [
            ("Lan duong:", f"{lane_state} ({conf:.0%})", (120, 240, 175)),
            ("Goc cua:", f"{c_state} ({c_dir})", (220, 228, 240)),
            ("Watchdog:", f"Lane:{lane_age_ms:.0f}ms | Sign:{sign_age_ms:.0f}ms", (205, 215, 230)),
            ("Anh sang:", f"{light_mode}", (255, 210, 100)),
            ("ESP32:", esp_txt, (85, 225, 255) if esp_ok else (160, 170, 185)),
        ]
        for idx, (k, v, v_col) in enumerate(rows):
            ry = box_y + 102 + idx * 19
            cv2.putText(panel, k, (rx + 12, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150, 162, 178), 1, cv2.LINE_AA)
            cv2.putText(panel, v[:30], (rx + 98, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.37, v_col, 1, cv2.LINE_AA)

        cv2.rectangle(panel, (0, 0), (self.CAM_W - 1, self.LOWER_H - 1), (55, 62, 75), 1)
        return panel

    def _build_compact_hud_strip(self, speed_pct, err_val, curve_val, cmd_str, lane_state, sign_action):
        strip = np.full((self.OBS_HUD_H, self.TOTAL_W, 3), (18, 22, 29), dtype=np.uint8)
        cv2.line(strip, (0, 0), (self.TOTAL_W, 0), (58, 66, 80), 1)
        spd_col = (70, 120, 255) if speed_pct == 0 else ((0, 215, 255) if speed_pct < 60 else (0, 225, 130))
        self._draw_progress_bar(strip, 16, 22, 260, 12, abs(speed_pct) / 100.0, spd_col, "Toc do", f"{speed_pct}%")
        self._draw_bipolar_bar(strip, 300, 22, 320, 12, err_val, "Goc lai", f"Err:{err_val:+.2f} | C:{curve_val:+.2f}")

        act_txt = sign_action if sign_action else "Khong co bien bao uu tien"
        cv2.putText(strip, f"CMD: {cmd_str}  |  Lan: {lane_state}", (650, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 238, 248), 1, cv2.LINE_AA)
        cv2.putText(strip, f"Hanh dong bien bao: {act_txt}", (650, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 225, 255) if sign_action else (150, 160, 175), 1, cv2.LINE_AA)
        return strip

    def _build_bottom_bar(self, width, pkt, lane_state, sign_action):
        bar = np.full((self.BOT_H, width, 3), (20, 24, 31), dtype=np.uint8)
        cv2.line(bar, (0, 0), (width, 0), (62, 70, 84), 1)

        is_stop = "STOP" in pkt
        pkt_col = (90, 140, 255) if is_stop else (90, 255, 140)
        cv2.putText(bar, f"UART: {pkt}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, pkt_col, 1, cv2.LINE_AA)

        act_txt = sign_action if sign_action else "Khong"
        right_txt = f"Lan: {lane_state}  |  Bien: {act_txt}  |  [C] Cam1  [V] Cam2  [X] Hoan doi  [+/-] FOV  [D] Che do  [Q] Thoat"
        (tw, _), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(bar, right_txt, (max(230, width - tw - 12), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 235), 1, cv2.LINE_AA)
        return bar

    def render_and_poll(self, state, cam_rig, sign_info):
        """
        Ve giao dien quan sat 2 Camera (CAM 1 Lan duong | CAM 2 Bien bao YOLO11n)
        va xu ly su kien chuot/phim. Tra ve False neu nguoi dung bam thoat ('q' / ESC).
        """
        if not self.available:
            return True

        now = time.time()
        with state.cam_lock:
            l_cam_fps = state.lane_cam_fps
            s_cam_fps = state.sign_cam_fps
            fov_pct = state.cam_fov_pct
            lane_cam_ok = state.lane_cam_ok
            sign_cam_ok = state.sign_cam_ok
            shared_cam = state.shared_cam
            raw_lane_frame = state.lane_latest_frame.copy() if state.lane_latest_frame is not None else None
            raw_sign_frame = state.sign_latest_frame.copy() if state.sign_latest_frame is not None else None

        with state.lane_lock:
            lane_vis = state.debug_lane_img.copy() if state.debug_lane_img is not None else raw_lane_frame
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

        # Khung CAM 2 (Bien bao): uu tien ve khung xac nhan len anh goc cua Cam 2 (khong de len Cam 1!)
        sign_vis = raw_sign_frame.copy() if raw_sign_frame is not None else (
            sign_dbg["yolo_vis"].copy() if ("yolo_vis" in sign_dbg and sign_dbg["yolo_vis"] is not None) else None
        )
        if sign_vis is not None:
            # Ve nhe cac ung vien duoi nguong (mau cam dut) de de quan sat
            for prev in sign_dbg.get("yolo_previews", []):
                if not prev.get("accepted", False):
                    bx, by, bw, bh = prev["box"]
                    cv2.rectangle(sign_vis, (bx, by), (bx + bw, by + bh), (0, 150, 255), 1, cv2.LINE_AA)
            # Ve dam cac bien bao da xac nhan on dinh qua SignTracker
            for cls_name, x, y, w, h, score in signs:
                info = sign_info.get(cls_name, {"vn": cls_name, "color": (0, 255, 0)})
                col = info["color"]
                label = f"{info['vn']} ({score:.0%})"
                cv2.rectangle(sign_vis, (x, y), (x + w, y + h), col, 2, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                tx = max(4, min(x, sign_vis.shape[1] - tw - 10))
                ty = max(th + 8, y - 5)
                cv2.rectangle(sign_vis, (tx - 2, ty - th - 5), (tx + tw + 6, ty + 3), (18, 22, 28), -1)
                cv2.putText(sign_vis, label, (tx + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

        lane_cam_lbl = cam_rig.get_lane_label()
        sign_cam_lbl = cam_rig.get_sign_label()
        conf = float(l_info.get("confidence", 0.0))
        cam_h = self.CAM_H if self.detail_mode else self.OBS_CAM_H

        cam1_panel = self._build_camera_viewport(
            lane_vis, self.CAM_W, cam_h,
            title="CAM 1: QUAN SAT LAN DUONG",
            cam_label=lane_cam_lbl,
            accent=(0, 225, 140),
            status_dot_col=(0, 235, 130) if lane_cam_ok else (70, 70, 255),
            status_text="LIVE" if lane_cam_ok else "MAT TIN HIEU",
            hud_left=f"Trang thai: {l_state} | Sang: {light_mode} | Tin cay: {conf:.0%}",
            hud_right=f"Err: {err_val:+.2f}",
            placeholder="Dang cho Camera 1 (Lan duong)...",
        )

        sign_hud_left = f"YOLO11n | Phat hien: {len(signs)} bien" if sign_dbg else "Chua nap model yolo11n_10signs.onnx"
        sign_hud_right = sign_action if sign_action else "Khong co lenh uu tien"
        cam2_panel = self._build_camera_viewport(
            sign_vis, self.CAM_W, cam_h,
            title="CAM 2: QUAN SAT BIEN BAO (YOLO11n)",
            cam_label=sign_cam_lbl,
            accent=(0, 195, 255),
            status_dot_col=(255, 200, 60) if shared_cam else ((0, 220, 255) if sign_cam_ok else (70, 70, 255)),
            status_text="CHUNG CAM 1" if shared_cam else ("LIVE" if sign_cam_ok else "MAT TIN HIEU"),
            hud_left=sign_hud_left,
            hud_right=sign_hud_right,
            placeholder="Dang cho Camera 2 (Bien bao)...",
        )

        cams_row = np.hstack([cam1_panel, cam2_panel])
        top_bar = self._build_top_bar(
            self.TOTAL_W, lane_cam_lbl, sign_cam_lbl, fov_pct,
            lane_cam_ok, sign_cam_ok, shared_cam, esp_ok,
            l_cam_fps, s_cam_fps, l_fps, s_fps,
        )
        bot_bar = self._build_bottom_bar(self.TOTAL_W, pkt, l_state, sign_action)

        if self.detail_mode:
            left_lower = self._build_lane_pipeline_panel(lane_thres, lane_warp, lane_hist, l_info)
            right_lower = self._build_sign_and_control_panel(
                signs, sign_dbg, sign_info, sign_action,
                l_state, l_info, c_info, light_mode,
                curve_val, speed_pct, err_val, cmd_str,
                lane_age_ms, sign_age_ms, esp_ok, esp_port,
            )
            lower_row = np.hstack([left_lower, right_lower])
            dashboard = np.vstack([top_bar, cams_row, lower_row, bot_bar])
        else:
            hud_strip = self._build_compact_hud_strip(speed_pct, err_val, curve_val, cmd_str, l_state, sign_action)
            dashboard = np.vstack([top_bar, cams_row, hud_strip, bot_bar])

        act = self._pending_action
        if act is not None:
            self._pending_action = None
            if act == "switch_lane_cam":
                cam_rig.switch_lane_camera()
            elif act == "switch_sign_cam":
                cam_rig.switch_sign_camera()
                with state.cam_lock:
                    state.shared_cam = cam_rig.shared_mode
            elif act == "swap_cams":
                cam_rig.swap_cameras()
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
                cam_rig.switch_lane_camera()
            elif key == ord("v"):
                cam_rig.switch_sign_camera()
                with state.cam_lock:
                    state.shared_cam = cam_rig.shared_mode
            elif key == ord("x"):
                cam_rig.swap_cameras()
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
