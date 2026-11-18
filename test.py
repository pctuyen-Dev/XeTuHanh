import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
P1_DIR = os.path.join(BASE_DIR, "P1.NhanDienLanDuong")
P2_DIR = os.path.join(BASE_DIR, "P2.NhanDienBienBao")
IMG_DIR = os.path.join(BASE_DIR, "test_images")
VID_DIR = os.path.join(BASE_DIR, "test_videos")

for d in (BASE_DIR, P1_DIR, P2_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import cameraModule
import utils
from LaneDetectionModule import LaneDetector

try:
    from MotorModule import decide_pi_to_esp32_command
    MOTOR_LOGIC_AVAILABLE = True
except Exception:
    MOTOR_LOGIC_AVAILABLE = False

try:
    from TrafficSignDetectionModule import (
        preprocess_lighting,
        compute_sign_color_masks,
        get_boxes_from_mask,
        detect_traffic_signs,
        load_sign_model,
        SignTracker,
        SIGN_INFO,
        CLASSES_15,
    )
    SIGN_MODULE_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Khong the tai module TrafficSignDetectionModule: {e}")
    SIGN_MODULE_AVAILABLE = False
    SIGN_INFO = {}
    CLASSES_15 = []


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
HIGHWAY_KEYWORDS = (
    "solid", "white", "yellow", "challenge", "demo",
    "night", "seoul", "driving", "sample", "lane", "highway", "road"
)

PRESET_HIGHWAY = {
    "name": "Duong cao toc / Xe that",
    "w_top": 140, "h_top": 145, "w_bot": 25, "h_bot": 230,
    "h_min": 0, "h_max": 179, "s_min": 0, "s_max": 50, "v_min": 185, "v_max": 255,
}

PRESET_ROBOT_TOY = {
    "name": "Xe Robot san nha",
    "w_top": 105, "h_top": 105, "w_bot": 20, "h_bot": 230,
    "h_min": 0, "h_max": 179, "s_min": 0, "s_max": 85, "v_min": 170, "v_max": 255,
}


def is_highway_source(src):
    if not isinstance(src, str):
        return False
    name = os.path.basename(src).lower()
    if "videoline" in name or "robot" in name or "san" in name:
        return False
    return any(k in name for k in HIGHWAY_KEYWORDS)


class PiCameraCapture:
    def __init__(self, size):
        self.size = size

    def read(self):
        frame = cameraModule.getImg(size=self.size)
        return frame is not None, frame

    def release(self):
        cameraModule.release_camera()


def draw_fps_badge(img, text, text_color=(0, 255, 120), border_color=(0, 200, 100), y_offset=5):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42 if w >= 360 else 0.35
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 7, 4
    x2 = w - 6
    x1 = max(4, x2 - tw - pad_x * 2)
    y1 = y_offset
    y2 = y1 + th + pad_y * 2
    cv2.rectangle(img, (x1, y1), (x2, y2), (18, 22, 28), cv2.FILLED)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)
    cv2.putText(img, text, (x1 + pad_x, y2 - pad_y), font, scale, text_color, thickness, cv2.LINE_AA)


def build_intermediate_grid(current_frame, img_thres, img_warp, img_hist,
                            detected_signs, sign_debug, is_highway, wT, hT,
                            lane_fps=0.0, sign_fps=0.0, boundary_traces=None):
    """Tao luoi 8 buoc xu ly trung gian (4 buoc Lan duong + 4 buoc Bien bao)."""
    cw, ch = wT // 2, hT
    cell_size = (cw, ch)

    contrast, _, _ = utils.compute_glare_contrast(current_frame)
    contrast_vis = cv2.applyColorMap(cv2.convertScaleAbs(contrast, alpha=3.0), cv2.COLORMAP_TURBO)
    p_l1 = cv2.resize(contrast_vis, cell_size)

    thres_bgr = cv2.cvtColor(img_thres, cv2.COLOR_GRAY2BGR)
    thres_bgr[img_thres > 0] = (180, 255, 180)
    p_l2 = cv2.resize(thres_bgr, cell_size)

    p_l3 = cv2.resize(img_warp, cell_size)
    for gx in range(cw // 4, cw, cw // 4):
        cv2.line(p_l3, (gx, 0), (gx, ch), (60, 60, 60), 1)

    hist_vis = img_hist.copy()
    if boundary_traces:
        for tr in boundary_traces:
            pts = np.asarray(tr, dtype=np.int32)
            cv2.polylines(hist_vis, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        skel = utils.thin_mask(img_thres) if not is_highway else cv2.Canny(img_thres, 50, 150)
        hist_vis[skel > 0] = (0, 255, 255)
    p_l4 = cv2.resize(hist_vis, cell_size)

    if SIGN_MODULE_AVAILABLE:
        if sign_debug and "proc_img" in sign_debug:
            clahe_img = sign_debug["proc_img"]
            mask_r = sign_debug["mask_r"]
            mask_b = sign_debug["mask_b"]
            mask_clean = sign_debug["mask_clean"]
            candidate_boxes = sign_debug.get("candidate_boxes", [])
            onnx_previews = sign_debug.get("onnx_previews", [])
        else:
            clahe_img, l_mode = preprocess_lighting(current_frame, return_mode=True)
            mask_r, mask_b, mask_clean = compute_sign_color_masks(
                current_frame, use_clahe=True, precomputed_lighting=(clahe_img, l_mode)
            )
            candidate_boxes = get_boxes_from_mask(mask_clean, image=clahe_img, verify_geometry=True)
            onnx_previews = []

        src_h, src_w = clahe_img.shape[:2]
        sx = wT / float(max(1, src_w))
        sy = hT / float(max(1, src_h))

        p_b1 = cv2.resize(clahe_img, cell_size)

        hsv_mask_vis = np.zeros_like(clahe_img)
        hsv_mask_vis[mask_r > 0] = (0, 60, 255)
        hsv_mask_vis[mask_b > 0] = (255, 150, 0)
        p_b2 = cv2.resize(hsv_mask_vis, cell_size)

        contour_vis = current_frame.copy()
        mask_clean_resized = cv2.resize(mask_clean, (wT, hT), interpolation=cv2.INTER_NEAREST) if (src_w != wT or src_h != hT) else mask_clean
        cnts, _ = cv2.findContours(mask_clean_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = [c for c in cnts if cv2.contourArea(c) >= 35]
        cv2.drawContours(contour_vis, valid_cnts, -1, (0, 200, 255), 1)
        for bx, by, bw, bh in candidate_boxes:
            x1, y1 = int(bx * sx), int(by * sy)
            x2, y2 = int((bx + bw) * sx), int((by + bh) * sy)
            cv2.rectangle(contour_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        p_b3 = cv2.resize(contour_vis, cell_size)

        p_b4 = np.full((ch, cw, 3), 24, dtype=np.uint8)
        items_to_render = []
        if detected_signs:
            for cls_name, bx, by, bw, bh, sc in detected_signs[:3]:
                x0, y0 = max(0, int(bx)), max(0, int(by))
                x1, y1 = min(src_w, int(bx + bw)), min(src_h, int(by + bh))
                items_to_render.append((cls_name, float(sc), clahe_img[y0:y1, x0:x1], True))
        elif onnx_previews:
            for prev in onnx_previews[:3]:
                items_to_render.append((prev["cls_name"], prev["score"], prev["crop_bgr"], prev["accepted"]))

        if items_to_render:
            for idx, (cls_name, sc, sub, is_ok) in enumerate(items_to_render):
                if sub is not None and sub.size > 0:
                    crop_32 = cv2.resize(sub, (32, 32), interpolation=cv2.INTER_AREA)
                    crop_disp = cv2.resize(crop_32, (56, 56), interpolation=cv2.INTER_NEAREST)
                    top_y = 34 + idx * 66
                    if top_y + 56 < ch:
                        p_b4[top_y:top_y + 56, 10:66] = crop_disp
                        border_c = (0, 255, 120) if is_ok else (0, 165, 255)
                        cv2.rectangle(p_b4, (10, top_y), (66, top_y + 56), border_c, 1)
                        vn_name = SIGN_INFO.get(cls_name, {"vn": cls_name})["vn"]
                        cv2.putText(p_b4, "32x32 RGB -> ONNX", (72, top_y + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1)
                        cv2.putText(p_b4, f"{vn_name}", (72, top_y + 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, border_c, 1)
                        cv2.putText(p_b4, f"Conf: {sc:.0%}", (72, top_y + 52),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1)
        else:
            cv2.putText(p_b4, "Khong co bien bao", (cw // 2 - 62, ch // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 130, 130), 1)
    else:
        blank = np.zeros((ch, cw, 3), dtype=np.uint8)
        p_b1, p_b2, p_b3, p_b4 = blank.copy(), blank.copy(), blank.copy(), blank.copy()

    lane_cells = [
        (p_l1, "L1. Top-Hat Khu Bong"),
        (p_l2, "L2. Mat Na Nhi Phan"),
        (p_l3, "L3. Bird's Eye Warp"),
        (p_l4, "L4. Xuong & Histogram"),
    ]
    sign_cells = [
        (p_b1, "B1. Can Bang Sang/Toi"),
        (p_b2, "B2. Mask Mau Do/Xanh"),
        (p_b3, "B3. Loc Contour & ROI"),
        (p_b4, "B4. Crop 32x32 ONNX"),
    ]

    for idx, (cell, title) in enumerate(lane_cells):
        cv2.rectangle(cell, (0, 0), (cw - 1, 24), (20, 25, 20), cv2.FILLED)
        cv2.rectangle(cell, (0, 0), (cw - 1, ch - 1), (70, 110, 70), 1)
        cv2.putText(cell, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 255, 120), 1, cv2.LINE_AA)
        if idx == 3 and lane_fps > 0:
            draw_fps_badge(cell, f"{lane_fps:.1f} FPS", y_offset=2)

    for idx, (cell, title) in enumerate(sign_cells):
        cv2.rectangle(cell, (0, 0), (cw - 1, 24), (25, 20, 30), cv2.FILLED)
        cv2.rectangle(cell, (0, 0), (cw - 1, ch - 1), (70, 90, 130), 1)
        cv2.putText(cell, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 210, 255), 1, cv2.LINE_AA)
        if idx == 3 and sign_fps > 0:
            draw_fps_badge(cell, f"{sign_fps:.1f} FPS", text_color=(0, 215, 255), border_color=(0, 180, 255), y_offset=2)

    return np.vstack([np.hstack([c for c, _ in lane_cells]), np.hstack([c for c, _ in sign_cells])])


class XeTuHanhTestApp:
    """Giao dien Desktop truc quan All-in-One cho kiem thu Lan duong & Bien bao."""

    BG_DARK = "#0b1320"
    BG_PANEL = "#111b2d"
    BG_CARD = "#162238"
    BG_BORDER = "#243552"
    ACCENT_BLUE = "#1d6fe8"
    ACCENT_GREEN = "#1f9d55"
    ACCENT_CYAN = "#00d4ff"
    TEXT_PRIMARY = "#eef4ff"
    TEXT_MUTED = "#8fa3c4"

    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.wT = args.width
        self.hT = args.height

        self.root.title("Hệ Thống Kiểm Thử Xe Tự Hành - Làn Đường & Biển Báo")
        self.root.configure(bg=self.BG_DARK)
        self.root.geometry("1260x820")
        self.root.minsize(1040, 680)

        # Trang thai xu ly
        self.cap = None
        self.video_source = None
        self.is_static_image = False
        self.static_img_raw = None
        self.current_frame = None
        self.last_result_bgr = None

        self.paused = False
        self.cam_fov_pct = 100
        self.fps = 0.0
        self.lane_fps = 0.0
        self.sign_fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_proc_ms = 0.0
        self.last_terminal_sign_sig = None
        self.last_table_sig = None

        self.detector = LaneDetector(avg_window=8)
        self.current_preset = PRESET_ROBOT_TOY
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="PerceptionWorker")
        self.async_sign_future = None
        self.latest_detected_signs = []
        self.latest_sign_debug = {}

        # Nap model bien bao
        self.sign_model = None
        self.sign_tracker = None
        if SIGN_MODULE_AVAILABLE:
            try:
                self.sign_model = load_sign_model()
                self.sign_tracker = SignTracker(history_size=5, min_hits=2, max_age=3)
                print(f"[INFO] Da nap thanh cong Model ONNX ({len(CLASSES_15) - 1} Bien bao) + SignTracker!")
            except Exception as e:
                print(f"[WARNING] Loi khoi tao model bien bao: {e}")

        # Bien giao dien
        self.var_cam_idx = tk.StringVar(value="Cam 1")
        self.var_preset = tk.StringVar(value="Robot sân nhà")
        self.var_view_mode = tk.StringVar(value="Biển báo + Làn đường")
        self.var_roi_mode = tk.StringVar(value="Mở rộng cua")
        self.var_enable_lane = tk.BooleanVar(value=True)
        self.var_enable_sign = tk.BooleanVar(value=True)

        self._photo_left = None
        self._photo_right = None

        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()

        # Tu dong mo nguon neu truyen tu CLI, hoac mo san anh 21 bien bao neu co
        self._init_startup_source()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(20, self._update_loop)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.TCombobox",
            fieldbackground=self.BG_CARD,
            background=self.BG_CARD,
            foreground=self.TEXT_PRIMARY,
            bordercolor=self.BG_BORDER,
            arrowcolor=self.TEXT_PRIMARY,
            selectbackground=self.ACCENT_BLUE,
            selectforeground="#ffffff",
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", self.BG_CARD)],
            foreground=[("readonly", self.TEXT_PRIMARY)],
        )
        style.configure(
            "Sign.Treeview",
            background="#0e1726",
            foreground=self.TEXT_PRIMARY,
            fieldbackground="#0e1726",
            borderwidth=0,
            rowheight=26,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Sign.Treeview.Heading",
            background="#192842",
            foreground="#b9d1f5",
            borderwidth=1,
            relief="flat",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Sign.Treeview", background=[("selected", "#214578")], foreground=[("selected", "#ffffff")])

    def _build_ui(self):
        # 1. HEADER + TOOLBAR
        header = tk.Frame(self.root, bg=self.BG_DARK, pady=6, padx=12)
        header.pack(fill="x")

        title_row = tk.Frame(header, bg=self.BG_DARK)
        title_row.pack(fill="x", pady=(0, 6))
        tk.Label(
            title_row, text="🚗  Kiểm thử Xe Tự Hành — Làn Đường & Biển Báo",
            font=("Segoe UI", 13, "bold"), fg=self.TEXT_PRIMARY, bg=self.BG_DARK
        ).pack(side="left")

        self.lbl_source_pill = tk.Label(
            title_row, text="Nguồn: Chưa chọn",
            font=("Segoe UI", 9, "bold"), fg=self.ACCENT_CYAN, bg=self.BG_CARD,
            padx=10, pady=3, relief="flat"
        )
        self.lbl_source_pill.pack(side="right")

        toolbar = tk.Frame(header, bg=self.BG_PANEL, padx=8, pady=7, highlightbackground=self.BG_BORDER, highlightthickness=1)
        toolbar.pack(fill="x")

        btn_cfg = {"font": ("Segoe UI", 9, "bold"), "bd": 0, "padx": 10, "pady": 5, "cursor": "hand2", "relief": "flat"}

        # Nhom nut chon nguon dau vao
        tk.Button(toolbar, text="📁 Chọn Ảnh", bg=self.ACCENT_BLUE, fg="white", activebackground="#2b7ef7",
                  command=self.action_pick_image, **btn_cfg).pack(side="left", padx=(0, 5))
        tk.Button(toolbar, text="🎞 Chọn Video", bg="#6f42c1", fg="white", activebackground="#8458d6",
                  command=self.action_pick_video, **btn_cfg).pack(side="left", padx=(0, 5))
        tk.Button(toolbar, text="📷 Mở Camera", bg=self.ACCENT_GREEN, fg="white", activebackground="#27b867",
                  command=self.action_open_camera, **btn_cfg).pack(side="left", padx=(0, 4))

        cb_cam = ttk.Combobox(toolbar, textvariable=self.var_cam_idx, values=["Cam 0", "Cam 1", "Cam 2", "Cam 3"],
                              width=7, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 9))
        cb_cam.pack(side="left", padx=(0, 6))
        cb_cam.bind("<<ComboboxSelected>>", lambda e: self.action_open_camera())

        tk.Button(toolbar, text="🪧 Bảng 21 Biển", bg="#d97706", fg="white", activebackground="#f59e0b",
                  command=self.action_open_board_21, **btn_cfg).pack(side="left", padx=(0, 5))
        tk.Button(toolbar, text="💾 Lưu Ảnh", bg=self.BG_CARD, fg=self.TEXT_PRIMARY, activebackground="#243552",
                  command=self.action_save_snapshot, **btn_cfg).pack(side="left", padx=(0, 10))

        # Divider
        tk.Frame(toolbar, bg=self.BG_BORDER, width=1, height=24).pack(side="left", padx=6)

        # Combobox che do
        tk.Label(toolbar, text="Chế độ:", fg=self.TEXT_MUTED, bg=self.BG_PANEL, font=("Segoe UI", 9)).pack(side="left", padx=(2, 4))
        cb_preset = ttk.Combobox(toolbar, textvariable=self.var_preset, values=["Robot sân nhà", "Đường cao tốc"],
                                 width=14, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 9))
        cb_preset.pack(side="left", padx=(0, 8))
        cb_preset.bind("<<ComboboxSelected>>", lambda e: self.action_change_preset())

        tk.Label(toolbar, text="Hiển thị:", fg=self.TEXT_MUTED, bg=self.BG_PANEL, font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        cb_view = ttk.Combobox(
            toolbar, textvariable=self.var_view_mode,
            values=["Biển báo + Làn đường", "Chỉ Biển báo", "Chỉ Làn đường", "8 Bước Trung Gian"],
            width=19, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 9)
        )
        cb_view.pack(side="left", padx=(0, 8))

        tk.Label(toolbar, text="ROI:", fg=self.TEXT_MUTED, bg=self.BG_PANEL, font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        cb_roi = ttk.Combobox(toolbar, textvariable=self.var_roi_mode, values=["Mở rộng cua", "Chỉ trong ROI"],
                              width=12, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 9))
        cb_roi.pack(side="left", padx=(0, 6))
        cb_roi.bind("<<ComboboxSelected>>", lambda e: self.detector.reset())

        # FOV +/-
        tk.Button(toolbar, text="-", bg=self.BG_CARD, fg="white", width=2, bd=0, cursor="hand2",
                  command=lambda: self._adjust_fov(-10)).pack(side="left", padx=(2, 1))
        self.lbl_fov = tk.Label(toolbar, text="FOV:100%", fg=self.ACCENT_CYAN, bg=self.BG_CARD,
                                font=("Segoe UI", 8, "bold"), padx=6, pady=4)
        self.lbl_fov.pack(side="left")
        tk.Button(toolbar, text="+", bg=self.BG_CARD, fg="white", width=2, bd=0, cursor="hand2",
                  command=lambda: self._adjust_fov(10)).pack(side="left", padx=(1, 0))

        # 2. KHUNG CHINH 2 MAN HINH (ANH GOC | KET QUA)
        views_frame = tk.Frame(self.root, bg=self.BG_DARK, padx=12)
        views_frame.pack(fill="both", expand=True, pady=(4, 6))
        views_frame.columnconfigure(0, weight=1)
        views_frame.columnconfigure(1, weight=1)
        views_frame.rowconfigure(0, weight=1)

        # Khung Trai: Anh goc & ROI
        left_card = tk.Frame(views_frame, bg=self.BG_PANEL, highlightbackground=self.BG_BORDER, highlightthickness=1)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_hdr = tk.Frame(left_card, bg="#152238", height=28)
        left_hdr.pack(fill="x")
        tk.Label(left_hdr, text="Ảnh gốc & Vùng ROI", font=("Segoe UI", 9, "bold"), fg=self.TEXT_PRIMARY, bg="#152238", padx=10, pady=4).pack(side="left")
        self.lbl_light_badge = tk.Label(left_hdr, text="Ánh sáng: NORMAL", font=("Segoe UI", 8, "bold"), fg="#fcd34d", bg="#152238", padx=10)
        self.lbl_light_badge.pack(side="right")

        self.canvas_left = tk.Label(left_card, bg="#080e18", text="Chọn Ảnh, Video hoặc Mở Camera ở thanh trên",
                                    fg=self.TEXT_MUTED, font=("Segoe UI", 10))
        self.canvas_left.pack(fill="both", expand=True, padx=6, pady=6)

        # Khung Phai: Ket qua nhan dien
        right_card = tk.Frame(views_frame, bg=self.BG_PANEL, highlightbackground=self.BG_BORDER, highlightthickness=1)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right_hdr = tk.Frame(right_card, bg="#152238", height=28)
        right_hdr.pack(fill="x")
        tk.Label(right_hdr, text="Kết quả Nhận diện (Biển báo & Làn đường)", font=("Segoe UI", 9, "bold"), fg=self.TEXT_PRIMARY, bg="#152238", padx=10, pady=4).pack(side="left")
        self.lbl_fps_badge = tk.Label(right_hdr, text="FPS: 0.0 | Lane: 0.0 | Sign: 0.0", font=("Segoe UI", 8, "bold"), fg="#4ade80", bg="#152238", padx=10)
        self.lbl_fps_badge.pack(side="right")

        self.canvas_right = tk.Label(right_card, bg="#080e18", text="Chưa có dữ liệu xử lý",
                                     fg=self.TEXT_MUTED, font=("Segoe UI", 10))
        self.canvas_right.pack(fill="both", expand=True, padx=6, pady=6)

        # 3. THANH THONG TIN LAN DUONG & DIEU KHIEN XE (TELEMETRY STRIP)
        tele_card = tk.Frame(self.root, bg=self.BG_PANEL, padx=10, pady=6, highlightbackground=self.BG_BORDER, highlightthickness=1)
        tele_card.pack(fill="x", padx=12, pady=(0, 6))

        self.lbl_lane_state = self._make_metric(tele_card, "Trạng thái làn:", "UNKNOWN", "#38bdf8")
        self.lbl_offset = self._make_metric(tele_card, "Độ lệch tâm (e):", "+0.0 px (CÂN GIỮA)", "#4ade80")
        self.lbl_corner = self._make_metric(tele_card, "Góc cua:", "KHÔNG", "#facc15")
        self.lbl_esp_cmd = self._make_metric(tele_card, "Lệnh ESP32 giả lập:", "<0, +0.00, NORM>", "#c084fc")
        self.lbl_sign_override = self._make_metric(tele_card, "Ưu tiên Biển báo:", "Không có", "#fb923c")

        # 4. BANG DANH SACH BIEN BAO PHAT HIEN (GIONG MAU)
        table_card = tk.Frame(self.root, bg=self.BG_PANEL, highlightbackground=self.BG_BORDER, highlightthickness=1)
        table_card.pack(fill="x", padx=12, pady=(0, 6))

        table_hdr = tk.Frame(table_card, bg="#152238")
        table_hdr.pack(fill="x")
        self.lbl_table_title = tk.Label(table_hdr, text="Biển báo phát hiện (0)", font=("Segoe UI", 9, "bold"),
                                        fg=self.TEXT_PRIMARY, bg="#152238", padx=10, pady=4)
        self.lbl_table_title.pack(side="left")

        tk.Checkbutton(table_hdr, text="Nhận diện Biển báo", variable=self.var_enable_sign,
                       bg="#152238", fg=self.TEXT_PRIMARY, selectcolor=self.BG_CARD,
                       activebackground="#152238", activeforeground="white", font=("Segoe UI", 8)).pack(side="right", padx=8)
        tk.Checkbutton(table_hdr, text="Nhận diện Làn đường", variable=self.var_enable_lane,
                       bg="#152238", fg=self.TEXT_PRIMARY, selectcolor=self.BG_CARD,
                       activebackground="#152238", activeforeground="white", font=("Segoe UI", 8)).pack(side="right", padx=8)

        tree_wrap = tk.Frame(table_card, bg=self.BG_PANEL)
        tree_wrap.pack(fill="x", padx=6, pady=6)

        cols = ("idx", "vn", "cls", "action", "center", "bbox", "conf")
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=5, style="Sign.Treeview")
        headings = [
            ("idx", "#", 45, "center"),
            ("vn", "Biển báo", 160, "w"),
            ("cls", "Mã lớp", 130, "center"),
            ("action", "Hành động xe", 230, "w"),
            ("center", "Tọa độ tâm (x, y)", 140, "center"),
            ("bbox", "Bounding Box (x, y, w, h)", 180, "center"),
            ("conf", "Độ chính xác (%)", 120, "center"),
        ]
        for cid, ctext, cwidth, calign in headings:
            self.tree.heading(cid, text=ctext)
            self.tree.column(cid, width=cwidth, anchor=calign, stretch=(cid in ("vn", "action")))

        sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")

        # 6. THANH NUT HANH DONG DUOI CUNG & STATUS BAR
        action_bar = tk.Frame(self.root, bg=self.BG_DARK, pady=4)
        action_bar.pack(fill="x", padx=12)

        btn_center = tk.Frame(action_bar, bg=self.BG_DARK)
        btn_center.pack()
        self.btn_pause = tk.Button(btn_center, text="⏸ Tạm dừng", bg=self.ACCENT_GREEN, fg="white",
                                   activebackground="#27b867", command=self.action_toggle_pause, **btn_cfg)
        self.btn_pause.pack(side="left", padx=5)
        tk.Button(btn_center, text="↻ Đặt lại (Reset)", bg=self.BG_CARD, fg=self.TEXT_PRIMARY,
                  activebackground="#243552", command=self.action_reset, **btn_cfg).pack(side="left", padx=5)

        status_bar = tk.Frame(self.root, bg="#080d17", height=24)
        status_bar.pack(fill="x", side="bottom")
        self.lbl_status_left = tk.Label(status_bar, text="● Sẵn sàng — Chọn nguồn Ảnh, Video hoặc Camera để bắt đầu",
                                        fg="#4ade80", bg="#080d17", font=("Segoe UI", 8), padx=10, pady=3)
        self.lbl_status_left.pack(side="left")
        self.lbl_status_right = tk.Label(status_bar, text="0 biển báo | 0 ms",
                                         fg=self.TEXT_MUTED, bg="#080d17", font=("Segoe UI", 8), padx=10, pady=3)
        self.lbl_status_right.pack(side="right")

    def _make_metric(self, parent, title, val, color):
        box = tk.Frame(parent, bg=self.BG_CARD, padx=8, pady=4, highlightbackground=self.BG_BORDER, highlightthickness=1)
        box.pack(side="left", fill="x", expand=True, padx=3)
        tk.Label(box, text=title, font=("Segoe UI", 8), fg=self.TEXT_MUTED, bg=self.BG_CARD).pack(anchor="w")
        lbl = tk.Label(box, text=val, font=("Segoe UI", 9, "bold"), fg=color, bg=self.BG_CARD)
        lbl.pack(anchor="w")
        return lbl

    def _bind_shortcuts(self):
        self.root.bind("<space>", lambda e: self.action_toggle_pause())
        self.root.bind("<Escape>", lambda e: self._on_close())

    def _init_startup_source(self):
        if self.args.cam >= 0:
            self.var_cam_idx.set(f"Cam {self.args.cam}")
            self.open_media_source(self.args.cam)
        elif self.args.video:
            self.open_media_source(self.args.video)
        elif self.args.video_direct:
            self.open_media_source(self.args.video_direct)
        else:
            # Mo san anh mau dau tien trong DuLieuBienBaoPNG (neu co) de giao dien hien thi ngay truc quan
            png_dir = os.path.join(P2_DIR, "data", "DuLieuBienBaoPNG")
            default_img = os.path.join(png_dir, "ReTrai.png")
            if os.path.exists(default_img):
                self.open_media_source(default_img)

    # --- CAC HANH DONG NUT BAM ---

    def action_pick_image(self):
        init_d = os.path.join(P2_DIR, "data") if os.path.isdir(os.path.join(P2_DIR, "data")) else BASE_DIR
        f = filedialog.askopenfilename(
            title="Chọn ảnh kiểm thử", initialdir=init_d,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if f:
            self.open_media_source(f)

    def action_pick_video(self):
        init_d = VID_DIR if os.path.isdir(VID_DIR) else BASE_DIR
        f = filedialog.askopenfilename(
            title="Chọn video kiểm thử", initialdir=init_d,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if f:
            self.open_media_source(f)

    def action_open_camera(self):
        txt = self.var_cam_idx.get()
        try:
            idx = int(txt.replace("Cam", "").strip())
        except Exception:
            idx = 1
        self.open_media_source(idx)

    def action_open_board_21(self):
        cand_21 = os.path.join(P2_DIR, "data", "21_BienBaoGiaoThong.png")
        cand_15 = os.path.join(IMG_DIR, "BienBao15Test.jpg")
        cand_png = os.path.join(P2_DIR, "data", "DuLieuBienBaoPNG", "ReTrai.png")
        for target in (cand_21, cand_15, cand_png):
            if os.path.exists(target):
                self.open_media_source(target)
                return
        self.action_pick_image()

    def action_save_snapshot(self):
        if self.last_result_bgr is None:
            return
        out_path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả", defaultextension=".png",
            initialfile="ket_qua_nhan_dien.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if out_path:
            cv2.imwrite(out_path, self.last_result_bgr)
            self.lbl_status_left.config(text=f"● Đã lưu ảnh kết quả: {os.path.basename(out_path)}")

    def action_change_preset(self):
        is_hw = self.var_preset.get() == "Đường cao tốc"
        self.current_preset = PRESET_HIGHWAY if is_hw else PRESET_ROBOT_TOY
        self.var_roi_mode.set("Chỉ trong ROI" if is_hw else "Mở rộng cua")
        self.detector.reset()

    def action_toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(
            text="▶ Tiếp tục" if self.paused else "⏸ Tạm dừng",
            bg="#d97706" if self.paused else self.ACCENT_GREEN
        )

    def action_reset(self):
        self.detector.reset()
        if self.sign_tracker is not None:
            self.sign_tracker.reset()
        self.latest_detected_signs = []
        self.latest_sign_debug = {}
        self.last_terminal_sign_sig = None
        self.last_table_sig = None
        self._update_sign_table([])

    def _adjust_fov(self, delta):
        self.cam_fov_pct = max(40, min(100, self.cam_fov_pct + delta))
        self.lbl_fov.config(text=f"FOV:{self.cam_fov_pct}%")

    # --- MO NGUON DAU VAO ---

    def open_media_source(self, src):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.sign_tracker is not None:
            self.sign_tracker.reset()
        self.detector.reset()
        self.latest_detected_signs = []
        self.latest_sign_debug = {}
        self.last_terminal_sign_sig = None
        self.last_table_sig = None
        self.paused = False
        self.btn_pause.config(text="⏸ Tạm dừng", bg=self.ACCENT_GREEN)

        if isinstance(src, int):
            self.is_static_image = False
            model = Path("/proc/device-tree/model")
            if src == 0 and model.is_file() and b"Raspberry Pi" in model.read_bytes():
                if not cameraModule.boot_camera(size=(self.wT, self.hT), max_retries=2):
                    self.lbl_status_left.config(text=f"● Lỗi: Không mở được Pi Camera {src}", fg="#f87171")
                    return False
                self.cap = PiCameraCapture((self.wT, self.hT))
            else:
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(src)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    self.lbl_status_left.config(text=f"● Lỗi: Không tìm thấy Camera {src}", fg="#f87171")
                    return False
                cameraModule.configure_wide_fov_webcam(cap)
                self.cap = cap
            self.video_source = src
            self.lbl_source_pill.config(text=f"Nguồn: Camera {src} (Live)")
            self.lbl_status_left.config(text=f"● Đang chạy Camera {src}", fg="#4ade80")
            return True

        src = str(src).strip('"\'')
        if not os.path.exists(src):
            self.lbl_status_left.config(text=f"● Không tìm thấy file: {src}", fg="#f87171")
            return False

        if src.lower().endswith(IMAGE_EXTENSIONS):
            raw_img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if raw_img is None:
                return False
            if raw_img.ndim == 3 and raw_img.shape[2] == 4:
                alpha = raw_img[:, :, 3:4].astype(np.float32) / 255.0
                bgr = raw_img[:, :, :3].astype(np.float32)
                img = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
                # Them vien trang nho neu anh bieu tuong chiem gan het khung hinh (> 80%)
                ys, xs = np.nonzero(raw_img[:, :, 3] > 20)
                if xs.size > 0 and ((xs.max() - xs.min()) > 0.78 * img.shape[1] or (ys.max() - ys.min()) > 0.78 * img.shape[0]):
                    pad = int(max(img.shape[:2]) * 0.14)
                    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            else:
                img = raw_img[:, :, :3] if raw_img.ndim == 3 else cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
            self.static_img_raw = img
            self.is_static_image = True
            self.video_source = src
            self.lbl_source_pill.config(text=f"Nguồn: Ảnh ({os.path.basename(src)})")
            self.lbl_status_left.config(text=f"● Đang phân tích ảnh: {os.path.basename(src)}", fg="#4ade80")
            return True

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.lbl_status_left.config(text=f"● Không mở được video: {os.path.basename(src)}", fg="#f87171")
            return False
        self.cap = cap
        self.is_static_image = False
        self.video_source = src

        is_hw = is_highway_source(src)
        self.var_preset.set("Đường cao tốc" if is_hw else "Robot sân nhà")
        self.action_change_preset()
        self.lbl_source_pill.config(text=f"Nguồn: Video ({os.path.basename(src)})")
        self.lbl_status_left.config(text=f"● Đang phát video: {os.path.basename(src)}", fg="#4ade80")
        return True

    def _get_roi_and_hsv(self):
        wT, hT = self.wT, self.hT
        p = self.current_preset
        wt, ht, wb, hb = p["w_top"], p["h_top"], p["w_bot"], p["h_bot"]
        points = np.float32([(wt, ht), (wT - wt, ht), (wb, hb), (wT - wb, hb)])
        hsv_range = (
            np.array([p["h_min"], p["s_min"], p["v_min"]]),
            np.array([p["h_max"], p["s_max"], p["v_max"]]),
        )
        return points, hsv_range

    def _run_sign_branch(self, frame_bgr, raw_static):
        t0_s = time.perf_counter()
        dets, dbg = [], {}
        if SIGN_MODULE_AVAILABLE and self.var_enable_sign.get() and self.sign_model is not None:
            if self.is_static_image and raw_static is not None:
                dets, dbg = detect_traffic_signs(
                    raw_static, model=self.sign_model,
                    tracker=None, conf_thresh=0.60, return_scores=True, return_debug=True
                )
            else:
                dets, dbg = detect_traffic_signs(
                    frame_bgr, model=self.sign_model,
                    tracker=self.sign_tracker, conf_thresh=0.60, return_scores=True, return_debug=True
                )
        dt_s = max(1e-4, time.perf_counter() - t0_s)
        return dets, dbg, 1.0 / dt_s

    def _print_signs_to_terminal(self, dets):
        src_name = f"Camera {self.video_source}" if isinstance(self.video_source, int) else os.path.basename(str(self.video_source))
        if self.is_static_image:
            print("\n" + "=" * 74)
            print(f"[KET QUA NHAN DIEN BIEN BAO TREN ANH] Nguon: {src_name}")
            if not dets:
                print(" -> Khong phat hien bien bao nao trong anh.")
            else:
                sorted_dets = sorted(dets, key=lambda d: (d[2] // 120, d[1]))
                print(f" -> Phat hien {len(sorted_dets)} bien bao:")
                for idx, (cls_name, bx, by, bw, bh, score) in enumerate(sorted_dets, start=1):
                    info = SIGN_INFO.get(cls_name, {"vn": cls_name, "action": "N/A"})
                    print(
                        f"    {idx:2d}. [{cls_name:<15}] {info.get('vn', cls_name):<16} | "
                        f"Conf: {score:6.1%} | Action: {info.get('action', ''):<24} | Box: ({bx}, {by}, {bw}, {bh})"
                    )
            print("=" * 74, flush=True)
        elif dets:
            summary = ", ".join(
                f"{SIGN_INFO.get(c, {'vn': c}).get('vn', c)} [{c}] ({sc:.0%})"
                for c, _, _, _, _, sc in dets
            )
            print(f"[BIEN BAO - {src_name}] Phat hien ({len(dets)}): {summary}", flush=True)

    def _update_sign_table(self, dets):
        sig = tuple((d[0], d[1], d[2], d[3], d[4], round(d[5], 3)) for d in dets)
        if sig == self.last_table_sig:
            return
        self.last_table_sig = sig
        for item in self.tree.get_children():
            self.tree.delete(item)

        sorted_dets = sorted(dets, key=lambda d: (d[2] // 120, d[1]))
        self.lbl_table_title.config(text=f"Biển báo phát hiện ({len(sorted_dets)})")
        for idx, (cls_name, bx, by, bw, bh, score) in enumerate(sorted_dets, start=1):
            info = SIGN_INFO.get(cls_name, {"vn": cls_name, "action": "N/A"})
            cx, cy = bx + bw // 2, by + bh // 2
            self.tree.insert(
                "", "end",
                values=(
                    idx,
                    info.get("vn", cls_name),
                    cls_name,
                    info.get("action", ""),
                    f"({cx}, {cy})",
                    f"({bx}, {by}, {bw}, {bh})",
                    f"{score:.1%}",
                ),
            )

    @staticmethod
    def _render_bgr_to_label(label_widget, bgr_img):
        if bgr_img is None or bgr_img.size == 0:
            return None
        lw = max(320, label_widget.winfo_width() - 6)
        lh = max(180, label_widget.winfo_height() - 6)
        ih, iw = bgr_img.shape[:2]
        scale = min(lw / float(iw), lh / float(ih))
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label_widget.configure(image=photo, text="")
        return photo

    def _update_loop(self):
        t_loop_start = time.perf_counter()

        raw_for_sign = None
        if self.is_static_image and self.static_img_raw is not None:
            raw_for_sign = self.static_img_raw
            self.current_frame = cv2.resize(self.static_img_raw, (self.wT, self.hT))
            self.fps = 30.0
        elif self.cap is not None:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    if not isinstance(self.cap, PiCameraCapture):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.detector.reset()
                        ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = cameraModule.apply_fov_width(frame, self.cam_fov_pct)
                    self.current_frame = cv2.resize(frame, (self.wT, self.hT))
                    self.frame_count += 1
                    if self.frame_count % 5 == 0:
                        elapsed = time.time() - self.start_time
                        self.fps = 5.0 / elapsed if elapsed > 0 else 30.0
                        self.start_time = time.time()

        if self.current_frame is not None:
            points, hsv_range = self._get_roi_and_hsv()
            active_profile = "highway" if self.var_preset.get() == "Đường cao tốc" else "robot"
            restrict_roi = (self.var_roi_mode.get() == "Chỉ trong ROI")

            # Nhan dien bien bao
            if not self.var_enable_sign.get():
                self.latest_detected_signs = []
                self.latest_sign_debug = {}
                self.sign_fps = 0.0
            elif self.is_static_image:
                self.latest_detected_signs, self.latest_sign_debug, self.sign_fps = self._run_sign_branch(
                    self.current_frame, raw_for_sign
                )
            else:
                if self.async_sign_future is None:
                    self.async_sign_future = self.executor.submit(self._run_sign_branch, self.current_frame.copy(), None)
                elif self.async_sign_future.done():
                    try:
                        self.latest_detected_signs, self.latest_sign_debug, s_fps_inst = self.async_sign_future.result()
                        self.sign_fps = 0.7 * self.sign_fps + 0.3 * s_fps_inst if self.sign_fps > 0 else s_fps_inst
                    except Exception:
                        pass
                    if not self.paused:
                        self.async_sign_future = self.executor.submit(self._run_sign_branch, self.current_frame.copy(), None)

            # Nhan dien lan duong
            t0_lane = time.perf_counter()
            curve_val, img_roi, img_thres, img_warp, img_hist, img_lane_result = self.detector.process_frame(
                self.current_frame, points, hsv_range,
                use_dual_edge=True, profile=active_profile,
                fresh_observation=(not self.is_static_image and not self.paused),
                restrict_roi=restrict_roi,
            )
            dt_lane = max(1e-4, time.perf_counter() - t0_lane)
            self.lane_fps = 0.75 * self.lane_fps + 0.25 * (1.0 / dt_lane) if self.lane_fps > 0 else (1.0 / dt_lane)

            detected_signs = self.latest_detected_signs
            if SIGN_MODULE_AVAILABLE and self.var_enable_sign.get():
                cur_sig = (
                    (str(self.video_source), tuple((d[0], d[1], d[2], d[3], d[4]) for d in detected_signs))
                    if self.is_static_image
                    else (str(self.video_source), tuple(sorted(d[0] for d in detected_signs)))
                )
                if cur_sig != self.last_terminal_sign_sig:
                    self.last_terminal_sign_sig = cur_sig
                    if self.is_static_image or detected_signs:
                        self._print_signs_to_terminal(detected_signs)

            # Ve khung ket qua bien bao tren anh nguon (giu do phan giai cao neu la anh tinh)
            sign_vis = raw_for_sign.copy() if (self.is_static_image and raw_for_sign is not None) else self.current_frame.copy()
            if SIGN_MODULE_AVAILABLE and self.var_enable_sign.get() and detected_signs:
                for cls_name, bx, by, bw, bh, score in detected_signs:
                    info = SIGN_INFO.get(cls_name, {"vn": cls_name, "color": (0, 255, 0)})
                    col = info.get("color", (0, 255, 0))
                    label = f"{info.get('vn', cls_name)} ({score:.0%})"
                    thick = max(2, sign_vis.shape[1] // 450)
                    fscale = max(0.45, sign_vis.shape[1] / 1200.0)
                    cv2.rectangle(sign_vis, (bx, by), (bx + bw, by + bh), col, thick, cv2.LINE_AA)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
                    tx = max(4, min(bx, sign_vis.shape[1] - tw - 10))
                    ty = max(th + 10, by - 6)
                    cv2.rectangle(sign_vis, (tx - 2, ty - th - 6), (tx + tw + 6, ty + 3), (20, 20, 20), -1)
                    cv2.putText(sign_vis, label, (tx + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 255, 255), thick, cv2.LINE_AA)

            # Chon anh hien thi cho 2 khung (Trai: Anh goc/ROI | Phai: Ket qua)
            left_bgr = raw_for_sign if (self.is_static_image and raw_for_sign is not None) else img_roi
            view_mode = self.var_view_mode.get()
            if view_mode == "Chỉ Biển báo":
                right_bgr = sign_vis
            elif view_mode == "Chỉ Làn đường":
                right_bgr = img_lane_result
            elif view_mode == "8 Bước Trung Gian":
                right_bgr = build_intermediate_grid(
                    self.current_frame, img_thres, img_warp, img_hist,
                    detected_signs, self.latest_sign_debug, active_profile == "highway",
                    self.wT, self.hT, lane_fps=self.lane_fps, sign_fps=self.sign_fps,
                    boundary_traces=self.detector.boundary_traces,
                )
            else:
                # Bien bao + Lan duong ghep canh nhau ro net
                sign_panel = cv2.resize(sign_vis, (self.wT, self.hT))
                lane_panel = cv2.resize(img_lane_result, (self.wT, self.hT))
                right_bgr = np.vstack([sign_panel, lane_panel]) if self.is_static_image else np.hstack([sign_panel, lane_panel])

            self.last_result_bgr = right_bgr
            self._photo_left = self._render_bgr_to_label(self.canvas_left, left_bgr)
            self._photo_right = self._render_bgr_to_label(self.canvas_right, right_bgr)

            # Cap nhat Telemetry Lan duong & Lenh ESP32
            det_lane = self.detector.last_detection
            light_mode = det_lane.get("lighting", {}).get("mode", "NORMAL")
            lane_state = det_lane.get("lane_state", "UNKNOWN")
            e_px = float(det_lane.get("center_offset_px", 0.0))
            e_dir = det_lane.get("offset_direction", "CAN GIUA")
            corner = det_lane.get("corner", {})
            c_dir = corner.get("direction", "UNKNOWN")
            c_zone = corner.get("distance_zone", "NONE")

            self.lbl_light_badge.config(text=f"Ánh sáng: {light_mode}")
            self.lbl_fps_badge.config(text=f"FPS: {self.fps:.1f} | Lane: {self.lane_fps:.1f} | Sign: {self.sign_fps:.1f}")
            self.lbl_lane_state.config(text=lane_state)
            self.lbl_offset.config(text=f"{e_px:+.1f} px ({e_dir})")
            self.lbl_corner.config(text=f"{c_dir} ({c_zone})" if c_dir != "UNKNOWN" else "ĐOẠN THẲNG")

            if MOTOR_LOGIC_AVAILABLE:
                spd, err, cmd, sign_ov = decide_pi_to_esp32_command(
                    curve_val, detected_signs, SIGN_INFO, lane_info=det_lane
                )
                self.lbl_esp_cmd.config(text=f"<{spd}%, {err:+.2f}, {cmd}>")
                self.lbl_sign_override.config(text=sign_ov if sign_ov else "Không có")

            self._update_sign_table(detected_signs)
            self.last_proc_ms = (time.perf_counter() - t_loop_start) * 1000.0
            self.lbl_status_right.config(text=f"{len(detected_signs)} biển báo | {self.last_proc_ms:.0f} ms")

        delay_ms = 60 if (self.is_static_image or self.paused) else 15
        self.root.after(delay_ms, self._update_loop)

    def _on_close(self):
        try:
            self.executor.shutdown(wait=False)
            if self.cap is not None:
                self.cap.release()
        finally:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Kiem thu Lan duong & Bien bao All-in-One")
    parser.add_argument("--video", "-v", type=str, default="", help="Duong dan video/anh")
    parser.add_argument("video_direct", nargs="?", default="", help="Duong dan video/anh truc tiep")
    parser.add_argument("--cam", "-c", type=int, default=-1, help="Index webcam")
    parser.add_argument("--page", type=int, default=1, choices=[1, 2], help="Che do xem ban dau")
    parser.add_argument("--width", type=int, default=480, help="Chieu rong frame (default: 480)")
    parser.add_argument("--height", type=int, default=240, help="Chieu cao frame (default: 240)")
    args = parser.parse_args()

    root = tk.Tk()
    XeTuHanhTestApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
