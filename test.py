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
        detect_traffic_signs,
        load_sign_model,
        SignTracker,
        SIGN_INFO,
        YOLO_CLASSES,
    )
    SIGN_MODULE_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Khong the tai module TrafficSignDetectionModule: {e}")
    SIGN_MODULE_AVAILABLE = False
    SIGN_INFO = {}
    YOLO_CLASSES = []


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
        self._mgr = cameraModule.CameraManager()
        self.ok = self._mgr.boot(size=size, max_retries=2, retry_delay=0.5, allow_csi=True)

    def read(self):
        frame = self._mgr.read_frame(display=False, size=self.size)
        return frame is not None, frame

    def release(self):
        self._mgr.close()


def _decorate_cell(cell, title, accent_text, bg_hdr, border_col):
    ch, cw = cell.shape[:2]
    cv2.rectangle(cell, (0, 0), (cw - 1, 22), bg_hdr, cv2.FILLED)
    cv2.rectangle(cell, (0, 0), (cw - 1, ch - 1), border_col, 1)
    cv2.putText(cell, title, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, accent_text, 1, cv2.LINE_AA)
    return cell


def build_lane_steps_grid(lane_frame, img_thres, img_warp, img_hist, is_highway,
                          wT, hT, boundary_traces=None):
    """Luoi 2x2 hien thi 4 buoc trung gian cua Camera 1 (Lan duong)."""
    cw, ch = wT // 2, hT // 2
    cw, ch = max(240, cw), max(150, ch)
    cell_size = (cw, ch)

    contrast, _, _ = utils.compute_glare_contrast(lane_frame)
    contrast_vis = cv2.applyColorMap(cv2.convertScaleAbs(contrast, alpha=3.0), cv2.COLORMAP_TURBO)
    p1 = _decorate_cell(cv2.resize(contrast_vis, cell_size), "1. Top-Hat Khu Bong", (120, 255, 160), (18, 28, 22), (60, 110, 75))

    thres_bgr = cv2.cvtColor(img_thres, cv2.COLOR_GRAY2BGR)
    thres_bgr[img_thres > 0] = (160, 255, 175)
    p2 = _decorate_cell(cv2.resize(thres_bgr, cell_size), "2. Mat Na Nhi Phan", (120, 255, 160), (18, 28, 22), (60, 110, 75))

    p3 = cv2.resize(img_warp, cell_size)
    for gx in range(cw // 4, cw, cw // 4):
        cv2.line(p3, (gx, 22), (gx, ch), (55, 65, 65), 1)
    p3 = _decorate_cell(p3, "3. Bird-Eye Warp", (120, 255, 160), (18, 28, 22), (60, 110, 75))

    hist_vis = img_hist.copy()
    if boundary_traces:
        for tr in boundary_traces:
            pts = np.asarray(tr, dtype=np.int32)
            cv2.polylines(hist_vis, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        skel = utils.thin_mask(img_thres) if not is_highway else cv2.Canny(img_thres, 50, 150)
        hist_vis[skel > 0] = (0, 255, 255)
    p4 = _decorate_cell(cv2.resize(hist_vis, cell_size), "4. Xuong & Histogram", (120, 255, 160), (18, 28, 22), (60, 110, 75))

    return np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])


def build_sign_steps_grid(sign_frame, detected_signs, sign_debug, wT, hT):
    """Luoi 2x2 hien thi 4 buoc xu ly cua Camera 2 (Bien bao YOLO11n)."""
    cw, ch = max(240, wT // 2), max(160, hT // 2)
    cell_size = (cw, ch)

    p1 = _decorate_cell(cv2.resize(sign_frame, cell_size), "1. Anh Goc Cam 2", (110, 220, 255), (18, 24, 34), (65, 95, 130))

    lb = sign_debug.get("letterbox_img") if sign_debug else None
    p2_raw = cv2.resize(lb, cell_size) if (lb is not None and lb.size > 0) else np.zeros((ch, cw, 3), dtype=np.uint8)
    p2 = _decorate_cell(p2_raw, "2. Letterbox YOLO11n", (110, 220, 255), (18, 24, 34), (65, 95, 130))

    yvis = sign_debug.get("yolo_vis") if sign_debug else None
    p3_raw = cv2.resize(yvis, cell_size) if (yvis is not None and yvis.size > 0) else cv2.resize(sign_frame, cell_size)
    p3 = _decorate_cell(p3_raw, "3. Phat Hien YOLO11n", (110, 220, 255), (18, 24, 34), (65, 95, 130))

    p4 = np.full((ch, cw, 3), (18, 22, 30), dtype=np.uint8)
    previews = sign_debug.get("yolo_previews", []) if sign_debug else []
    items = []
    if detected_signs:
        sh, sw = sign_frame.shape[:2]
        for cls_name, bx, by, bw, bh, sc in detected_signs[:2]:
            x0, y0 = max(0, int(bx)), max(0, int(by))
            x1, y1 = min(sw, int(bx + bw)), min(sh, int(by + bh))
            sub = sign_frame[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else None
            items.append((cls_name, float(sc), sub, True))
    elif previews:
        for prev in previews[:2]:
            items.append((prev["cls_name"], float(prev["score"]), prev["crop_bgr"], bool(prev["accepted"])))

    if items:
        for idx, (cls_name, sc, sub, is_ok) in enumerate(items):
            ty = 28 + idx * 62
            if ty + 54 < ch:
                if sub is not None and sub.size > 0:
                    p4[ty:ty + 54, 10:64] = cv2.resize(sub, (54, 54), interpolation=cv2.INTER_AREA)
                b_col = (0, 240, 130) if is_ok else (0, 165, 255)
                cv2.rectangle(p4, (10, ty), (64, ty + 54), b_col, 1)
                vn_name = SIGN_INFO.get(cls_name, {"vn": cls_name})["vn"]
                st = "NHAN" if is_ok else "THAP"
                cv2.putText(p4, vn_name[:16], (72, ty + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 242, 250), 1, cv2.LINE_AA)
                cv2.putText(p4, f"{sc:.0%} [{st}]", (72, ty + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, b_col, 1, cv2.LINE_AA)
    else:
        msg = "Chua nap YOLO11n" if not sign_debug else "Khong co bien bao"
        cv2.putText(p4, msg, (cw // 2 - 58, ch // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 140, 155), 1, cv2.LINE_AA)
    p4 = _decorate_cell(p4, "4. Crop & Do Tin Cay", (110, 220, 255), (18, 24, 34), (65, 95, 130))

    return np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])


def _load_static_image(path):
    raw_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        return None
    if raw_img.ndim == 3 and raw_img.shape[2] == 4:
        alpha = raw_img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = raw_img[:, :, :3].astype(np.float32)
        img = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        ys, xs = np.nonzero(raw_img[:, :, 3] > 20)
        if xs.size > 0 and ((xs.max() - xs.min()) > 0.78 * img.shape[1] or (ys.max() - ys.min()) > 0.78 * img.shape[0]):
            pad = int(max(img.shape[:2]) * 0.14)
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return img
    return raw_img[:, :, :3] if raw_img.ndim == 3 else cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)


def _open_camera_capture(idx, size):
    model = Path("/proc/device-tree/model")
    if idx == 0 and model.is_file() and b"Raspberry Pi" in model.read_bytes():
        pi_cap = PiCameraCapture(size)
        if pi_cap.ok:
            return pi_cap
        pi_cap.release()
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cameraModule.configure_wide_fov_webcam(cap)
    return cap


class XeTuHanhTestApp:
    """Man hinh Quan sat & Kiem thu 2 Camera Doc lap (Cam 1 Lan duong | Cam 2 Bien bao YOLO11n)."""

    BG_DARK = "#0a101b"
    BG_PANEL = "#111a2b"
    BG_CARD = "#172338"
    BG_BORDER = "#243552"
    ACCENT_BLUE = "#1d6fe8"
    ACCENT_GREEN = "#16a34a"
    ACCENT_CYAN = "#06b6d4"
    ACCENT_AMBER = "#d97706"
    TEXT_PRIMARY = "#eef4ff"
    TEXT_MUTED = "#8fa3c4"

    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.wT = args.width
        self.hT = args.height
        self.sign_wT = 480
        self.sign_hT = 320

        self.root.title("Trạm Quan Sát 2 Camera — Xe Tự Hành (Làn Đường & Biển Báo YOLO11n)")
        self.root.configure(bg=self.BG_DARK)
        self.root.geometry("1320x840")
        self.root.minsize(1100, 700)

        # Nguon CAM 1 (Lan duong)
        self.lane_cap = None
        self.lane_source = None
        self.lane_is_static = False
        self.lane_static_raw = None
        self.lane_frame = None
        self.last_lane_vis = None

        # Nguon CAM 2 (Bien bao YOLO11n)
        self.sign_cap = None
        self.sign_source = None
        self.sign_is_static = False
        self.sign_static_raw = None
        self.sign_frame = None
        self.sign_share_lane = False
        self.last_sign_vis = None

        self.paused = False
        self.cam_fov_pct = 100
        self.lane_fps = 0.0
        self.sign_fps = 0.0
        self.last_proc_ms = 0.0
        self.last_terminal_sign_sig = None
        self.last_table_sig = None

        self.detector = LaneDetector(avg_window=8)
        self.current_preset = PRESET_ROBOT_TOY
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SignWorker")
        self.async_sign_future = None
        self.latest_detected_signs = []
        self.latest_sign_debug = {}

        # Nap model YOLO11n
        self.sign_model = None
        self.sign_tracker = None
        if SIGN_MODULE_AVAILABLE:
            try:
                self.sign_model = load_sign_model()
                self.sign_tracker = SignTracker(history_size=5, min_hits=2, max_age=3)
                print(f"[INFO] Da nap thanh cong Model YOLO11n ({len(self.sign_model.class_names)} Bien bao) + SignTracker!")
            except Exception as e:
                print(f"[WARNING] Loi khoi tao model YOLO11n: {e}")

        # Bien giao dien
        self.var_lane_cam = tk.StringVar(value="Cam 0")
        self.var_sign_cam = tk.StringVar(value="Cam 1")
        self.var_lane_view = tk.StringVar(value="Kết quả Làn")
        self.var_sign_view = tk.StringVar(value="Kết quả YOLO11n")
        self.var_preset = tk.StringVar(value="Robot sân nhà")
        self.var_roi_mode = tk.StringVar(value="Mở rộng cua")
        self.var_enable_lane = tk.BooleanVar(value=True)
        self.var_enable_sign = tk.BooleanVar(value=True)

        self._photo_left = None
        self._photo_right = None

        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()

        self._init_startup_sources()
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
            arrowcolor=self.ACCENT_CYAN,
            bordercolor=self.BG_BORDER,
            lightcolor=self.BG_BORDER,
            darkcolor=self.BG_BORDER,
        )
        style.map("Dark.TCombobox", fieldbackground=[("readonly", self.BG_CARD)], foreground=[("readonly", self.TEXT_PRIMARY)])
        style.configure(
            "Sign.Treeview",
            background="#0e1726",
            foreground=self.TEXT_PRIMARY,
            fieldbackground="#0e1726",
            borderwidth=0,
            rowheight=24,
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
        btn_cfg = {"font": ("Segoe UI", 8, "bold"), "bd": 0, "padx": 8, "pady": 4, "cursor": "hand2", "relief": "flat"}

        # 1. TOP HEADER (TRANG THAI 2 CAM + NUT NHANH)
        header = tk.Frame(self.root, bg=self.BG_DARK, pady=6, padx=12)
        header.pack(fill="x")

        top_row = tk.Frame(header, bg=self.BG_DARK)
        top_row.pack(fill="x", pady=(0, 6))
        tk.Label(
            top_row, text="🛰  TRẠM QUAN SÁT 2 CAMERA — XE TỰ HÀNH",
            font=("Segoe UI", 12, "bold"), fg=self.TEXT_PRIMARY, bg=self.BG_DARK
        ).pack(side="left")

        # Nhom nut thao tac nhanh ben phai header
        quick_bar = tk.Frame(top_row, bg=self.BG_DARK)
        quick_bar.pack(side="right")
        tk.Button(quick_bar, text="📷 Mở 2 Camera", bg=self.ACCENT_GREEN, fg="white", activebackground="#22c55e",
                  command=self.action_open_both_cameras, **btn_cfg).pack(side="left", padx=3)
        tk.Button(quick_bar, text="⇄ Đổi chỗ 2 Cam", bg=self.ACCENT_AMBER, fg="white", activebackground="#f59e0b",
                  command=self.action_swap_cameras, **btn_cfg).pack(side="left", padx=3)
        self.btn_pause = tk.Button(quick_bar, text="⏸ Tạm dừng", bg=self.ACCENT_BLUE, fg="white",
                                   activebackground="#3b82f6", command=self.action_toggle_pause, **btn_cfg)
        self.btn_pause.pack(side="left", padx=3)
        tk.Button(quick_bar, text="↻ Reset", bg=self.BG_CARD, fg=self.TEXT_PRIMARY, activebackground="#243552",
                  command=self.action_reset, **btn_cfg).pack(side="left", padx=3)
        tk.Button(quick_bar, text="💾 Lưu 2 Khung", bg=self.BG_CARD, fg=self.TEXT_PRIMARY, activebackground="#243552",
                  command=self.action_save_snapshot, **btn_cfg).pack(side="left", padx=(3, 0))

        # 2. THANH CONG CU 2 CUM RIENG BIET (CAM 1 LAN DUONG | CAM 2 BIEN BAO)
        ctrl_row = tk.Frame(header, bg=self.BG_DARK)
        ctrl_row.pack(fill="x")
        ctrl_row.columnconfigure(0, weight=1)
        ctrl_row.columnconfigure(1, weight=1)

        # Cum trai: Dieu khien CAM 1 (Lan duong)
        lane_tb = tk.Frame(ctrl_row, bg=self.BG_PANEL, padx=8, pady=5, highlightbackground="#1f5c3a", highlightthickness=1)
        lane_tb.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        tk.Label(lane_tb, text="CAM 1 (Làn):", fg="#4ade80", bg=self.BG_PANEL, font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))
        cb_lcam = ttk.Combobox(lane_tb, textvariable=self.var_lane_cam, values=["Cam 0", "Cam 1", "Cam 2", "Cam 3"],
                               width=6, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 8))
        cb_lcam.pack(side="left", padx=(0, 3))
        cb_lcam.bind("<<ComboboxSelected>>", lambda e: self.action_open_lane_cam())
        tk.Button(lane_tb, text="Mở", bg="#15803d", fg="white", command=self.action_open_lane_cam, **btn_cfg).pack(side="left", padx=(0, 4))
        tk.Button(lane_tb, text="🎞 Video", bg="#4f46e5", fg="white", command=self.action_pick_lane_video, **btn_cfg).pack(side="left", padx=(0, 3))
        tk.Button(lane_tb, text="📁 Ảnh", bg=self.ACCENT_BLUE, fg="white", command=self.action_pick_lane_image, **btn_cfg).pack(side="left", padx=(0, 6))

        cb_lview = ttk.Combobox(lane_tb, textvariable=self.var_lane_view,
                                values=["Kết quả Làn", "4 Bước Làn", "Ảnh gốc + ROI"],
                                width=12, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 8))
        cb_lview.pack(side="left", padx=(0, 4))
        cb_preset = ttk.Combobox(lane_tb, textvariable=self.var_preset, values=["Robot sân nhà", "Đường cao tốc"],
                                 width=12, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 8))
        cb_preset.pack(side="left", padx=(0, 4))
        cb_preset.bind("<<ComboboxSelected>>", lambda e: self.action_change_preset())
        cb_roi = ttk.Combobox(lane_tb, textvariable=self.var_roi_mode, values=["Mở rộng cua", "Chỉ trong ROI"],
                              width=11, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 8))
        cb_roi.pack(side="left")
        cb_roi.bind("<<ComboboxSelected>>", lambda e: self.detector.reset())

        # Cum phai: Dieu khien CAM 2 (Bien bao YOLO11n)
        sign_tb = tk.Frame(ctrl_row, bg=self.BG_PANEL, padx=8, pady=5, highlightbackground="#165a72", highlightthickness=1)
        sign_tb.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        tk.Label(sign_tb, text="CAM 2 (Biển):", fg="#38bdf8", bg=self.BG_PANEL, font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))
        cb_scam = ttk.Combobox(sign_tb, textvariable=self.var_sign_cam,
                               values=["Cam 1", "Cam 0", "Cam 2", "Cam 3", "Chung Cam 1"],
                               width=11, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 8))
        cb_scam.pack(side="left", padx=(0, 3))
        cb_scam.bind("<<ComboboxSelected>>", lambda e: self.action_open_sign_cam())
        tk.Button(sign_tb, text="Mở", bg="#0284c7", fg="white", command=self.action_open_sign_cam, **btn_cfg).pack(side="left", padx=(0, 4))
        tk.Button(sign_tb, text="🎞 Video", bg="#6d28d9", fg="white", command=self.action_pick_sign_video, **btn_cfg).pack(side="left", padx=(0, 3))
        tk.Button(sign_tb, text="📁 Ảnh", bg=self.ACCENT_BLUE, fg="white", command=self.action_pick_sign_image, **btn_cfg).pack(side="left", padx=(0, 3))
        tk.Button(sign_tb, text="🪧 Mẫu", bg=self.ACCENT_AMBER, fg="white", command=self.action_open_sample_sign, **btn_cfg).pack(side="left", padx=(0, 6))

        cb_sview = ttk.Combobox(sign_tb, textvariable=self.var_sign_view,
                                values=["Kết quả YOLO11n", "4 Bước YOLO11n"],
                                width=14, state="readonly", style="Dark.TCombobox", font=("Segoe UI", 8))
        cb_sview.pack(side="left", padx=(0, 6))

        tk.Button(sign_tb, text="-", bg=self.BG_CARD, fg="white", width=2, bd=0, cursor="hand2",
                  command=lambda: self._adjust_fov(-10)).pack(side="left")
        self.lbl_fov = tk.Label(sign_tb, text="FOV:100%", fg=self.ACCENT_CYAN, bg=self.BG_CARD,
                                font=("Segoe UI", 8, "bold"), padx=5, pady=3)
        self.lbl_fov.pack(side="left", padx=1)
        tk.Button(sign_tb, text="+", bg=self.BG_CARD, fg="white", width=2, bd=0, cursor="hand2",
                  command=lambda: self._adjust_fov(10)).pack(side="left")

        # 3. HAI MAN HINH QUAN SAT LON SONG SONG (50% CAM 1 LAN DUONG | 50% CAM 2 BIEN BAO)
        views_frame = tk.Frame(self.root, bg=self.BG_DARK, padx=12)
        views_frame.pack(fill="both", expand=True, pady=(4, 6))
        views_frame.columnconfigure(0, weight=1)
        views_frame.columnconfigure(1, weight=1)
        views_frame.rowconfigure(0, weight=1)

        # MAN HINH TRAI: CAM 1 - LAN DUONG
        left_card = tk.Frame(views_frame, bg=self.BG_PANEL, highlightbackground="#1f5c3a", highlightthickness=1)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_hdr = tk.Frame(left_card, bg="#122620", height=30)
        left_hdr.pack(fill="x")
        self.lbl_cam1_title = tk.Label(
            left_hdr, text="● CAM 1: QUAN SÁT LÀN ĐƯỜNG  |  Chưa chọn nguồn",
            font=("Segoe UI", 9, "bold"), fg="#4ade80", bg="#122620", padx=10, pady=5
        )
        self.lbl_cam1_title.pack(side="left")
        self.lbl_lane_badge = tk.Label(
            left_hdr, text="Sáng: NORMAL  |  0.0 FPS",
            font=("Segoe UI", 8, "bold"), fg="#fcd34d", bg="#122620", padx=10
        )
        self.lbl_lane_badge.pack(side="right")

        self.canvas_left = tk.Label(
            left_card, bg="#070c14",
            text="CAM 1 (Làn đường)\nChọn Camera / Video / Ảnh ở cụm trái phía trên",
            fg=self.TEXT_MUTED, font=("Segoe UI", 10)
        )
        self.canvas_left.pack(fill="both", expand=True, padx=5, pady=5)

        # MAN HINH PHAI: CAM 2 - BIEN BAO YOLO11n
        right_card = tk.Frame(views_frame, bg=self.BG_PANEL, highlightbackground="#165a72", highlightthickness=1)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_hdr = tk.Frame(right_card, bg="#122334", height=30)
        right_hdr.pack(fill="x")
        self.lbl_cam2_title = tk.Label(
            right_hdr, text="● CAM 2: QUAN SÁT BIỂN BÁO (YOLO11n)  |  Chưa chọn nguồn",
            font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#122334", padx=10, pady=5
        )
        self.lbl_cam2_title.pack(side="left")
        self.lbl_sign_badge = tk.Label(
            right_hdr, text="YOLO11n  |  0.0 FPS",
            font=("Segoe UI", 8, "bold"), fg="#4ade80", bg="#122334", padx=10
        )
        self.lbl_sign_badge.pack(side="right")

        self.canvas_right = tk.Label(
            right_card, bg="#070c14",
            text="CAM 2 (Biển báo YOLO11n)\nChọn Camera / Video / Ảnh ở cụm phải phía trên",
            fg=self.TEXT_MUTED, font=("Segoe UI", 10)
        )
        self.canvas_right.pack(fill="both", expand=True, padx=5, pady=5)

        # 4. THANH TELEMETRY DIEU KHIEN XE
        tele_card = tk.Frame(self.root, bg=self.BG_PANEL, padx=8, pady=5, highlightbackground=self.BG_BORDER, highlightthickness=1)
        tele_card.pack(fill="x", padx=12, pady=(0, 6))

        self.lbl_lane_state = self._make_metric(tele_card, "Trạng thái làn (Cam 1):", "UNKNOWN", "#38bdf8")
        self.lbl_offset = self._make_metric(tele_card, "Độ lệch tâm (e):", "+0.0 px (CÂN GIỮA)", "#4ade80")
        self.lbl_corner = self._make_metric(tele_card, "Góc cua:", "ĐOẠN THẲNG", "#facc15")
        self.lbl_esp_cmd = self._make_metric(tele_card, "Lệnh ESP32 giả lập:", "<0%, +0.00, NORM>", "#c084fc")
        self.lbl_sign_override = self._make_metric(tele_card, "Ưu tiên Biển báo (Cam 2):", "Không có", "#fb923c")

        # 5. BANG DANH SACH BIEN BAO PHAT HIEN (GON 4 DONG DE UU TIEN KHONG GIAN 2 MAN HINH CAM)
        table_card = tk.Frame(self.root, bg=self.BG_PANEL, highlightbackground=self.BG_BORDER, highlightthickness=1)
        table_card.pack(fill="x", padx=12, pady=(0, 6))

        table_hdr = tk.Frame(table_card, bg="#152238")
        table_hdr.pack(fill="x")
        self.lbl_table_title = tk.Label(
            table_hdr, text="Biển báo phát hiện từ CAM 2 — YOLO11n (0)",
            font=("Segoe UI", 9, "bold"), fg=self.TEXT_PRIMARY, bg="#152238", padx=10, pady=4
        )
        self.lbl_table_title.pack(side="left")

        tk.Checkbutton(table_hdr, text="Bật YOLO11n (Cam 2)", variable=self.var_enable_sign,
                       bg="#152238", fg=self.TEXT_PRIMARY, selectcolor=self.BG_CARD,
                       activebackground="#152238", activeforeground="white", font=("Segoe UI", 8)).pack(side="right", padx=8)
        tk.Checkbutton(table_hdr, text="Bật Bám làn (Cam 1)", variable=self.var_enable_lane,
                       bg="#152238", fg=self.TEXT_PRIMARY, selectcolor=self.BG_CARD,
                       activebackground="#152238", activeforeground="white", font=("Segoe UI", 8)).pack(side="right", padx=8)

        tree_wrap = tk.Frame(table_card, bg=self.BG_PANEL)
        tree_wrap.pack(fill="x", padx=6, pady=5)

        cols = ("idx", "vn", "cls", "action", "center", "bbox", "conf")
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=4, style="Sign.Treeview")
        headings = [
            ("idx", "#", 40, "center"),
            ("vn", "Biển báo", 150, "w"),
            ("cls", "Mã lớp", 110, "center"),
            ("action", "Hành động xe", 220, "w"),
            ("center", "Tâm (x, y)", 120, "center"),
            ("bbox", "Bounding Box (x, y, w, h)", 175, "center"),
            ("conf", "Độ tin cậy", 105, "center"),
        ]
        for cid, ctext, cwidth, calign in headings:
            self.tree.heading(cid, text=ctext)
            self.tree.column(cid, width=cwidth, anchor=calign, stretch=(cid in ("vn", "action")))

        sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")

        # 6. STATUS BAR DUOI CUNG
        status_bar = tk.Frame(self.root, bg="#070c14", height=24)
        status_bar.pack(fill="x", side="bottom")
        self.lbl_status_left = tk.Label(
            status_bar, text="● Sẵn sàng — Chọn nguồn độc lập cho CAM 1 (Làn đường) và CAM 2 (Biển báo)",
            fg="#4ade80", bg="#070c14", font=("Segoe UI", 8), padx=10, pady=3
        )
        self.lbl_status_left.pack(side="left")
        self.lbl_status_right = tk.Label(
            status_bar, text="0 biển báo | 0 ms",
            fg=self.TEXT_MUTED, bg="#070c14", font=("Segoe UI", 8), padx=10, pady=3
        )
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

    def _init_startup_sources(self):
        if self.args.lane_cam >= 0 or self.args.sign_cam >= 0:
            if self.args.lane_cam >= 0:
                self.var_lane_cam.set(f"Cam {self.args.lane_cam}")
                self.open_lane_source(self.args.lane_cam)
            if self.args.sign_cam >= 0:
                self.var_sign_cam.set(f"Cam {self.args.sign_cam}")
                self.open_sign_source(self.args.sign_cam)
            return

        if self.args.cam >= 0:
            self.var_lane_cam.set(f"Cam {self.args.cam}")
            self.open_lane_source(self.args.cam)
            self.var_sign_cam.set("Chung Cam 1")
            self.open_sign_source("SHARED")
        elif self.args.video or self.args.video_direct:
            src = self.args.video or self.args.video_direct
            self.open_lane_source(src)
            self.var_sign_cam.set("Chung Cam 1")
            self.open_sign_source("SHARED")
        else:
            # Mo san anh mau cho Cam 2 (Bien bao) neu co de giao dien hien thi truc quan ngay
            png_dir = os.path.join(P2_DIR, "data", "DuLieuBienBaoPNG")
            default_img = os.path.join(png_dir, "ReTrai.png")
            if os.path.exists(default_img):
                self.open_sign_source(default_img)

    # --- MO NGUON DOC LAP CHO CAM 1 (LAN DUONG) & CAM 2 (BIEN BAO) ---

    def _close_lane_source(self):
        if self.lane_cap is not None:
            try:
                self.lane_cap.release()
            except Exception:
                pass
            self.lane_cap = None

    def _close_sign_source(self):
        if self.sign_cap is not None:
            try:
                self.sign_cap.release()
            except Exception:
                pass
            self.sign_cap = None

    def open_lane_source(self, src):
        self._close_lane_source()
        self.detector.reset()
        self.lane_frame = None
        self.lane_static_raw = None

        if isinstance(src, int):
            cap = _open_camera_capture(src, (self.wT, self.hT))
            if cap is None:
                self.lbl_status_left.config(text=f"● Lỗi CAM 1: Không mở được Camera {src}", fg="#f87171")
                return False
            self.lane_cap = cap
            self.lane_is_static = False
            self.lane_source = src
            self.lbl_cam1_title.config(text=f"● CAM 1: QUAN SÁT LÀN ĐƯỜNG  |  Camera {src} (LIVE)")
            self.lbl_status_left.config(text=f"● Đã mở CAM 1 (Làn đường): Camera {src}", fg="#4ade80")
            return True

        src = str(src).strip('"\'')
        if not os.path.exists(src):
            self.lbl_status_left.config(text=f"● Không tìm thấy file cho CAM 1: {src}", fg="#f87171")
            return False

        if src.lower().endswith(IMAGE_EXTENSIONS):
            img = _load_static_image(src)
            if img is None:
                return False
            self.lane_static_raw = img
            self.lane_is_static = True
            self.lane_source = src
            self.lbl_cam1_title.config(text=f"● CAM 1: QUAN SÁT LÀN ĐƯỜNG  |  Ảnh: {os.path.basename(src)}")
            return True

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.lbl_status_left.config(text=f"● Không mở được video cho CAM 1: {os.path.basename(src)}", fg="#f87171")
            return False
        self.lane_cap = cap
        self.lane_is_static = False
        self.lane_source = src
        is_hw = is_highway_source(src)
        self.var_preset.set("Đường cao tốc" if is_hw else "Robot sân nhà")
        self.action_change_preset()
        self.lbl_cam1_title.config(text=f"● CAM 1: QUAN SÁT LÀN ĐƯỜNG  |  Video: {os.path.basename(src)}")
        return True

    def open_sign_source(self, src):
        self._close_sign_source()
        if self.sign_tracker is not None:
            self.sign_tracker.reset()
        self.latest_detected_signs = []
        self.latest_sign_debug = {}
        self.last_terminal_sign_sig = None
        self.last_table_sig = None
        self.sign_frame = None
        self.sign_static_raw = None
        self.sign_share_lane = False

        if src == "SHARED":
            self.sign_share_lane = True
            self.sign_is_static = False
            self.sign_source = "SHARED"
            self.lbl_cam2_title.config(text="● CAM 2: QUAN SÁT BIỂN BÁO (YOLO11n)  |  Dùng chung CAM 1")
            self.lbl_status_left.config(text="● CAM 2 đang dùng chung khung hình từ CAM 1", fg="#fcd34d")
            return True

        if isinstance(src, int):
            if isinstance(self.lane_source, int) and self.lane_source == src and self.lane_cap is not None:
                self.sign_share_lane = True
                self.sign_source = "SHARED"
                self.var_sign_cam.set("Chung Cam 1")
                self.lbl_cam2_title.config(text=f"● CAM 2: QUAN SÁT BIỂN BÁO (YOLO11n)  |  Chung Camera {src}")
                return True
            cap = _open_camera_capture(src, (self.sign_wT, self.sign_hT))
            if cap is None:
                self.lbl_status_left.config(text=f"● Lỗi CAM 2: Không mở được Camera {src}", fg="#f87171")
                return False
            self.sign_cap = cap
            self.sign_is_static = False
            self.sign_source = src
            self.lbl_cam2_title.config(text=f"● CAM 2: QUAN SÁT BIỂN BÁO (YOLO11n)  |  Camera {src} (LIVE)")
            self.lbl_status_left.config(text=f"● Đã mở CAM 2 (Biển báo): Camera {src}", fg="#38bdf8")
            return True

        src = str(src).strip('"\'')
        if not os.path.exists(src):
            self.lbl_status_left.config(text=f"● Không tìm thấy file cho CAM 2: {src}", fg="#f87171")
            return False

        if src.lower().endswith(IMAGE_EXTENSIONS):
            img = _load_static_image(src)
            if img is None:
                return False
            self.sign_static_raw = img
            self.sign_is_static = True
            self.sign_source = src
            self.lbl_cam2_title.config(text=f"● CAM 2: QUAN SÁT BIỂN BÁO (YOLO11n)  |  Ảnh: {os.path.basename(src)}")
            return True

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.lbl_status_left.config(text=f"● Không mở được video cho CAM 2: {os.path.basename(src)}", fg="#f87171")
            return False
        self.sign_cap = cap
        self.sign_is_static = False
        self.sign_source = src
        self.lbl_cam2_title.config(text=f"● CAM 2: QUAN SÁT BIỂN BÁO (YOLO11n)  |  Video: {os.path.basename(src)}")
        return True

    # --- CAC HANH DONG NUT BAM ---

    @staticmethod
    def _parse_cam_combo(text, default_idx=0):
        if "Chung" in text:
            return "SHARED"
        try:
            return int(text.replace("Cam", "").strip())
        except Exception:
            return default_idx

    def action_open_lane_cam(self):
        idx = self._parse_cam_combo(self.var_lane_cam.get(), 0)
        if isinstance(idx, int):
            self.open_lane_source(idx)

    def action_open_sign_cam(self):
        src = self._parse_cam_combo(self.var_sign_cam.get(), 1)
        self.open_sign_source(src)

    def action_open_both_cameras(self):
        l_idx = self._parse_cam_combo(self.var_lane_cam.get(), 0)
        s_src = self._parse_cam_combo(self.var_sign_cam.get(), 1)
        ok1 = self.open_lane_source(l_idx if isinstance(l_idx, int) else 0)
        ok2 = self.open_sign_source(s_src)
        if ok1 and not ok2 and s_src != "SHARED":
            self.var_sign_cam.set("Chung Cam 1")
            self.open_sign_source("SHARED")

    def action_swap_cameras(self):
        """Hoan doi ngay lap tuc 2 luong Camera (CAM 1 <-> CAM 2)."""
        if self.sign_share_lane:
            self.lbl_status_left.config(text="● Đang dùng chung 1 camera nên không thể hoán đổi", fg="#fcd34d")
            return
        self.lane_cap, self.sign_cap = self.sign_cap, self.lane_cap
        self.lane_source, self.sign_source = self.sign_source, self.lane_source
        self.lane_is_static, self.sign_is_static = self.sign_is_static, self.lane_is_static
        self.lane_static_raw, self.sign_static_raw = self.sign_static_raw, self.lane_static_raw
        self.detector.reset()
        if self.sign_tracker is not None:
            self.sign_tracker.reset()

        l_txt = f"Camera {self.lane_source} (LIVE)" if isinstance(self.lane_source, int) else (os.path.basename(str(self.lane_source)) if self.lane_source else "Chưa chọn")
        s_txt = f"Camera {self.sign_source} (LIVE)" if isinstance(self.sign_source, int) else (os.path.basename(str(self.sign_source)) if self.sign_source else "Chưa chọn")
        if isinstance(self.lane_source, int):
            self.var_lane_cam.set(f"Cam {self.lane_source}")
        if isinstance(self.sign_source, int):
            self.var_sign_cam.set(f"Cam {self.sign_source}")
        self.lbl_cam1_title.config(text=f"● CAM 1: QUAN SÁT LÀN ĐƯỜNG  |  {l_txt}")
        self.lbl_cam2_title.config(text=f"● CAM 2: QUAN SÁT BIỂN BÁO (YOLO11n)  |  {s_txt}")
        self.lbl_status_left.config(text="● Đã hoán đổi nguồn CAM 1 ↔ CAM 2", fg="#4ade80")

    def action_pick_lane_image(self):
        init_d = IMG_DIR if os.path.isdir(IMG_DIR) else BASE_DIR
        f = filedialog.askopenfilename(title="Chọn ảnh cho CAM 1 (Làn đường)", initialdir=init_d,
                                       filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if f:
            self.open_lane_source(f)

    def action_pick_lane_video(self):
        init_d = VID_DIR if os.path.isdir(VID_DIR) else BASE_DIR
        f = filedialog.askopenfilename(title="Chọn video cho CAM 1 (Làn đường)", initialdir=init_d,
                                       filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All files", "*.*")])
        if f:
            self.open_lane_source(f)

    def action_pick_sign_image(self):
        init_d = os.path.join(P2_DIR, "data") if os.path.isdir(os.path.join(P2_DIR, "data")) else BASE_DIR
        f = filedialog.askopenfilename(title="Chọn ảnh cho CAM 2 (Biển báo)", initialdir=init_d,
                                       filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if f:
            self.open_sign_source(f)

    def action_pick_sign_video(self):
        init_d = VID_DIR if os.path.isdir(VID_DIR) else BASE_DIR
        f = filedialog.askopenfilename(title="Chọn video cho CAM 2 (Biển báo)", initialdir=init_d,
                                       filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All files", "*.*")])
        if f:
            self.open_sign_source(f)

    def action_open_sample_sign(self):
        cand_21 = os.path.join(P2_DIR, "data", "DuLieuBienBaoPNG", "21_BienBaoGiaoThong.png")
        cand_png = os.path.join(P2_DIR, "data", "DuLieuBienBaoPNG", "ReTrai.png")
        for target in (cand_21, cand_png):
            if os.path.exists(target):
                self.open_sign_source(target)
                return
        self.action_pick_sign_image()

    def action_save_snapshot(self):
        imgs = [im for im in (self.last_lane_vis, self.last_sign_vis) if im is not None]
        if not imgs:
            return
        if len(imgs) == 2:
            h = max(imgs[0].shape[0], imgs[1].shape[0])
            w1 = int(round(imgs[0].shape[1] * h / float(imgs[0].shape[0])))
            w2 = int(round(imgs[1].shape[1] * h / float(imgs[1].shape[0])))
            merged = np.hstack([cv2.resize(imgs[0], (w1, h)), cv2.resize(imgs[1], (w2, h))])
        else:
            merged = imgs[0]
        out_path = filedialog.asksaveasfilename(
            title="Lưu ảnh quan sát 2 Camera", defaultextension=".png",
            initialfile="quan_sat_2_camera.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if out_path:
            cv2.imwrite(out_path, merged)
            self.lbl_status_left.config(text=f"● Đã lưu ảnh: {os.path.basename(out_path)}", fg="#4ade80")

    def action_change_preset(self):
        is_hw = self.var_preset.get() == "Đường cao tốc"
        self.current_preset = PRESET_HIGHWAY if is_hw else PRESET_ROBOT_TOY
        self.var_roi_mode.set("Chỉ trong ROI" if is_hw else "Mở rộng cua")
        self.detector.reset()

    def action_toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(
            text="▶ Tiếp tục" if self.paused else "⏸ Tạm dừng",
            bg=self.ACCENT_AMBER if self.paused else self.ACCENT_BLUE
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

    # --- XU LY & HIEN THI ---

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

    def _run_sign_branch(self, frame_bgr, use_tracker=True):
        t0_s = time.perf_counter()
        dets, dbg = [], {}
        if SIGN_MODULE_AVAILABLE and self.var_enable_sign.get() and self.sign_model is not None:
            dets, dbg = detect_traffic_signs(
                frame_bgr,
                model=self.sign_model,
                tracker=(self.sign_tracker if use_tracker else None),
                conf_thresh=0.50,
                return_scores=True,
                return_debug=True,
            )
        dt_s = max(1e-4, time.perf_counter() - t0_s)
        return dets, dbg, 1.0 / dt_s

    def _update_sign_table(self, dets):
        sig = tuple((d[0], int(d[1]), int(d[2]), int(d[3]), int(d[4]), round(float(d[5]), 2)) for d in dets)
        if sig == self.last_table_sig:
            return
        self.last_table_sig = sig
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.lbl_table_title.config(text=f"Biển báo phát hiện từ CAM 2 — YOLO11n ({len(dets)})")
        for idx, (cls_name, bx, by, bw, bh, score) in enumerate(dets, start=1):
            info = SIGN_INFO.get(cls_name, {"vn": cls_name, "action": "N/A"})
            cx, cy = int(bx + bw // 2), int(by + bh // 2)
            self.tree.insert("", "end", values=(
                idx, info.get("vn", cls_name), cls_name, info.get("action", "N/A"),
                f"({cx}, {cy})", f"({bx}, {by}, {bw}, {bh})", f"{score * 100:.1f}%"
            ))

    def _render_bgr_to_label(self, label_widget, bgr_img):
        if bgr_img is None or bgr_img.size == 0:
            return None
        w = max(320, label_widget.winfo_width() - 6)
        h = max(220, label_widget.winfo_height() - 6)
        ih, iw = bgr_img.shape[:2]
        scale = min(w / float(iw), h / float(ih))
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        label_widget.config(image=photo, text="")
        return photo

    def _read_next_frame(self, cap, is_static, static_raw, target_size):
        if is_static and static_raw is not None:
            return static_raw.copy()
        if cap is not None and not self.paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not isinstance(cap, PiCameraCapture):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
            if ret and frame is not None:
                frame = cameraModule.apply_fov_width(frame, self.cam_fov_pct)
                return cv2.resize(frame, target_size)
        return None

    def _update_loop(self):
        t_loop_start = time.perf_counter()

        # 1. Doc frame CAM 1 (Lan duong)
        new_lane = self._read_next_frame(self.lane_cap, self.lane_is_static, self.lane_static_raw, (self.wT, self.hT))
        if new_lane is not None:
            self.lane_frame = cv2.resize(new_lane, (self.wT, self.hT)) if (new_lane.shape[1] != self.wT or new_lane.shape[0] != self.hT) else new_lane

        # 2. Doc frame CAM 2 (Bien bao YOLO11n) doc lap (hoac dung chung CAM 1)
        if self.sign_share_lane:
            if self.lane_frame is not None:
                self.sign_frame = self.lane_frame.copy()
        else:
            new_sign = self._read_next_frame(self.sign_cap, self.sign_is_static, self.sign_static_raw, (self.sign_wT, self.sign_hT))
            if new_sign is not None:
                self.sign_frame = new_sign

        # 3. Xu ly & hien thi MAN HINH TRAI (CAM 1: LAN DUONG)
        curve_val = 0.0
        det_lane = {}
        if self.lane_frame is not None:
            if self.var_enable_lane.get():
                points, hsv_range = self._get_roi_and_hsv()
                active_profile = "highway" if self.var_preset.get() == "Đường cao tốc" else "robot"
                restrict_roi = (self.var_roi_mode.get() == "Chỉ trong ROI")

                t0_lane = time.perf_counter()
                curve_val, img_roi, img_thres, img_warp, img_hist, img_lane_result = self.detector.process_frame(
                    self.lane_frame, points, hsv_range,
                    use_dual_edge=True, profile=active_profile,
                    fresh_observation=(not self.lane_is_static and not self.paused),
                    restrict_roi=restrict_roi,
                )
                dt_lane = max(1e-4, time.perf_counter() - t0_lane)
                self.lane_fps = 0.75 * self.lane_fps + 0.25 * (1.0 / dt_lane) if self.lane_fps > 0 else (1.0 / dt_lane)

                lview = self.var_lane_view.get()
                if lview == "4 Bước Làn":
                    left_bgr = build_lane_steps_grid(
                        self.lane_frame, img_thres, img_warp, img_hist,
                        active_profile == "highway", self.wT, self.hT,
                        boundary_traces=self.detector.boundary_traces,
                    )
                elif lview == "Ảnh gốc + ROI":
                    left_bgr = img_roi
                else:
                    left_bgr = img_lane_result

                det_lane = self.detector.last_detection
                light_mode = det_lane.get("lighting", {}).get("mode", "NORMAL")
                lane_state = det_lane.get("lane_state", "UNKNOWN")
                e_px = float(det_lane.get("center_offset_px", 0.0))
                e_dir = det_lane.get("offset_direction", "CAN GIUA")
                corner = det_lane.get("corner", {})
                c_dir = corner.get("direction", "UNKNOWN")
                c_zone = corner.get("distance_zone", "NONE")

                self.lbl_lane_badge.config(text=f"Sáng: {light_mode}  |  {self.lane_fps:.1f} FPS")
                self.lbl_lane_state.config(text=lane_state)
                self.lbl_offset.config(text=f"{e_px:+.1f} px ({e_dir})")
                self.lbl_corner.config(text=f"{c_dir} ({c_zone})" if c_dir != "UNKNOWN" else "ĐOẠN THẲNG")
            else:
                left_bgr = self.lane_frame
                self.lbl_lane_badge.config(text="Bám làn: TẮT")

            self.last_lane_vis = left_bgr
            self._photo_left = self._render_bgr_to_label(self.canvas_left, left_bgr)

        # 4. Xu ly & hien thi MAN HINH PHAI (CAM 2: BIEN BAO YOLO11n)
        if self.sign_frame is not None:
            if not self.var_enable_sign.get():
                self.latest_detected_signs = []
                self.latest_sign_debug = {}
                self.sign_fps = 0.0
            elif self.sign_is_static:
                self.latest_detected_signs, self.latest_sign_debug, self.sign_fps = self._run_sign_branch(
                    self.sign_frame, use_tracker=False
                )
            else:
                if self.async_sign_future is None:
                    self.async_sign_future = self.executor.submit(self._run_sign_branch, self.sign_frame.copy(), True)
                elif self.async_sign_future.done():
                    try:
                        self.latest_detected_signs, self.latest_sign_debug, s_fps_inst = self.async_sign_future.result()
                        self.sign_fps = 0.7 * self.sign_fps + 0.3 * s_fps_inst if self.sign_fps > 0 else s_fps_inst
                    except Exception:
                        pass
                    if not self.paused:
                        self.async_sign_future = self.executor.submit(self._run_sign_branch, self.sign_frame.copy(), True)

            detected_signs = self.latest_detected_signs
            sign_vis = self.sign_frame.copy()
            if SIGN_MODULE_AVAILABLE and self.var_enable_sign.get():
                # Ve nhe cac ung vien duoi nguong
                for prev in self.latest_sign_debug.get("yolo_previews", []):
                    if not prev.get("accepted", False):
                        bx, by, bw, bh = prev["box"]
                        cv2.rectangle(sign_vis, (bx, by), (bx + bw, by + bh), (0, 155, 255), 1, cv2.LINE_AA)
                # Ve dam cac bien bao da xac nhan
                for cls_name, bx, by, bw, bh, score in detected_signs:
                    info = SIGN_INFO.get(cls_name, {"vn": cls_name, "color": (0, 255, 0)})
                    col = info.get("color", (0, 255, 0))
                    label = f"{info.get('vn', cls_name)} ({score:.0%})"
                    thick = max(2, sign_vis.shape[1] // 450)
                    fscale = max(0.48, sign_vis.shape[1] / 1100.0)
                    cv2.rectangle(sign_vis, (bx, by), (bx + bw, by + bh), col, thick, cv2.LINE_AA)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
                    tx = max(4, min(bx, sign_vis.shape[1] - tw - 10))
                    ty = max(th + 10, by - 6)
                    cv2.rectangle(sign_vis, (tx - 2, ty - th - 6), (tx + tw + 6, ty + 3), (18, 22, 28), -1)
                    cv2.putText(sign_vis, label, (tx + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 255, 255), thick, cv2.LINE_AA)

            if self.var_sign_view.get() == "4 Bước YOLO11n":
                right_bgr = build_sign_steps_grid(self.sign_frame, detected_signs, self.latest_sign_debug, self.sign_wT, self.sign_hT)
            else:
                right_bgr = sign_vis

            self.last_sign_vis = right_bgr
            self._photo_right = self._render_bgr_to_label(self.canvas_right, right_bgr)

            mdl_txt = "YOLO11n" if self.sign_model is not None else "Chưa có model"
            self.lbl_sign_badge.config(text=f"{mdl_txt}  |  {len(detected_signs)} biển  |  {self.sign_fps:.1f} FPS")
            self._update_sign_table(detected_signs)

        # 5. Cap nhat lenh ESP32 gia lap tong hop tu ca 2 Camera
        if MOTOR_LOGIC_AVAILABLE:
            spd, err, cmd, sign_ov = decide_pi_to_esp32_command(
                curve_val, self.latest_detected_signs, SIGN_INFO, lane_info=det_lane
            )
            self.lbl_esp_cmd.config(text=f"<{spd}%, {err:+.2f}, {cmd}>")
            self.lbl_sign_override.config(text=sign_ov if sign_ov else "Không có")

        self.last_proc_ms = (time.perf_counter() - t_loop_start) * 1000.0
        self.lbl_status_right.config(text=f"{len(self.latest_detected_signs)} biển báo | {self.last_proc_ms:.0f} ms")

        delay_ms = 55 if (self.paused or (self.lane_is_static and self.sign_is_static)) else 16
        self.root.after(delay_ms, self._update_loop)

    def _on_close(self):
        try:
            self.executor.shutdown(wait=False)
            self._close_lane_source()
            self._close_sign_source()
        finally:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Tram Quan Sat 2 Camera — Lan duong & Bien bao YOLO11n")
    parser.add_argument("--lane-cam", type=int, default=-1, help="Index webcam cho CAM 1 (Lan duong)")
    parser.add_argument("--sign-cam", type=int, default=-1, help="Index webcam cho CAM 2 (Bien bao YOLO11n)")
    parser.add_argument("--cam", "-c", type=int, default=-1, help="Mo 1 webcam dung chung (tuong thich cu)")
    parser.add_argument("--video", "-v", type=str, default="", help="Duong dan video/anh")
    parser.add_argument("video_direct", nargs="?", default="", help="Duong dan video/anh truc tiep")
    parser.add_argument("--width", type=int, default=480, help="Chieu rong frame Lan duong (default: 480)")
    parser.add_argument("--height", type=int, default=240, help="Chieu cao frame Lan duong (default: 240)")
    args = parser.parse_args()

    root = tk.Tk()
    XeTuHanhTestApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
