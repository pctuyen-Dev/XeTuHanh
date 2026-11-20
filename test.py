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

BGR_SUBCARD = (26, 21, 18)
BGR_HEADER_BAR = (42, 34, 30)
BGR_BORDER_CARD = (63, 52, 45)
BGR_CYAN = (250, 165, 96)
BGR_GREEN = (153, 211, 52)
BGR_AMBER = (65, 184, 245)
BGR_WHITE = (242, 236, 232)
BGR_MUTED = (135, 119, 110)


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


def _decorate_cell(cell, title, accent_text=BGR_CYAN):
    ch, cw = cell.shape[:2]
    cv2.rectangle(cell, (0, 0), (cw - 1, 22), BGR_HEADER_BAR, cv2.FILLED)
    cv2.rectangle(cell, (0, 0), (cw - 1, ch - 1), BGR_BORDER_CARD, 1)
    cv2.putText(cell, title, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, accent_text, 1, cv2.LINE_AA)
    return cell


def build_lane_steps_grid(lane_frame, img_thres, img_warp, img_hist, is_highway,
                          wT, hT, boundary_traces=None):
    cw, ch = max(240, wT // 2), max(150, hT // 2)
    cell_size = (cw, ch)

    contrast, _, _ = utils.compute_glare_contrast(lane_frame)
    contrast_vis = cv2.applyColorMap(cv2.convertScaleAbs(contrast, alpha=3.0), cv2.COLORMAP_TURBO)
    p1 = _decorate_cell(cv2.resize(contrast_vis, cell_size), "1. Top-Hat Khu Bong", BGR_GREEN)

    thres_bgr = np.full((img_thres.shape[0], img_thres.shape[1], 3), (18, 15, 13), dtype=np.uint8)
    thres_bgr[img_thres > 0] = BGR_GREEN
    p2 = _decorate_cell(cv2.resize(thres_bgr, cell_size), "2. Mat Na Nhi Phan", BGR_GREEN)

    p3 = cv2.resize(img_warp, cell_size)
    for gx in range(cw // 4, cw, cw // 4):
        cv2.line(p3, (gx, 22), (gx, ch), BGR_BORDER_CARD, 1)
    p3 = _decorate_cell(p3, "3. Bird-Eye Warp", BGR_GREEN)

    hist_vis = img_hist.copy()
    if boundary_traces:
        for tr in boundary_traces:
            pts = np.asarray(tr, dtype=np.int32)
            cv2.polylines(hist_vis, [pts], False, BGR_AMBER, 2, cv2.LINE_AA)
    else:
        skel = utils.thin_mask(img_thres) if not is_highway else cv2.Canny(img_thres, 50, 150)
        hist_vis[skel > 0] = BGR_AMBER
    p4 = _decorate_cell(cv2.resize(hist_vis, cell_size), "4. Xuong & Histogram", BGR_GREEN)

    return np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])


def build_sign_steps_grid(sign_frame, detected_signs, sign_debug, wT, hT):
    cw, ch = max(240, wT // 2), max(160, hT // 2)
    cell_size = (cw, ch)

    p1 = _decorate_cell(cv2.resize(sign_frame, cell_size), "1. Anh Goc Cam 2", BGR_CYAN)

    lb = sign_debug.get("letterbox_img") if sign_debug else None
    p2_raw = cv2.resize(lb, cell_size) if (lb is not None and lb.size > 0) else np.full((ch, cw, 3), BGR_SUBCARD, dtype=np.uint8)
    p2 = _decorate_cell(p2_raw, "2. Letterbox YOLO11n", BGR_CYAN)

    yvis = sign_debug.get("yolo_vis") if sign_debug else None
    p3_raw = cv2.resize(yvis, cell_size) if (yvis is not None and yvis.size > 0) else cv2.resize(sign_frame, cell_size)
    p3 = _decorate_cell(p3_raw, "3. Phat Hien YOLO11n", BGR_CYAN)

    p4 = np.full((ch, cw, 3), BGR_SUBCARD, dtype=np.uint8)
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
                b_col = BGR_GREEN if is_ok else BGR_AMBER
                cv2.rectangle(p4, (10, ty), (64, ty + 54), b_col, 1)
                vn_name = SIGN_INFO.get(cls_name, {"vn": cls_name})["vn"]
                st = "NHAN" if is_ok else "THAP"
                cv2.putText(p4, vn_name[:16], (72, ty + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BGR_WHITE, 1, cv2.LINE_AA)
                cv2.putText(p4, f"{sc:.0%} [{st}]", (72, ty + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, b_col, 1, cv2.LINE_AA)
    else:
        msg = "Chua nap YOLO11n" if not sign_debug else "Khong co bien bao"
        cv2.putText(p4, msg, (cw // 2 - 58, ch // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BGR_MUTED, 1, cv2.LINE_AA)
    p4 = _decorate_cell(p4, "4. Crop & Do Tin Cay", BGR_CYAN)

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
    WIN_W = 1280
    WIN_H = 740
    PAD_X = 12
    GAP_X = 10
    CARD_W = (WIN_W - PAD_X * 2 - GAP_X) // 2
    CAM_CARD_H = 426
    VIEW_W = CARD_W - 14
    VIEW_H = 345

    BG_CANVAS = "#0f1115"
    BG_HEADER_TOP = "#15181e"
    BG_CARD = "#181b21"
    BG_SUBCARD = "#12151a"
    BG_HEADER_BAR = "#1e222a"
    BG_PILL = "#21262f"
    BG_PILL_HOVER = "#2b313d"

    BORDER_CARD = "#2d343f"
    BORDER_SUB = "#252a34"
    BORDER_BTN = "#38404e"
    BORDER_DIM = "#222730"

    COL_ACCENT = "#3b82f6"
    COL_CYAN = "#60a5fa"
    COL_BLUE = "#93c5fd"
    COL_TINT_BG = "#182538"
    COL_TINT_HOVER = "#1f314a"
    COL_TINT_BORDER = "#2d4c78"

    COL_GREEN = "#34d399"
    COL_AMBER = "#f5b841"
    COL_ORANGE = "#f0843c"
    COL_RED = "#f87171"

    TXT_WHITE = "#e8ecf2"
    TXT_SOFT = "#a8b2c1"
    TXT_MUTED = "#6e7787"

    SUBCARD_RGB = (18, 21, 26)

    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.wT = args.width
        self.hT = args.height
        self.sign_wT = 480
        self.sign_hT = 320

        self.root.title("Xe Tự Hành — Kiểm Thử 2 Camera")
        self.root.configure(bg=self.BG_CANVAS)
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.root.resizable(False, False)

        self.lane_cap = None
        self.lane_source = None
        self.lane_is_static = False
        self.lane_static_raw = None
        self.lane_frame = None
        self.last_lane_vis = None

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
        self.last_table_sig = None

        self.detector = LaneDetector(avg_window=8)
        self.current_preset = PRESET_ROBOT_TOY
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SignWorker")
        self.async_sign_future = None
        self.latest_detected_signs = []
        self.latest_sign_debug = {}

        self.sign_model = None
        self.sign_tracker = None
        if SIGN_MODULE_AVAILABLE:
            try:
                self.sign_model = load_sign_model()
                self.sign_tracker = SignTracker(history_size=5, min_hits=2, max_age=3)
                print(f"[INFO] Da nap thanh cong Model YOLO11n ({len(self.sign_model.class_names)} Bien bao) + SignTracker!")
            except Exception as e:
                print(f"[WARNING] Loi khoi tao model YOLO11n: {e}")

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
            "Main.TCombobox",
            fieldbackground=self.BG_PILL,
            background=self.BG_PILL,
            foreground=self.TXT_WHITE,
            arrowcolor=self.COL_CYAN,
            bordercolor=self.BORDER_BTN,
            lightcolor=self.BORDER_BTN,
            darkcolor=self.BORDER_BTN,
            selectbackground=self.COL_TINT_BG,
            selectforeground=self.TXT_WHITE,
        )
        style.map(
            "Main.TCombobox",
            fieldbackground=[("readonly", self.BG_PILL)],
            foreground=[("readonly", self.TXT_WHITE)],
            bordercolor=[("focus", self.COL_TINT_BORDER)],
        )
        self.root.option_add("*TCombobox*Listbox.background", self.BG_CARD)
        self.root.option_add("*TCombobox*Listbox.foreground", self.TXT_WHITE)
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.COL_TINT_BG)
        self.root.option_add("*TCombobox*Listbox.selectForeground", self.COL_BLUE)

        style.configure(
            "Sign.Treeview",
            background=self.BG_SUBCARD,
            foreground=self.TXT_WHITE,
            fieldbackground=self.BG_SUBCARD,
            borderwidth=0,
            rowheight=24,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Sign.Treeview.Heading",
            background=self.BG_HEADER_BAR,
            foreground=self.TXT_SOFT,
            borderwidth=0,
            relief="flat",
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Sign.Treeview",
            background=[("selected", self.COL_TINT_BG)],
            foreground=[("selected", self.COL_BLUE)],
        )
        style.configure(
            "Main.Vertical.TScrollbar",
            background=self.BG_PILL,
            troughcolor=self.BG_SUBCARD,
            bordercolor=self.BG_SUBCARD,
            arrowcolor=self.TXT_SOFT,
        )

    def _make_btn(self, parent, text, command, tinted=False, width=None, padx=9, pady=3):
        bg = self.COL_TINT_BG if tinted else self.BG_PILL
        hover = self.COL_TINT_HOVER if tinted else self.BG_PILL_HOVER
        fg = self.COL_BLUE if tinted else self.TXT_WHITE
        border = self.COL_TINT_BORDER if tinted else self.BORDER_BTN

        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            font=("Segoe UI", 8, "bold"),
            bd=0,
            padx=padx,
            pady=pady,
            cursor="hand2",
            relief="flat",
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            width=width,
        )
        btn._base_bg = bg
        btn._hover_bg = hover
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg=b._hover_bg))
        btn.bind("<Leave>", lambda e, b=btn: b.config(bg=b._base_bg))
        return btn

    def _set_btn_style(self, btn, tinted=False, text=None):
        bg = self.COL_TINT_BG if tinted else self.BG_PILL
        hover = self.COL_TINT_HOVER if tinted else self.BG_PILL_HOVER
        fg = self.COL_BLUE if tinted else self.TXT_WHITE
        border = self.COL_TINT_BORDER if tinted else self.BORDER_BTN
        btn._base_bg = bg
        btn._hover_bg = hover
        cfg = {
            "bg": bg,
            "fg": fg,
            "activebackground": hover,
            "activeforeground": fg,
            "highlightbackground": border,
            "highlightcolor": border,
        }
        if text is not None:
            cfg["text"] = text
        btn.config(**cfg)

    def _build_ui(self):
        header = tk.Frame(self.root, bg=self.BG_HEADER_TOP, height=46)
        header.pack(fill="x")
        header.pack_propagate(False)

        header_inner = tk.Frame(header, bg=self.BG_HEADER_TOP)
        header_inner.pack(fill="both", expand=True, padx=self.PAD_X)

        tk.Label(
            header_inner,
            text="XE TỰ HÀNH — KIỂM THỬ 2 CAMERA",
            font=("Segoe UI", 12, "bold"),
            fg=self.TXT_WHITE,
            bg=self.BG_HEADER_TOP,
        ).pack(side="left", pady=10)

        quick_bar = tk.Frame(header_inner, bg=self.BG_HEADER_TOP)
        quick_bar.pack(side="right", pady=8)

        self._make_btn(quick_bar, "Mở 2 Cam", self.action_open_both_cameras, tinted=True).pack(side="left", padx=(0, 6))
        self._make_btn(quick_bar, "Đổi chỗ", self.action_swap_cameras).pack(side="left", padx=(0, 6))
        self.btn_pause = self._make_btn(quick_bar, "Tạm dừng", self.action_toggle_pause)
        self.btn_pause.pack(side="left", padx=(0, 6))
        self._make_btn(quick_bar, "Reset", self.action_reset).pack(side="left", padx=(0, 6))
        self._make_btn(quick_bar, "Lưu ảnh", self.action_save_snapshot).pack(side="left", padx=(0, 10))

        self._make_btn(quick_bar, "-", lambda: self._adjust_fov(-10), width=2, padx=2).pack(side="left")
        self.btn_fov = self._make_btn(quick_bar, "FOV: 100%", self._reset_fov, tinted=True, padx=7)
        self.btn_fov.pack(side="left", padx=2)
        self._make_btn(quick_bar, "+", lambda: self._adjust_fov(10), width=2, padx=2).pack(side="left")

        tk.Frame(self.root, bg=self.BORDER_DIM, height=1).pack(fill="x")

        views_frame = tk.Frame(self.root, bg=self.BG_CANVAS, height=self.CAM_CARD_H)
        views_frame.pack(fill="x", padx=self.PAD_X, pady=(8, 0))
        views_frame.pack_propagate(False)

        left_card = tk.Frame(
            views_frame, bg=self.BG_CARD, width=self.CARD_W, height=self.CAM_CARD_H,
            highlightbackground=self.BORDER_CARD, highlightthickness=1,
        )
        left_card.pack(side="left")
        left_card.pack_propagate(False)

        left_hdr = tk.Frame(left_card, bg=self.BG_HEADER_BAR, height=32)
        left_hdr.pack(fill="x")
        left_hdr.pack_propagate(False)
        tk.Label(
            left_hdr, text="CAM 1 – LÀN ĐƯỜNG",
            font=("Segoe UI", 9, "bold"), fg=self.TXT_WHITE, bg=self.BG_HEADER_BAR,
        ).pack(side="left", padx=(10, 8))
        self.lbl_cam1_src = tk.Label(
            left_hdr, text="Chưa chọn",
            font=("Segoe UI", 8, "bold"), fg=self.TXT_MUTED, bg=self.BG_SUBCARD,
            padx=7, pady=1, highlightbackground=self.BORDER_SUB, highlightthickness=1,
        )
        self.lbl_cam1_src.pack(side="left", pady=5)
        self.lbl_lane_badge = tk.Label(
            left_hdr, text="NORMAL  |  0.0 FPS",
            font=("Segoe UI", 8, "bold"), fg=self.COL_CYAN, bg=self.BG_HEADER_BAR,
        )
        self.lbl_lane_badge.pack(side="right", padx=10)

        tk.Frame(left_card, bg=self.BORDER_DIM, height=1).pack(fill="x")

        left_tb = tk.Frame(left_card, bg=self.BG_HEADER_TOP, height=34)
        left_tb.pack(fill="x")
        left_tb.pack_propagate(False)

        cb_lcam = ttk.Combobox(
            left_tb, textvariable=self.var_lane_cam, values=["Cam 0", "Cam 1", "Cam 2", "Cam 3"],
            width=6, state="readonly", style="Main.TCombobox", font=("Segoe UI", 8),
        )
        cb_lcam.pack(side="left", padx=(8, 4), pady=5)
        cb_lcam.bind("<<ComboboxSelected>>", lambda e: self.action_open_lane_cam())

        self._make_btn(left_tb, "Video", self.action_pick_lane_video, padx=7, pady=2).pack(side="left", padx=(0, 4), pady=4)
        self._make_btn(left_tb, "Ảnh", self.action_pick_lane_image, padx=7, pady=2).pack(side="left", padx=(0, 8), pady=4)

        ttk.Combobox(
            left_tb, textvariable=self.var_lane_view,
            values=["Kết quả Làn", "4 Bước Làn", "Ảnh gốc + ROI"],
            width=12, state="readonly", style="Main.TCombobox", font=("Segoe UI", 8),
        ).pack(side="left", padx=(0, 4), pady=5)

        cb_preset = ttk.Combobox(
            left_tb, textvariable=self.var_preset, values=["Robot sân nhà", "Đường cao tốc"],
            width=12, state="readonly", style="Main.TCombobox", font=("Segoe UI", 8),
        )
        cb_preset.pack(side="left", padx=(0, 4), pady=5)
        cb_preset.bind("<<ComboboxSelected>>", lambda e: self.action_change_preset())

        cb_roi = ttk.Combobox(
            left_tb, textvariable=self.var_roi_mode, values=["Mở rộng cua", "Chỉ trong ROI"],
            width=11, state="readonly", style="Main.TCombobox", font=("Segoe UI", 8),
        )
        cb_roi.pack(side="left", padx=(0, 4), pady=5)
        cb_roi.bind("<<ComboboxSelected>>", lambda e: self.detector.reset())

        tk.Checkbutton(
            left_tb, text="Bật", variable=self.var_enable_lane,
            bg=self.BG_HEADER_TOP, fg=self.TXT_SOFT, selectcolor=self.BG_PILL,
            activebackground=self.BG_HEADER_TOP, activeforeground=self.TXT_WHITE,
            font=("Segoe UI", 8), cursor="hand2",
        ).pack(side="right", padx=(0, 6))

        left_vp = tk.Frame(
            left_card, bg=self.BG_SUBCARD, width=self.VIEW_W, height=self.VIEW_H,
            highlightbackground=self.BORDER_DIM, highlightthickness=1,
        )
        left_vp.pack(padx=6, pady=6)
        left_vp.pack_propagate(False)
        self.canvas_left = tk.Label(
            left_vp, bg=self.BG_SUBCARD,
            text="Đang chờ tín hiệu CAM 1 (Làn đường)...",
            fg=self.TXT_MUTED, font=("Segoe UI", 10),
        )
        self.canvas_left.pack(fill="both", expand=True)

        right_card = tk.Frame(
            views_frame, bg=self.BG_CARD, width=self.CARD_W, height=self.CAM_CARD_H,
            highlightbackground=self.BORDER_CARD, highlightthickness=1,
        )
        right_card.pack(side="right")
        right_card.pack_propagate(False)

        right_hdr = tk.Frame(right_card, bg=self.BG_HEADER_BAR, height=32)
        right_hdr.pack(fill="x")
        right_hdr.pack_propagate(False)
        tk.Label(
            right_hdr, text="CAM 2 – BIỂN BÁO (YOLO11n)",
            font=("Segoe UI", 9, "bold"), fg=self.TXT_WHITE, bg=self.BG_HEADER_BAR,
        ).pack(side="left", padx=(10, 8))
        self.lbl_cam2_src = tk.Label(
            right_hdr, text="Chưa chọn",
            font=("Segoe UI", 8, "bold"), fg=self.TXT_MUTED, bg=self.BG_SUBCARD,
            padx=7, pady=1, highlightbackground=self.BORDER_SUB, highlightthickness=1,
        )
        self.lbl_cam2_src.pack(side="left", pady=5)
        self.lbl_sign_badge = tk.Label(
            right_hdr, text="YOLO11n  |  0.0 FPS",
            font=("Segoe UI", 8, "bold"), fg=self.COL_GREEN, bg=self.BG_HEADER_BAR,
        )
        self.lbl_sign_badge.pack(side="right", padx=10)

        tk.Frame(right_card, bg=self.BORDER_DIM, height=1).pack(fill="x")

        right_tb = tk.Frame(right_card, bg=self.BG_HEADER_TOP, height=34)
        right_tb.pack(fill="x")
        right_tb.pack_propagate(False)

        cb_scam = ttk.Combobox(
            right_tb, textvariable=self.var_sign_cam,
            values=["Cam 1", "Cam 0", "Cam 2", "Cam 3", "Chung Cam 1"],
            width=11, state="readonly", style="Main.TCombobox", font=("Segoe UI", 8),
        )
        cb_scam.pack(side="left", padx=(8, 4), pady=5)
        cb_scam.bind("<<ComboboxSelected>>", lambda e: self.action_open_sign_cam())

        self._make_btn(right_tb, "Video", self.action_pick_sign_video, padx=7, pady=2).pack(side="left", padx=(0, 4), pady=4)
        self._make_btn(right_tb, "Ảnh", self.action_pick_sign_image, padx=7, pady=2).pack(side="left", padx=(0, 8), pady=4)

        ttk.Combobox(
            right_tb, textvariable=self.var_sign_view,
            values=["Kết quả YOLO11n", "4 Bước YOLO11n"],
            width=15, state="readonly", style="Main.TCombobox", font=("Segoe UI", 8),
        ).pack(side="left", pady=5)

        tk.Checkbutton(
            right_tb, text="Bật", variable=self.var_enable_sign,
            bg=self.BG_HEADER_TOP, fg=self.TXT_SOFT, selectcolor=self.BG_PILL,
            activebackground=self.BG_HEADER_TOP, activeforeground=self.TXT_WHITE,
            font=("Segoe UI", 8), cursor="hand2",
        ).pack(side="right", padx=(0, 6))

        right_vp = tk.Frame(
            right_card, bg=self.BG_SUBCARD, width=self.VIEW_W, height=self.VIEW_H,
            highlightbackground=self.BORDER_DIM, highlightthickness=1,
        )
        right_vp.pack(padx=6, pady=6)
        right_vp.pack_propagate(False)
        self.canvas_right = tk.Label(
            right_vp, bg=self.BG_SUBCARD,
            text="Đang chờ tín hiệu CAM 2 (Biển báo YOLO11n)...",
            fg=self.TXT_MUTED, font=("Segoe UI", 10),
        )
        self.canvas_right.pack(fill="both", expand=True)

        tele_card = tk.Frame(
            self.root, bg=self.BG_CARD, height=54,
            highlightbackground=self.BORDER_CARD, highlightthickness=1,
        )
        tele_card.pack(fill="x", padx=self.PAD_X, pady=(8, 0))
        tele_card.pack_propagate(False)

        tele_grid = tk.Frame(tele_card, bg=self.BG_CARD)
        tele_grid.pack(fill="both", expand=True, padx=6, pady=5)
        for col_i in range(5):
            tele_grid.columnconfigure(col_i, weight=1, uniform="tele")
        tele_grid.rowconfigure(0, weight=1)

        self.lbl_lane_state = self._make_metric(tele_grid, 0, "Làn đường", "UNKNOWN", self.COL_GREEN)
        self.lbl_offset = self._make_metric(tele_grid, 1, "Lệch tâm", "+0.0 px (CAN GIUA)", self.COL_CYAN)
        self.lbl_corner = self._make_metric(tele_grid, 2, "Góc cua", "ĐOẠN THẲNG", self.TXT_WHITE)
        self.lbl_esp_cmd = self._make_metric(tele_grid, 3, "Lệnh ESP32", "<0%, +0.00, NORM>", self.COL_BLUE)
        self.lbl_sign_override = self._make_metric(tele_grid, 4, "Ưu tiên biển báo", "Không có", self.COL_AMBER)

        table_card = tk.Frame(
            self.root, bg=self.BG_CARD, height=152,
            highlightbackground=self.BORDER_CARD, highlightthickness=1,
        )
        table_card.pack(fill="x", padx=self.PAD_X, pady=(8, 0))
        table_card.pack_propagate(False)

        table_hdr = tk.Frame(table_card, bg=self.BG_HEADER_BAR, height=28)
        table_hdr.pack(fill="x")
        table_hdr.pack_propagate(False)
        self.lbl_table_title = tk.Label(
            table_hdr, text="BIỂN BÁO PHÁT HIỆN (0)",
            font=("Segoe UI", 9, "bold"), fg=self.TXT_WHITE, bg=self.BG_HEADER_BAR,
        )
        self.lbl_table_title.pack(side="left", padx=10, pady=4)

        tk.Frame(table_card, bg=self.BORDER_DIM, height=1).pack(fill="x")

        tree_wrap = tk.Frame(table_card, bg=self.BG_CARD)
        tree_wrap.pack(fill="both", expand=True, padx=6, pady=5)

        cols = ("idx", "vn", "cls", "action", "bbox", "conf")
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=4, style="Sign.Treeview")
        headings = [
            ("idx", "#", 45, "center"),
            ("vn", "Biển báo", 220, "w"),
            ("cls", "Mã lớp", 130, "center"),
            ("action", "Hành động xe", 360, "w"),
            ("bbox", "Vùng (x, y, w, h)", 180, "center"),
            ("conf", "Độ tin cậy", 110, "center"),
        ]
        for cid, ctext, cwidth, calign in headings:
            self.tree.heading(cid, text=ctext)
            self.tree.column(cid, width=cwidth, anchor=calign, stretch=(cid in ("vn", "action")))

        sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview, style="Main.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        status_bar = tk.Frame(self.root, bg=self.BG_HEADER_TOP, height=32)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        tk.Frame(status_bar, bg=self.BORDER_DIM, height=1).pack(fill="x")

        status_inner = tk.Frame(status_bar, bg=self.BG_HEADER_TOP)
        status_inner.pack(fill="both", expand=True, padx=self.PAD_X)

        self.lbl_status_left = tk.Label(
            status_inner, text="● Sẵn sàng",
            fg=self.COL_GREEN, bg=self.BG_HEADER_TOP, font=("Segoe UI", 8, "bold"),
        )
        self.lbl_status_left.pack(side="left", pady=5)
        self.lbl_status_right = tk.Label(
            status_inner, text="[Space] Tạm dừng   |   [+/-] FOV   |   [Esc] Thoát   |   0 ms",
            fg=self.TXT_SOFT, bg=self.BG_HEADER_TOP, font=("Segoe UI", 8),
        )
        self.lbl_status_right.pack(side="right", pady=5)

    def _make_metric(self, parent, col_idx, title, val, color):
        box = tk.Frame(
            parent, bg=self.BG_SUBCARD, padx=9, pady=3,
            highlightbackground=self.BORDER_SUB, highlightthickness=1,
        )
        padx = (0, 6) if col_idx < 4 else (0, 0)
        box.grid(row=0, column=col_idx, sticky="nsew", padx=padx)
        tk.Label(box, text=title, font=("Segoe UI", 8), fg=self.TXT_MUTED, bg=self.BG_SUBCARD).pack(anchor="w")
        lbl = tk.Label(box, text=val, font=("Segoe UI", 9, "bold"), fg=color, bg=self.BG_SUBCARD)
        lbl.pack(anchor="w")
        return lbl

    def _set_source_pill(self, pill_label, text, active=True):
        if active:
            pill_label.config(
                text=text[:24],
                fg=self.COL_BLUE,
                bg=self.COL_TINT_BG,
                highlightbackground=self.COL_TINT_BORDER,
            )
        else:
            pill_label.config(
                text=text[:24],
                fg=self.TXT_MUTED,
                bg=self.BG_SUBCARD,
                highlightbackground=self.BORDER_SUB,
            )

    def _bind_shortcuts(self):
        self.root.bind("<space>", lambda e: self.action_toggle_pause())
        self.root.bind("<Escape>", lambda e: self._on_close())
        self.root.bind("-", lambda e: self._adjust_fov(-10))
        self.root.bind("_", lambda e: self._adjust_fov(-10))
        self.root.bind("+", lambda e: self._adjust_fov(10))
        self.root.bind("=", lambda e: self._adjust_fov(10))

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
            for cand in (
                os.path.join(P2_DIR, "data", "DuLieuBienBaoPNG", "ReTrai.png"),
            ):
                if os.path.exists(cand):
                    self.open_sign_source(cand)
                    break

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
                self._set_source_pill(self.lbl_cam1_src, "Lỗi kết nối", False)
                self.lbl_status_left.config(text=f"● Lỗi CAM 1: Không mở được Camera {src}", fg=self.COL_RED)
                return False
            self.lane_cap = cap
            self.lane_is_static = False
            self.lane_source = src
            self._set_source_pill(self.lbl_cam1_src, f"Camera {src}", True)
            self.lbl_status_left.config(text=f"● Đã mở CAM 1: Camera {src}", fg=self.COL_GREEN)
            return True

        src = str(src).strip('"\'')
        if not os.path.exists(src):
            self.lbl_status_left.config(text=f"● Không tìm thấy file CAM 1: {src}", fg=self.COL_RED)
            return False

        if src.lower().endswith(IMAGE_EXTENSIONS):
            img = _load_static_image(src)
            if img is None:
                return False
            self.lane_static_raw = img
            self.lane_is_static = True
            self.lane_source = src
            self._set_source_pill(self.lbl_cam1_src, os.path.basename(src), True)
            self.lbl_status_left.config(text=f"● Đã mở ảnh CAM 1: {os.path.basename(src)}", fg=self.COL_GREEN)
            return True

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.lbl_status_left.config(text=f"● Không mở được video CAM 1: {os.path.basename(src)}", fg=self.COL_RED)
            return False
        self.lane_cap = cap
        self.lane_is_static = False
        self.lane_source = src
        is_hw = is_highway_source(src)
        self.var_preset.set("Đường cao tốc" if is_hw else "Robot sân nhà")
        self.action_change_preset()
        self._set_source_pill(self.lbl_cam1_src, os.path.basename(src), True)
        self.lbl_status_left.config(text=f"● Đã mở video CAM 1: {os.path.basename(src)}", fg=self.COL_GREEN)
        return True

    def open_sign_source(self, src):
        self._close_sign_source()
        if self.sign_tracker is not None:
            self.sign_tracker.reset()
        self.latest_detected_signs = []
        self.latest_sign_debug = {}
        self.last_table_sig = None
        self.sign_frame = None
        self.sign_static_raw = None
        self.sign_share_lane = False

        if src == "SHARED":
            self.sign_share_lane = True
            self.sign_is_static = False
            self.sign_source = "SHARED"
            self._set_source_pill(self.lbl_cam2_src, "Chung CAM 1", True)
            self.lbl_status_left.config(text="● CAM 2 đang dùng chung khung hình từ CAM 1", fg=self.COL_AMBER)
            return True

        if isinstance(src, int):
            if isinstance(self.lane_source, int) and self.lane_source == src and self.lane_cap is not None:
                self.sign_share_lane = True
                self.sign_source = "SHARED"
                self.var_sign_cam.set("Chung Cam 1")
                self._set_source_pill(self.lbl_cam2_src, f"Chung Camera {src}", True)
                return True
            cap = _open_camera_capture(src, (self.sign_wT, self.sign_hT))
            if cap is None:
                self._set_source_pill(self.lbl_cam2_src, "Lỗi kết nối", False)
                self.lbl_status_left.config(text=f"● Lỗi CAM 2: Không mở được Camera {src}", fg=self.COL_RED)
                return False
            self.sign_cap = cap
            self.sign_is_static = False
            self.sign_source = src
            self._set_source_pill(self.lbl_cam2_src, f"Camera {src}", True)
            self.lbl_status_left.config(text=f"● Đã mở CAM 2: Camera {src}", fg=self.COL_CYAN)
            return True

        src = str(src).strip('"\'')
        if not os.path.exists(src):
            self.lbl_status_left.config(text=f"● Không tìm thấy file CAM 2: {src}", fg=self.COL_RED)
            return False

        if src.lower().endswith(IMAGE_EXTENSIONS):
            img = _load_static_image(src)
            if img is None:
                return False
            self.sign_static_raw = img
            self.sign_is_static = True
            self.sign_source = src
            self._set_source_pill(self.lbl_cam2_src, os.path.basename(src), True)
            self.lbl_status_left.config(text=f"● Đã mở ảnh CAM 2: {os.path.basename(src)}", fg=self.COL_CYAN)
            return True

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.lbl_status_left.config(text=f"● Không mở được video CAM 2: {os.path.basename(src)}", fg=self.COL_RED)
            return False
        self.sign_cap = cap
        self.sign_is_static = False
        self.sign_source = src
        self._set_source_pill(self.lbl_cam2_src, os.path.basename(src), True)
        self.lbl_status_left.config(text=f"● Đã mở video CAM 2: {os.path.basename(src)}", fg=self.COL_CYAN)
        return True

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
        if self.sign_share_lane:
            self.lbl_status_left.config(text="● Đang dùng chung 1 camera nên không thể hoán đổi", fg=self.COL_AMBER)
            return
        self.lane_cap, self.sign_cap = self.sign_cap, self.lane_cap
        self.lane_source, self.sign_source = self.sign_source, self.lane_source
        self.lane_is_static, self.sign_is_static = self.sign_is_static, self.lane_is_static
        self.lane_static_raw, self.sign_static_raw = self.sign_static_raw, self.lane_static_raw
        self.detector.reset()
        if self.sign_tracker is not None:
            self.sign_tracker.reset()

        l_txt = f"Camera {self.lane_source}" if isinstance(self.lane_source, int) else (os.path.basename(str(self.lane_source)) if self.lane_source else "Chưa chọn")
        s_txt = f"Camera {self.sign_source}" if isinstance(self.sign_source, int) else (os.path.basename(str(self.sign_source)) if self.sign_source else "Chưa chọn")
        if isinstance(self.lane_source, int):
            self.var_lane_cam.set(f"Cam {self.lane_source}")
        if isinstance(self.sign_source, int):
            self.var_sign_cam.set(f"Cam {self.sign_source}")
        self._set_source_pill(self.lbl_cam1_src, l_txt, self.lane_source is not None)
        self._set_source_pill(self.lbl_cam2_src, s_txt, self.sign_source is not None)
        self.lbl_status_left.config(text="● Đã hoán đổi nguồn CAM 1 ↔ CAM 2", fg=self.COL_GREEN)

    def action_pick_lane_image(self):
        init_d = IMG_DIR if os.path.isdir(IMG_DIR) else BASE_DIR
        f = filedialog.askopenfilename(
            title="Chọn ảnh cho CAM 1 (Làn đường)", initialdir=init_d,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if f:
            self.open_lane_source(f)

    def action_pick_lane_video(self):
        init_d = VID_DIR if os.path.isdir(VID_DIR) else BASE_DIR
        f = filedialog.askopenfilename(
            title="Chọn video cho CAM 1 (Làn đường)", initialdir=init_d,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All files", "*.*")],
        )
        if f:
            self.open_lane_source(f)

    def action_pick_sign_image(self):
        test_d = os.path.join(P2_DIR, "data", "test")
        data_d = os.path.join(P2_DIR, "data")
        init_d = test_d if os.path.isdir(test_d) else (data_d if os.path.isdir(data_d) else BASE_DIR)
        f = filedialog.askopenfilename(
            title="Chọn ảnh cho CAM 2 (Biển báo)", initialdir=init_d,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if f:
            self.open_sign_source(f)

    def action_pick_sign_video(self):
        init_d = VID_DIR if os.path.isdir(VID_DIR) else BASE_DIR
        f = filedialog.askopenfilename(
            title="Chọn video cho CAM 2 (Biển báo)", initialdir=init_d,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All files", "*.*")],
        )
        if f:
            self.open_sign_source(f)

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
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if out_path:
            cv2.imwrite(out_path, merged)
            self.lbl_status_left.config(text=f"● Đã lưu ảnh: {os.path.basename(out_path)}", fg=self.COL_GREEN)

    def action_change_preset(self):
        is_hw = self.var_preset.get() == "Đường cao tốc"
        self.current_preset = PRESET_HIGHWAY if is_hw else PRESET_ROBOT_TOY
        self.var_roi_mode.set("Chỉ trong ROI" if is_hw else "Mở rộng cua")
        self.detector.reset()

    def action_toggle_pause(self):
        self.paused = not self.paused
        self._set_btn_style(
            self.btn_pause,
            tinted=self.paused,
            text="Tiếp tục" if self.paused else "Tạm dừng",
        )

    def action_reset(self):
        self.detector.reset()
        if self.sign_tracker is not None:
            self.sign_tracker.reset()
        self.latest_detected_signs = []
        self.latest_sign_debug = {}
        self.last_table_sig = None
        self._update_sign_table([])

    def _adjust_fov(self, delta):
        self.cam_fov_pct = max(40, min(100, self.cam_fov_pct + delta))
        self.btn_fov.config(text=f"FOV: {self.cam_fov_pct}%")

    def _reset_fov(self):
        self.cam_fov_pct = 100
        self.btn_fov.config(text="FOV: 100%")

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
        self.lbl_table_title.config(text=f"BIỂN BÁO PHÁT HIỆN ({len(dets)})")
        for idx, (cls_name, bx, by, bw, bh, score) in enumerate(dets, start=1):
            info = SIGN_INFO.get(cls_name, {"vn": cls_name, "action": "N/A"})
            self.tree.insert("", "end", values=(
                idx, info.get("vn", cls_name), cls_name, info.get("action", "N/A"),
                f"({bx}, {by}, {bw}, {bh})", f"{score * 100:.1f}%",
            ))

    def _render_bgr_to_label(self, label_widget, bgr_img):
        if bgr_img is None or bgr_img.size == 0:
            return None
        tw, th = self.VIEW_W - 2, self.VIEW_H - 2
        ih, iw = bgr_img.shape[:2]
        scale = min(tw / float(iw), th / float(ih))
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        resized = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        canvas = np.full((th, tw, 3), self.SUBCARD_RGB, dtype=np.uint8)
        ox = (tw - nw) // 2
        oy = (th - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = rgb
        photo = ImageTk.PhotoImage(image=Image.fromarray(canvas))
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

        new_lane = self._read_next_frame(self.lane_cap, self.lane_is_static, self.lane_static_raw, (self.wT, self.hT))
        if new_lane is not None:
            self.lane_frame = cv2.resize(new_lane, (self.wT, self.hT)) if (new_lane.shape[1] != self.wT or new_lane.shape[0] != self.hT) else new_lane

        if self.sign_share_lane:
            if self.lane_frame is not None:
                self.sign_frame = self.lane_frame.copy()
        else:
            new_sign = self._read_next_frame(self.sign_cap, self.sign_is_static, self.sign_static_raw, (self.sign_wT, self.sign_hT))
            if new_sign is not None:
                self.sign_frame = new_sign

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

                self.lbl_lane_badge.config(text=f"{light_mode}  |  {self.lane_fps:.1f} FPS", fg=self.COL_CYAN)
                self.lbl_lane_state.config(text=lane_state)
                self.lbl_offset.config(text=f"{e_px:+.1f} px ({e_dir})")
                self.lbl_corner.config(text=f"{c_dir} ({c_zone})" if c_dir != "UNKNOWN" else "ĐOẠN THẲNG")
            else:
                left_bgr = self.lane_frame
                self.lbl_lane_badge.config(text="TẮT", fg=self.TXT_MUTED)

            self.last_lane_vis = left_bgr
            self._photo_left = self._render_bgr_to_label(self.canvas_left, left_bgr)

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
                for prev in self.latest_sign_debug.get("yolo_previews", []):
                    if not prev.get("accepted", False):
                        bx, by, bw, bh = prev["box"]
                        cv2.rectangle(sign_vis, (bx, by), (bx + bw, by + bh), BGR_AMBER, 1, cv2.LINE_AA)
                for cls_name, bx, by, bw, bh, score in detected_signs:
                    info = SIGN_INFO.get(cls_name, {"vn": cls_name, "color": BGR_GREEN})
                    col = info.get("color", BGR_GREEN)
                    label = f"{info.get('vn', cls_name)} ({score:.0%})"
                    thick = max(2, sign_vis.shape[1] // 450)
                    fscale = max(0.48, sign_vis.shape[1] / 1100.0)
                    cv2.rectangle(sign_vis, (bx, by), (bx + bw, by + bh), col, thick, cv2.LINE_AA)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
                    tx = max(4, min(bx, sign_vis.shape[1] - tw - 10))
                    ty = max(th + 10, by - 6)
                    cv2.rectangle(sign_vis, (tx - 2, ty - th - 6), (tx + tw + 6, ty + 3), BGR_SUBCARD, -1)
                    cv2.putText(sign_vis, label, (tx + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fscale, BGR_AMBER, thick, cv2.LINE_AA)

            if self.var_sign_view.get() == "4 Bước YOLO11n":
                right_bgr = build_sign_steps_grid(self.sign_frame, detected_signs, self.latest_sign_debug, self.sign_wT, self.sign_hT)
            else:
                right_bgr = sign_vis

            self.last_sign_vis = right_bgr
            self._photo_right = self._render_bgr_to_label(self.canvas_right, right_bgr)

            if not self.var_enable_sign.get():
                self.lbl_sign_badge.config(text="TẮT", fg=self.TXT_MUTED)
            else:
                mdl_txt = "YOLO11n" if self.sign_model is not None else "Chưa có model"
                self.lbl_sign_badge.config(text=f"{mdl_txt}  |  {self.sign_fps:.1f} FPS", fg=self.COL_GREEN)
            self._update_sign_table(detected_signs)

        if MOTOR_LOGIC_AVAILABLE:
            spd, err, cmd, sign_ov = decide_pi_to_esp32_command(
                curve_val, self.latest_detected_signs, SIGN_INFO, lane_info=det_lane
            )
            self.lbl_esp_cmd.config(text=f"<{spd}%, {err:+.2f}, {cmd}>")
            self.lbl_sign_override.config(text=sign_ov if sign_ov else "Không có")

        self.last_proc_ms = (time.perf_counter() - t_loop_start) * 1000.0
        self.lbl_status_right.config(
            text=f"[Space] Tạm dừng   |   [+/-] FOV   |   [Esc] Thoát   |   {self.last_proc_ms:.0f} ms"
        )

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
    parser = argparse.ArgumentParser(description="Xe Tu Hanh — Kiem Thu 2 Camera")
    parser.add_argument("--lane-cam", type=int, default=-1, help="Index webcam cho CAM 1 (Lan duong)")
    parser.add_argument("--sign-cam", type=int, default=-1, help="Index webcam cho CAM 2 (Bien bao YOLO11n)")
    parser.add_argument("--cam", "-c", type=int, default=-1, help="Mo 1 webcam dung chung")
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
