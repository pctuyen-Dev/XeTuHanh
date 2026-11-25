import os
import time
import unicodedata
import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

# Bang mau Dark phoi mau hai hoa (Obsidian Graphite + Cobalt + Emerald - dinh dang BGR)
BG_CANVAS     = (21, 17, 15)      # #0f1115 - Nen chinh toi sau
BG_HEADER_TOP = (30, 24, 21)      # #15181e - Thanh tieu de tren cung
BG_CARD       = (33, 27, 24)      # #181b21 - Nen the noi dung chinh
BG_SUBCARD    = (26, 21, 18)      # #12151a - Khung phu ben trong
BG_HEADER_BAR = (42, 34, 30)      # #1e222a - Dai tieu de moi the
BG_PILL       = (47, 38, 33)      # #21262f - Nut bam mac dinh
BG_PILL_HOVER = (61, 49, 43)      # #2b313d - Nut bam khi di chuot
BG_TRACK      = (50, 40, 35)      # #232832 - Nen thanh tien trinh

BORDER_GLOW   = (38, 32, 28)      # Khong dung vien neon
BORDER_CARD   = (63, 52, 45)      # #2d343f - Vien the chinh
BORDER_SUB    = (52, 42, 37)      # #252a34 - Vien khung phu
BORDER_BTN    = (78, 64, 56)      # #38404e - Vien nut bam
BORDER_DIM    = (48, 39, 34)      # #222730 - Duong ke chia manh

# He mau nhan phoi hop (Cobalt diu + Emerald + Amber + Crimson)
COL_ACCENT      = (246, 130, 59)  # #3b82f6 - Xanh cobalt nhan chinh
COL_CYAN        = (250, 165, 96)  # #60a5fa - Xanh thep sang de doc thong so
COL_BLUE        = (253, 197, 147) # #93c5fd - Xanh nhat cho chu nut nhan
COL_TINT_BG     = (56, 37, 24)    # #182538 - Nen nut nhan xanh dam
COL_TINT_HOVER  = (74, 49, 31)    # #1f314a - Nen nut nhan khi di chuot
COL_TINT_BORDER = (120, 76, 45)   # #2d4c78 - Vien nut nhan xanh dam

COL_GREEN       = (153, 211, 52)  # #34d399 - Xanh ngoc trang thai tot
COL_GREEN_DARK  = (74, 163, 22)   # #16a34a - Xanh la dam (dau thanh toc do)
COL_LIME        = (153, 211, 52)  # #34d399 - Dong bo mau xanh ngoc
COL_AMBER       = (65, 184, 245)  # #f5b841 - Vang am canh bao / bien bao
COL_AMBER_DARK  = (18, 128, 212)  # #d48012 - Vang dam (dau thanh toc do thap)
COL_ORANGE      = (60, 132, 240)  # #f0843c - Cam lech goc lai cao
COL_RED         = (113, 113, 248) # #f87171 - Do mem de doc tren nen toi
COL_RED_DARK    = (55, 55, 180)   # #b43737 - Do dam (dau thanh khi dung)
COL_PURPLE      = (250, 165, 96)  # Dong bo ve xanh thep (khong dung tim neon)

# Mau nut Tat he thong (Do tham noi bat co kiem soat)
BTN_QUIT_BG       = (43, 43, 181) # #b52b2b
BTN_QUIT_BG_HOVER = (52, 52, 207) # #cf3434
BTN_QUIT_BORDER   = (85, 85, 224) # #e05555

TXT_WHITE     = (242, 236, 232)   # #e8ecf2 - Chu chinh sang ro
TXT_SOFT      = (193, 178, 168)   # #a8b2c1 - Chu phu
TXT_MUTED     = (135, 119, 110)   # #6e7787 - Nhan mo


def _strip_accents(text: str) -> str:
    """Chuyen tieng Viet co dau -> khong dau cho che do fallback OpenCV."""
    s = text.replace("đ", "d").replace("Đ", "D").replace("–", "-").replace("—", "-")
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


class _TextEngine:
    """Ve chu tieng Viet sac net bang Pillow + cache alpha mask (fallback OpenCV)."""

    def __init__(self):
        self._use_pil = False
        self._fonts = {}
        self._cache = {}
        self._max_cache = 512
        self._reg_path = None
        self._bold_path = None

        if _PIL_AVAILABLE:
            self._reg_path, self._bold_path = self._find_fonts()
            if self._reg_path:
                self._use_pil = True

    @staticmethod
    def _find_fonts():
        candidates = [
            # Windows
            (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
            (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
            (r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\tahomabd.ttf"),
            # Linux / Raspberry Pi
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
            ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
             "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
        ]
        for reg, bold in candidates:
            if os.path.isfile(reg):
                return reg, bold if os.path.isfile(bold) else reg
        return None, None

    def _get_font(self, size: int, bold: bool):
        key = (size, bold)
        f = self._fonts.get(key)
        if f is None:
            path = self._bold_path if bold else self._reg_path
            f = ImageFont.truetype(path, size)
            self._fonts[key] = f
        return f

    def _get_mask(self, text: str, size: int, bold: bool):
        key = (text, size, bold)
        item = self._cache.get(key)
        if item is not None:
            return item

        font = self._get_font(size, bold)
        bbox = font.getbbox(text)
        x0, y0, x1, y1 = bbox
        w = max(1, x1 - x0 + 2)
        h = max(1, y1 - y0 + 2)

        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        draw.text((-x0 + 1, -y0 + 1), text, font=font, fill=255)
        mask = np.array(img, dtype=np.uint8)

        if len(self._cache) >= self._max_cache:
            # Xoa 1/4 cache cu nhat de giu bo nho gon nhe
            for k in list(self._cache.keys())[: self._max_cache // 4]:
                del self._cache[k]

        item = (mask, w, h, y0)
        self._cache[key] = item
        return item

    def measure(self, text: str, size: int = 13, bold: bool = False):
        if not text:
            return 0, size
        if self._use_pil:
            _, w, h, _ = self._get_mask(text, size, bold)
            return w, h
        scale = max(0.32, size / 32.0)
        thick = 2 if bold and size >= 16 else 1
        (tw, th), _ = cv2.getTextSize(_strip_accents(text), cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        return tw, th

    def draw(self, canvas, text: str, x: int, y: int, size: int = 13,
             color=TXT_WHITE, bold: bool = False, align: str = "left"):
        """Ve text voi (x, y) la goc tren-trai (hoac giua/phai theo align)."""
        if not text:
            return 0

        if not self._use_pil:
            clean = _strip_accents(text)
            scale = max(0.32, size / 32.0)
            thick = 2 if bold and size >= 16 else 1
            (tw, th), _ = cv2.getTextSize(clean, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            if align == "center":
                x = int(x - tw // 2)
            elif align == "right":
                x = int(x - tw)
            cv2.putText(canvas, clean, (int(x), int(y + th)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            return tw

        mask, w, h, y0 = self._get_mask(text, size, bold)
        if align == "center":
            x = int(x - w // 2)
        elif align == "right":
            x = int(x - w)
        else:
            x = int(x)
        y = int(y + y0)

        ch, cw = canvas.shape[:2]
        sx0 = max(0, -x)
        sy0 = max(0, -y)
        dx0 = max(0, x)
        dy0 = max(0, y)
        dx1 = min(cw, x + w)
        dy1 = min(ch, y + h)
        if dx1 <= dx0 or dy1 <= dy0:
            return w

        alpha = mask[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)].astype(np.uint16)[..., None]
        roi = canvas[dy0:dy1, dx0:dx1].astype(np.uint16)
        col = np.array(color, dtype=np.uint16).reshape(1, 1, 3)
        canvas[dy0:dy1, dx0:dx1] = ((roi * (255 - alpha) + col * alpha) >> 8).astype(np.uint8)
        return w

def _fill_rounded_rect(img, x1, y1, x2, y2, r, color):
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if r == 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, cv2.FILLED)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, cv2.FILLED)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, cv2.FILLED, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, cv2.FILLED, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, cv2.FILLED, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, cv2.FILLED, cv2.LINE_AA)


def _stroke_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=1):
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if r == 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)


def _draw_card(img, x1, y1, x2, y2, r=8, bg=BG_CARD, border=BORDER_CARD, glow=False):
    if glow:
        _stroke_rounded_rect(img, x1 - 1, y1 - 1, x2 + 1, y2 + 1, r + 1, BORDER_GLOW, 1)
    _fill_rounded_rect(img, x1, y1, x2, y2, r, bg)
    _stroke_rounded_rect(img, x1, y1, x2, y2, r, border, 1)


def _draw_car_icon(img, cx, cy):
    # Vom noc xe
    roof = np.array([
        [cx - 17, cy + 1],
        [cx - 11, cy - 11],
        [cx + 10, cy - 11],
        [cx + 18, cy + 1],
    ], dtype=np.int32)
    cv2.polylines(img, [roof], False, COL_CYAN, 2, cv2.LINE_AA)
    # Than xe
    _stroke_rounded_rect(img, cx - 26, cy + 1, cx + 26, cy + 13, 5, TXT_WHITE, 2)
    # Kinh chan gio
    cv2.line(img, (cx - 11, cy - 2), (cx + 11, cy - 2), COL_TINT_BORDER, 1, cv2.LINE_AA)
    # Den pha
    cv2.circle(img, (cx - 18, cy + 7), 3, COL_AMBER, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx + 18, cy + 7), 3, COL_AMBER, cv2.FILLED, cv2.LINE_AA)
    # Banh xe
    cv2.ellipse(img, (cx - 15, cy + 14), (5, 3), 0, 0, 180, TXT_SOFT, 2, cv2.LINE_AA)
    cv2.ellipse(img, (cx + 15, cy + 14), (5, 3), 0, 0, 180, TXT_SOFT, 2, cv2.LINE_AA)
    # Duong lan ben duoi
    cv2.line(img, (cx - 30, cy + 19), (cx + 30, cy + 19), COL_TINT_BORDER, 1, cv2.LINE_AA)


def _draw_camera_icon(img, x, y, color=COL_CYAN):
    _fill_rounded_rect(img, x, y + 3, x + 18, y + 15, 3, color)
    cv2.rectangle(img, (x + 5, y + 1), (x + 12, y + 4), color, cv2.FILLED)
    cv2.circle(img, (x + 9, y + 9), 4, BG_HEADER_BAR, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (x + 9, y + 9), 2, color, cv2.FILLED, cv2.LINE_AA)


def _draw_pipeline_icon(img, x, y, color=COL_CYAN):
    cv2.circle(img, (x + 4, y + 5), 2, color, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (x + 13, y + 5), 2, color, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (x + 8, y + 13), 2, COL_GREEN, cv2.FILLED, cv2.LINE_AA)
    cv2.line(img, (x + 4, y + 6), (x + 8, y + 12), color, 1, cv2.LINE_AA)
    cv2.line(img, (x + 13, y + 6), (x + 8, y + 12), color, 1, cv2.LINE_AA)


def _draw_shield_icon(img, x, y, color=COL_CYAN):
    pts = np.array([
        [x + 8, y + 1],
        [x + 15, y + 4],
        [x + 14, y + 11],
        [x + 8, y + 16],
        [x + 2, y + 11],
        [x + 1, y + 4],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.line(img, (x + 8, y + 5), (x + 8, y + 10), BG_HEADER_BAR, 2, cv2.LINE_AA)
    cv2.circle(img, (x + 8, y + 13), 1, BG_HEADER_BAR, cv2.FILLED)


def _draw_warning_triangle(img, cx, cy, size=24, color=TXT_MUTED):
    pts = np.array([
        [cx, cy - size],
        [cx + int(size * 1.15), cy + int(size * 0.85)],
        [cx - int(size * 1.15), cy + int(size * 0.85)],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - int(size * 0.38)), (cx, cy + int(size * 0.25)), color, 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy + int(size * 0.56)), 2, color, cv2.FILLED, cv2.LINE_AA)


def _draw_uart_icon(img, x, y, color=COL_CYAN):
    _stroke_rounded_rect(img, x, y, x + 22, y + 16, 3, color, 1)
    cv2.circle(img, (x + 6, y + 6), 2, COL_GREEN, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (x + 14, y + 6), 2, color, cv2.FILLED, cv2.LINE_AA)
    cv2.line(img, (x + 5, y + 12), (x + 17, y + 12), color, 1, cv2.LINE_AA)


def _draw_power_icon(img, cx, cy, r=6, color=TXT_WHITE):
    cv2.ellipse(img, (cx, cy + 1), (r, r), 0, -35, 215, color, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - r), (cx, cy - 1), color, 2, cv2.LINE_AA)

class RobotDashboardUI:
    TOTAL_W  = 1280
    TOTAL_H  = 720
    PAD_X    = 12
    GAP      = 8

    HEADER_H = 58
    TOOL_H   = 40
    CAM_H    = 332
    LOWER_H  = 228
    BOT_H    = 38

    # Che do Quan sat thuan (giu nguyen tong chieu cao 720px, phong to 2 camera)
    OBS_CAM_H = 508
    OBS_HUD_H = 52

    CARD_W = (TOTAL_W - PAD_X * 2 - 10) // 2   # 623px moi card trai/phai

    def __init__(self, win_name="Xe Tu Hanh", detail_mode=True):
        self.win_name = win_name
        self.available = False
        self.detail_mode = detail_mode
        self._pending_action = None
        self._hover_action = None
        self._buttons = {}
        self._txt = _TextEngine()
        self._last_render_t = time.time()
        self._ui_fps = 0.0

        # Tao nen tinh 1 lan de tiet kiem CPU moi frame
        self._base_canvas = self._make_base_canvas()

        try:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.win_name, self.TOTAL_W, self.TOTAL_H)
            cv2.setMouseCallback(self.win_name, self._on_mouse)
            self.available = True
        except cv2.error:
            self.available = False

    def _make_base_canvas(self):
        canvas = np.full((self.TOTAL_H, self.TOTAL_W, 3), BG_CANVAS, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (self.TOTAL_W, self.HEADER_H - 1), BG_HEADER_TOP, cv2.FILLED)
        cv2.line(canvas, (0, self.HEADER_H - 1), (self.TOTAL_W, self.HEADER_H - 1), BORDER_DIM, 1)
        return canvas

    def _on_mouse(self, event, x, y, flags, param):
        hovered = None
        for action, (x1, y1, x2, y2) in self._buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                hovered = action
                break
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover_action = hovered
        elif event == cv2.EVENT_LBUTTONDOWN and hovered is not None:
            self._pending_action = hovered

    def _draw_button(self, canvas, rect, text, bg=BG_PILL, border=BORDER_BTN,
                     text_color=TXT_WHITE, bold=False, icon=None, icon_color=None):
        x1, y1, x2, y2 = rect
        _fill_rounded_rect(canvas, x1, y1, x2, y2, 6, bg)
        _stroke_rounded_rect(canvas, x1, y1, x2, y2, 6, border, 1)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ty = cy - 9
        ic = icon_color if icon_color is not None else TXT_SOFT
        if icon == "swap":
            tw, _ = self._txt.measure(text, 13, bold)
            start_x = cx - (tw + 16) // 2
            cv2.line(canvas, (start_x, cy - 2), (start_x + 10, cy - 2), ic, 1, cv2.LINE_AA)
            cv2.line(canvas, (start_x + 7, cy - 5), (start_x + 10, cy - 2), ic, 1, cv2.LINE_AA)
            cv2.line(canvas, (start_x, cy + 3), (start_x + 10, cy + 3), ic, 1, cv2.LINE_AA)
            cv2.line(canvas, (start_x + 3, cy + 6), (start_x, cy + 3), ic, 1, cv2.LINE_AA)
            self._txt.draw(canvas, text, start_x + 15, ty, 13, text_color, bold)
        elif icon == "dot":
            tw, _ = self._txt.measure(text, 13, bold)
            start_x = cx - (tw + 12) // 2
            cv2.circle(canvas, (start_x + 3, cy), 4, ic, cv2.FILLED, cv2.LINE_AA)
            self._txt.draw(canvas, text, start_x + 12, ty, 13, text_color, bold)
        elif icon == "power":
            tw, _ = self._txt.measure(text, 13, bold)
            start_x = cx - (tw + 20) // 2
            _draw_power_icon(canvas, start_x + 6, cy, r=6, color=ic)
            self._txt.draw(canvas, text, start_x + 20, ty, 13, text_color, bold)
        else:
            self._txt.draw(canvas, text, cx, ty, 13, text_color, bold, align="center")

    def _draw_header(self, canvas, l_cam_fps, s_cam_fps, l_fps, s_fps):
        _draw_car_icon(canvas, 46, 28)
        self._txt.draw(
            canvas,
            "XE ROBOT NHẬN DIỆN BIỂN BÁO",
            86, 7, size=20, color=TXT_WHITE, bold=True,
        )

        bx2 = self.TOTAL_W - self.PAD_X
        bx1 = bx2 - 284
        _draw_card(canvas, bx1, 7, bx2, self.HEADER_H - 8, r=6, bg=BG_SUBCARD, border=BORDER_SUB, glow=False)
        self._txt.draw(canvas, "FPS:", bx1 + 12, 11, size=13, color=TXT_SOFT, bold=True)
        self._txt.draw(canvas, f"{self._ui_fps:4.1f}", bx1 + 48, 10, size=15, color=COL_GREEN, bold=True)

        sub_y = 30
        parts = [
            ("C1:", f"{l_cam_fps:.0f}"),
            ("C2:", f"{s_cam_fps:.0f}"),
            ("Lane:", f"{l_fps:.0f}"),
            ("YOLO:", f"{s_fps:.0f}"),
        ]
        px = bx1 + 12
        for idx, (lbl, val) in enumerate(parts):
            w1 = self._txt.draw(canvas, lbl, px, sub_y, size=12, color=TXT_MUTED)
            px += w1 + 2
            w2 = self._txt.draw(canvas, val, px, sub_y, size=12, color=COL_CYAN, bold=True)
            px += w2 + 8
            if idx < len(parts) - 1:
                self._txt.draw(canvas, "|", px - 5, sub_y, size=12, color=BORDER_DIM)
                px += 4

    def _draw_toolbar(self, canvas, lane_cam_lbl, sign_cam_lbl, fov_pct, lane_cam_ok, sign_cam_ok):
        y1 = self.HEADER_H + 5
        y2 = self.HEADER_H + self.TOOL_H - 5

        def _pill_bg(action, tinted=False):
            if tinted:
                return COL_TINT_HOVER if self._hover_action == action else COL_TINT_BG
            return BG_PILL_HOVER if self._hover_action == action else BG_PILL

        x = self.PAD_X
        self._buttons = {}
        b_lane = (x, y1, x + 178, y2)
        self._buttons["switch_lane_cam"] = b_lane
        self._draw_button(
            canvas, b_lane, f"Làn: {lane_cam_lbl[:10]} [C]",
            bg=_pill_bg("switch_lane_cam"), border=BORDER_BTN, text_color=TXT_WHITE,
            icon="dot", icon_color=COL_GREEN if lane_cam_ok else COL_RED,
        )
        x = b_lane[2] + 8
        b_sign = (x, y1, x + 204, y2)
        self._buttons["switch_sign_cam"] = b_sign
        self._draw_button(
            canvas, b_sign, f"Biển: {sign_cam_lbl[:13]} [V]",
            bg=_pill_bg("switch_sign_cam"), border=BORDER_BTN, text_color=TXT_WHITE,
            icon="dot", icon_color=COL_AMBER if sign_cam_ok else COL_RED,
        )
        x = b_sign[2] + 8
        b_swap = (x, y1, x + 124, y2)
        self._buttons["swap_cams"] = b_swap
        self._draw_button(
            canvas, b_swap, "Đổi chỗ [X]",
            bg=_pill_bg("swap_cams"), border=BORDER_BTN, text_color=TXT_WHITE,
            icon="swap", icon_color=COL_CYAN,
        )
        x = b_swap[2] + 8
        b_fmin = (x, y1, x + 30, y2)
        b_frst = (x + 33, y1, x + 127, y2)
        b_fpls = (x + 130, y1, x + 160, y2)
        self._buttons["fov_minus"] = b_fmin
        self._buttons["fov_reset"] = b_frst
        self._buttons["fov_plus"] = b_fpls
        self._draw_button(canvas, b_fmin, "-", bg=_pill_bg("fov_minus"), border=BORDER_BTN, text_color=TXT_SOFT, bold=True)
        self._draw_button(
            canvas, b_frst, f"FOV: {fov_pct}%",
            bg=_pill_bg("fov_reset", tinted=True),
            border=COL_TINT_BORDER,
            text_color=COL_BLUE, bold=True,
        )
        self._draw_button(canvas, b_fpls, "+", bg=_pill_bg("fov_plus"), border=BORDER_BTN, text_color=TXT_SOFT, bold=True)
        x = b_fpls[2] + 8

        b_mode = (x, y1, x + 138, y2)
        self._buttons["toggle_mode"] = b_mode
        mode_lbl = "Quan sát [D]" if self.detail_mode else "Chi tiết [D]"
        self._draw_button(
            canvas, b_mode, mode_lbl,
            bg=_pill_bg("toggle_mode", tinted=not self.detail_mode),
            border=COL_TINT_BORDER if not self.detail_mode else BORDER_BTN,
            text_color=COL_BLUE if not self.detail_mode else TXT_WHITE,
        )

        # Nut tat he thong
        st_x2 = self.TOTAL_W - self.PAD_X
        st_x1 = st_x2 - 126
        b_quit = (st_x1, y1, st_x2, y2)
        self._buttons["quit_app"] = b_quit
        self._draw_button(
            canvas, b_quit, "Tắt [Q]",
            bg=BTN_QUIT_BG_HOVER if self._hover_action == "quit_app" else BTN_QUIT_BG,
            border=BTN_QUIT_BORDER, text_color=TXT_WHITE, bold=True,
            icon="power", icon_color=TXT_WHITE,
        )

    def _draw_camera_card(self, canvas, x1, y1, w, h, frame, title, source_pill,
                          live_ok, status_text, status_col, hud_left="", hud_right="",
                          hud_right_col=TXT_SOFT, placeholder="Đang chờ tín hiệu Camera..."):
        x2, y2 = x1 + w, y1 + h
        _draw_card(canvas, x1, y1, x2, y2, r=8, bg=BG_CARD, border=BORDER_CARD, glow=False)

        hdr_h = 32
        # Dai tieu de the noi bat nhe de phan tach phan tieu de va khung video
        _fill_rounded_rect(canvas, x1 + 1, y1 + 1, x2 - 1, y1 + hdr_h, 7, BG_HEADER_BAR)
        cv2.rectangle(canvas, (x1 + 1, y1 + hdr_h - 6), (x2 - 1, y1 + hdr_h), BG_HEADER_BAR, cv2.FILLED)
        cv2.line(canvas, (x1 + 1, y1 + hdr_h), (x2 - 1, y1 + hdr_h), BORDER_DIM, 1)

        _draw_camera_icon(canvas, x1 + 12, y1 + 8, color=COL_CYAN)
        self._txt.draw(canvas, title, x1 + 38, y1 + 7, size=14, color=TXT_WHITE, bold=True)

        pw, _ = self._txt.measure(source_pill, 12, bold=True)
        pill_w = pw + 16
        px2 = x2 - 10
        px1 = px2 - pill_w
        _fill_rounded_rect(canvas, px1, y1 + 6, px2, y1 + 26, 5, COL_TINT_BG if live_ok else BG_SUBCARD)
        _stroke_rounded_rect(canvas, px1, y1 + 6, px2, y1 + 26, 5, COL_TINT_BORDER if live_ok else BORDER_SUB, 1)
        self._txt.draw(canvas, source_pill, (px1 + px2) // 2, y1 + 8, size=12, color=COL_BLUE if live_ok else TXT_MUTED, bold=True, align="center")

        vx1, vy1 = x1 + 6, y1 + hdr_h + 6
        vx2, vy2 = x2 - 6, y2 - 6
        vw, vh = vx2 - vx1, vy2 - vy1

        if frame is not None and frame.size > 0:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
            canvas[vy1:vy2, vx1:vx2] = cv2.resize(bgr, (vw, vh), interpolation=cv2.INTER_AREA)
        else:
            _fill_rounded_rect(canvas, vx1, vy1, vx2, vy2, 6, BG_SUBCARD)
            self._txt.draw(canvas, placeholder, (vx1 + vx2) // 2, (vy1 + vy2) // 2 - 8,
                           size=14, color=TXT_MUTED, align="center")

        if hud_left or hud_right:
            ov_h = 26
            oy1 = vy2 - ov_h
            roi = canvas[oy1:vy2, vx1:vx2]
            dark = np.full_like(roi, BG_SUBCARD)
            cv2.addWeighted(dark, 0.78, roi, 0.22, 0, roi)
            if hud_left:
                self._txt.draw(canvas, hud_left, vx1 + 10, oy1 + 5, size=13, color=TXT_WHITE)
            if hud_right:
                self._txt.draw(canvas, hud_right, vx2 - 10, oy1 + 5, size=13, color=hud_right_col, bold=True, align="right")

        cv2.rectangle(canvas, (vx1, vy1), (vx2 - 1, vy2 - 1), BORDER_DIM, 1)

    def _draw_pipeline_card(self, canvas, x1, y1, w, h, lane_thres, lane_warp, lane_hist):
        x2, y2 = x1 + w, y1 + h
        _draw_card(canvas, x1, y1, x2, y2, r=8, bg=BG_CARD, border=BORDER_CARD, glow=False)

        hdr_h = 30
        _fill_rounded_rect(canvas, x1 + 1, y1 + 1, x2 - 1, y1 + hdr_h, 7, BG_HEADER_BAR)
        cv2.rectangle(canvas, (x1 + 1, y1 + hdr_h - 6), (x2 - 1, y1 + hdr_h), BG_HEADER_BAR, cv2.FILLED)
        cv2.line(canvas, (x1 + 1, y1 + hdr_h), (x2 - 1, y1 + hdr_h), BORDER_DIM, 1)

        _draw_pipeline_icon(canvas, x1 + 12, y1 + 7, color=COL_CYAN)
        self._txt.draw(canvas, "XỬ LÝ LÀN ĐƯỜNG (LANE PIPELINE)",
                       x1 + 36, y1 + 6, size=13, color=TXT_WHITE, bold=True)

        # Chuyen Threshold thanh xanh ngoc dong bo tren nen toi sau
        thres_vis = None
        if lane_thres is not None and lane_thres.size > 0:
            thres_vis = np.full((lane_thres.shape[0], lane_thres.shape[1], 3), (18, 15, 13), dtype=np.uint8)
            thres_vis[lane_thres > 0] = COL_GREEN

        pad = 10
        gap = 8
        cell_w = (w - pad * 2 - gap * 2) // 3
        cell_y1 = y1 + hdr_h + 8
        cell_y2 = y2 - 10
        cell_h = cell_y2 - cell_y1
        title_h = 26

        items = [
            ("1. Ngưỡng (Threshold)", thres_vis),
            ("2. Biến dạng mắt chim", lane_warp),
            ("3. Biểu đồ (Histogram)", lane_hist),
        ]

        for idx, (label, img) in enumerate(items):
            cx1 = x1 + pad + idx * (cell_w + gap)
            cx2 = cx1 + cell_w
            _fill_rounded_rect(canvas, cx1, cell_y1, cx2, cell_y2, 6, BG_SUBCARD)
            _stroke_rounded_rect(canvas, cx1, cell_y1, cx2, cell_y2, 6, BORDER_SUB, 1)

            self._txt.draw(canvas, label, (cx1 + cx2) // 2, cell_y1 + 5,
                           size=12, color=TXT_SOFT, bold=True, align="center")

            ix1, iy1 = cx1 + 3, cell_y1 + title_h
            ix2, iy2 = cx2 - 3, cell_y2 - 3
            iw, ih = ix2 - ix1, iy2 - iy1
            if img is not None and img.size > 0:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
                canvas[iy1:iy2, ix1:ix2] = cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA)
            else:
                self._txt.draw(canvas, "Chờ dữ liệu...", (ix1 + ix2) // 2, (iy1 + iy2) // 2 - 7,
                               size=12, color=TXT_MUTED, align="center")

    def _draw_progress_bar(self, canvas, x, y, w, h, ratio, fill_col_start, fill_col_end):
        ratio = float(np.clip(ratio, 0.0, 1.0))
        _fill_rounded_rect(canvas, x, y, x + w, y + h, 4, BG_TRACK)
        draw_w = min(int(round(w * ratio)), max(0, w - 2))
        if draw_w > 0:
            grad = np.zeros((h - 2, draw_w, 3), dtype=np.uint8)
            t = np.linspace(0.0, 1.0, draw_w, dtype=np.float32).reshape(1, draw_w, 1)
            c0 = np.array(fill_col_start, dtype=np.float32).reshape(1, 1, 3)
            c1 = np.array(fill_col_end, dtype=np.float32).reshape(1, 1, 3)
            grad[:] = (c0 * (1.0 - t) + c1 * t).astype(np.uint8)
            canvas[y + 1:y + h - 1, x + 1:x + 1 + draw_w] = grad
        _stroke_rounded_rect(canvas, x, y, x + w, y + h, 4, BORDER_SUB, 1)

    def _draw_bipolar_bar(self, canvas, x, y, w, h, val):
        val = float(np.clip(val, -1.0, 1.0))
        _fill_rounded_rect(canvas, x, y, x + w, y + h, 4, BG_TRACK)
        mid_x = x + w // 2
        fill_x = mid_x + int(round((w // 2 - 2) * val))
        col = COL_GREEN if abs(val) < 0.25 else (COL_AMBER if abs(val) < 0.6 else COL_ORANGE)
        if fill_x > mid_x:
            cv2.rectangle(canvas, (mid_x, y + 2), (fill_x, y + h - 2), col, cv2.FILLED)
        elif fill_x < mid_x:
            cv2.rectangle(canvas, (fill_x, y + 2), (mid_x, y + h - 2), col, cv2.FILLED)
        cv2.line(canvas, (mid_x, y - 2), (mid_x, y + h + 2), TXT_SOFT, 1, cv2.LINE_AA)
        _stroke_rounded_rect(canvas, x, y, x + w, y + h, 4, BORDER_SUB, 1)

    def _draw_sign_control_card(self, canvas, x1, y1, w, h, sign_dbg, sign_info,
                                lane_state, lane_info, corner_info, light_mode,
                                curve_val, speed_pct, err_val,
                                lane_age_ms, sign_age_ms, esp_ok, esp_port):
        x2, y2 = x1 + w, y1 + h
        _draw_card(canvas, x1, y1, x2, y2, r=8, bg=BG_CARD, border=BORDER_CARD, glow=False)

        hdr_h = 30
        _fill_rounded_rect(canvas, x1 + 1, y1 + 1, x2 - 1, y1 + hdr_h, 7, BG_HEADER_BAR)
        cv2.rectangle(canvas, (x1 + 1, y1 + hdr_h - 6), (x2 - 1, y1 + hdr_h), BG_HEADER_BAR, cv2.FILLED)
        cv2.line(canvas, (x1 + 1, y1 + hdr_h), (x2 - 1, y1 + hdr_h), BORDER_DIM, 1)

        _draw_shield_icon(canvas, x1 + 12, y1 + 6, color=COL_CYAN)
        self._txt.draw(canvas, "BIỂN BÁO & ĐIỀU KHIỂN XE",
                       x1 + 36, y1 + 6, size=13, color=TXT_WHITE, bold=True)

        pad = 10
        by1 = y1 + hdr_h + 8
        by2 = y2 - 10

        # --- Hop trai: Hien thi bien bao YOLO11n ---
        left_w = 236
        lx1, lx2 = x1 + pad, x1 + pad + left_w
        _fill_rounded_rect(canvas, lx1, by1, lx2, by2, 6, BG_SUBCARD)
        _stroke_rounded_rect(canvas, lx1, by1, lx2, by2, 6, BORDER_SUB, 1)

        previews = sign_dbg.get("yolo_previews", sign_dbg.get("onnx_previews", [])) if sign_dbg else []
        if previews:
            for idx, prev in enumerate(previews[:3]):
                ry = by1 + 10 + idx * 56
                crop = prev.get("crop_bgr")
                cls_name = prev.get("cls_name", "?")
                sc = float(prev.get("score", 0.0))
                ok = bool(prev.get("accepted", False))

                if crop is not None and crop.size > 0:
                    canvas[ry:ry + 44, lx1 + 10:lx1 + 54] = cv2.resize(crop, (44, 44), interpolation=cv2.INTER_AREA)
                b_col = COL_GREEN if ok else COL_AMBER
                cv2.rectangle(canvas, (lx1 + 10, ry), (lx1 + 54, ry + 44), b_col, 1, cv2.LINE_AA)

                vn_name = sign_info.get(cls_name, {"vn": cls_name})["vn"]
                self._txt.draw(canvas, vn_name[:18], lx1 + 64, ry + 4, size=13, color=TXT_WHITE, bold=True)
                st_txt = "NHẬN" if ok else "THẤP"
                self._txt.draw(canvas, f"{sc:.0%}  [{st_txt}]", lx1 + 64, ry + 24, size=12, color=b_col, bold=True)
        else:
            cx = (lx1 + lx2) // 2
            cy = (by1 + by2) // 2 - 14
            _draw_warning_triangle(canvas, cx, cy, size=24, color=COL_AMBER)
            self._txt.draw(canvas, "Chưa phát hiện biển báo", cx, cy + 32,
                           size=13, color=TXT_SOFT, bold=True, align="center")

        # --- Hop phai: Dieu khien dong co & thong so telemetry ---
        rx1 = lx2 + 10
        rx2 = x2 - pad
        rw = rx2 - rx1
        _fill_rounded_rect(canvas, rx1, by1, rx2, by2, 6, BG_SUBCARD)
        _stroke_rounded_rect(canvas, rx1, by1, rx2, by2, 6, BORDER_SUB, 1)

        bx = rx1 + 12
        bw = rw - 24

        # 1. Thanh toc do dong co
        self._txt.draw(canvas, "Tốc độ động cơ", bx, by1 + 6, size=13, color=TXT_WHITE, bold=True)
        self._txt.draw(canvas, f"{speed_pct}%", bx + bw, by1 + 6, size=14, color=TXT_WHITE, bold=True, align="right")
        if speed_pct == 0:
            c0, c1 = COL_RED_DARK, COL_RED
        elif speed_pct < 60:
            c0, c1 = COL_AMBER_DARK, COL_AMBER
        else:
            c0, c1 = COL_GREEN_DARK, COL_GREEN
        self._draw_progress_bar(canvas, bx, by1 + 26, bw, 13, abs(speed_pct) / 100.0, c0, c1)

        # 2. Thanh goc lai / lech tam
        self._txt.draw(canvas, "Góc lái / Lệch tâm", bx, by1 + 45, size=12, color=TXT_SOFT)
        self._txt.draw(canvas, f"Err: {err_val:+.2f}  |  Curve: {curve_val:+.2f}",
                       bx + bw, by1 + 45, size=12, color=TXT_WHITE, bold=True, align="right")
        self._draw_bipolar_bar(canvas, bx, by1 + 63, bw, 9, err_val)

        # 3. Bang 5 dong thong so
        conf = float(lane_info.get("confidence", 0.0))
        c_state = corner_info.get("state", "NONE") if isinstance(corner_info, dict) else "NONE"
        c_dir = corner_info.get("direction", "-") if isinstance(corner_info, dict) else "-"
        esp_txt = f"Kết nối ({esp_port})" if esp_ok else "Mô phỏng (Chưa có cổng)"
        delay_col = COL_RED if lane_age_ms > 250 else TXT_WHITE

        rows = [
            ("Làn đường :", f"{lane_state} ({conf:.0%})", COL_GREEN),
            ("Góc cua :",   f"{c_state} ({c_dir})",        TXT_WHITE),
            ("Thời trễ :",  f"Lane: {lane_age_ms:.0f} ms  |  Sign: {sign_age_ms:.0f} ms", delay_col),
            ("Ánh sáng :",  f"{light_mode}",               COL_CYAN),
            ("ESP32 :",     esp_txt,                       COL_GREEN if esp_ok else TXT_SOFT),
        ]
        for idx, (k, v, v_col) in enumerate(rows):
            ry = by1 + 80 + idx * 19
            self._txt.draw(canvas, k, bx, ry, size=12, color=TXT_MUTED)
            self._txt.draw(canvas, v[:34], bx + 84, ry, size=12, color=v_col, bold=(idx in (0, 3)))

    def _draw_compact_hud_strip(self, canvas, y1, speed_pct, err_val, curve_val,
                                lane_state, conf, light_mode, esp_ok, esp_port):
        x1 = self.PAD_X
        x2 = self.TOTAL_W - self.PAD_X
        y2 = y1 + self.OBS_HUD_H
        _draw_card(canvas, x1, y1, x2, y2, r=8, bg=BG_CARD, border=BORDER_CARD, glow=False)

        # Thanh toc do
        self._txt.draw(canvas, "Tốc độ", x1 + 14, y1 + 6, size=12, color=TXT_SOFT)
        self._txt.draw(canvas, f"{speed_pct}%", x1 + 244, y1 + 6, size=13, color=TXT_WHITE, bold=True, align="right")
        c0, c1 = (COL_RED_DARK, COL_RED) if speed_pct == 0 else (COL_GREEN_DARK, COL_GREEN)
        self._draw_progress_bar(canvas, x1 + 14, y1 + 26, 230, 12, abs(speed_pct) / 100.0, c0, c1)

        # Thanh goc lai
        self._txt.draw(canvas, "Góc lái", x1 + 272, y1 + 6, size=12, color=TXT_SOFT)
        self._txt.draw(canvas, f"Err: {err_val:+.2f} | C: {curve_val:+.2f}",
                       x1 + 542, y1 + 6, size=12, color=TXT_WHITE, bold=True, align="right")
        self._draw_bipolar_bar(canvas, x1 + 272, y1 + 27, 270, 11, err_val)

        # Thong so cot loi (khi bang chi tiet dang an)
        esp_lbl = f"ESP32 ({esp_port})" if esp_ok else "Mô phỏng (SIM)"
        self._txt.draw(canvas, f"Làn: {lane_state} ({conf:.0%})", x1 + 576, y1 + 16, size=13, color=COL_GREEN, bold=True)
        self._txt.draw(canvas, f"Ánh sáng: {light_mode}", x1 + 810, y1 + 16, size=13, color=COL_CYAN, bold=True)
        self._txt.draw(canvas, esp_lbl, x2 - 16, y1 + 16, size=13,
                       color=COL_GREEN if esp_ok else TXT_SOFT, bold=True, align="right")

    def _draw_bottom_bar(self, canvas, pkt):
        y1 = self.TOTAL_H - self.BOT_H
        cv2.rectangle(canvas, (0, y1), (self.TOTAL_W, self.TOTAL_H), BG_HEADER_TOP, cv2.FILLED)
        cv2.line(canvas, (0, y1), (self.TOTAL_W, y1), BORDER_DIM, 1)

        _draw_uart_icon(canvas, self.PAD_X + 2, y1 + 11, color=COL_CYAN)
        is_stop = "STOP" in pkt
        pkt_col = COL_RED if is_stop else COL_GREEN
        self._txt.draw(canvas, f"UART: {pkt}", self.PAD_X + 34, y1 + 10, size=14, color=pkt_col, bold=True)

        keys_txt = "[C] Cam1    [V] Cam2    [X] Hoán đổi    [+/-] FOV    [D] Chế độ    [Q] Thoát"
        self._txt.draw(canvas, keys_txt, self.TOTAL_W - self.PAD_X, y1 + 11,
                       size=13, color=TXT_SOFT, align="right")

    def render_and_poll(self, state, cam_rig, sign_info):
        if not self.available:
            return True

        now = time.time()
        dt = now - self._last_render_t
        self._last_render_t = now
        if dt > 0.001:
            inst_fps = min(120.0, 1.0 / dt)
            self._ui_fps = inst_fps if self._ui_fps == 0.0 else (self._ui_fps * 0.85 + inst_fps * 0.15)

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

        # Ve bounding box bien bao len khung hinh Cam 2
        sign_vis = raw_sign_frame.copy() if raw_sign_frame is not None else (
            sign_dbg["yolo_vis"].copy() if ("yolo_vis" in sign_dbg and sign_dbg["yolo_vis"] is not None) else None
        )
        if sign_vis is not None:
            for prev in sign_dbg.get("yolo_previews", []):
                if not prev.get("accepted", False):
                    bx, by, bw, bh = prev["box"]
                    cv2.rectangle(sign_vis, (bx, by), (bx + bw, by + bh), COL_AMBER, 1, cv2.LINE_AA)
            for cls_name, x, y, w, h, score in signs:
                info = sign_info.get(cls_name, {"vn": cls_name, "color": COL_GREEN})
                col = info["color"]
                label = f"{info['vn']} ({score:.0%})"
                cv2.rectangle(sign_vis, (x, y), (x + w, y + h), col, 2, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                tx = max(4, min(x, sign_vis.shape[1] - tw - 10))
                ty = max(th + 8, y - 5)
                cv2.rectangle(sign_vis, (tx - 2, ty - th - 5), (tx + tw + 6, ty + 3), BG_SUBCARD, -1)
                cv2.putText(sign_vis, label, (tx + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_AMBER, 1, cv2.LINE_AA)

        lane_cam_lbl = cam_rig.get_lane_label()
        sign_cam_lbl = cam_rig.get_sign_label()
        conf = float(l_info.get("confidence", 0.0))

        dashboard = self._base_canvas.copy()

        # 1. Thanh Tieu de & FPS
        self._draw_header(dashboard, l_cam_fps, s_cam_fps, l_fps, s_fps)

        # 2. Thanh cong cu dieu khien
        self._draw_toolbar(dashboard, lane_cam_lbl, sign_cam_lbl, fov_pct, lane_cam_ok, sign_cam_ok or shared_cam)

        # 3. Hang 2 Camera
        cam_y = self.HEADER_H + self.TOOL_H + self.GAP
        cam_h = self.CAM_H if self.detail_mode else self.OBS_CAM_H
        left_x = self.PAD_X
        right_x = self.PAD_X + self.CARD_W + 10

        # Cam 1 khong lap lai HUD Trang thai/Sang/Err vi da co trong bang Dieu khien ben duoi
        self._draw_camera_card(
            dashboard, left_x, cam_y, self.CARD_W, cam_h,
            frame=lane_vis,
            title="CAM 1 – QUAN SÁT LÀN ĐƯỜNG",
            source_pill=lane_cam_lbl,
            live_ok=lane_cam_ok,
            status_text="LIVE" if lane_cam_ok else "MẤT TÍN HIỆU",
            status_col=COL_GREEN if lane_cam_ok else COL_RED,
            placeholder="Đang chờ Camera 1 (Làn đường)...",
        )

        sign_hud_left = f"Phát hiện: {len(signs)} biển" if sign_dbg else "Chưa nạp model YOLO11n"
        sign_hud_right = sign_action if sign_action else "Không có lệnh ưu tiên"
        self._draw_camera_card(
            dashboard, right_x, cam_y, self.CARD_W, cam_h,
            frame=sign_vis,
            title="CAM 2 – QUAN SÁT BIỂN BÁO (YOLO11n)",
            source_pill=sign_cam_lbl,
            live_ok=sign_cam_ok or shared_cam,
            status_text="LIVE" if (sign_cam_ok or shared_cam) else "MẤT TÍN HIỆU",
            status_col=COL_GREEN if (sign_cam_ok or shared_cam) else COL_RED,
            hud_left=sign_hud_left,
            hud_right=sign_hud_right,
            hud_right_col=COL_AMBER if sign_action else TXT_MUTED,
            placeholder="Đang chờ Camera 2 (Biển báo)...",
        )

        # 4. Hang duoi (Chi tiet hoac Dai HUD thu gon)
        lower_y = cam_y + cam_h + self.GAP
        if self.detail_mode:
            self._draw_pipeline_card(
                dashboard, left_x, lower_y, self.CARD_W, self.LOWER_H,
                lane_thres, lane_warp, lane_hist,
            )
            self._draw_sign_control_card(
                dashboard, right_x, lower_y, self.CARD_W, self.LOWER_H,
                sign_dbg, sign_info, l_state, l_info, c_info, light_mode,
                curve_val, speed_pct, err_val, lane_age_ms, sign_age_ms, esp_ok, esp_port,
            )
        else:
            self._draw_compact_hud_strip(
                dashboard, lower_y, speed_pct, err_val, curve_val,
                l_state, conf, light_mode, esp_ok, esp_port,
            )

        # 5. Thanh trang thai UART & Phim tat duoi cung
        self._draw_bottom_bar(dashboard, pkt)

        # Xu ly su kien click chuot
        act = self._pending_action
        if act is not None:
            self._pending_action = None
            if act == "quit_app":
                state.running = False
                return False
            elif act == "switch_lane_cam":
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
            if cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
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
