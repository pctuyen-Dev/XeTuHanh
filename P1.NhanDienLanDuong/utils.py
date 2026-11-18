import cv2
import numpy as np


def resize_robot_frame(img, max_side=640):
    """Bound processing work without changing camera aspect or adding pixels."""
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    return cv2.resize(img, (max(1, round(w*scale)), max(1, round(h*scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()


def analyze_lighting(img, roi_points=None):
    """
    Tu dong phan tich dieu kien anh sang mat duong (Troi toi / Binh thuong / Troi sang loa).
    Tra ve dict: {"mode": "DARK" | "NORMAL" | "BRIGHT", "med_l": float, "p98_l": float, "med_v": float, "p98_v": float}.
    """
    h, w = img.shape[:2]
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    L = hls[:, :, 1]
    V = hsv[:, :, 2]

    top_y = int(h * 0.50)
    bot_y = int(h * 0.96)
    if roi_points is not None:
        pts = np.asarray(roi_points, dtype=np.float32)
        top_y = max(int(h * 0.35), min(int(min(pts[0][1], pts[1][1])), int(h * 0.75)))
        bot_y = min(int(h * 0.98), max(int(max(pts[2][1], pts[3][1])), top_y + 10))

    x_margin = int(w * 0.10)
    roi_l = L[top_y:bot_y, x_margin:w - x_margin]
    roi_v = V[top_y:bot_y, x_margin:w - x_margin]
    if roi_l.size == 0:
        roi_l = L
        roi_v = V

    med_l = float(np.median(roi_l))
    p98_l = float(np.percentile(roi_l, 98))
    med_v = float(np.median(roi_v))
    p98_v = float(np.percentile(roi_v, 98))

    if med_l < 75.0 and p98_l < 168.0:
        mode = "DARK"
    elif med_l > 152.0 or (med_l > 135.0 and p98_l > 235.0):
        mode = "BRIGHT"
    else:
        mode = "NORMAL"

    return {
        "mode": mode,
        "med_l": med_l,
        "p98_l": p98_l,
        "med_v": med_v,
        "p98_v": p98_v,
    }


def build_roi_polygon_mask(shape, roi_points, expand_x_ratio=0.0, min_top_ratio=0.05):
    """
    Tao mat na da giac hinh thang tu 4 diem ROI [(x_tl, y_tl), (x_tr, y_tr), (x_bl, y_bl), (x_br, y_br)].
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if roi_points is None:
        mask[int(h * min_top_ratio):int(h * 0.98), :] = 255
        return mask

    pts = np.asarray(roi_points, dtype=np.float32)
    pad_x = w * float(expand_x_ratio)
    tl = (max(0, int(round(pts[0][0] - pad_x))), max(int(h * min_top_ratio), int(round(pts[0][1]))))
    tr = (min(w - 1, int(round(pts[1][0] + pad_x))), max(int(h * min_top_ratio), int(round(pts[1][1]))))
    bl = (max(0, int(round(pts[2][0] - pad_x))), min(h - 1, int(round(pts[2][1]))))
    br = (min(w - 1, int(round(pts[3][0] + pad_x))), min(h - 1, int(round(pts[3][1]))))

    poly = np.array([[tl, tr, br, bl]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    return mask


def _build_zhang_suen_luts():
    lut0 = np.zeros(256, dtype=np.uint8)
    lut1 = np.zeros(256, dtype=np.uint8)
    for idx in range(256):
        p = [(idx >> i) & 1 for i in range(8)]
        cnt = sum(p)
        trans = sum((p[i] == 0 and p[(i + 1) % 8] == 1) for i in range(8))
        if 2 <= cnt <= 6 and trans == 1:
            if (p[0] * p[2] * p[4] == 0) and (p[2] * p[4] * p[6] == 0):
                lut0[idx] = 1
            if (p[0] * p[2] * p[6] == 0) and (p[0] * p[4] * p[6] == 0):
                lut1[idx] = 1
    return lut0, lut1


_ZS_LUT0, _ZS_LUT1 = _build_zhang_suen_luts()


def thin_mask(mask):
    """Zhang-Suen centerlines accelerated via 256-entry LUT on active foreground coordinates (~12x faster)."""
    skel = (mask > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if ys.size == 0:
        return np.zeros_like(mask)
    y0, y1 = max(0, int(ys.min()) - 1), min(skel.shape[0], int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(skel.shape[1], int(xs.max()) + 2)
    roi = np.pad(skel[y0:y1, x0:x1], 1)
    ay, ax = np.where(roi[1:-1, 1:-1] > 0)
    ay += 1
    ax += 1
    while ay.size > 0:
        changed = False
        for lut in (_ZS_LUT0, _ZS_LUT1):
            code = (
                (roi[ay - 1, ax] << 0)
                | (roi[ay - 1, ax + 1] << 1)
                | (roi[ay, ax + 1] << 2)
                | (roi[ay + 1, ax + 1] << 3)
                | (roi[ay + 1, ax] << 4)
                | (roi[ay + 1, ax - 1] << 5)
                | (roi[ay, ax - 1] << 6)
                | (roi[ay - 1, ax - 1] << 7)
            )
            drop = lut[code] > 0
            if np.any(drop):
                roi[ay[drop], ax[drop]] = 0
                keep = ~drop
                ay, ax = ay[keep], ax[keep]
                changed = True
        if not changed:
            break
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = roi[1:-1, 1:-1] * 255
    return out


def compute_glare_contrast(img_or_gray):
    """Perspective-scaled 2D Top-Hat contrast that suppresses glossy glare pools without bright-border bleed."""
    gray = cv2.cvtColor(img_or_gray, cv2.COLOR_BGR2GRAY) if img_or_gray.ndim == 3 else img_or_gray
    h, w = gray.shape[:2]
    d_top = max(9, int(round(min(w, h) * 0.080)) | 1)
    d_bot = max(15, int(round(min(w, h) * 0.130)) | 1)
    bg_top = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_top, d_top)))
    bg_bot = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_bot, d_bot)))
    y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    alpha = np.clip((y_norm - 0.30) / 0.40, 0.0, 1.0)
    background = ((1.0 - alpha) * bg_top + alpha * bg_bot).astype(np.uint8)
    contrast = cv2.subtract(gray, background)
    return contrast, background, y_norm


def thresholding(img, hsv_range=None, roi_points=None, restrict_roi=False):
    """Tape candidates with automatic Day/Night adaptation, specular glare suppression, and thick-blob removal."""
    h, w = img.shape[:2]
    light_info = analyze_lighting(img, roi_points=roi_points)
    mode = light_info["mode"]
    med_v = light_info["med_v"]
    p98_v = light_info["p98_v"]

    proc_img = img
    if mode == "DARK" and p98_v < 170.0:
        gamma = float(np.clip(np.log(125.0 / 255.0) / np.log(max(med_v, 12.0) / 255.0), 0.42, 0.85))
        lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        proc_img = cv2.LUT(img, lut)

    hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
    light = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    contrast, background, y_norm = compute_glare_contrast(light)

    # Local edge sharpness (Sobel gradient max within half-tape window) rejects smooth specular glare domes
    gx = cv2.Sobel(light, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(light, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy) * 0.25
    k_edge = max(13, int(round(min(w, h) * 0.088)) | 1)
    local_edge_max = cv2.dilate(gmag, np.ones((k_edge, k_edge), np.uint8))

    road_top = int(h * 0.38)
    road_bot = int(h * 0.96)
    road_gray = light[road_top:road_bot, :] if road_bot > road_top else light
    road_bg = background[road_top:road_bot, :] if road_bot > road_top else background
    # Use lower-to-mid background percentile so widespread glossy glare does not inflate mat baseline
    mat_dark_g = float(np.percentile(road_bg, 35))
    p98_g = float(np.percentile(road_gray, 99.5))

    is_dark_scene = (mode == "DARK" and p98_v < 170.0)
    contrast_thresh = max(20, min(46, int(round(max(22.0, (p98_g - mat_dark_g) * 0.26)))))
    if is_dark_scene:
        contrast_thresh = max(11, int(round(18.0 * (max(p98_v, 60.0) / 175.0))))
        v_min = max(105, int(med_v * 1.35 + 15))
        s_max = 105
        edge_min = 12.0
    elif p98_g >= 218.0:
        # Reject dim glossy sheen/crease reflections when real white tape hits near-white
        v_glare_floor = int(round(np.clip(0.81 * p98_g, 188.0, 208.0)))
        v_min = max(165, v_glare_floor)
        if hsv_range is not None:
            v_min = max(v_min, int(hsv_range[0][2]))
            s_max = min(int(hsv_range[1][1]), 58)
        else:
            s_max = 58
        edge_min = 24.0
    else:
        v_min = 165 if hsv_range is None else int(hsv_range[0][2])
        s_max = 85 if hsv_range is None else int(hsv_range[1][1])
        edge_min = 18.0

    # Neutrality check + tight colored-surface guard reject tinted specular highlights (e.g. glossy red hood)
    # without erasing clean high-contrast white tape near the orange chassis at the bottom border.
    v_minus_l = hsv[:, :, 2].astype(np.int16) - light.astype(np.int16)
    neutral_ok = v_minus_l <= (24 if is_dark_scene else 18)
    b_ch, g_ch, r_ch = [c.astype(np.float32) for c in cv2.split(proc_img)]
    chroma_dom = np.maximum(
        np.maximum(0.0, r_ch - np.maximum(g_ch, b_ch)),
        np.maximum(0.0, b_ch - np.maximum(r_ch, g_ch)),
    )
    chroma_local = cv2.boxFilter(chroma_dom, -1, (9, 9))
    hi_contrast_white = (contrast >= 70) & (hsv[:, :, 1] <= 85) & (v_minus_l <= 24)
    # Also recover shaded white tape lying over dark road (e.g. near camera edge/hood) when contrast is very sharp
    shaded_dark_mat_tape = (
        (background <= 95) & (contrast >= 80) & (hsv[:, :, 2] >= 158) & (hsv[:, :, 1] <= 65) & (v_minus_l <= 20)
    )
    not_colored_surface_glare = (chroma_local < 28.0) | ((chroma_dom < 26.0) & hi_contrast_white)

    color = (
        (
            ((((hsv[:, :, 1] <= s_max) & neutral_ok) | hi_contrast_white) & (hsv[:, :, 2] >= v_min))
            | shaded_dark_mat_tape
        )
        & not_colored_surface_glare
    )

    y_v_min = 60 if is_dark_scene else 90
    yellow = cv2.inRange(hsv, np.array([14, 48, y_v_min]), np.array([38, 255, 255])) > 0

    # Allow compressed contrast where real bright white tape passes directly through an elevated glare patch
    contrast_ok = (contrast >= contrast_thresh) | (
        (background >= 135)
        & (hsv[:, :, 2] >= 232)
        & (hsv[:, :, 1] <= 40)
        & neutral_ok
        & (contrast >= max(16, int(round(contrast_thresh * 0.45))))
    )
    mask = np.uint8((color | yellow) & contrast_ok & (local_edge_max >= edge_min)) * 255

    # Erase thick 2D glare pools, bright floor, and hood glare cores while detaching narrow lane tapes.
    # Replicate border first so wide bright floor regions touching image edges get their true thickness.
    pad = 28
    mask_pad = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    dist_pad = cv2.distanceTransform(mask_pad, cv2.DIST_L2, 3)
    max_half_w = 9.0 + 8.5 * y_norm
    max_half_w_pad = np.pad(max_half_w, ((pad, pad), (0, 0)), mode="edge")
    fat_core_pad = (dist_pad > max_half_w_pad).astype(np.uint8) * 255
    if np.any(fat_core_pad):
        dist_to_core = cv2.distanceTransform(255 - fat_core_pad, cv2.DIST_L2, 3)[pad : pad + h, pad : pad + w]
        mask[dist_to_core <= (max_half_w + 2.5)] = 0

    # Bridge tiny 2-4px breaks along the tape (such as a thin wire crossing the white line)
    kernel_bridge = np.ones((4, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_bridge)

    if restrict_roi and roi_points is not None:
        roi_mask = build_roi_polygon_mask((h, w), roi_points, expand_x_ratio=0.04, min_top_ratio=0.05)
        mask = cv2.bitwise_and(mask, roi_mask)
    else:
        mask[:3] = 0
        mask[int(h * .995):] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    min_area_std = max(14, int(w * h * 0.00012))
    min_span_std = max(10, int(w * 0.025))
    min_frag_v = min(226.0, max(v_min + 12.0, p98_g * 0.88))
    for i in range(1, count):
        x, y, cw, ch, area = stats[i]
        # Giu lai cac net dut ngan o xa (nua tren khung hinh) co be rong hop ly
        is_dash_upper = (y < int(h * 0.62)) and (area >= 14) and (max(cw, ch) >= 7) and (2 <= min(cw, ch) <= 18)
        if not ((area >= min_area_std and max(cw, ch) >= min_span_std) or is_dash_upper):
            continue
        comp_sub = labels[y : y + ch, x : x + cw] == i
        if not is_dark_scene and area < 140:
            p75_c = float(np.percentile(contrast[y : y + ch, x : x + cw][comp_sub], 75))
            p75_e = float(np.percentile(local_edge_max[y : y + ch, x : x + cw][comp_sub], 75))
            p75_v = float(np.percentile(hsv[y : y + ch, x : x + cw, 2][comp_sub], 75))
            if p75_c < max(contrast_thresh * 1.15, 45.0) or p75_e < 40.0 or p75_v < min_frag_v:
                continue
        clean[y : y + ch, x : x + cw][comp_sub] = 255
    return clean


def thresholding_highway(img, hsv_range=None, roi_points=None, restrict_roi=True):
    """
    Bo loc Vach Ke Duong Toi Uu 5 Tang ket hop TU DONG THICH UNG TROI TOI & TROI SANG:
    1. Tu dong do do sang nen mat duong (DARK / NORMAL / BRIGHT) de dieu chinh nguong L, V va Top-Hat ngang.
    2. Ket hop Morphological Horizontal Top-Hat (25x3) & BoxFilter (31x5) de tach vach dut net ban dem, loai bo den hau xe.
    3. Gioi han Vung Da Giac Mat Duong (Road Polygon Mask) mo rong theo 4 diem ROI.
    4. Bo loc Be Rong Ngang tung dong (Horizontal Run-Length Filter) & xoa dom nhieu.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    V = hsv[:, :, 2]

    top_y = int(h * 0.56)
    bot_y = int(h * 0.96)
    if roi_points is not None:
        pts = np.asarray(roi_points, dtype=np.float32)
        top_y = max(int(h * 0.42), min(int(min(pts[0][1], pts[1][1])), int(h * 0.68)))
        bot_y = min(int(h * 0.98), max(int(max(pts[2][1], pts[3][1])), int(h * 0.82)))

    light_info = analyze_lighting(img, roi_points=roi_points)
    mode = light_info["mode"]
    med_l = light_info["med_l"]
    p98_l = light_info["p98_l"]

    # 1. Vach trang: do tuong phan cuc bo ngang 31x5 + Morphology Top-Hat 25x3
    local_mean = cv2.boxFilter(L, cv2.CV_32F, (31, 5))
    diff = L.astype(np.float32) - local_mean

    if mode == "DARK":
        # CHE DO TROI TOI / BAN DEM: Dung Top-Hat ngang (25x3) + gioi han do bao hoa HSV de bat vach trang dut net, loai den xe
        kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, kernel_th)
        l_min = max(48.0, min(130.0, med_l * 1.25 + 10.0))
        diff_min = 6.8 if med_l < 55.0 else 8.5
        v_min_hsv = max(90, min(175, int(p98_l * 0.90)))
        white_local = (L >= l_min) & ((diff >= diff_min) | (tophat >= 10)) & (hsv[:, :, 1] <= 80)
        white_hsv = (hsv[:, :, 1] <= 55) & (V >= v_min_hsv) & (diff >= 5.0)
        mask_yellow = cv2.inRange(
            hsv,
            np.array([14, 55, max(55, int(med_l * 1.25))]),
            np.array([38, 255, 255])
        )
    elif mode == "BRIGHT":
        # CHE DO TROI SANG CHOI / NANG GAT: Tang nguong chong loa mat duong
        l_min = max(155.0, min(205.0, med_l * 1.10))
        white_local = (L >= l_min) & (diff >= 15.0) & (S <= 65)
        white_hsv = (hsv[:, :, 1] <= 40) & (V >= 200)
        mask_yellow = cv2.inRange(hsv, np.array([14, 65, 110]), np.array([38, 255, 255]))
    else:
        # CHE DO ANH SANG BINH THUONG
        white_local = (L >= 140) & (diff >= 12.0) & (S <= 85)
        white_hsv = (hsv[:, :, 1] <= 50) & (V >= 185)
        mask_yellow = cv2.inRange(hsv, np.array([14, 55, 90]), np.array([38, 255, 255]))

    mask_white = np.where(white_local | white_hsv, 255, 0).astype(np.uint8)
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    # 3. Gioi han hinh thang mat duong (Road ROI mo rong theo 4 diem ROI)
    road_roi = np.zeros_like(mask)
    if roi_points is not None and restrict_roi:
        pts = np.asarray(roi_points, dtype=np.float32)
        x_tl = max(int(w * 0.02), min(int(pts[0][0]), int(w * 0.45)))
        x_tr = min(int(w * 0.98), max(int(pts[1][0]), int(w * 0.55)))
        x_bl = max(int(w * 0.01), min(int(pts[2][0]), int(w * 0.30)))
        x_br = min(int(w * 0.99), max(int(pts[3][0]), int(w * 0.70)))
        poly = np.array([[
            (x_bl, bot_y),
            (x_tl, top_y),
            (x_tr, top_y),
            (x_br, bot_y)
        ]], dtype=np.int32)
    else:
        poly = np.array([[
            (int(w * 0.03), bot_y),
            (int(w * 0.28), top_y),
            (int(w * 0.72), top_y),
            (int(w * 0.97), bot_y)
        ]], dtype=np.int32)
    cv2.fillPoly(road_roi, poly, 255)
    mask = cv2.bitwise_and(mask, road_roi)

    # 4. Loc chieu rong doan ngang tren tung dong quet y
    clean = np.zeros_like(mask)
    span_y = float(max(1, bot_y - top_y))
    w_scale = 0.038 if mode == "DARK" else 0.050
    w_base = 0.022 if mode == "DARK" else 0.035
    for y in range(top_y, bot_y):
        row = np.flatnonzero(mask[y])
        if len(row) == 0:
            continue
        prog = (y - top_y) / span_y
        max_w = max(5, int(w * (w_base + w_scale * prog)))
        groups = np.split(row, np.flatnonzero(np.diff(row) > 2) + 1)
        for g in groups:
            if 2 <= len(g) <= max_w:
                clean[y, g] = 255

    # 5. Xoa cac dom nhieu co lap dien tich nho (< 14 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    filtered = np.zeros_like(clean)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 14 and stats[i, cv2.CC_STAT_HEIGHT] >= 4:
            filtered[labels == i] = 255

    return filtered


def warpImg(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts2, pts1) if inv else cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h))


def nothing(a):
    pass


def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], wT // 2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)


def valTrackbars(wT=480, hT=240, preset="robot"):
    try:
        widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
        heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
        widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
        heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
        if widthTop < 0 or heightTop <= 0:
            raise cv2.error
    except cv2.error:
        if preset == "robot":
            widthTop, heightTop, widthBottom, heightBottom = 105, 105, 20, 230
        else:
            widthTop, heightTop, widthBottom, heightBottom = 140, 145, 25, 230
    return np.float32([
        (widthTop, heightTop),
        (wT - widthTop, heightTop),
        (widthBottom, heightBottom),
        (wT - widthBottom, heightBottom),
    ])


def drawPoints(img, points):
    img_out = img.copy()
    pts = points.astype(np.int32)
    poly_pts = np.array([pts[0], pts[1], pts[3], pts[2]], np.int32)
    cv2.polylines(img_out, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=2)
    for p in pts:
        cv2.circle(img_out, (int(p[0]), int(p[1])), 6, (0, 0, 255), cv2.FILLED)
    return img_out


def getHistogram(img_warp, display=False, minPer=0.2, region=1):
    h, w = img_warp.shape[:2]
    if region == 1:
        hist_values = np.sum(img_warp, axis=0)
    else:
        hist_values = np.sum(img_warp[int(h - h // region):, :], axis=0)

    max_val = np.max(hist_values)
    min_val = minPer * max_val

    if max_val == 0 or np.isnan(max_val):
        base_point = w // 2
    else:
        index_array = np.where(hist_values >= min_val)[0]
        base_point = w // 2 if len(index_array) == 0 else int(np.average(index_array))

    if display:
        img_hist = np.zeros((h, w, 3), np.uint8)
        norm_factor = max_val if max_val > 0 else 1
        for x, intensity in enumerate(hist_values):
            bar_height = int((intensity / norm_factor) * (h - 20))
            cv2.line(img_hist, (x, h), (x, h - bar_height), (255, 0, 255), 1)
        cv2.circle(img_hist, (base_point, h - 15), 8, (0, 255, 255), cv2.FILLED)
        return base_point, img_hist

    return base_point


def imgHelper(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)
    return ver
