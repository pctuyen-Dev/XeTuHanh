from functools import lru_cache
import cv2
import numpy as np

# Preallocated constants and LUT bases to avoid per-frame allocations in hot loops
_LUT_BASE_F = np.arange(256, dtype=np.float32) / 255.0
_YELLOW_HI = np.array([38, 255, 255], dtype=np.uint8)
_YELLOW_LO_DARK = np.array([14, 48, 60], dtype=np.uint8)
_YELLOW_LO_NORM = np.array([14, 48, 90], dtype=np.uint8)
_YELLOW_LO_HWY_BRIGHT = np.array([14, 65, 110], dtype=np.uint8)
_YELLOW_LO_HWY_NORM = np.array([14, 55, 90], dtype=np.uint8)
_KERNEL_BRIDGE = np.ones((4, 2), dtype=np.uint8)
_STRUCT_ELEM_CACHE = {}


def configure_opencv(num_threads=None, use_optimized=True):
    """Configure OpenCV SIMD/threading runtime (useful on Raspberry Pi 5 Cortex-A76)."""
    cv2.setUseOptimized(bool(use_optimized))
    if num_threads is not None:
        cv2.setNumThreads(int(num_threads))


def _get_cached_kernel(shape, ksize):
    key = (shape, ksize)
    kern = _STRUCT_ELEM_CACHE.get(key)
    if kern is None:
        if shape == "erode_h":
            kern = np.ones((1, int(ksize)), dtype=np.uint8)
        else:
            kern = cv2.getStructuringElement(shape, ksize)
        _STRUCT_ELEM_CACHE[key] = kern
    return kern


def _uint8_quantiles(arr, probs):
    """Exact linear-interpolated quantiles on uint8 arrays via bincount (avoids sorting)."""
    n = arr.size
    if n == 0:
        return [0.0] * len(probs)
    cdf = np.cumsum(np.bincount(arr.ravel(), minlength=256))
    out = []
    for p in probs:
        idx = p * (n - 1)
        lo = int(idx)
        hi = lo + 1 if lo + 1 < n else lo
        frac = idx - lo
        v_lo = int(np.searchsorted(cdf, lo + 1, side="left"))
        if frac == 0.0:
            out.append(float(v_lo))
        else:
            v_hi = int(np.searchsorted(cdf, hi + 1, side="left"))
            out.append(float(v_lo + frac * (v_hi - v_lo)))
    return out


def _row_runs(row_u8):
    """Return (starts, ends) arrays for contiguous nonzero runs in a 1D uint8 row."""
    nz = np.flatnonzero(row_u8)
    if nz.size == 0:
        return None, None
    breaks = np.flatnonzero(np.diff(nz) > 1)
    if breaks.size == 0:
        return np.array([nz[0]], dtype=np.intp), np.array([nz[-1] + 1], dtype=np.intp)
    starts = np.empty(breaks.size + 1, dtype=np.intp)
    ends = np.empty(breaks.size + 1, dtype=np.intp)
    starts[0] = nz[0]
    ends[-1] = nz[-1] + 1
    starts[1:] = nz[breaks + 1]
    ends[:-1] = nz[breaks] + 1
    return starts, ends


def _detach_wide_outer_objects(mask):
    """
    Detach wide outer objects (e.g. traffic sign base touching the outer edge of the tape)
    while preserving horizontal corner arms (vertical thickness <= max_vert_tape_h).
    Modifies mask in place.
    """
    h, w = mask.shape
    if h <= 13:
        return

    min_max_w = int(round(16 + 24 * min(1.0, 11.0 / float(h))))
    # Extract all horizontal runs across rows 11..h-1 in one vectorized pass
    bin_all = (mask > 0).astype(np.uint8)
    padded = np.zeros((h - 11, w + 2), dtype=np.uint8)
    padded[:, 1:-1] = bin_all[11:]
    r_t, c_t = np.nonzero(padded[:, 1:] != padded[:, :-1])
    if r_t.size == 0:
        return

    r_s = r_t[0::2] + 11
    c_s = c_t[0::2]
    c_e = c_t[1::2]
    r_ws = c_e - c_s

    wide_mask = (r_s < h - 2) & (r_ws > min_max_w)
    if not np.any(wide_mask):
        return
    wide_idx = np.flatnonzero(wide_mask)

    row_bounds = np.searchsorted(r_s, np.arange(h + 1), side="left")
    csum = np.cumsum(bin_all, axis=0, dtype=np.int16)
    modified_rows = None

    w_third = w // 3
    w_twothird = (2 * w) // 3
    wi = wide_idx.size - 1
    while wi >= 0:
        last_w = int(wide_idx[wi])
        y = int(r_s[last_w])
        while wi > 0 and r_s[wide_idx[wi - 1]] == y:
            wi -= 1
        first_w = int(wide_idx[wi])
        wi -= 1

        ratio = y / float(h)
        if ratio < 0.0:
            ratio = 0.0
        elif ratio > 1.0:
            ratio = 1.0
        max_tape_w = int(round(16 + 24 * ratio))
        max_vert_tape_h = int(round(12 + 14 * ratio))
        y_win_hi_m1 = (h - 1) if (y + max_vert_tape_h + 1 > h) else (y + max_vert_tape_h)
        y_win_lo_m1 = y - max_vert_tape_h - 1
        hi_row = csum[y_win_hi_m1]
        lo_row = csum[y_win_lo_m1] if y_win_lo_m1 >= 0 else None
        below_cache = None
        row_modified = False

        for k in range(first_w, last_w + 1):
            if int(r_ws[k]) <= max_tape_w:
                continue
            s = int(c_s[k])
            e = int(c_e[k])
            if s <= w_third and e >= w_twothird:
                continue

            cc = (hi_row[s:e] - lo_row[s:e]) if lo_row is not None else hi_row[s:e]
            nc = cc.size
            n_le = int(np.count_nonzero(cc <= max_vert_tape_h))
            if n_le * 2 > nc:
                continue
            if n_le * 2 == nc:
                ccs = np.sort(cc)
                if 0.5 * (int(ccs[(nc >> 1) - 1]) + int(ccs[nc >> 1])) <= max_vert_tape_h:
                    continue

            if below_cache is None:
                yb = h - 1 if y + 2 >= h else y + 2
                if modified_rows is not None and yb in modified_rows:
                    bs, be = _row_runs(mask[yb])
                    if bs is None:
                        below_cache = ()
                    else:
                        bw = be - bs
                        vb = np.flatnonzero((bw >= 3) & (bw <= max_tape_w))
                        below_cache = [(int(bs[bi]), int(be[bi]), int(bw[bi])) for bi in vb]
                else:
                    p0, p1 = int(row_bounds[yb]), int(row_bounds[yb + 1])
                    if p0 == p1:
                        below_cache = ()
                    else:
                        below_cache = [
                            (int(c_s[p]), int(c_e[p]), int(r_ws[p]))
                            for p in range(p0, p1)
                            if 3 <= int(r_ws[p]) <= max_tape_w
                        ]
            if not below_cache:
                break

            for sb, eb, wb in below_cache:
                if abs(s - sb) <= 6 and s > w_third:
                    x0 = s + wb + 2
                    if x0 < e:
                        sub_b = bin_all[y, x0:e]
                        if np.any(sub_b):
                            csum[y:, x0:e] -= sub_b
                            sub_b[:] = 0
                            mask[y, x0:e] = 0
                            row_modified = True
                elif abs(e - eb) <= 6 and e < w_twothird:
                    x1 = max(0, e - wb - 2)
                    if s < x1:
                        sub_b = bin_all[y, s:x1]
                        if np.any(sub_b):
                            csum[y:, s:x1] -= sub_b
                            sub_b[:] = 0
                            mask[y, s:x1] = 0
                            row_modified = True

        if row_modified:
            if modified_rows is None:
                modified_rows = {y}
            else:
                modified_rows.add(y)


def resize_robot_frame(img, max_side=640):
    """Bound processing work without changing camera aspect or adding pixels."""
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()


def _classify_lighting_mode(med_l, p98_l, prev_mode=None):
    """Classify road lighting condition with optional hysteresis to prevent mode flicker."""
    if prev_mode == "DARK":
        if med_l > 160.0 or (med_l > 142.0 and p98_l > 240.0):
            return "BRIGHT"
        if med_l < 83.0 and p98_l < 178.0:
            return "DARK"
        return "NORMAL"
    if prev_mode == "BRIGHT":
        if med_l < 68.0 and p98_l < 158.0:
            return "DARK"
        if med_l > 142.0 or (med_l > 127.0 and p98_l > 225.0):
            return "BRIGHT"
        return "NORMAL"
    if med_l < 75.0 and p98_l < 168.0:
        return "DARK"
    if med_l > 152.0 or (med_l > 135.0 and p98_l > 235.0):
        return "BRIGHT"
    return "NORMAL"


def analyze_lighting(img, roi_points=None, prev_mode=None):
    """
    Tu dong phan tich dieu kien anh sang mat duong (Troi toi / Binh thuong / Troi sang loa).
    Tra ve dict: {"mode": "DARK" | "NORMAL" | "BRIGHT", "med_l": float, "p98_l": float, "med_v": float, "p98_v": float}.
    """
    h, w = img.shape[:2]
    top_y = int(h * 0.50)
    bot_y = int(h * 0.96)
    if roi_points is not None:
        pts = np.asarray(roi_points, dtype=np.float32)
        top_y = max(int(h * 0.35), min(int(min(pts[0][1], pts[1][1])), int(h * 0.75)))
        bot_y = min(int(h * 0.98), max(int(max(pts[2][1], pts[3][1])), top_y + 10))

    x_margin = int(w * 0.10)
    roi = img[top_y:bot_y, x_margin:w - x_margin]
    if roi.size == 0:
        roi = img

    hls_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    if roi.ndim == 3:
        b_roi, g_roi, r_roi = cv2.split(roi)
        v_roi = cv2.max(cv2.max(b_roi, g_roi), r_roi)
    else:
        v_roi = roi
    med_l, p98_l = _uint8_quantiles(hls_roi[:, :, 1], (0.50, 0.98))
    med_v, p98_v = _uint8_quantiles(v_roi, (0.50, 0.98))

    mode = _classify_lighting_mode(med_l, p98_l, prev_mode=prev_mode)

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


def _filter_specular_tarp_glare_components(proc_img, light, labels, stats, keep, med_g, p98_g, mode, is_bw_synthetic, grad_l1=None):
    """
    Bo loc chong anh sang bong loa mat bat den (Specular Tarp Glare) & phan chieu tren can truoc do (Red Bumper):
      1. Loai bo dom trang/logo phan chieu nam tren can xe mau do o day anh (vong ngoai co R - max(G,B) >= nguong).
      2. Loai bo cac vet bong den / loa anh sang / dai phan chieu mem tren mat bat bong (dac trung boi mep
         gradient mem, ty le sac net p75_g1/dL thap, do tuong phan cuc bo dL thap do quang loa halo, hoac do sang
         kem han vach trang that — ke ca trong tinh huong chay sang / washout nen bat len cao).
      3. An toan tuyet doi trong che do toi (DARK) va mo phong do hoa (is_bw_synthetic).
    """
    h, w = light.shape[:2]
    active_ids = np.flatnonzero(keep) + 1
    if active_ids.size == 0:
        return

    dyn = float(p98_g - med_g)
    run_glare_check = (not is_bw_synthetic) and (mode != "DARK") and (p98_g >= 185.0) and (dyn >= 35.0)
    has_color = (proc_img is not None) and (proc_img.ndim == 3) and (proc_img.shape[2] >= 3)
    if not run_glare_check and not has_color:
        return

    if run_glare_check and grad_l1 is None:
        gx = cv2.Sobel(light, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(light, cv2.CV_16S, 0, 1, ksize=3)
        grad_l1 = cv2.add(cv2.absdiff(gx, 0), cv2.absdiff(gy, 0))

    k3 = _get_cached_kernel(cv2.MORPH_RECT, (3, 3))
    k5 = _get_cached_kernel(cv2.MORPH_RECT, (5, 5))
    k15 = _get_cached_kernel(cv2.MORPH_RECT, (15, 15))
    pad = 8
    y_bumper_top = int(h * 0.68)
    y_bumper_bot = int(h * 0.84)

    dl_soft_th = max(22.0, 0.40 * dyn)
    g1_soft_th = max(145.0, min(220.0, 2.15 * dyn)) if dyn < 110.0 else max(220.0, 1.85 * dyn)
    dl_dim_th = max(38.0, 0.64 * dyn)
    g1_dim_th = max(190.0, 2.45 * dyn)
    dl_halo_th = max(34.0, 0.52 * dyn)
    g1_halo_th = max(140.0, 1.55 * dyn)
    g1_blunt_th = min(185.0, max(125.0, 1.45 * dyn))
    dl_protect_th = max(55.0, 0.62 * dyn)
    g1_protect_th = max(195.0, 2.00 * dyn)
    is_washout = (med_g >= 145.0) and (p98_g >= 235.0)

    for cid in active_ids:
        bx = int(stats[cid, cv2.CC_STAT_LEFT])
        by = int(stats[cid, cv2.CC_STAT_TOP])
        cw = int(stats[cid, cv2.CC_STAT_WIDTH])
        ch = int(stats[cid, cv2.CC_STAT_HEIGHT])

        in_bumper_zone = has_color and (by >= y_bumper_top) and ((by + ch) >= y_bumper_bot)
        if not run_glare_check and not in_bumper_zone:
            continue

        y0 = max(0, by - pad)
        y1 = min(h, by + ch + pad)
        x0 = max(0, bx - pad)
        x1 = min(w, bx + cw + pad)
        sub_lbl = (labels[y0:y1, x0:x1] == cid).astype(np.uint8)
        ring = (cv2.dilate(sub_lbl, k15) > 0) & (cv2.dilate(sub_lbl, k5) == 0)
        if not np.any(ring):
            continue

        # Tang A: Loai bo phan chieu/logo trang nam gon tren can truoc mau do cua xe
        if in_bumper_zone:
            sub_bgr = proc_img[y0:y1, x0:x1]
            ring_px = sub_bgr[ring]
            red_dom = ring_px[:, 2].astype(np.int16) - np.maximum(ring_px[:, 1], ring_px[:, 0]).astype(np.int16)
            med_red_dom = float(np.median(red_dom))
            if med_red_dom >= 22.0:
                keep[cid - 1] = False
                continue
            if is_washout and med_red_dom >= 11.0:
                sub_g_b = light[y0:y1, x0:x1]
                if float(np.median(sub_g_b[sub_lbl > 0])) - float(np.median(sub_g_b[ring])) < max(42.0, 0.55 * dyn):
                    keep[cid - 1] = False
                    continue

        if not run_glare_check:
            continue

        # Tang B: Phan loai vat ly bong loa bat (tuong phan dL + do sac mep Sobel L1 + ty le sac net + nguong sang)
        sub_g = light[y0:y1, x0:x1]
        fg_med = float(np.median(sub_g[sub_lbl > 0]))
        bg_med = float(np.median(sub_g[ring]))
        dL = fg_med - bg_med
        if dL < 14.0:
            keep[cid - 1] = False
            continue

        border = (cv2.dilate(sub_lbl, k3) > 0) & (cv2.erode(sub_lbl, k3) == 0)
        if not np.any(border):
            continue
        p75_g1 = float(np.percentile(grad_l1[y0:y1, x0:x1][border], 75))

        # Bao ve tuyet doi vach lan that dai & sac net
        if ch >= 45 and dL >= dl_protect_th and p75_g1 >= g1_protect_th:
            continue

        fg_gap = float(p98_g) - fg_med
        sharpness = p75_g1 / max(dL, 1.0)
        is_soft_glare = (dL < dl_soft_th) and (p75_g1 < g1_soft_th)
        is_blunt_glare = (p75_g1 < g1_blunt_th) and (sharpness < 1.95)
        is_dim_glare_band = (fg_gap >= 24.0) and (dL < dl_dim_th) and (p75_g1 < g1_dim_th)
        is_halo_glare_pool = (bg_med >= float(med_g) + 38.0) and (dL < dl_halo_th) and (p75_g1 < g1_halo_th)
        if is_soft_glare or is_blunt_glare or is_dim_glare_band or is_halo_glare_pool:
            keep[cid - 1] = False


def thresholding(img, hsv_range=None, roi_points=None, restrict_roi=False, lighting_info=None,
                 prev_mode=None, return_aux=False):
    """Tape candidates with automatic Day/Night adaptation, glare suppression, and outer-blob detachment."""
    h, w = img.shape[:2]
    light_info = (
        lighting_info
        if lighting_info is not None
        else analyze_lighting(img, roi_points=roi_points, prev_mode=prev_mode)
    )
    mode = light_info["mode"]
    med_v = light_info["med_v"]
    p98_v = light_info["p98_v"]

    proc_img = img
    gamma_applied = False
    is_bw_frame = (med_v < 15.0) and (int(img.max()) >= 180 if img.size > 0 else False)
    if mode == "DARK" and p98_v < 170.0 and not is_bw_frame:
        gamma = float(np.clip(np.log(125.0 / 255.0) / np.log(max(med_v, 12.0) / 255.0), 0.42, 0.85))
        lut = np.clip((_LUT_BASE_F ** gamma) * 255.0, 0.0, 255.0).astype(np.uint8)
        proc_img = cv2.LUT(img, lut)
        gamma_applied = True

    hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
    light = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    raw_gray = light if not gamma_applied else None

    road_top = int(h * 0.38)
    road_bot = int(h * 0.96)
    road_gray = light[road_top:road_bot, :] if road_bot > road_top else light
    p20_g, med_g, p88_g, p98_g = _uint8_quantiles(road_gray, (0.20, 0.50, 0.88, 0.995))
    is_global_washout = (med_g > 135.0) and (p20_g >= 115.0) and (p98_g >= 244.0)

    if min(w, h) >= 96:
        # Downsampled 3x morphological open for ~1.6x faster background illumination estimation
        hs, ws = max(16, h // 3), max(16, w // 3)
        light_s = cv2.resize(light, (ws, hs), interpolation=cv2.INTER_AREA)
        d_s = max(5, int(round(min(ws, hs) * 0.13)) | 1)
        bg_s = cv2.morphologyEx(
            light_s,
            cv2.MORPH_OPEN,
            _get_cached_kernel(cv2.MORPH_ELLIPSE, (d_s, d_s)),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=int(round(med_g)),
        )
        background = cv2.resize(bg_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        diameter = max(9, int(round(min(w, h) * 0.13)) | 1)
        background = cv2.morphologyEx(
            light,
            cv2.MORPH_OPEN,
            _get_cached_kernel(cv2.MORPH_ELLIPSE, (diameter, diameter)),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=int(round(med_g)),
        )
    contrast = cv2.subtract(light, background)

    contrast_thresh = max(18, int(round(max(22.0, (p98_g - med_g) * 0.32))))
    if mode == "DARK" and p98_v < 170.0:
        contrast_thresh = max(11, int(round(18.0 * (max(p98_v, 60.0) / 175.0))))
    elif med_g >= 125.0 and p98_g >= 208.0:
        # Bright glare-washed mat: keep highlight-compressed tape above background glare without over-inflating threshold
        contrast_thresh = min(contrast_thresh, 28)

    adaptive_v_min = max(165, int(round(max(p88_g + 15.0, 0.42 * med_g + 0.54 * p98_g))))
    is_bw_synthetic = (med_g < 18.0) and (p98_g >= 140.0 or (int(light.max()) >= 180 if light.size > 0 else False))

    if hsv_range is None:
        s_max = 105 if (mode == "DARK" and p98_v < 170.0) else (65 if med_g > 115 else 85)
        v_min = adaptive_v_min if med_g >= 25.0 else max(95, int(med_g * 1.5 + 85))
        color_u8 = cv2.inRange(
            hsv,
            np.array([0, 0, v_min], dtype=np.uint8),
            np.array([179, s_max, 255], dtype=np.uint8),
        )
    else:
        lower = np.asarray(hsv_range[0], dtype=np.int32).copy()
        upper = np.asarray(hsv_range[1], dtype=np.int32).copy()
        if is_bw_synthetic and lower[2] >= 150:
            # Nen duong den tuyet doi / mo phong do hoa: giu tron ven vien anti-aliasing cua vach trang mong
            lower[2] = min(int(lower[2]), 100)
            contrast_thresh = min(contrast_thresh, 14)
        elif mode == "DARK" and p98_v < 170.0 and lower[2] >= 165:
            if gamma_applied:
                xm = int(w * 0.10)
                v_roi_post = hsv[int(h * 0.50):int(h * 0.96), xm:w - xm, 2]
                med_v_eff = _uint8_quantiles(v_roi_post if v_roi_post.size > 0 else hsv[:, :, 2], (0.50,))[0]
                lower[2] = min(int(max(135.0, p98_g - 12.0)), max(105, int(med_v_eff * 1.25 + 10)))
            else:
                lower[2] = max(105, int(med_v * 1.35 + 15))
            upper[1] = max(int(upper[1]), 105)
        elif p98_g >= 200.0 and med_g >= 55.0 and lower[2] <= 188:
            v_floor_cap = min(232.0, 186.0 + (med_g - 135.0) * 0.90) if is_global_washout else 186.0
            v_floor = int(round(min(v_floor_cap, max(float(adaptive_v_min), 0.30 * med_g + 0.68 * p98_g))))
            lower[2] = max(int(lower[2]), v_floor)
            s_cap = 42 if (is_global_washout and med_g >= 150.0) else 68
            upper[1] = min(int(upper[1]), s_cap)
        elif p98_g < 175.0 and lower[2] >= 160:
            lower[2] = min(int(lower[2]), adaptive_v_min)
        color_u8 = cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8))

    # Only merge yellow lane candidates when caller did not restrict HSV saturation to white tape
    # (prevents synthetic yellow UI/ROI overlays or indoor yellow clutter from polluting white-tape masks).
    allow_yellow = (hsv_range is None) or (int(np.asarray(hsv_range[1])[1]) > 120)
    if allow_yellow:
        yellow_lo = _YELLOW_LO_DARK if (mode == "DARK" and p98_v < 170.0) else _YELLOW_LO_NORM
        yellow_u8 = cv2.inRange(hsv, yellow_lo, _YELLOW_HI)
        mask = cv2.bitwise_or(color_u8, yellow_u8)
    else:
        mask = color_u8
    contrast_ok = cv2.compare(contrast, int(contrast_thresh), cv2.CMP_GE)
    mask = cv2.bitwise_and(mask, contrast_ok)

    # Bo loc chong bong loa 3 tang muc pixel (Glare-Halo Gate + Sharp-Edge Support + Saturated Tape Rescue):
    #   - Tai vung bat bi anh sang phan chieu (background > med_g + 16), nang nguong sang & tuong phan de loai quang loa.
    #   - Cat dut cac dai/dom bong den mem (gradient Sobel L1 thap) TRUOC khi noi thanh phan lien thong, tranh dinh vao vach that.
    #   - Giu nguyen (rescue) doan vach trang bao hoa (light >= 246) khi quang loa rat manh chieu thang qua vach/goc cua.
    grad_l1 = None
    dyn_g = float(p98_g - med_g)
    if (not is_bw_synthetic) and (mode != "DARK") and (p98_g >= 185.0) and (dyn_g >= 35.0):
        gx = cv2.Sobel(light, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(light, cv2.CV_16S, 0, 1, ksize=3)
        grad_l1 = cv2.add(cv2.absdiff(gx, 0), cv2.absdiff(gy, 0))

    if (not is_bw_synthetic) and (mode != "DARK") and (p98_g >= 195.0) and (med_g >= 45.0):
        glare_halo = cv2.subtract(background, int(round(med_g + 16.0)))
        sharp_pixel_th = max(125, int(min(175.0, round(1.30 * dyn_g))))
        sharp_u8 = (grad_l1 >= sharp_pixel_th).astype(np.uint8) if grad_l1 is not None else None

        if np.any(glare_halo):
            glare_v_cap = min(236.0, 218.0 + (med_g - 135.0) * 0.45) if is_global_washout else 218.0
            glare_v_min = int(min(glare_v_cap, round(p98_g - 30.0)))
            req_contrast = np.clip(
                int(contrast_thresh) + (glare_halo.astype(np.int16) >> 1),
                int(contrast_thresh),
                120,
            ).astype(np.uint8)
            halo_reject = (glare_halo > 0) & ((light < glare_v_min) | (contrast < req_contrast))
            mask[halo_reject] = 0

            # Giai cuu vach trang bao hoa (light >= 246) nam trong vung loa/suong nang cuc manh nhung van co mep buoc sac net
            if sharp_u8 is not None and dyn_g >= 35.0:
                bg_rescue_min = max(160, int(round(med_g + 22.0)))
                headroom = np.maximum(255 - background.astype(np.int16), 1)
                rescue_g_th = min(sharp_pixel_th, max(65, int(round(1.50 * dyn_g))))
                rescue_src = sharp_u8 if rescue_g_th == sharp_pixel_th else (grad_l1 >= rescue_g_th).astype(np.uint8)
                rescue_support = cv2.dilate(rescue_src, _get_cached_kernel(cv2.MORPH_RECT, (15, 15)))
                con_i16 = contrast.astype(np.int16)
                tape_in_glare = (
                    (color_u8 > 0)
                    & (light >= 246)
                    & (background >= bg_rescue_min)
                    & (((contrast >= 11) & (con_i16 * 100 >= headroom * 72)) | ((light >= 252) & (contrast >= 6) & (con_i16 * 100 >= headroom * 85)))
                    & (rescue_support > 0)
                )
                y_rescue_top = int(h * 0.15)
                if y_rescue_top > 0:
                    tape_in_glare[:y_rescue_top, :] = False
                mask[tape_in_glare] = 255

        # Cat dut dai bong den mem (khong nam gan bat ky mep buoc sac net nao) truoc khi noi CC
        if sharp_u8 is not None and dyn_g >= 45.0:
            sharp_support = cv2.dilate(sharp_u8, _get_cached_kernel(cv2.MORPH_RECT, (23, 23)))
            low_con_th = int(max(45, round(0.72 * dyn_g)))
            blunt_reject = (sharp_support == 0) & ((glare_halo > 0) | (contrast < low_con_th))
            mask[blunt_reject] = 0

    _detach_wide_outer_objects(mask)

    # Bridge tiny 2-4px breaks along the tape (such as a thin wire crossing the white line)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_BRIDGE)

    if restrict_roi and roi_points is not None:
        roi_mask = build_roi_polygon_mask((h, w), roi_points, expand_x_ratio=0.04, min_top_ratio=0.05)
        mask = cv2.bitwise_and(mask, roi_mask)
    else:
        mask[:3] = 0
        mask[int(h * 0.995):] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        out_mask = np.zeros_like(mask)
        if return_aux:
            return out_mask, {"lighting_info": light_info, "gray": raw_gray, "cc_labels": None, "cc_stats": None}
        return out_mask

    min_area_std = max(12, int(w * h * 0.00008))
    min_span_std = max(8, int(w * 0.025))
    st = stats[1:]
    ys = st[:, cv2.CC_STAT_TOP]
    cws = st[:, cv2.CC_STAT_WIDTH]
    chs = st[:, cv2.CC_STAT_HEIGHT]
    areas = st[:, cv2.CC_STAT_AREA]
    max_span = np.maximum(cws, chs)
    # Giu lai cac net dut ngan o xa (nua tren khung hinh) co be rong hop ly, loai bo cham nhieu nho le loi
    keep = ((areas >= min_area_std) & (max_span >= min_span_std)) | (
        (ys < int(h * 0.62)) & (areas >= 16) & (max_span >= 7) & (cws >= 2) & (cws <= 24)
    )
    _filter_specular_tarp_glare_components(
        proc_img, light, labels, stats, keep, med_g, p98_g, mode, is_bw_synthetic, grad_l1=grad_l1
    )
    lut = np.zeros(count, dtype=np.uint8)
    lut[1:][keep] = 255
    out_mask = lut[labels]
    if return_aux:
        return out_mask, {"lighting_info": light_info, "gray": raw_gray, "cc_labels": labels, "cc_stats": stats}
    return out_mask


def thresholding_highway(img, hsv_range=None, roi_points=None, restrict_roi=True,
                         lighting_info=None, prev_mode=None, return_aux=False):
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

    if lighting_info is not None:
        light_info = lighting_info
    else:
        lt = int(h * 0.50)
        lb = int(h * 0.96)
        if roi_points is not None:
            pts = np.asarray(roi_points, dtype=np.float32)
            lt = max(int(h * 0.35), min(int(min(pts[0][1], pts[1][1])), int(h * 0.75)))
            lb = min(int(h * 0.98), max(int(max(pts[2][1], pts[3][1])), lt + 10))
        xm = int(w * 0.10)
        rl = L[lt:lb, xm:w - xm]
        rv = V[lt:lb, xm:w - xm]
        if rl.size == 0:
            rl, rv = L, V
        ml, p98l = _uint8_quantiles(rl, (0.50, 0.98))
        mv, p98v = _uint8_quantiles(rv, (0.50, 0.98))
        lmode = _classify_lighting_mode(ml, p98l, prev_mode=prev_mode)
        light_info = {"mode": lmode, "med_l": ml, "p98_l": p98l, "med_v": mv, "p98_v": p98v}

    mode = light_info["mode"]
    med_l = light_info["med_l"]
    p98_l = light_info["p98_l"]

    # 1. Vach trang: do tuong phan cuc bo ngang 31x5 + Morphology Top-Hat 25x3
    local_mean = cv2.boxFilter(L, cv2.CV_32F, (31, 5))
    diff = L.astype(np.float32) - local_mean

    if mode == "DARK":
        # CHE DO TROI TOI / BAN DEM: Dung Top-Hat ngang (25x3) + gioi han do bao hoa HSV de bat vach trang dut net, loai den xe
        kernel_th = _get_cached_kernel(cv2.MORPH_RECT, (25, 3))
        tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, kernel_th)
        l_min = max(48.0, min(130.0, med_l * 1.25 + 10.0))
        diff_min = 6.8 if med_l < 55.0 else 8.5
        v_min_hsv = max(90, min(175, int(p98_l * 0.90)))
        white_local = (L >= l_min) & ((diff >= diff_min) | (tophat >= 10)) & (hsv[:, :, 1] <= 80)
        white_hsv = (hsv[:, :, 1] <= 55) & (V >= v_min_hsv) & (diff >= 5.0)
        mask_yellow = cv2.inRange(
            hsv,
            np.array([14, 55, max(55, int(med_l * 1.25))], dtype=np.uint8),
            _YELLOW_HI,
        )
    elif mode == "BRIGHT":
        # CHE DO TROI SANG CHOI / NANG GAT: Tang nguong chong loa mat duong + kiem tra tuong phan cuc bo
        l_min = max(155.0, min(205.0, med_l * 1.10))
        white_local = (L >= l_min) & (diff >= 15.0) & (S <= 65)
        white_hsv = (hsv[:, :, 1] <= 40) & (V >= 200) & (diff >= 6.0)
        mask_yellow = cv2.inRange(hsv, _YELLOW_LO_HWY_BRIGHT, _YELLOW_HI)
    else:
        # CHE DO ANH SANG BINH THUONG
        white_local = (L >= 140) & (diff >= 12.0) & (S <= 85)
        white_hsv = (hsv[:, :, 1] <= 50) & (V >= 185)
        mask_yellow = cv2.inRange(hsv, _YELLOW_LO_HWY_NORM, _YELLOW_HI)

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
    active_ys = np.flatnonzero(np.any(mask[top_y:bot_y], axis=1)) + top_y
    for y in active_ys:
        y = int(y)
        row = np.flatnonzero(mask[y])
        prog = (y - top_y) / span_y
        max_w = max(5, int(w * (w_base + w_scale * prog)))
        breaks = np.flatnonzero(np.diff(row) > 2)
        if breaks.size == 0:
            if 2 <= row.size <= max_w:
                clean[y, row] = 255
        else:
            s_idx = 0
            for b in breaks:
                e_idx = int(b) + 1
                cnt = e_idx - s_idx
                if 2 <= cnt <= max_w:
                    clean[y, row[s_idx:e_idx]] = 255
                s_idx = e_idx
            if 2 <= (row.size - s_idx) <= max_w:
                clean[y, row[s_idx:]] = 255

    # 5. Xoa cac dom nhieu co lap dien tich nho (< 14 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if num_labels <= 1:
        out_mask = np.zeros_like(clean)
        if return_aux:
            return out_mask, {"lighting_info": light_info}
        return out_mask
    st = stats[1:]
    keep = (st[:, cv2.CC_STAT_AREA] >= 14) & (st[:, cv2.CC_STAT_HEIGHT] >= 4)
    lut = np.zeros(num_labels, dtype=np.uint8)
    lut[1:][keep] = 255
    out_mask = lut[labels]
    if return_aux:
        return out_mask, {"lighting_info": light_info}
    return out_mask


@lru_cache(maxsize=16)
def _get_perspective_matrix(pts_flat, w, h, inv):
    pts1 = np.array(pts_flat, dtype=np.float32).reshape(4, 2)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    return cv2.getPerspectiveTransform(pts2, pts1) if inv else cv2.getPerspectiveTransform(pts1, pts2)


def warpImg(img, points, w, h, inv=False, flags=cv2.INTER_LINEAR):
    pts_arr = np.asarray(points, dtype=np.float32)
    matrix = _get_perspective_matrix(tuple(float(v) for v in pts_arr.ravel()), int(w), int(h), bool(inv))
    return cv2.warpPerspective(img, matrix, (w, h), flags=flags)


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
    src = img_warp if region == 1 else img_warp[int(h - h // region):, :]
    hist_values = np.sum(src, axis=0)

    max_val = np.max(hist_values)
    min_val = minPer * max_val

    if max_val == 0 or np.isnan(max_val):
        base_point = w // 2
    else:
        index_array = np.flatnonzero(hist_values >= min_val)
        base_point = w // 2 if index_array.size == 0 else int(np.mean(index_array))

    if display:
        img_hist = np.zeros((h, w, 3), np.uint8)
        norm_factor = float(max_val) if max_val > 0 else 1.0
        bar_heights = ((hist_values / norm_factor) * (h - 20)).astype(np.int32)
        max_bh = int(np.max(bar_heights)) if bar_heights.size else 0
        if max_bh > 0:
            y_start = max(0, h - max_bh)
            ys = np.arange(y_start, h, dtype=np.int32)[:, None]
            m = (ys >= (h - bar_heights)[None, :]).view(np.uint8) * 255
            img_hist[y_start:h, :, 0] = m
            img_hist[y_start:h, :, 2] = m
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
