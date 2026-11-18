import os
import cv2
import numpy as np

# DANH SÁCH 21 BIỂN BÁO GIAO THÔNG (15 BIỂN GỐC + 6 BIỂN MỚI) + LỚP UNKNOWN (0)
CLASSES_15 = [
    'unknown',          # 0  - Nhiễu môi trường / Không phải biển báo
    'left',             # 1  - Rẽ trái bắt buộc
    'no_left',          # 2  - Cấm rẽ trái
    'right',            # 3  - Rẽ phải bắt buộc
    'no_right',         # 4  - Cấm rẽ phải
    'straight',         # 5  - Đi thẳng bắt buộc
    'stop',             # 6  - Dừng lại (STOP)
    'no_straight',      # 7  - Cấm đi thẳng
    'no_entry',         # 8  - Cấm đi ngược chiều
    'speed_30',         # 9  - Giới hạn tốc độ tối đa 30 km/h
    'speed_60',         # 10 - Giới hạn tốc độ tối đa 60 km/h
    'roundabout',       # 11 - Nơi giao nhau chạy theo vòng xuyến
    'yield',            # 12 - Nhường đường
    'pedestrian',       # 13 - Người đi bộ cắt ngang
    'road_work',        # 14 - Công trường đang thi công
    'danger',           # 15 - Chú ý nguy hiểm (!)
    'speed_5',          # 16 - Giới hạn tốc độ tối đa 5 km/h
    'speed_10',         # 17 - Giới hạn tốc độ tối đa 10 km/h
    'speed_20',         # 18 - Giới hạn tốc độ tối đa 20 km/h
    'end_speed_limit',  # 19 - Hết giới hạn tốc độ
    'no_parking',       # 20 - Cấm đỗ xe
    'intersection',     # 21 - Giao lộ / Giao nhau
]

SIGN_INFO = {
    'left':            {'vn': 'ReTrai',          'action': 'RE TRAI',       'speed_factor': 0.85, 'steer_bias': -0.45, 'color': (255, 140, 0)},
    'no_left':         {'vn': 'CamReTrai',      'action': 'CAM RE TRAI',            'speed_factor': 0.90, 'steer_bias': +0.20, 'color': (0, 80, 255)},
    'right':           {'vn': 'RePhai',          'action': 'RE PHAI',       'speed_factor': 0.85, 'steer_bias': +0.45, 'color': (255, 140, 0)},
    'no_right':        {'vn': 'CamRePhai',      'action': 'CAM RE PHAI',            'speed_factor': 0.90, 'steer_bias': -0.20, 'color': (0, 80, 255)},
    'straight':        {'vn': 'DiThang',         'action': 'DI THANG',      'speed_factor': 1.00, 'steer_bias':  0.00, 'color': (255, 180, 0)},
    'stop':            {'vn': 'DUNG(STOP)',      'action': 'DUNG - STOP',     'speed_factor': 0.00, 'steer_bias':  0.00, 'color': (0, 0, 255)},
    'no_straight':     {'vn': 'CamDiThang',     'action': 'CAM DI THANG',           'speed_factor': 0.50, 'steer_bias':  0.00, 'color': (0, 80, 255)},
    'no_entry':        {'vn': 'CamNguocChieu',  'action': 'CAM DI NGUOC CHIEU',      'speed_factor': 0.00, 'steer_bias':  0.00, 'color': (0, 0, 255)},
    'speed_5':         {'vn': 'TocDo5',         'action': 'TOC DO DUOI 5km/h',    'speed_factor': 0.25, 'steer_bias':  0.00, 'color': (0, 140, 255)},
    'speed_10':        {'vn': 'TocDo10',        'action': 'TOC DO DUOI 10km/h',   'speed_factor': 0.35, 'steer_bias':  0.00, 'color': (0, 155, 255)},
    'speed_20':        {'vn': 'TocDo20',        'action': 'TOC DO DUOI 20km/h',   'speed_factor': 0.45, 'steer_bias':  0.00, 'color': (0, 165, 255)},
    'speed_30':        {'vn': 'TocDo30',        'action': 'TOC DO DUOI 30km/h',   'speed_factor': 0.55, 'steer_bias':  0.00, 'color': (0, 165, 255)},
    'speed_60':        {'vn': 'TocDo60',        'action': 'TOC DO DUOI 60km/h',     'speed_factor': 1.25, 'steer_bias':  0.00, 'color': (0, 220, 120)},
    'end_speed_limit': {'vn': 'HetGHTocDo',    'action': 'KHOI PHUC TOC DO','speed_factor': 1.00, 'steer_bias':  0.00, 'color': (180, 230, 120)},
    'no_parking':      {'vn': 'CamDoXe',        'action': 'KHONG DUNG / DO XE',     'speed_factor': 1.00, 'steer_bias':  0.00, 'color': (180, 60, 255)},
    'intersection':    {'vn': 'GiaoLo',      'action': 'GIAM TOC QUA GIAO LO',   'speed_factor': 0.50, 'steer_bias':  0.00, 'color': (0, 200, 255)},
    'roundabout':      {'vn': 'VongXuyen',       'action': 'DI THEO VONG XUYEN',     'speed_factor': 0.70, 'steer_bias': -0.25, 'color': (255, 160, 0)},
    'yield':           {'vn': 'NhuongDuong',     'action': 'GIAM TOC NHUONG DUONG',  'speed_factor': 0.50, 'steer_bias':  0.00, 'color': (0, 215, 255)},
    'pedestrian':      {'vn': 'NguoiDiBo',      'action': 'GIAM TOC QUA DUONG',     'speed_factor': 0.40, 'steer_bias':  0.00, 'color': (0, 215, 255)},
    'road_work':       {'vn': 'CongTruong',      'action': 'DI CHAM CONG TRUONG',    'speed_factor': 0.45, 'steer_bias':  0.00, 'color': (0, 180, 255)},
    'danger':          {'vn': 'NguyHiem',    'action': 'CANH BAO NGUY HIEM',     'speed_factor': 0.55, 'steer_bias':  0.00, 'color': (0, 140, 255)},
}

# Phân nhóm màu hợp lệ để chống nhận diện nhầm chéo nhóm màu
SPEED_LIMIT_SIGNS = {'speed_5', 'speed_10', 'speed_20', 'speed_30', 'speed_60'}
SPEED_CLASS_TO_IDX = {
    'speed_30': 9,
    'speed_60': 10,
    'speed_5': 16,
    'speed_10': 17,
    'speed_20': 18,
}
BLUE_SIGNS = {'left', 'right', 'straight', 'roundabout'}
SOLID_RED_SIGNS = {'stop', 'no_entry'}
WHITE_CORE_RED_SIGNS = {'no_left', 'no_right', 'no_straight', 'speed_5', 'speed_10', 'speed_20', 'speed_30', 'speed_60', 'yield'}
WARNING_TRIANGLE_SIGNS = {'pedestrian', 'road_work', 'danger', 'intersection'}
NO_PARKING_SIGNS = {'no_parking'}
END_LIMIT_SIGNS = {'end_speed_limit'}
RED_YELLOW_SIGNS = SOLID_RED_SIGNS | WHITE_CORE_RED_SIGNS | WARNING_TRIANGLE_SIGNS | NO_PARKING_SIGNS

_clahe_normal = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_clahe_dark = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
_clahe_bright = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
_default_model_cache = {}
_speed_digit_templates_cache = None


def load_sign_model(model_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(base_dir, "data_train_model.onnx")

    if model_path in _default_model_cache:
        return _default_model_cache[model_path]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Khong tim thay model ONNX tai: {model_path}")

    net = cv2.dnn.readNetFromONNX(model_path)
    _default_model_cache[model_path] = net
    return net


def preprocess_lighting(image, return_mode=False):
    """
    Tự động nhận biết môi trường Trời tối (DARK) / Bình thường (NORMAL) / Trời sáng chói (BRIGHT)
    và cân bằng sáng Gamma + CLAHE trên kênh L (không gian LAB) kết hợp khử nhiễu hạt vùng tối.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l = float(np.mean(l))

    if mean_l < 85.0:
        mode = "DARK"
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        l_d, a_d, b_d = cv2.split(cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB))
        gamma = float(np.clip(np.log(122.0 / 255.0) / np.log(max(mean_l, 18.0) / 255.0), 0.50, 0.85))
        lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        lut[:20] = np.arange(20, dtype=np.uint8)
        l_corr = cv2.LUT(l_d, lut)
        l_eq = _clahe_dark.apply(l_corr)
        bgr = cv2.cvtColor(cv2.merge((l_eq, a_d, b_d)), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        bright_enough = hsv[:, :, 2] >= 52.0
        hsv[:, :, 1] = np.where(bright_enough, np.clip(hsv[:, :, 1] * 1.18, 0, 255), hsv[:, :, 1] * 0.85)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif mean_l > 170.0:
        mode = "BRIGHT"
        gamma = float(np.clip(np.log(135.0 / 255.0) / np.log(min(mean_l, 248.0) / 255.0), 1.15, 1.85))
        lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        l_corr = cv2.LUT(l, lut)
        l_eq = _clahe_bright.apply(l_corr)
        out = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
    else:
        mode = "NORMAL"
        l_eq = _clahe_normal.apply(l)
        out = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    if return_mode:
        return out, mode
    return out


def _stretch_sign_crop(sub_bgr):
    """
    Kéo giãn dải động cục bộ (Local Dynamic Range Stretch) riêng cho từng vùng cắt biển báo:
    Khi biển báo chụp ban đêm hoặc ở xa bị tối/xám (p95 < 205), tự động kéo nền trắng lên 250
    và phục hồi độ bão hòa màu viền đỏ/xanh để mạng ONNX nhận diện chính xác.
    """
    if sub_bgr is None or sub_bgr.size == 0:
        return sub_bgr
    lab = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p5, p95 = np.percentile(l, [5, 95])
    if p95 < 205.0 or (p95 - p5) < 115.0:
        l_s = np.clip((l.astype(np.float32) - p5) * (245.0 / max(p95 - p5, 15.0)) + 8.0, 0, 255).astype(np.uint8)
        bgr_s = cv2.cvtColor(cv2.merge((l_s, a, b)), cv2.COLOR_LAB2BGR)
        hsv_s = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_s[:, :, 1] = np.clip(hsv_s[:, :, 1] * 1.40, 0, 255)
        return cv2.cvtColor(hsv_s.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return sub_bgr


def compute_sign_color_masks(image, use_clahe=True, precomputed_lighting=None):
    """
    Tách mặt nạ màu Đỏ (mask_r), Xanh dương (mask_b) và mặt nạ tổng hợp đã khử nhiễu (mask_clean).
    Trong chế độ Trời tối (DARK), áp dụng thêm bộ lọc độ lệch màu thực (True RGB Chroma Dominance)
    để loại bỏ hoàn toàn nhiễu mặt đường nhựa phản chiếu đèn vàng/đỏ và nhiễu hạt cảm biến.
    """
    if precomputed_lighting is not None:
        proc, mode = precomputed_lighting
    elif use_clahe:
        proc, mode = preprocess_lighting(image, return_mode=True)
    else:
        proc, mode = image, "NORMAL"

    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    b_ch, g_ch, r_ch = [c.astype(np.int16) for c in cv2.split(proc)]
    red_chroma = r_ch - np.maximum(g_ch, b_ch)
    blue_chroma = b_ch - np.maximum(r_ch, g_ch)

    if mode == "DARK":
        lower_r1, upper_r1 = np.array([0, 82, 52]), np.array([11, 255, 255])
        lower_r2, upper_r2 = np.array([165, 82, 52]), np.array([180, 255, 255])
        mask_r_hsv = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1),
                                    cv2.inRange(hsv, lower_r2, upper_r2))
        chroma_r_ok = ((red_chroma >= 25) & (r_ch >= (g_ch * 13) // 10)).astype(np.uint8) * 255
        mask_r = cv2.bitwise_and(mask_r_hsv, chroma_r_ok)

        lower_b, upper_b = np.array([96, 90, 44]), np.array([138, 255, 255])
        mask_b_hsv = cv2.inRange(hsv, lower_b, upper_b)
        chroma_b_ok = (blue_chroma >= 16).astype(np.uint8) * 255
        mask_b = cv2.bitwise_and(mask_b_hsv, chroma_b_ok)
    else:
        s_red_min = 62
        v_min = 35
        s_blue_min = 68
        lower_r1, upper_r1 = np.array([0, s_red_min, v_min]), np.array([13, 255, 255])
        lower_r2, upper_r2 = np.array([164, s_red_min, v_min]), np.array([180, 255, 255])
        mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1),
                                cv2.inRange(hsv, lower_r2, upper_r2))
        chroma_r_ok = (red_chroma >= 15).astype(np.uint8) * 255
        mask_r = cv2.bitwise_and(mask_r, chroma_r_ok)

        lower_b, upper_b = np.array([94, s_blue_min, v_min]), np.array([140, 255, 255])
        mask_b = cv2.inRange(hsv, lower_b, upper_b)
        chroma_b_ok = (blue_chroma >= 14).astype(np.uint8) * 255
        mask_b = cv2.bitwise_and(mask_b, chroma_b_ok)

        if use_clahe:
            hsv_orig = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            b0, g0, r0 = [c.astype(np.int16) for c in cv2.split(image)]
            rc0 = ((r0 - np.maximum(g0, b0)) >= 15).astype(np.uint8) * 255
            bc0 = ((b0 - np.maximum(r0, g0)) >= 14).astype(np.uint8) * 255
            mask_r_orig = cv2.bitwise_and(
                cv2.bitwise_or(cv2.inRange(hsv_orig, lower_r1, upper_r1),
                               cv2.inRange(hsv_orig, lower_r2, upper_r2)),
                rc0
            )
            mask_b_orig = cv2.bitwise_and(cv2.inRange(hsv_orig, lower_b, upper_b), bc0)
            mask_r = cv2.bitwise_or(mask_r, mask_r_orig)
            mask_b = cv2.bitwise_or(mask_b, mask_b_orig)

    mask_raw = cv2.bitwise_or(mask_r, mask_b)
    if not use_clahe:
        return mask_r, mask_b, mask_raw

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
    mask_r = cv2.bitwise_and(mask_r, mask_clean)
    mask_b = cv2.bitwise_and(mask_b, mask_clean)
    return mask_r, mask_b, mask_clean


def filter_signs_by_color(image, use_clahe=True):
    """Lọc các đối tượng màu Đỏ, Xanh dương và Vàng/Đỏ cảnh báo kèm khử nhiễu hình thái học thích ứng sáng/tối."""
    _, _, mask_clean = compute_sign_color_masks(image, use_clahe=use_clahe)
    return mask_clean


def _non_max_suppression_boxes(boxes, iou_thresh=0.45):
    """Loại bỏ các khung bao chồng lấn nhau trên cùng một biển báo."""
    if not boxes:
        return []
    arr = np.array(boxes, dtype=np.float32)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 0] + arr[:, 2]
    y2 = arr[:, 1] + arr[:, 3]
    areas = arr[:, 2] * arr[:, 3]
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes[int(i)])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def _find_achromatic_end_limit_boxes(image):
    """
    Tìm ứng viên biển 'Hết giới hạn tốc độ' (DP.134/DP.135):
    Biển có dạng hình tròn nền trắng, viền đen/xám đậm và dải gạch chéo đen từ phải-trên xuống trái-dưới
    (không có màu đỏ/xanh nên cần bộ dò hình tròn viền tối riêng).
    """
    if image is None or image.size == 0:
        return []
    im_h, im_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mặt nạ nét tối (viền đen + gạch chéo đen) có độ bão hòa màu thấp
    dark_mask = ((gray < 105) & (hsv[:, :, 1] < 110)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dark_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 18 or h < 18 or w > int(0.80 * im_w) or h > int(0.80 * im_h):
            continue
        aspect = w / float(h)
        if not (0.78 <= aspect <= 1.28):
            continue

        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_perim = cv2.arcLength(hull, True)
        if hull_area <= 0 or hull_perim <= 0:
            continue

        # Biển tròn nên bao lồi (convex hull) có độ tròn (circularity) cao > 0.72 và lấp đầy khung vuông ~ pi/4 (0.785)
        circularity = (4.0 * np.pi * hull_area) / (hull_perim * hull_perim)
        rect_fill = hull_area / float(w * h)
        if circularity < 0.72 or not (0.62 <= rect_fill <= 0.90):
            continue

        # Kiểm tra cấu trúc bên trong khung bao: phải có nền trắng sáng + vạch chéo từ phải-trên xuống trái-dưới
        sub_gray = gray[y:y + h, x:x + w]
        sub_s = hsv[y:y + h, x:x + w, 1]
        white_ratio = float(np.count_nonzero((sub_gray >= 145) & (sub_s <= 85))) / float(w * h)
        dark_ratio = float(np.count_nonzero(sub_gray < 105)) / float(w * h)
        if not (0.28 <= white_ratio <= 0.82 and 0.12 <= dark_ratio <= 0.58):
            continue

        # Kiểm tra 2 nửa trắng (Trên-Trái & Dưới-Phải) phải sáng rõ hơn dải gạch chéo (Phải-Trên xuống Trái-Dưới)
        g64 = cv2.resize(sub_gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        tl_lobe = float(np.mean(g64[14:27, 14:27]))
        br_lobe = float(np.mean(g64[37:50, 37:50]))
        anti_diag = float(np.mean([g64[63 - i - 2:63 - i + 3, i - 2:i + 3].mean() for i in range(16, 48)]))
        if min(tl_lobe, br_lobe) < 155.0 or (min(tl_lobe, br_lobe) - anti_diag) < 38.0:
            continue

        cand_boxes.append([int(x), int(y), int(w), int(h)])

    return cand_boxes


def get_boxes_from_mask(mask, image=None, verify_geometry=True):
    """
    Tìm các khung bao ứng viên biển báo kết hợp lọc hình học Contour chống nhiễu môi trường.
    Hỗ trợ ghép cặp 2 nửa biển báo bị mũi tên trắng cắt đôi theo chiều dọc (như biển Đi thẳng)
    và phát hiện biển Hết giới hạn tốc độ (viền đen nền trắng gạch chéo).
    """
    bboxes = []
    im_height, im_width = mask.shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    half_candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 14 or w < 8:
            continue
        if w > 0.85 * im_width or h > 0.85 * im_height:
            continue

        area = cv2.contourArea(cnt)
        bbox_area = float(w * h)
        fill_ratio = area / max(bbox_area, 1.0)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1.0)

        if area >= 45 and fill_ratio >= 0.28 and solidity >= 0.55:
            half_candidates.append((int(x), int(y), int(w), int(h), float(area), float(fill_ratio), float(solidity)))

        if w < 15 or h < 15:
            continue
        aspect = w / float(h)
        if aspect > 1.70 or aspect < 0.62:
            continue

        if verify_geometry:
            if fill_ratio < 0.28:
                continue
            if solidity < 0.68 and not (fill_ratio >= 0.42 and solidity >= 0.52):
                continue

            if image is not None:
                sub = image[y:y+h, x:x+w]
                gray_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
                if float(np.std(gray_sub)) < 10.0:
                    continue

        bboxes.append([int(x), int(y), int(w), int(h)])

    # Ghép cặp 2 nửa trái/phải của biển báo tròn bị mũi tên trắng dọc cắt đôi (VD: Biển Đi thẳng)
    n_half = len(half_candidates)
    paired_indices = set()
    for i in range(n_half):
        x1, y1, w1, h1, a1, _, _ = half_candidates[i]
        for j in range(i + 1, n_half):
            x2, y2, w2, h2, a2, _, _ = half_candidates[j]
            if x1 > x2:
                lx, ly, lw, lh, la = x2, y2, w2, h2, a2
                rx, ry, rw, rh, ra = x1, y1, w1, h1, a1
            else:
                lx, ly, lw, lh, la = x1, y1, w1, h1, a1
                rx, ry, rw, rh, ra = x2, y2, w2, h2, a2

            ov_y = min(ly + lh, ry + rh) - max(ly, ry)
            min_h = min(lh, rh)
            max_h = max(lh, rh)
            gap_x = rx - (lx + lw)
            if ov_y < 0.35 * min_h or gap_x < -4 or gap_x > 0.60 * max_h:
                continue

            ux = min(lx, rx)
            uy = min(ly, ry)
            uw = max(lx + lw, rx + rw) - ux
            uh = max(ly + lh, ry + rh) - uy
            if uw < 16 or uh < 16:
                continue
            u_aspect = uw / float(uh)
            if not (0.60 <= u_aspect <= 1.70):
                continue
            if (la + ra) / float(uw * uh) < 0.24:
                continue
            if image is not None:
                sub = image[uy:uy+uh, ux:ux+uw]
                if float(np.std(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY))) < 10.0:
                    continue
            bboxes.append([int(ux), int(uy), int(uw), int(uh)])
            paired_indices.add(i)
            paired_indices.add(j)

    # Nếu chỉ thấy 1 nửa hình bán nguyệt đứng (do nửa kia sát mép ảnh hoặc tối), mở rộng khung vuông thử nghiệm
    for idx, (hx, hy, hw, hh, ha, hfill, hsol) in enumerate(half_candidates):
        if idx in paired_indices:
            continue
        h_aspect = hw / float(hh)
        if 0.40 <= h_aspect < 0.65 and hh >= 22 and hfill >= 0.45 and hsol >= 0.75:
            w_sq = int(round(hh * 0.96))
            for cand_x in (hx, max(0, hx + hw - w_sq)):
                cw = min(w_sq, im_width - cand_x)
                if cw >= int(hh * 0.68):
                    bboxes.append([int(cand_x), int(hy), int(cw), int(hh)])

    # Bổ sung ứng viên biển Hết giới hạn tốc độ (nền trắng viền đen gạch chéo)
    if image is not None and verify_geometry:
        bboxes.extend(_find_achromatic_end_limit_boxes(image))

    return _non_max_suppression_boxes(bboxes, iou_thresh=0.45)

# BỘ XÁC THỰC & CHỐNG NHẬN DIỆN SAI BIỂN TỐC ĐỘ (5, 10, 20, 30, 60 KM/H)
def _has_enclosed_bottom_loop(bin_patch_32x48):
    """
    Kiểm tra xem chữ số bên trái (đã chuẩn hóa 32x48, 1=mực đen, 0=nền trắng)
    có lỗ kín (closed loop) ở nửa dưới hay không -> Đặc trưng độc nhất của số '6' (60 km/h),
    hoàn toàn không có ở số '1', '2', '3'.
    """
    h, w = bin_patch_32x48.shape
    # Tạo ảnh nền trắng = 255, mực đen = 0, có thêm viền 1px xung quanh để floodFill từ góc ngoài
    inv = np.full((h + 2, w + 2), 255, dtype=np.uint8)
    inv[1:h + 1, 1:w + 1] = np.where(bin_patch_32x48 > 0, 0, 255).astype(np.uint8)

    # Đóng nhẹ khe hẹp 1px nếu ảnh hơi đứt nét nhưng không làm bít bụng chữ '3' hay '2'
    ff = inv.copy()
    mask_ff = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(ff, mask_ff, (0, 0), 128)

    # Các pixel vẫn giữ giá trị 255 là các lỗ trắng khép kín nằm bên trong nét chữ
    holes = (ff[1:h + 1, 1:w + 1] == 255)
    lower_holes = holes[int(h * 0.38):int(h * 0.92), int(w * 0.15):int(w * 0.85)]
    hole_px = int(np.count_nonzero(lower_holes))
    return hole_px >= 14, hole_px


def _classify_left_digit_topology(left_bin):
    """
    Phân tích hình thái học & cấu trúc nét chữ ở độ phân giải cao (32x48)
    để phân biệt chính xác chữ số hàng chục trong {'1', '2', '3', '6'} của biển 10, 20, 30, 60:
      - '1' (10 km/h): Tỷ lệ khung bao hẹp (aspect < 0.49), nét thẳng đứng.
      - '6' (60 km/h): Có bụng tròn khép kín ở nửa dưới (enclosed bottom loop) + sống lưng bên trái liền mạch (UL > UR).
      - '2' (20 km/h): Đáy phẳng ngang đặc (bottom bar > 0.72) + nét chéo nửa dưới nằm bên TRÁI (LL > LR).
      - '3' (30 km/h): Cả nửa trên và nửa dưới đều mở bên trái, nét cong tập trung bên PHẢI (UR > UL và LR > LL) + có ngấn ngang ở giữa.
    """
    lh, lw = left_bin.shape[:2]
    aspect = lw / float(max(1, lh))

    # Chuẩn hóa về lưới 32x48
    norm = cv2.resize(left_bin.astype(np.uint8), (32, 48), interpolation=cv2.INTER_NEAREST)
    norm_f = (norm > 0).astype(np.float32)

    # 1. Kiểm tra chữ số '1' (hẹp ngang HOẶC góc dưới-trái hoàn toàn trống vì số '1' chỉ có móc ở trên-trái và thân dọc bên phải)
    bot_left_tail = float(np.mean(norm_f[36:47, 2:13]))
    right_stem = float(np.mean(norm_f[8:44, 17:30]))
    if aspect < 0.48 or (bot_left_tail < 0.18 and right_stem > 0.65):
        return 'speed_10', 0.95

    # Tính mật độ mực tại các vùng đặc trưng hình học của chữ số hàng chục (1, 2, 3, 6)
    ul = float(np.mean(norm_f[10:23, 2:13]))      # Trên - Trái
    ur = float(np.mean(norm_f[10:23, 19:30]))     # Trên - Phải
    ll = float(np.mean(norm_f[28:40, 1:12]))      # Dưới - Trái (nét chéo xuống của số 2 & bụng trái số 6)
    lr = float(np.mean(norm_f[28:39, 22:32]))     # Bụng Dưới - Phải ngoài cùng (đặc ở số 3 & 6, rỗng ở số 2!)
    bot_bar = float(np.mean(norm_f[40:48, 2:30])) # Thanh ngang đáy (đặc trưng số 2)
    mid_waist = float(np.mean(norm_f[20:28, 9:24]))  # Eo ngang giữa (đặc trưng số 3 & 6)
    left_spine = float(np.mean(norm_f[12:38, 1:11])) # Sống lưng dọc bên trái (đặc trưng số 6)

    has_loop, hole_px = _has_enclosed_bottom_loop(norm)

    scores = {
        'speed_10': 0.0,
        'speed_20': 0.0,
        'speed_30': 0.0,
        'speed_60': 0.0,
    }

    # Điểm cho '10': hẹp ngang
    if aspect < 0.56:
        scores['speed_10'] += (0.58 - aspect) * 5.0

    # Điểm cho '60': có lỗ khép kín ở bụng dưới + sống lưng trái đậm hơn góc trên phải
    if has_loop and ul >= ur - 0.05:
        scores['speed_60'] += 0.65 + min(0.25, hole_px / 120.0)
    scores['speed_60'] += 0.45 * left_spine + 0.30 * max(0.0, ul - ur) + 0.20 * min(ll, lr)

    # Điểm cho '20': góc trên phải đậm (ur > ul), bụng dưới phải rỗng (ll > lr), đáy ngang phẳng (bot_bar cao)
    scores['speed_20'] += (
        0.55 * max(0.0, ll - lr)
        + 0.35 * max(0.0, ur - ul)
        + 0.45 * max(0.0, bot_bar - 0.55)
        + (0.30 if (ll > lr + 0.08 and bot_bar > 0.72 and not has_loop) else 0.0)
    )

    # Điểm cho '30': bụng dưới phải phình ra rất đậm (lr > ll), góc trên mở bên trái, có ngấn giữa
    scores['speed_30'] += (
        0.55 * max(0.0, lr - ll)
        + 0.30 * max(0.0, ur - ul)
        + 0.25 * mid_waist
        + (0.30 if (lr > ll + 0.12 and not has_loop) else 0.0)
    )

    best_cls = max(scores.keys(), key=lambda k: scores[k])
    sorted_vals = sorted(scores.values(), reverse=True)
    margin = sorted_vals[0] - sorted_vals[1]
    conf = float(np.clip(0.65 + 0.55 * margin, 0.60, 0.98))
    return best_cls, conf


def _verify_and_refine_speed_sign(sub_bgr, pred_cls, pred_score, onnx_probs=None):
    """
    Phương án khắc phục triệt để nhận diện sai biển tốc độ (5, 10, 20, 30, 60 km/h):
    Phân tích trực tiếp vùng lõi trắng của ảnh gốc độ phân giải cao (96x96) trước khi bị thu nhỏ 32x32:
      1. Loại bỏ biển cấm có gạch chéo đỏ xuyên tâm (cấm đi thẳng / cấm rẽ).
      2. Tách thành phần liên thông (Connected Components) của chữ số màu đen trong lòng biển trắng:
         - 1 chữ số nằm giữa lòng biển -> Chắc chắn là 'speed_5' (5 km/h).
         - 2 chữ số (chữ số hàng chục + chữ số '0' bên phải) -> Phân tích hình thái học 32x48 của chữ số hàng chục
           để phân biệt chính xác '10', '20', '30', '60' kết hợp xác suất ONNX.
    Trả về: (refined_cls_name, refined_score, is_valid_speed_sign)
    """
    if sub_bgr is None or sub_bgr.size == 0:
        return pred_cls, pred_score, False

    # Phóng lên 96x96 để giữ trọn chi tiết nét chữ số
    work_96 = cv2.resize(sub_bgr, (96, 96), interpolation=cv2.INTER_CUBIC)
    hsv_96 = cv2.cvtColor(work_96, cv2.COLOR_BGR2HSV)
    gray_96 = cv2.cvtColor(work_96, cv2.COLOR_BGR2GRAY)
    b_ch, g_ch, r_ch = [c.astype(np.int16) for c in cv2.split(work_96)]
    red_chroma = r_ch - np.maximum(g_ch, b_ch)

    # 1. Kiểm tra vùng tâm (r <= 24px): Biển tốc độ KHÔNG được có gạch chéo đỏ xuyên qua tâm
    yy, xx = np.ogrid[:96, :96]
    dist_sq = (xx - 47.5) ** 2 + (yy - 47.5) ** 2
    center_r24 = dist_sq <= (23.0 ** 2)
    red_in_center = (
        (((hsv_96[:, :, 0] <= 13) | (hsv_96[:, :, 0] >= 163)) & (hsv_96[:, :, 1] >= 65) & (red_chroma >= 20))
        & center_r24
    )
    if float(np.count_nonzero(red_in_center)) / float(np.count_nonzero(center_r24)) >= 0.08:
        # Có vạch đỏ cắt ngang tâm -> Đây là biển cấm (no_left/no_right/no_straight), không phải biển tốc độ
        return pred_cls, pred_score, False

    # 2. Tách mặt nạ chữ số màu đen nằm trong lòng trắng (r <= 31px)
    inner_mask = dist_sq <= (31.0 ** 2)
    inner_pixels = gray_96[inner_mask]
    if inner_pixels.size == 0:
        return pred_cls, pred_score, False

    p3, p90_in = np.percentile(inner_pixels, [3, 90])
    if (p90_in - p3) < 28.0:
        return pred_cls, pred_score, False

    thresh_val = float(p3 + 0.33 * (p90_in - p3))
    dark_ink = ((gray_96 < thresh_val) & (hsv_96[:, :, 1] < 125) & inner_mask).astype(np.uint8) * 255

    # Lọc các thành phần liên thông của chữ số
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_ink, connectivity=8)
    digit_comps = []
    for lbl in range(1, num_labels):
        cx_i, cy_i = centroids[lbl]
        x_i, y_i, w_i, h_i, area_i = stats[lbl]
        # Chữ số tốc độ nằm gần tâm theo chiều dọc và có chiều cao hợp lý
        if area_i < 20 or h_i < 14 or h_i > 58 or w_i < 4 or w_i > 56:
            continue
        if not (18 <= cy_i <= 78 and 14 <= cx_i <= 82):
            continue
        digit_comps.append((int(x_i), int(y_i), int(w_i), int(h_i), int(area_i), float(cx_i), float(cy_i), lbl))

    if not digit_comps:
        return pred_cls, pred_score, True

    # Sắp xếp các thành phần chữ số từ trái sang phải
    digit_comps.sort(key=lambda item: item[0])

    # Tính khung bao tổng của toàn bộ chữ số ở tâm
    min_x = min(c[0] for c in digit_comps)
    min_y = min(c[1] for c in digit_comps)
    max_x = max(c[0] + c[2] for c in digit_comps)
    max_y = max(c[1] + c[3] for c in digit_comps)
    total_w = max_x - min_x
    total_h = max(1, max_y - min_y)
    total_aspect = total_w / float(total_h)

    # TRƯỜNG HỢP A: Chỉ có 1 chữ số ở giữa (Biển tốc độ 5 km/h)
    if len(digit_comps) == 1 and total_aspect < 0.86:
        comp = digit_comps[0]
        cx_0 = comp[5]
        # Chữ số '5' đơn lẻ nằm ngay vùng tâm (34 <= cx <= 62)
        if 34.0 <= cx_0 <= 62.0 and comp[3] >= 18:
            onnx_s5 = float(onnx_probs[SPEED_CLASS_TO_IDX['speed_5']]) if onnx_probs is not None else 0.0
            final_sc = max(pred_score, 0.85, onnx_s5)
            return 'speed_5', final_sc, True

    # TRƯỜNG HỢP B: Có 2 chữ số (10, 20, 30, 60 km/h) hoặc 2 chữ số dính nhẹ nhau
    left_bin = None
    if len(digit_comps) >= 2:
        # Lấy thành phần chữ số bên trái và bên phải lớn nhất
        top2 = sorted(digit_comps, key=lambda c: c[4], reverse=True)[:2]
        top2.sort(key=lambda c: c[0])
        left_c, right_c = top2[0], top2[1]
        # Kiểm tra 2 chữ số nằm ngang hàng nhau
        if abs(left_c[6] - right_c[6]) <= 12.0 and right_c[5] > left_c[5] + 6.0:
            lx, ly, lw, lh, _, _, _, llbl = left_c
            left_bin = (labels[ly:ly + lh, lx:lx + lw] == llbl).astype(np.uint8)
    elif len(digit_comps) == 1 and total_aspect >= 0.86:
        # Hai chữ số bị dính nhẹ vào nhau khi mờ -> tìm khe hẹp nhất theo tổng cột (Vertical Projection Valley)
        lx, ly, lw, lh, _, _, _, _ = digit_comps[0]
        merged_patch = (dark_ink[ly:ly + lh, lx:lx + lw] > 0).astype(np.uint8)
        col_sums = np.sum(merged_patch, axis=0)
        s_min = max(3, int(lw * 0.24))
        s_max = max(s_min + 2, int(lw * 0.60))
        split_w = s_min + int(np.argmin(col_sums[s_min:s_max]))
        raw_left = merged_patch[:, :split_w]
        # Cắt sát lại khung bao thực của chữ số hàng chục sau khi tách khe
        ys, xs = np.nonzero(raw_left)
        if xs.size >= 8:
            left_bin = raw_left[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        else:
            left_bin = raw_left

    if left_bin is not None and left_bin.size > 0:
        topo_cls, topo_conf = _classify_left_digit_topology(left_bin)
        if onnx_probs is not None:
            # Kết hợp điểm hình thái học độ phân giải cao (96x96) và xác suất ONNX (32x32)
            combined_scores = {}
            for s_cls in ('speed_10', 'speed_20', 'speed_30', 'speed_60'):
                idx = SPEED_CLASS_TO_IDX[s_cls]
                p_onnx = float(onnx_probs[idx]) if idx < len(onnx_probs) else 0.0
                p_topo = topo_conf if s_cls == topo_cls else 0.15
                combined_scores[s_cls] = 0.45 * p_onnx + 0.55 * p_topo
            best_speed_cls = max(combined_scores.keys(), key=lambda k: combined_scores[k])
            best_speed_sc = float(max(pred_score, combined_scores[best_speed_cls], topo_conf))
            return best_speed_cls, best_speed_sc, True
        return topo_cls, max(pred_score, topo_conf), True

    return pred_cls, pred_score, True


def _check_color_class_consistency(sub_bgr, cls_name, light_mode="NORMAL"):
    """
    Kiểm tra màu sắc và cấu trúc vùng lõi của vùng cắt có khớp với biển báo thực tế hay không.
    Loại bỏ đèn hậu xe (taillights), đèn đường vàng cam (streetlamps) và mảng phản quang mặt đường ban đêm.
    """
    if sub_bgr is None or sub_bgr.size == 0:
        return False

    h, w = sub_bgr.shape[:2]
    hsv = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)
    b_ch, g_ch, r_ch = [c.astype(np.int16) for c in cv2.split(sub_bgr)]
    red_chroma = r_ch - np.maximum(g_ch, b_ch)
    blue_chroma = b_ch - np.maximum(r_ch, g_ch)

    total_px = float(h * w)

    if light_mode == "DARK":
        very_dark_ratio = float(np.count_nonzero(hsv[:, :, 2] < 42)) / total_px
        if very_dark_ratio > 0.45:
            return False

    blue_mask = (
        (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 140) &
        (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 38) &
        (blue_chroma >= 15)
    )
    red_mask = (
        ((hsv[:, :, 0] <= 13) | (hsv[:, :, 0] >= 163)) &
        (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 38) &
        (red_chroma >= 18) & (r_ch >= (g_ch * 12) // 10)
    )

    blue_ratio = float(np.count_nonzero(blue_mask)) / total_px
    red_ratio = float(np.count_nonzero(red_mask)) / total_px

    # Kiểm tra vùng lõi trung tâm (50% diện tích giữa biển báo)
    cy0, cy1 = max(1, h // 4), max(2, (3 * h) // 4)
    cx0, cx1 = max(1, w // 4), max(2, (3 * w) // 4)
    core_hsv = hsv[cy0:cy1, cx0:cx1]
    core_gray = cv2.cvtColor(sub_bgr[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)
    core_total = float(max(1, core_hsv.shape[0] * core_hsv.shape[1]))

    neutral_white_core = float(np.count_nonzero((core_hsv[:, :, 1] <= 95) & (core_hsv[:, :, 2] >= 95))) / core_total
    yellow_core = float(np.count_nonzero(
        (core_hsv[:, :, 0] >= 18) & (core_hsv[:, :, 0] <= 36) &
        (core_hsv[:, :, 1] >= 70) & (core_hsv[:, :, 2] >= 90)
    )) / core_total
    dark_symbol_core = float(np.count_nonzero(core_gray < 115)) / core_total
    p10, p90 = np.percentile(core_gray, [10, 90])

    if cls_name in BLUE_SIGNS:
        if not (blue_ratio >= 0.12 and blue_ratio > red_ratio * 1.25):
            return False
        return neutral_white_core >= 0.04

    if cls_name in SOLID_RED_SIGNS:
        if red_ratio < 0.22 or red_ratio < blue_ratio * 1.5:
            return False
        if neutral_white_core < 0.04 or neutral_white_core > 0.65:
            return False
        if np.any(red_mask):
            mean_r = float(np.mean(r_ch[red_mask]))
            mean_g = float(np.mean(g_ch[red_mask]))
            if mean_r < mean_g * 1.32:
                return False
        return True

    if cls_name in NO_PARKING_SIGNS:
        # Biển Cấm đỗ xe (P.131a/P.130): bắt buộc có viền/gạch chéo Đỏ VÀ nền Xanh dương (hoặc nền trắng chữ P)
        if red_ratio < 0.08:
            return False
        return (blue_ratio >= 0.08) or (neutral_white_core >= 0.12 and dark_symbol_core >= 0.04)

    if cls_name in END_LIMIT_SIGNS:
        # Biển Hết giới hạn tốc độ: nền trắng sáng trung tính + vạch chéo đen ở tâm, hầu như không có màu đỏ rực
        if red_ratio > 0.08:
            return False
        p5_c, p95_c = np.percentile(core_gray, [5, 95])
        return neutral_white_core >= 0.22 and dark_symbol_core >= 0.04 and (p95_c - p5_c) >= 35.0

    if cls_name in WHITE_CORE_RED_SIGNS:
        if not (red_ratio >= 0.06 and red_ratio >= blue_ratio * 0.60):
            return False
        if neutral_white_core < 0.12:
            return False
        if (p90 - p10) < 25.0:
            return False
        return True

    if cls_name in WARNING_TRIANGLE_SIGNS:
        if not (red_ratio >= 0.06 and red_ratio >= blue_ratio * 0.60):
            return False
        if (neutral_white_core + yellow_core) < 0.14:
            return False
        if dark_symbol_core < 0.05 or (p90 - p10) < 35.0:
            return False
        return True

    return True


class SignTracker:
    """
    Bộ lọc theo thời gian (Temporal Voting & Box Smoothing) giúp chống nhiễu nhấp nháy trên Video/Webcam.
    Tích hợp trọng số diện tích (Size-Weighted Voting): khi xe tiến lại gần biển báo (khung bao lớn, chữ số rõ),
    phiếu bầu sẽ có trọng số cao hơn lúc biển còn ở xa mờ, giúp tự động sửa sai nếu ở xa nhận nhầm tốc độ.
    """

    def __init__(self, history_size=6, min_hits=2, max_age=4, iou_threshold=0.30):
        self.history_size = history_size
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []

    def reset(self):
        self.tracks.clear()

    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = b1[2] * b1[3]
        a2 = b2[2] * b2[3]
        return inter / float(a1 + a2 - inter + 1e-6)

    def update(self, raw_detections):
        matched_track_indices = set()

        for det in raw_detections:
            cls_name, x, y, w, h, score = det
            box = [x, y, w, h]
            # Trọng số theo kích thước khung biển (biển càng gần/rõ nét thì trọng số càng cao, từ 1.0 -> 1.85)
            size_weight = float(np.clip(min(w, h) / 28.0, 0.85, 1.85))

            best_idx = -1
            best_iou = self.iou_threshold
            for idx, trk in enumerate(self.tracks):
                if idx in matched_track_indices:
                    continue
                iou_val = self._iou(box, trk['box'])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx

            if best_idx >= 0:
                trk = self.tracks[best_idx]
                alpha = 0.65
                trk['box'] = [
                    int(round(alpha * box[i] + (1 - alpha) * trk['box'][i]))
                    for i in range(4)
                ]
                trk['votes'].append((cls_name, float(score), size_weight))
                if len(trk['votes']) > self.history_size:
                    trk['votes'].pop(0)
                trk['age'] = 0
                trk['hits'] += 1
                matched_track_indices.add(best_idx)
            else:
                self.tracks.append({
                    'box': box,
                    'votes': [(cls_name, float(score), size_weight)],
                    'age': 0,
                    'hits': 1,
                })
                matched_track_indices.add(len(self.tracks) - 1)

        active_tracks = []
        confirmed_signs = []
        for idx, trk in enumerate(self.tracks):
            if idx not in matched_track_indices:
                trk['age'] += 1
            if trk['age'] <= self.max_age:
                active_tracks.append(trk)
                max_vote_sc = max(v[1] for v in trk['votes'])
                if (trk['hits'] >= self.min_hits or max_vote_sc >= 0.82) and trk['age'] <= 2:
                    weighted_sum = {}
                    raw_score_sum = {}
                    counts = {}
                    for v_idx, vote_item in enumerate(trk['votes']):
                        c_name, sc = vote_item[0], vote_item[1]
                        sz_w = vote_item[2] if len(vote_item) > 2 else 1.0
                        # Ưu tiên frame mới hơn (recency) kết hợp kích thước khung biển (size_weight)
                        recency_w = 1.0 + 0.12 * v_idx
                        w_total = sz_w * recency_w
                        weighted_sum[c_name] = weighted_sum.get(c_name, 0.0) + sc * w_total
                        raw_score_sum[c_name] = raw_score_sum.get(c_name, 0.0) + sc
                        counts[c_name] = counts.get(c_name, 0) + 1
                    best_cls = max(weighted_sum.keys(), key=lambda k: weighted_sum[k])
                    avg_score = raw_score_sum[best_cls] / counts[best_cls]
                    bx, by, bw, bh = trk['box']
                    confirmed_signs.append([best_cls, bx, by, bw, bh, float(avg_score)])

        self.tracks = active_tracks
        return confirmed_signs


def detect_traffic_signs(img, model=None, draw=None, tracker=None, conf_thresh=0.60, return_scores=False, return_debug=False):
    """Phát hiện và phân loại 21 biển báo giao thông trên ảnh/video có tự động bù sáng & xác thực chữ số tốc độ."""
    if img is None or img.size == 0:
        if return_debug:
            return [], {}
        return []

    if model is None:
        model = load_sign_model()

    proc_img, light_mode = preprocess_lighting(img, return_mode=True)
    work_img = proc_img if light_mode in ("DARK", "BRIGHT") else img

    # Tạo mặt nạ màu trực tiếp (tái sử dụng proc_img đã tính để không chạy preprocess_lighting 2 lần)
    mask_r, mask_b, mask = compute_sign_color_masks(img, use_clahe=True, precomputed_lighting=(proc_img, light_mode))
    bboxes = get_boxes_from_mask(mask, image=work_img, verify_geometry=True)

    raw_detections = []
    onnx_previews = []
    for bbox in bboxes:
        x, y, w, h = bbox
        pad_x = int(w * 0.04)
        pad_y = int(h * 0.04)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(work_img.shape[1], x + w + pad_x)
        y1 = min(work_img.shape[0], y + h + pad_y)

        sub_orig = img[y0:y1, x0:x1]
        sub_work = work_img[y0:y1, x0:x1]
        if sub_work.shape[0] < 14 or sub_work.shape[1] < 14:
            continue

        candidates_bgr = [sub_work]
        stretched = _stretch_sign_crop(sub_orig)
        if stretched is not sub_orig:
            candidates_bgr.append(stretched)

        best_cls_idx = 0
        best_score = 0.0
        best_sub_bgr = sub_work
        best_preds_vec = None

        for cand_bgr in candidates_bgr:
            cand_rgb = cv2.cvtColor(cand_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = cv2.resize(cand_rgb, (32, 32), interpolation=cv2.INTER_AREA)
            inp = np.expand_dims(inp, axis=0)
            model.setInput(inp)
            preds = model.forward()[0]
            c_idx = int(preds.argmax())
            sc = float(preds[c_idx])
            if c_idx != 0 and sc > best_score:
                best_cls_idx = c_idx
                best_score = sc
                best_sub_bgr = cand_bgr
                best_preds_vec = preds

        cls_name = CLASSES_15[best_cls_idx] if best_cls_idx < len(CLASSES_15) else 'unknown'
        color_ok = (best_cls_idx != 0) and _check_color_class_consistency(best_sub_bgr, cls_name, light_mode=light_mode)

        # Áp dụng bộ xác thực & tinh chỉnh chữ số tốc độ trên ảnh gốc độ phân giải cao (96x96)
        if color_ok and cls_name in SPEED_LIMIT_SIGNS:
            # Nếu biển tốc độ ở quá xa (kích thước < 18px), yêu cầu độ tự tin cao hơn để tránh đoán mò chữ số mờ
            if min(w, h) < 18 and best_score < max(conf_thresh, 0.78):
                color_ok = False
            else:
                cls_name, best_score, speed_valid = _verify_and_refine_speed_sign(
                    best_sub_bgr, cls_name, best_score, onnx_probs=best_preds_vec
                )
                if not speed_valid:
                    color_ok = False

        if return_debug:
            onnx_previews.append({
                "box": [int(x), int(y), int(w), int(h)],
                "crop_bgr": best_sub_bgr,
                "cls_name": cls_name,
                "score": float(best_score),
                "accepted": bool(best_cls_idx != 0 and best_score >= conf_thresh and color_ok),
            })

        if best_cls_idx == 0 or best_score < conf_thresh or not color_ok:
            continue

        raw_detections.append([cls_name, int(x), int(y), int(w), int(h), best_score])

    final_detections = tracker.update(raw_detections) if tracker is not None else raw_detections

    if draw is not None:
        for det in final_detections:
            cls_name, x, y, w, h, score = det
            info = SIGN_INFO.get(cls_name, {'vn': cls_name, 'color': (0, 255, 0)})
            box_color = info['color']
            label_vn = info['vn']
            text = f"{label_vn} ({score:.2f})"

            cv2.rectangle(draw, (x, y), (x + w, y + h), box_color, 3, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            tx = max(4, min(x, draw.shape[1] - tw - 10))
            ty = max(y - 6, th + 8)
            cv2.rectangle(draw, (tx, ty - th - 6), (tx + tw + 8, ty + 2), (25, 25, 25), -1)
            cv2.putText(draw, text, (tx + 4, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

    out_dets = final_detections if return_scores else [[d[0], d[1], d[2], d[3], d[4]] for d in final_detections]
    if return_debug:
        debug_info = {
            "proc_img": proc_img,
            "light_mode": light_mode,
            "mask_r": mask_r,
            "mask_b": mask_b,
            "mask_clean": mask,
            "candidate_boxes": bboxes,
            "raw_detections": raw_detections,
            "onnx_previews": onnx_previews,
        }
        return out_dets, debug_info
    return out_dets
