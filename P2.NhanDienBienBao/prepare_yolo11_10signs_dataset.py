"""
Tạo bộ dữ liệu chuẩn YOLO11 cho 10 biển báo đầu tiên (Hybrid: Ảnh thật Việt Nam + Ảnh tổng hợp từ PNG).
Chạy:
    py P2.NhanDienBienBao/prepare_yolo11_10signs_dataset.py
Kết quả:
    - P2.NhanDienBienBao/yolo11_10signs_dataset/ (images/train, images/val, labels/train, labels/val, data.yaml)
    - P2.NhanDienBienBao/yolo11_10signs_dataset.zip (sẵn sàng upload lên Google Colab GPU)
"""

import os
import random
import shutil
import zipfile
from collections import Counter

import cv2
import numpy as np

# 10 biển báo đầu tiên theo đúng thứ tự 1 -> 10 trong bảng biển báo
YOLO_CLASSES_10 = [
    ("left",        "ReTrai.png",        52),  # 0: 1. Chỉ được rẽ trái
    ("no_left",     "CamReTrai.png",     17),  # 1: 2. Cấm rẽ trái
    ("right",       "RePhai.png",        53),  # 2: 3. Chỉ được rẽ phải
    ("no_right",    "CamRePhai.png",     18),  # 3: 4. Cấm rẽ phải
    ("straight",    "DiThang.png",       None),  # 4: 5. Chỉ được đi thẳng
    ("stop",        "DungStop.png",      64),  # 5: 6. Dừng lại (STOP)
    ("no_straight", "CamDiThang.png",    None),  # 6: 7. Cấm đi thẳng
    ("no_entry",    "CamNguocChieu.png", 0),   # 7: 8. Cấm đi ngược chiều
    ("speed_30",    "TocDo30.png",       62),  # 8: 9. Tốc độ tối đa 30 km/h
    ("speed_60",    "TocDo60.png",       12),  # 9: 10. Tốc độ tối đa 60 km/h
]

VN_TO_YOLO = {vn_id: idx for idx, (_, _, vn_id) in enumerate(YOLO_CLASSES_10) if vn_id is not None}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PNG_DIR = os.path.join(BASE_DIR, "data", "DuLieuBienBaoPNG")
VN_DATASET_DIR = os.path.join(BASE_DIR, "traffic-sign-detection-vietnam", "dataset")
OUT_DIR = os.path.join(BASE_DIR, "yolo11_10signs_dataset")
OUT_ZIP = os.path.join(BASE_DIR, "yolo11_10signs_dataset.zip")

IMG_W, IMG_H = 640, 480
SEED = 42


def load_sign_pngs():
    """Đọc 10 file PNG 4 kênh (BGRA) và cắt gọn sát biên alpha để bbox chính xác tuyệt đối."""
    signs = {}
    for cls_id, (cls_name, fname, _) in enumerate(YOLO_CLASSES_10):
        path = os.path.join(PNG_DIR, fname)
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f"Khong tim thay anh PNG bien bao: {path}")
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        elif im.shape[2] == 3:
            alpha = np.full((im.shape[0], im.shape[1], 1), 255, dtype=np.uint8)
            im = np.concatenate([im, alpha], axis=2)

        alpha = im[:, :, 3]
        ys, xs = np.where(alpha > 20)
        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            im = im[y0:y1, x0:x1]
        signs[cls_id] = im
    return signs


def make_synthetic_room_bg(rng, w=IMG_W, h=IMG_H):
    """Tạo phông nền phòng/sa bàn (tường xám, sàn, đường vạch trắng, bóng mờ) cho trường hợp test trong nhà."""
    base_val = rng.randint(35, 215)
    tint = np.array([rng.randint(-18, 18), rng.randint(-14, 14), rng.randint(-14, 22)], dtype=np.int16)
    bg = np.clip(np.full((h, w, 3), base_val, dtype=np.int16) + tint, 0, 255).astype(np.uint8)

    # Gradient ánh sáng ngang/dọc
    gx = np.linspace(rng.uniform(0.70, 1.15), rng.uniform(0.70, 1.15), w, dtype=np.float32)
    gy = np.linspace(rng.uniform(0.75, 1.15), rng.uniform(0.65, 1.10), h, dtype=np.float32)
    grad = np.outer(gy, gx)
    bg = np.clip(bg.astype(np.float32) * grad[:, :, None], 0, 255).astype(np.uint8)

    # Thêm các khối hình chữ nhật / vạch kẻ sa bàn ngẫu nhiên để mô phỏng tường, bàn, vạch đường
    for _ in range(rng.randint(2, 7)):
        x1, y1 = rng.randint(0, w - 20), rng.randint(0, h - 20)
        x2, y2 = rng.randint(x1 + 10, w), rng.randint(y1 + 10, h)
        col = (rng.randint(20, 235), rng.randint(20, 235), rng.randint(20, 235))
        cv2.rectangle(bg, (x1, y1), (x2, y2), col, -1)

    for _ in range(rng.randint(1, 4)):
        pt1 = (rng.randint(0, w), rng.randint(0, h))
        pt2 = (rng.randint(0, w), rng.randint(0, h))
        col = (rng.randint(160, 255), rng.randint(160, 255), rng.randint(160, 255))
        cv2.line(bg, pt1, pt2, col, rng.randint(2, 10))

    bg = cv2.GaussianBlur(bg, (rng.choice([3, 5, 7]), rng.choice([3, 5, 7])), 0)
    return bg


def load_real_bg_image(path, rng, w=IMG_W, h=IMG_H):
    """Đọc ảnh nền thực tế và crop/resize về (w, h)."""
    im = cv2.imread(path)
    if im is None or im.size == 0:
        return make_synthetic_room_bg(rng, w, h)
    ih, iw = im.shape[:2]
    if rng.random() < 0.5:
        im = cv2.flip(im, 1)  # Lật ngang ảnh nền (không có biển báo mục tiêu) để tăng đa dạng
    # Random crop 60%-100% khung hình rồi resize về 640x480
    scale = rng.uniform(0.60, 1.0)
    cw, ch = max(64, int(iw * scale)), max(64, int(ih * scale))
    x0 = rng.randint(0, max(0, iw - cw))
    y0 = rng.randint(0, max(0, ih - ch))
    crop = im[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def warp_sign_bgra(sign_bgra, target_size, rng):
    """Resize, xoay nhẹ (±14°) và biến dạng phối cảnh nhẹ giống góc nhìn camera thật."""
    sh, sw = sign_bgra.shape[:2]
    scale = target_size / float(max(sh, sw))
    nw = max(16, int(round(sw * scale)))
    nh = max(16, int(round(sh * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(sign_bgra, (nw, nh), interpolation=interp)

    pad = max(12, int(max(nw, nh) * 0.24))
    canvas = cv2.copyMakeBorder(resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    ch, cw = canvas.shape[:2]

    # Xoay nhẹ
    angle = rng.uniform(-14.0, 14.0)
    M_rot = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), angle, 1.0)
    warped = cv2.warpAffine(canvas, M_rot, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Biến dạng phối cảnh nhẹ (như cầm biển nghiêng trước webcam)
    if rng.random() < 0.70:
        jitter = max(2.0, min(nw, nh) * rng.uniform(0.03, 0.11))
        src_pts = np.float32([[pad, pad], [pad + nw, pad], [pad + nw, pad + nh], [pad, pad + nh]])
        dst_pts = src_pts + np.float32([
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
        ])
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(warped, M_persp, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Cắt gọn lại sát biên alpha sau khi biến dạng
    alpha = warped[:, :, 3]
    ys, xs = np.where(alpha > 25)
    if len(xs) < 10 or len(ys) < 10:
        return resized
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return warped[y0:y1, x0:x1]


def augment_sign_appearance(sign_bgr, alpha_mask, rng):
    """Biến đổi độ sáng, độ bão hòa màu (webcam nhạt màu), lóa đèn chiếu cục bộ và bóng mờ trên riêng biển báo."""
    out = sign_bgr.astype(np.float32)

    # Giả lập webcam nhạt màu (desaturation) hoặc màu chuẩn
    if rng.random() < 0.55:
        hsv = cv2.cvtColor(sign_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_scale = rng.uniform(0.32, 0.85) if rng.random() < 0.65 else rng.uniform(0.85, 1.25)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        # Dịch nhiệt độ màu nhẹ (đèn vàng phòng / ánh sáng lạnh)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Ám màu nhẹ theo môi trường đèn phòng
    if rng.random() < 0.50:
        tint = np.array([rng.uniform(0.88, 1.12), rng.uniform(0.90, 1.10), rng.uniform(0.88, 1.14)], dtype=np.float32)
        out = np.clip(out * tint, 0, 255)

    # Giả lập đèn chiếu mạnh trực tiếp tạo đốm lóa trên biển báo
    if rng.random() < 0.30:
        h, w = out.shape[:2]
        cx = rng.uniform(0.15, 0.85) * w
        cy = rng.uniform(0.15, 0.85) * h
        sigma = rng.uniform(0.22, 0.55) * max(h, w)
        amp = rng.uniform(55.0, 135.0)
        yy, xx = np.ogrid[:h, :w]
        spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)) * amp
        out = np.clip(out + spot[:, :, None], 0, 255)

    # Giả lập độ sáng tối cục bộ của biển báo
    if rng.random() < 0.45:
        gain = rng.uniform(0.45, 1.35)
        bias = rng.uniform(-22.0, 26.0)
        out = np.clip(out * gain + bias, 0, 255)

    out_u8 = out.astype(np.uint8)

    # Làm mờ nhẹ giống webcam lấy nét chưa căng hoặc xe đang chạy
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        if rng.random() < 0.5:
            out_u8 = cv2.GaussianBlur(out_u8, (k, k), 0)
        else:
            kernel = np.zeros((k, k), dtype=np.float32)
            if rng.random() < 0.5:
                kernel[k // 2, :] = 1.0 / k
            else:
                kernel[:, k // 2] = 1.0 / k
            out_u8 = cv2.filter2D(out_u8, -1, kernel)

    return out_u8


def maybe_draw_hand_occlusion(canvas, bx, by, bw, bh, rng):
    """Giả lập ngón tay / vật che nhẹ một góc mép biển báo (che ~8-16% mép) để YOLO không bị mất khi cầm tay."""
    if rng.random() > 0.28:
        return
    # Màu da tay thực tế (BGR)
    skin_bgr = (
        rng.randint(75, 145),
        rng.randint(100, 165),
        rng.randint(125, 205),
    )
    side = rng.choice(["bottom", "left", "right", "top"])
    rad_x = max(4, int(bw * rng.uniform(0.10, 0.19)))
    rad_y = max(4, int(bh * rng.uniform(0.10, 0.19)))
    if side == "bottom":
        cx = bx + int(bw * rng.uniform(0.25, 0.75))
        cy = by + bh - int(rad_y * 0.35)
    elif side == "top":
        cx = bx + int(bw * rng.uniform(0.25, 0.75))
        cy = by + int(rad_y * 0.35)
    elif side == "left":
        cx = bx + int(rad_x * 0.35)
        cy = by + int(bh * rng.uniform(0.25, 0.75))
    else:
        cx = bx + bw - int(rad_x * 0.35)
        cy = by + int(bh * rng.uniform(0.25, 0.75))

    overlay = canvas.copy()
    cv2.ellipse(overlay, (cx, cy), (rad_x, rad_y), rng.uniform(-30, 30), 0, 360, skin_bgr, -1)
    cv2.GaussianBlur(overlay, (3, 3), 0, dst=overlay)
    cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, dst=canvas)


def augment_full_frame_lighting(frame, rng):
    """Biến đổi ánh sáng toàn khung hình: bình thường / quá tối / quá sáng chói / webcam nhạt màu."""
    r = rng.random()
    out = frame.astype(np.float32)

    if r < 0.24:
        # Quá tối / thiếu sáng ban đêm
        gamma = rng.uniform(1.55, 2.55)
        scale = rng.uniform(0.35, 0.72)
        out = np.clip(((out / 255.0) ** gamma) * 255.0 * scale, 0, 255)
        # Thêm nhiễu hạt cảm biến ban đêm
        noise = np.random.normal(0, rng.uniform(3.0, 9.0), out.shape).astype(np.float32)
        out = np.clip(out + noise, 0, 255)
    elif r < 0.46:
        # Quá sáng / chói lóa / ngược sáng
        gamma = rng.uniform(0.45, 0.78)
        bias = rng.uniform(15.0, 62.0)
        out = np.clip(((out / 255.0) ** gamma) * 255.0 + bias, 0, 255)
        if rng.random() < 0.55:
            # Đốm sáng chói tổng thể
            h, w = out.shape[:2]
            cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
            sig = rng.uniform(80.0, 180.0)
            yy, xx = np.ogrid[:h, :w]
            glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig)) * rng.uniform(45.0, 110.0)
            out = np.clip(out + glare[:, :, None], 0, 255)
    elif r < 0.66:
        # Webcam mờ nhạt, độ tương phản thấp
        mean_v = float(np.mean(out))
        alpha_c = rng.uniform(0.55, 0.85)
        out = np.clip((out - mean_v) * alpha_c + mean_v + rng.uniform(-18, 18), 0, 255)
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.38, 0.75), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    out_u8 = out.astype(np.uint8)
    if rng.random() < 0.25:
        out_u8 = cv2.GaussianBlur(out_u8, (3, 3), 0)
    return out_u8


def boxes_overlap(box_a, box_b, pad=8):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    return not (ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0)


def collect_vn_real_samples(split, max_per_class, rng):
    """
    Quét dataset traffic-sign-detection-vietnam/dataset/{split}:
    - Trả về (matched_samples, negative_bg_paths)
    - matched_samples: list of (img_path, [(yolo_cls, cx, cy, w, h), ...])
    - Chỉ giữ bbox đủ rõ (w >= 0.025 và h >= 0.025) để tránh hộp 5x5px gây nhiễu.
    """
    lbl_dir = os.path.join(VN_DATASET_DIR, split, "labels")
    img_dir = os.path.join(VN_DATASET_DIR, split, "images")
    if not (os.path.isdir(lbl_dir) and os.path.isdir(img_dir)):
        return [], []

    all_txt = sorted(os.listdir(lbl_dir))
    rng.shuffle(all_txt)

    by_class = {c: [] for c in range(len(YOLO_CLASSES_10))}
    neg_paths = []

    for fn in all_txt:
        if not fn.endswith(".txt"):
            continue
        stem = os.path.splitext(fn)[0]
        img_path = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, stem + ".png")
            if not os.path.exists(img_path):
                continue

        lbl_path = os.path.join(lbl_dir, fn)
        kept_boxes = []
        has_any_target = False
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    vn_id = int(float(parts[0]))
                    if vn_id in VN_TO_YOLO:
                        has_any_target = True
                        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                        if bw >= 0.025 and bh >= 0.025:
                            cx = float(np.clip(cx, 0.0, 1.0))
                            cy = float(np.clip(cy, 0.0, 1.0))
                            bw = float(np.clip(bw, 0.0, 1.0))
                            bh = float(np.clip(bh, 0.0, 1.0))
                            kept_boxes.append((VN_TO_YOLO[vn_id], cx, cy, bw, bh))
        except Exception:
            continue

        if kept_boxes:
            # Gán vào lớp ít mẫu nhất có mặt trong ảnh
            primary_cls = min(kept_boxes, key=lambda b: len(by_class[b[0]]))[0]
            by_class[primary_cls].append((img_path, kept_boxes))
        elif not has_any_target:
            neg_paths.append(img_path)

    selected = []
    seen = set()
    for c in range(len(YOLO_CLASSES_10)):
        pool = by_class[c]
        rng.shuffle(pool)
        for item in pool[:max_per_class]:
            if item[0] not in seen:
                seen.add(item[0])
                selected.append(item)

    return selected, neg_paths


def generate_split(split, signs_bgra, real_samples, bg_pool, synth_per_class, neg_count, rng):
    img_out_dir = os.path.join(OUT_DIR, "images", split)
    lbl_out_dir = os.path.join(OUT_DIR, "labels", split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    cls_counts = Counter()
    total_imgs = 0

    # 1. Copy các ảnh thật đã lọc nhãn (giữ nguyên độ sắc nét, nén jpg chuẩn)
    for idx, (src_img, boxes) in enumerate(real_samples):
        dst_stem = f"real_{split}_{idx:05d}"
        dst_img = os.path.join(img_out_dir, dst_stem + ".jpg")
        dst_lbl = os.path.join(lbl_out_dir, dst_stem + ".txt")
        shutil.copy2(src_img, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for cid, cx, cy, bw, bh in boxes:
                f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                cls_counts[cid] += 1
        total_imgs += 1

    # 2. Sinh ảnh tổng hợp cho từng lớp theo chỉ tiêu cân bằng
    synth_queue = []
    for cid in range(len(YOLO_CLASSES_10)):
        target_n = synth_per_class[cid]
        synth_queue.extend([cid] * target_n)
    rng.shuffle(synth_queue)

    s_idx = 0
    q_pos = 0
    while q_pos < len(synth_queue):
        # Mỗi ảnh tổng hợp ghép 1 hoặc 2 biển (65% 1 biển, 35% 2 biển)
        n_objs = 1 if (rng.random() < 0.65 or q_pos + 1 >= len(synth_queue)) else 2
        chosen_classes = synth_queue[q_pos:q_pos + n_objs]
        q_pos += n_objs

        # Chọn phông nền: 75% ảnh đường phố thực tế (không có 10 biển), 25% nền phòng/sa bàn tổng hợp
        if bg_pool and rng.random() < 0.75:
            bg = load_real_bg_image(rng.choice(bg_pool), rng, IMG_W, IMG_H)
        else:
            bg = make_synthetic_room_bg(rng, IMG_W, IMG_H)

        placed_boxes = []
        yolo_lines = []

        for cid in chosen_classes:
            # Phân bố kích thước biển báo: nhỏ (xa) / vừa / lớn (cầm gần webcam)
            sr = rng.random()
            if sr < 0.25:
                t_size = rng.randint(32, 62)
            elif sr < 0.72:
                t_size = rng.randint(62, 138)
            else:
                t_size = rng.randint(138, 225) if n_objs == 1 else rng.randint(100, 165)

            warped = warp_sign_bgra(signs_bgra[cid], t_size, rng)
            sh, sw = warped.shape[:2]
            if sw >= IMG_W - 12 or sh >= IMG_H - 12:
                continue

            placed = False
            for _ in range(25):
                x0 = rng.randint(4, IMG_W - sw - 4)
                y0 = rng.randint(4, IMG_H - sh - 4)
                cand = (x0, y0, x0 + sw, y0 + sh)
                if all(not boxes_overlap(cand, pb, pad=10) for pb in placed_boxes):
                    placed = True
                    break
            if not placed:
                continue

            alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
            sign_bgr = augment_sign_appearance(warped[:, :, :3], warped[:, :, 3], rng)

            # Alpha blend mượt viền vào nền
            roi = bg[y0:y0 + sh, x0:x0 + sw].astype(np.float32)
            blended = sign_bgr.astype(np.float32) * alpha + roi * (1.0 - alpha)
            bg[y0:y0 + sh, x0:x0 + sw] = np.clip(blended, 0, 255).astype(np.uint8)

            # Giả lập ngón tay cầm che nhẹ mép
            maybe_draw_hand_occlusion(bg, x0, y0, sw, sh, rng)

            placed_boxes.append((x0, y0, x0 + sw, y0 + sh))
            cx = (x0 + sw / 2.0) / float(IMG_W)
            cy = (y0 + sh / 2.0) / float(IMG_H)
            bw = sw / float(IMG_W)
            bh = sh / float(IMG_H)
            yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            cls_counts[cid] += 1

        if not yolo_lines:
            continue

        bg = augment_full_frame_lighting(bg, rng)
        dst_stem = f"synth_{split}_{s_idx:05d}"
        cv2.imwrite(os.path.join(img_out_dir, dst_stem + ".jpg"), bg, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(78, 95)])
        with open(os.path.join(lbl_out_dir, dst_stem + ".txt"), "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)
        s_idx += 1
        total_imgs += 1

    # 3. Sinh ảnh nền âm (Negative samples - không chứa 10 biển báo) để giảm nhận nhầm
    for n_idx in range(neg_count):
        if bg_pool and rng.random() < 0.78:
            bg = load_real_bg_image(rng.choice(bg_pool), rng, IMG_W, IMG_H)
        else:
            bg = make_synthetic_room_bg(rng, IMG_W, IMG_H)
        bg = augment_full_frame_lighting(bg, rng)
        dst_stem = f"neg_{split}_{n_idx:05d}"
        cv2.imwrite(os.path.join(img_out_dir, dst_stem + ".jpg"), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        with open(os.path.join(lbl_out_dir, dst_stem + ".txt"), "w", encoding="utf-8") as f:
            pass
        total_imgs += 1

    return total_imgs, cls_counts


def write_data_yaml():
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        f"nc: {len(YOLO_CLASSES_10)}",
        "names:",
    ]
    for idx, (cls_name, _, _) in enumerate(YOLO_CLASSES_10):
        lines.append(f"  {idx}: {cls_name}")
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return yaml_path


def zip_dataset():
    if os.path.exists(OUT_ZIP):
        os.remove(OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(OUT_DIR):
            for fn in files:
                full_p = os.path.join(root, fn)
                rel_p = os.path.relpath(full_p, OUT_DIR)
                zf.write(full_p, rel_p)
    size_mb = os.path.getsize(OUT_ZIP) / (1024.0 * 1024.0)
    return size_mb


def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    print("[1/5] Dang nap 10 anh PNG bien bao goc...")
    signs_bgra = load_sign_pngs()

    print("[2/5] Dang quet anh that tu traffic-sign-detection-vietnam...")
    real_train, bg_train = collect_vn_real_samples("train", max_per_class=160, rng=rng)
    real_val, bg_val = collect_vn_real_samples("val", max_per_class=35, rng=rng)
    print(f"      -> Real train: {len(real_train)} anh | Real val: {len(real_val)} anh")
    print(f"      -> Background pool: train={len(bg_train)}, val={len(bg_val)}")

    # Số mẫu tổng hợp cho từng lớp (2 lớp chưa có ảnh thật trong VN dataset sẽ được bù thêm)
    synth_train_per_cls = {
        cid: (400 if vn_id is None else 260)
        for cid, (_, _, vn_id) in enumerate(YOLO_CLASSES_10)
    }
    synth_val_per_cls = {
        cid: (65 if vn_id is None else 45)
        for cid, (_, _, vn_id) in enumerate(YOLO_CLASSES_10)
    }

    print("[3/5] Dang sinh tap TRAIN (Hybrid Real + Synthetic + Negative)...")
    n_train, cnt_train = generate_split("train", signs_bgra, real_train, bg_train, synth_train_per_cls, neg_count=180, rng=rng)

    print("[4/5] Dang sinh tap VAL...")
    n_val, cnt_val = generate_split("val", signs_bgra, real_val, bg_val, synth_val_per_cls, neg_count=35, rng=rng)

    yaml_path = write_data_yaml()
    print(f"\n[OK] Da tao data.yaml: {yaml_path}")
    print(f"     Tong so anh: train={n_train}, val={n_val}")
    print("     Phan bo so khung nhan theo tung bien bao:")
    for cid, (cls_name, _, _) in enumerate(YOLO_CLASSES_10):
        print(f"       {cid}: {cls_name:12s} | train={cnt_train[cid]:4d} | val={cnt_val[cid]:3d}")

    print("\n[5/5] Dang dong goi thanh file ZIP de upload len Google Colab...")
    size_mb = zip_dataset()
    print(f"[DONE] File ZIP san sang tai: {OUT_ZIP} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
