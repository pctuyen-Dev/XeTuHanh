import ast
from collections import deque
import os
import cv2
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ort = None
    ORT_AVAILABLE = False

try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    UltralyticsYOLO = None
    ULTRALYTICS_AVAILABLE = False


# DANH SÁCH 21 BIỂN BÁO
YOLO_CLASSES = [
    'left',             # 0 - 1. Chỉ được rẽ trái
    'no_left',          # 1 - 2. Cấm rẽ trái
    'right',            # 2 - 3. Chỉ được rẽ phải
    'no_right',         # 3 - 4. Cấm rẽ phải
    'straight',         # 4 - 5. Chỉ được đi thẳng
    'stop',             # 5 - 6. Dừng lại (STOP)
    'no_straight',      # 6 - 7. Cấm đi thẳng
    'no_entry',         # 7 - 8. Cấm đi ngược chiều
    'speed_30',         # 8 - 9. Tốc độ tối đa cho phép 30 km/h
    'speed_60',         # 9 - 10. Tốc độ tối đa cho phép 60 km/h
    'no_u_turn',        # 10 - 11. Cấm quay đầu
    'yield',            # 11 - 12. Nhường đường tại giao lộ
    'intersection',     # 12 - 13. Giao nhau cùng cấp
    'roundabout',       # 13 - 14. Chỉ được đi theo vòng xuyến
    'danger_left',      # 14 - 15. Chỗ ngoặt nguy hiểm bên trái
    'danger_right',     # 15 - 16. Chỗ ngoặt nguy hiểm bên phải
    'speed_10',         # 16 - 17. Tốc độ tối đa 10 km/h
    'speed_20',         # 17 - 18. Tốc độ tối đa 20 km/h
    'end_speed_limit',  # 18 - 19. Hết tất cả các lệnh cấm
    'road_work',        # 19 - 20. Công trường đang thi công
    'danger',           # 20 - 21. Nguy hiểm khác (!)
]

# Bảng ánh xạ thông tin điều khiển & hiển thị cho các biển báo
SIGN_INFO = {
    'left':            {'vn': 'ReTrai',          'action': 'RE TRAI',                'speed_factor': 0.85, 'steer_bias': -0.45, 'color': (255, 140, 0)},
    'no_left':         {'vn': 'CamReTrai',       'action': 'CAM RE TRAI',            'speed_factor': 0.90, 'steer_bias': +0.20, 'color': (0, 80, 255)},
    'right':           {'vn': 'RePhai',          'action': 'RE PHAI',                'speed_factor': 0.85, 'steer_bias': +0.45, 'color': (255, 140, 0)},
    'no_right':        {'vn': 'CamRePhai',       'action': 'CAM RE PHAI',            'speed_factor': 0.90, 'steer_bias': -0.20, 'color': (0, 80, 255)},
    'straight':        {'vn': 'DiThang',         'action': 'DI THANG',               'speed_factor': 1.00, 'steer_bias':  0.00, 'color': (255, 180, 0)},
    'stop':            {'vn': 'DUNG(STOP)',      'action': 'DUNG - STOP',            'speed_factor': 0.00, 'steer_bias':  0.00, 'color': (0, 0, 255)},
    'no_straight':     {'vn': 'CamDiThang',      'action': 'CAM DI THANG',           'speed_factor': 0.50, 'steer_bias':  0.00, 'color': (0, 80, 255)},
    'no_entry':        {'vn': 'CamNguocChieu',   'action': 'CAM DI NGUOC CHIEU',     'speed_factor': 0.00, 'steer_bias':  0.00, 'color': (0, 0, 255)},
    'speed_30':        {'vn': 'TocDo30',         'action': 'TOC DO DUOI 30km/h',     'speed_factor': 0.55, 'steer_bias':  0.00, 'color': (0, 165, 255)},
    'speed_60':        {'vn': 'TocDo60',         'action': 'TOC DO DUOI 60km/h',     'speed_factor': 1.25, 'steer_bias':  0.00, 'color': (0, 220, 120)},
    'no_u_turn':       {'vn': 'CamQuayDau',      'action': 'CAM QUAY DAU',           'speed_factor': 0.90, 'steer_bias':  0.00, 'color': (0, 80, 255)},
    'yield':           {'vn': 'NhuongDuong',     'action': 'GIAM TOC NHUONG DUONG',  'speed_factor': 0.50, 'steer_bias':  0.00, 'color': (0, 215, 255)},
    'intersection':    {'vn': 'GiaoLo',          'action': 'GIAM TOC QUA GIAO LO',   'speed_factor': 0.50, 'steer_bias':  0.00, 'color': (0, 200, 255)},
    'roundabout':      {'vn': 'VongXuyen',       'action': 'DI THEO VONG XUYEN',     'speed_factor': 0.70, 'steer_bias': -0.25, 'color': (255, 160, 0)},
    'danger_left':     {'vn': 'NgoacTrai',       'action': 'GIAM TOC CUA TRAI',      'speed_factor': 0.55, 'steer_bias': -0.20, 'color': (0, 185, 255)},
    'danger_right':    {'vn': 'NgoacPhai',       'action': 'GIAM TOC CUA PHAI',      'speed_factor': 0.55, 'steer_bias': +0.20, 'color': (0, 185, 255)},
    'speed_10':        {'vn': 'TocDo10',         'action': 'TOC DO DUOI 10km/h',     'speed_factor': 0.35, 'steer_bias':  0.00, 'color': (0, 155, 255)},
    'speed_20':        {'vn': 'TocDo20',         'action': 'TOC DO DUOI 20km/h',     'speed_factor': 0.45, 'steer_bias':  0.00, 'color': (0, 165, 255)},
    'end_speed_limit': {'vn': 'HetCam',          'action': 'KHOI PHUC TOC DO',       'speed_factor': 1.00, 'steer_bias':  0.00, 'color': (180, 230, 120)},
    'road_work':       {'vn': 'CongTruong',      'action': 'DI CHAM CONG TRUONG',    'speed_factor': 0.45, 'steer_bias':  0.00, 'color': (0, 180, 255)},
    'danger':          {'vn': 'NguyHiem',        'action': 'CANH BAO NGUY HIEM',     'speed_factor': 0.55, 'steer_bias':  0.00, 'color': (0, 140, 255)},
}

_default_model_cache = {}


class YOLO11SignDetector:
    def __init__(self, model_path, class_names=None, input_size=416):
        self.model_path = model_path
        self.mode = None  # "ort" | "cv2_dnn" | "pt"
        self.session = None
        self.net = None
        self.pt_model = None
        self.input_name = None
        self.input_h = int(input_size)
        self.input_w = int(input_size)
        self.class_names = list(class_names) if class_names is not None else list(YOLO_CLASSES)

        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".pt":
            if not ULTRALYTICS_AVAILABLE:
                raise RuntimeError("Can cai dat 'ultralytics' de nap file .pt, hoac hay dung file .onnx!")
            self.pt_model = UltralyticsYOLO(model_path)
            self.mode = "pt"
            if class_names is None:
                parsed = self._parse_class_names(getattr(self.pt_model, "names", None))
                if parsed:
                    self.class_names = parsed
        elif ORT_AVAILABLE:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self.mode = "ort"
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            ishape = inp.shape
            if len(ishape) == 4:
                if isinstance(ishape[2], int) and ishape[2] > 0:
                    self.input_h = int(ishape[2])
                if isinstance(ishape[3], int) and ishape[3] > 0:
                    self.input_w = int(ishape[3])

            meta = self.session.get_modelmeta().custom_metadata_map or {}
            if class_names is None and "names" in meta:
                try:
                    parsed = self._parse_class_names(ast.literal_eval(meta["names"]))
                    if parsed:
                        self.class_names = parsed
                except Exception:
                    pass
        else:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.mode = "cv2_dnn"

    @staticmethod
    def _parse_class_names(raw):
        if isinstance(raw, dict) and raw:
            return [str(raw[k]) for k in sorted(raw.keys())]
        if isinstance(raw, (list, tuple)) and raw:
            return [str(x) for x in raw]
        return None

    @staticmethod
    def letterbox(img, new_shape=(416, 416), color=(114, 114, 114)):
        h, w = img.shape[:2]
        nh, nw = int(new_shape[0]), int(new_shape[1])
        r = min(nw / float(w), nh / float(h))
        unpad_w = int(round(w * r))
        unpad_h = int(round(h * r))
        dw = (nw - unpad_w) / 2.0
        dh = (nh - unpad_h) / 2.0

        resized = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR) if (w != unpad_w or h != unpad_h) else img
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, left, top

    def _infer_raw(self, blob):
        if self.mode == "ort":
            out = self.session.run(None, {self.input_name: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        return np.asarray(out, dtype=np.float32)

    def predict(self, img, conf_thresh=0.50, preview_thresh=0.25, iou_thresh=0.45):
        orig_h, orig_w = img.shape[:2]
        padded, ratio, pad_x, pad_y = self.letterbox(img, (self.input_h, self.input_w))
        min_thr = float(min(conf_thresh, preview_thresh))

        if self.mode == "pt":
            res = self.pt_model.predict(
                img, imgsz=max(self.input_h, self.input_w),
                conf=min_thr, iou=iou_thresh, verbose=False
            )[0]
            accepted, previews = [], []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(confs)):
                    x1, y1, x2, y2 = xyxy[i]
                    bx = int(max(0, round(x1)))
                    by = int(max(0, round(y1)))
                    bw = int(max(1, round(x2 - x1)))
                    bh = int(max(1, round(y2 - y1)))
                    if bw < 10 or bh < 10:
                        continue
                    cid = int(clss[i])
                    sc = float(confs[i])
                    cls_name = self.class_names[cid] if cid < len(self.class_names) else f"cls_{cid}"
                    det = [cls_name, bx, by, bw, bh, sc]
                    previews.append(det)
                    if sc >= conf_thresh:
                        accepted.append(det)
            accepted.sort(key=lambda d: d[5], reverse=True)
            previews.sort(key=lambda d: d[5], reverse=True)
            return accepted, previews, padded

        blob = cv2.dnn.blobFromImage(padded, scalefactor=1.0 / 255.0, swapRB=True, crop=False)

        preds = self._infer_raw(blob)
        if preds.ndim == 3:
            preds = preds[0]
        # YOLO11 ONNX output shape: (4 + nc, num_anchors) -> chuyển về (num_anchors, 4 + nc)
        if preds.ndim == 2 and preds.shape[0] < preds.shape[1] and preds.shape[0] <= 128:
            preds = preds.T

        if preds.ndim != 2 or preds.shape[1] < 5:
            return [], [], padded

        scores_mat = preds[:, 4:]
        max_scores = scores_mat.max(axis=1)
        keep_mask = max_scores >= min_thr
        if not np.any(keep_mask):
            return [], [], padded

        kept_boxes = preds[keep_mask, :4]
        kept_cls = np.argmax(scores_mat[keep_mask], axis=1)
        kept_scores = max_scores[keep_mask]

        if float(np.max(kept_boxes[:, 2:4])) <= 2.0:
            kept_boxes = kept_boxes.copy()
            kept_boxes[:, [0, 2]] *= float(self.input_w)
            kept_boxes[:, [1, 3]] *= float(self.input_h)

        cx, cy, bw, bh = kept_boxes[:, 0], kept_boxes[:, 1], kept_boxes[:, 2], kept_boxes[:, 3]

        x1 = np.clip((cx - bw * 0.5 - pad_x) / ratio, 0.0, float(orig_w - 1))
        y1 = np.clip((cy - bh * 0.5 - pad_y) / ratio, 0.0, float(orig_h - 1))
        x2 = np.clip((cx + bw * 0.5 - pad_x) / ratio, 0.0, float(orig_w))
        y2 = np.clip((cy + bh * 0.5 - pad_y) / ratio, 0.0, float(orig_h))

        boxes_xywh = np.stack([x1, y1, np.maximum(1.0, x2 - x1), np.maximum(1.0, y2 - y1)], axis=1)
        nms_boxes = np.round(boxes_xywh).astype(int).tolist()
        nms_scores = kept_scores.astype(float).tolist()
        idxs = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, min_thr, iou_thresh)
        if len(idxs) == 0:
            return [], [], padded

        accepted, previews = [], []
        for idx in np.asarray(idxs).flatten():
            bx, by, bww, bhh = nms_boxes[int(idx)]
            if bww < 10 or bhh < 10:
                continue
            cid = int(kept_cls[int(idx)])
            sc = float(kept_scores[int(idx)])
            cls_name = self.class_names[cid] if cid < len(self.class_names) else f"cls_{cid}"
            det = [cls_name, bx, by, bww, bhh, sc]
            previews.append(det)
            if sc >= conf_thresh:
                accepted.append(det)

        accepted.sort(key=lambda d: d[5], reverse=True)
        previews.sort(key=lambda d: d[5], reverse=True)
        return accepted, previews, padded


def load_sign_model(model_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        candidates = [
            os.path.join(base_dir, "DataTrain.onnx"),
            os.path.join(base_dir, "DataTrain.pt"),
            os.path.join(base_dir, "best.onnx"),
            os.path.join(base_dir, "best.pt"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)
        if model_path is None:
            default_path = os.path.join(base_dir, "DataTrain.onnx")
            raise FileNotFoundError(f"Chua co model YOLO11n tai: {default_path}")

    if model_path in _default_model_cache:
        return _default_model_cache[model_path]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Khong tim thay model YOLO11n tai: {model_path}")

    detector = YOLO11SignDetector(model_path)
    _default_model_cache[model_path] = detector
    return detector


class SignTracker:
    def __init__(self, history_size=6, min_hits=2, max_age=4, iou_threshold=0.30):
        self.history_size = history_size
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []

    def reset(self):
        self.tracks.clear()

    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0:
            return 0.0
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, raw_detections):
        for trk in self.tracks:
            trk['age'] += 1
            trk['matched_this_frame'] = False

        for det in raw_detections:
            cls_name, x, y, w, h, score = det
            det_box = [x, y, w, h]
            det_area = max(100.0, float(w * h))

            best_iou = 0.0
            best_trk = None
            for trk in self.tracks:
                if trk['matched_this_frame']:
                    continue
                iou = self._compute_iou(trk['box'], det_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_trk = trk

            if best_trk is not None:
                alpha = 0.6
                old_b = best_trk['box']
                best_trk['box'] = [
                    int(alpha * x + (1 - alpha) * old_b[0]),
                    int(alpha * y + (1 - alpha) * old_b[1]),
                    int(alpha * w + (1 - alpha) * old_b[2]),
                    int(alpha * h + (1 - alpha) * old_b[3]),
                ]
                best_trk['history'].append((cls_name, score, det_area))
                best_trk['age'] = 0
                best_trk['matched_this_frame'] = True
            else:
                self.tracks.append({
                    'box': det_box,
                    'history': deque([(cls_name, score, det_area)], maxlen=self.history_size),
                    'age': 0,
                    'matched_this_frame': True,
                })

        active_tracks = []
        confirmed_signs = []
        for trk in self.tracks:
            if trk['age'] <= self.max_age:
                active_tracks.append(trk)
                if len(trk['history']) >= self.min_hits:
                    weighted_scores = {}
                    raw_score_sum = {}
                    counts = {}
                    for idx_h, item in enumerate(trk['history']):
                        c, s, area = item
                        recency_w = 1.0 + 0.15 * idx_h
                        size_w = min(4.0, (area / 625.0) ** 0.7)
                        vote_w = s * recency_w * size_w
                        weighted_scores[c] = weighted_scores.get(c, 0.0) + vote_w
                        raw_score_sum[c] = raw_score_sum.get(c, 0.0) + s
                        counts[c] = counts.get(c, 0) + 1

                    best_cls = max(weighted_scores, key=weighted_scores.get)
                    avg_score = raw_score_sum[best_cls] / counts[best_cls]
                    bx, by, bw, bh = trk['box']
                    confirmed_signs.append([best_cls, bx, by, bw, bh, float(avg_score)])

        self.tracks = active_tracks
        return confirmed_signs


def detect_traffic_signs(img, model=None, draw=None, tracker=None, conf_thresh=0.50, return_scores=False, return_debug=False):
    if img is None or img.size == 0:
        if return_debug:
            return [], {}
        return []

    if model is None:
        model = load_sign_model()

    preview_thr = max(0.25, float(conf_thresh) * 0.45) if return_debug else float(conf_thresh)
    raw_detections, preview_detections, letterbox_img = model.predict(
        img,
        conf_thresh=float(conf_thresh),
        preview_thresh=preview_thr,
        iou_thresh=0.45,
    )

    final_detections = tracker.update(raw_detections) if tracker is not None else raw_detections

    if draw is not None:
        for cls_name, x, y, w, h, score in final_detections:
            info = SIGN_INFO.get(cls_name, {'vn': cls_name, 'color': (0, 255, 0)})
            text = f"{info['vn']} ({score:.2f})"
            cv2.rectangle(draw, (x, y), (x + w, y + h), info['color'], 3, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            tx = max(4, min(x, draw.shape[1] - tw - 10))
            ty = max(y - 6, th + 8)
            cv2.rectangle(draw, (tx, ty - th - 6), (tx + tw + 8, ty + 2), (25, 25, 25), -1)
            cv2.putText(draw, text, (tx + 4, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

    out_dets = final_detections if return_scores else [[d[0], d[1], d[2], d[3], d[4]] for d in final_detections]
    if return_debug:
        gray_mean = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        light_mode = "DARK" if gray_mean < 75.0 else ("BRIGHT" if gray_mean > 175.0 else "NORMAL")
        im_h, im_w = img.shape[:2]
        yolo_vis = img.copy()
        previews = []
        cand_boxes = []
        for cls_name, x, y, w, h, sc in preview_detections:
            cand_boxes.append([int(x), int(y), int(w), int(h)])
            is_ok = bool(sc >= conf_thresh)
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(im_w, int(x + w)), min(im_h, int(y + h))
            crop_bgr = img[y0:y1, x0:x1].copy() if (x1 > x0 and y1 > y0) else None
            previews.append({
                "box": [int(x), int(y), int(w), int(h)],
                "crop_bgr": crop_bgr,
                "cls_name": cls_name,
                "score": float(sc),
                "accepted": is_ok,
            })
            info = SIGN_INFO.get(cls_name, {'vn': cls_name, 'color': (0, 255, 130)})
            b_col = info['color'] if is_ok else (0, 155, 255)
            cv2.rectangle(yolo_vis, (x0, y0), (x1, y1), b_col, 2, cv2.LINE_AA)
            cv2.putText(yolo_vis, f"{info['vn']} {sc:.0%}", (x0, max(14, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, b_col, 1, cv2.LINE_AA)

        debug_info = {
            "backend": "YOLO11n",
            "input_size": (model.input_w, model.input_h),
            "light_mode": light_mode,
            "letterbox_img": letterbox_img,
            "yolo_vis": yolo_vis,
            "candidate_boxes": cand_boxes,
            "raw_detections": raw_detections,
            "yolo_previews": previews,
        }
        return out_dets, debug_info
    return out_dets
