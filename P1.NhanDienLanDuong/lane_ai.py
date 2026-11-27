"""
lane_ai.py - Suy luan mask lan duong bang MobileNetV2-UNet (ONNX Runtime) cho Raspberry Pi 5.
Chay song song nhe nhang voi YOLO11n bien bao (P2) va tra ve mask nhi phan tuong thich voi LaneDetector.
"""

import os
import cv2
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ort = None
    ORT_AVAILABLE = False

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lane_unet.onnx")
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_KERNEL_BRIDGE = np.ones((3, 3), dtype=np.uint8)


class AILaneSegmenter:
    """Bo tach mask vach lan duong bang ONNX Runtime (MobileNetV2-UNet)."""

    def __init__(self, model_path=None, prob_thresh=0.5, num_threads=2):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.prob_thresh = float(prob_thresh)
        self.num_threads = int(num_threads)
        self.session = None
        self.input_name = None
        self.input_h = 160
        self.input_w = 320
        self._load()

    def _load(self):
        if not ORT_AVAILABLE or not os.path.exists(self.model_path):
            return
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, self.num_threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if len(inp.shape) == 4:
            if isinstance(inp.shape[2], int) and inp.shape[2] > 0:
                self.input_h = int(inp.shape[2])
            if isinstance(inp.shape[3], int) and inp.shape[3] > 0:
                self.input_w = int(inp.shape[3])

    @property
    def ready(self):
        return self.session is not None

    def predict_mask(self, img, roi_points=None, restrict_roi=False, prev_mode=None):
        """
        Tra ve (out_mask, aux_dict) giong het dinh dang cua utils.thresholding(..., return_aux=True).
        """
        if not self.ready:
            raise RuntimeError("AILaneSegmenter chua san sang (thieu onnxruntime hoac file .onnx).")

        import utils

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        x = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None, :, :, :], dtype=np.float32)

        logits = self.session.run(None, {self.input_name: blob})[0][0, 0]
        # Logit threshold tuong duong sigmoid(logits) >= prob_thresh (tranh tinh exp toan anh)
        p = min(max(self.prob_thresh, 1e-4), 1.0 - 1e-4)
        logit_th = float(np.log(p / (1.0 - p)))
        small_mask = np.where(logits >= logit_th, 255, 0).astype(np.uint8)
        mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_BRIDGE)

        if restrict_roi and roi_points is not None:
            roi_mask = utils.build_roi_polygon_mask((h, w), roi_points, expand_x_ratio=0.04, min_top_ratio=0.05)
            mask = cv2.bitwise_and(mask, roi_mask)
        else:
            mask[:3] = 0
            mask[int(h * 0.995):] = 0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        light_info = utils.analyze_lighting(img, roi_points=roi_points, prev_mode=prev_mode)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return np.zeros_like(mask), {
                "lighting_info": light_info, "gray": gray, "cc_labels": None, "cc_stats": None
            }

        min_area = max(12, int(w * h * 0.00008))
        st = stats[1:]
        keep = st[:, cv2.CC_STAT_AREA] >= min_area
        lut = np.zeros(count, dtype=np.uint8)
        lut[1:][keep] = 255
        out_mask = lut[labels]
        return out_mask, {
            "lighting_info": light_info, "gray": gray, "cc_labels": labels, "cc_stats": stats
        }
