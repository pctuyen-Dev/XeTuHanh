"""
cameraModule.py - Quan ly Camera da backend cho Raspberry Pi 5 (CSI) & PC (USB).

Luon khoi dong o goc nhin rong nhat chuan (100% Full-FOV):
  1. Picamera2 (CSI): chon sensor mode Full-FOV (binned) + main stream cung ty le khung
     cam bien + ScalerCrop = toan bo cam bien (tranh bi libcamera tu crop hai ben).
  2. GStreamer (libcamerasrc): yeu cau khung 4:3 full-FOV truoc khi resize.
  3. rpicam-vid / libcamera-vid MJPEG pipe: ep --roi 0,0,1,1 de giu toan bo cam bien.
  4. USB / V4L2 / DirectShow (Webcam): uu tien MJPG 16:9 (1280x720 -> 1920x1080 -> 800x448 -> 640x360)
     va tat Zoom/Pan/Tilt de lay tron ven goc nhin rong nhat cua cam bien, fallback 640x480 (4:3).
"""

import os
import shutil
import subprocess
import threading
import time
import cv2
import numpy as np

frameWidth = 480
frameHeight = 240


WIDE_FOV_WEBCAM_RESOLUTIONS = (
    (1280, 720),
    (1920, 1080),
    (800, 448),
    (640, 360),
    (640, 480),
)


def configure_wide_fov_webcam(cap, target_fps=30):
    """
    Cau hinh Webcam USB/PC de lay goc nhin rong nhat (Full-FOV):
      - Dat buffer = 1 de giam do tre frame.
      - Reset Zoom/Pan/Tilt ve muc rong nhat / chinh giua.
      - Bat MJPG @ 30fps de ho tro do phan giai 16:9 Full-FOV khong bi nghen bang thong USB.
      - Thu cac do phan giai 16:9 Full-FOV (1280x720...) truoc khi fallback ve 4:3 (640x480).
    """
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    for prop, val in (
        (cv2.CAP_PROP_ZOOM, 0),
        (cv2.CAP_PROP_PAN, 0),
        (cv2.CAP_PROP_TILT, 0),
    ):
        try:
            cap.set(prop, val)
        except Exception:
            pass

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, target_fps)
    except Exception:
        pass

    for rw, rh in WIDE_FOV_WEBCAM_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
        aw = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        ah = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if abs(aw - rw) <= 16 and abs(ah - rh) <= 16:
            return aw, ah

    return (
        int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )


def apply_fov_width(frame, fov_pct=100):

    if frame is None or fov_pct >= 100:
        return frame
    h, w = frame.shape[:2]
    scale = max(0.4, min(1.0, float(fov_pct) / 100.0))
    cw = max(2, int(round(w * scale)))
    ch = max(2, int(round(h * scale)))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return frame[y0:y0 + ch, x0:x0 + cw]


def _capture_array_with_timeout(cam, timeout=1.8):
    """Doc frame tu Picamera2 co gioi han thoi gian (tranh treo vo han khi cap CSI long)."""
    result = [None]
    err = [None]

    def _worker():
        try:
            result[0] = cam.capture_array()
        except Exception as exc:
            err[0] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"Picamera2 capture_array timed out after {timeout}s (Kiem tra cap CSI Camera)")
    if err[0] is not None:
        raise err[0]
    return result[0]


def _select_full_fov_mode(cam):
    props = getattr(cam, "camera_properties", {}) or {}
    full_res = getattr(cam, "sensor_resolution", (2592, 1944))
    max_crop = props.get("ScalerCropMaximum")
    if max_crop and len(max_crop) == 4 and max_crop[2] > 0 and max_crop[3] > 0:
        fallback_limits = tuple(int(v) for v in max_crop)
    else:
        fallback_limits = (0, 0, int(full_res[0]), int(full_res[1]))

    modes = getattr(cam, "sensor_modes", None) or []
    best_mode = None
    best_area = -1
    best_pixels = 10**12
    for mode in modes:
        limits = mode.get("crop_limits", fallback_limits)
        size = mode.get("size", (0, 0))
        area = int(limits[2]) * int(limits[3])
        pixels = int(size[0]) * int(size[1])
        if area > best_area or (area == best_area and 0 < pixels < best_pixels):
            best_area = area
            best_pixels = pixels
            best_mode = mode

    if best_mode is not None:
        return best_mode.get("size"), tuple(int(v) for v in best_mode.get("crop_limits", fallback_limits))
    return (int(full_res[0]), int(full_res[1])), fallback_limits


def _full_fov_main_size(crop_limits, target_w=640):
    """Tinh kich thuoc main stream giu dung ty le cam bien de libcamera khong tu crop."""
    cw, ch = max(1, int(crop_limits[2])), max(1, int(crop_limits[3]))
    main_w = max(640, int(target_w) & ~1)
    main_h = max(64, (int(round(main_w * ch / cw)) + 1) & ~1)
    return (main_w, main_h)


class CameraManager:
    """Quan ly vong doi Camera va da backend fallback."""

    def __init__(self):
        self._active_backend = None   # "picamera2" | "opencv" | "rpicam_pipe"
        self._cap = None
        self._picam2 = None
        self._pipe_proc = None
        self._pipe_buf = b""
        self._fail_count = 0
        self._usb_idx = None
        self._io_lock = threading.Lock()

    def close(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None

        if self._pipe_proc is not None:
            try:
                self._pipe_proc.terminate()
                self._pipe_proc.wait(timeout=1.0)
            except Exception:
                pass
            self._pipe_proc = None

        self._pipe_buf = b""
        self._active_backend = None
        self._usb_idx = None

    def _try_picamera2(self):
        cam = None
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            raw_size, crop_limits = _select_full_fov_mode(cam)
            main_size = _full_fov_main_size(crop_limits, target_w=640)
            full_crop = (
                int(crop_limits[0]),
                int(crop_limits[1]),
                int(crop_limits[2]) & ~1,
                int(crop_limits[3]) & ~1,
            )

            config = None
            for extra in (
                {"sensor": {"output_size": raw_size}},
                {"raw": {"size": raw_size}},
                {},
            ):
                try:
                    config = cam.create_preview_configuration(
                        main={"size": main_size, "format": "RGB888"},
                        controls={"ScalerCrop": full_crop},
                        **extra,
                    )
                    break
                except Exception:
                    config = None

            if config is None:
                return False

            cam.configure(config)
            cam.start()
            try:
                cam.set_controls({"ScalerCrop": full_crop})
            except Exception:
                pass
            time.sleep(0.35)

            frame = _capture_array_with_timeout(cam, timeout=2.0)
            if frame is not None and frame.size > 0:
                self._picam2 = cam
                self._active_backend = "picamera2"
                print(
                    f"[CAMERA OK] Picamera2 Full-FOV! Sensor={raw_size}, "
                    f"Main={main_size}, ScalerCrop={full_crop}"
                )
                return True
        except Exception as exc:
            print(f"[CAMERA] Picamera2: {exc}")
        finally:
            if cam is not None and cam is not self._picam2:
                try:
                    cam.stop()
                except Exception:
                    pass
                try:
                    cam.close()
                except Exception as exc:
                    print(f"[CAMERA] Loi dong Picamera2: {exc}")
        return False

    def _try_gstreamer(self, w, h):
        cap_w, cap_h = (640, 480) if (w * 3 > h * 4) else (w, h)
        pipeline = (
            "libcamerasrc ! "
            f"video/x-raw,width={cap_w},height={cap_h},format=NV12,framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true"
        )
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self._cap = cap
                        self._active_backend = "opencv"
                        print("[CAMERA OK] Da ket noi Pi Camera qua GStreamer!")
                        return True
                    time.sleep(0.1)
                cap.release()
        except Exception:
            pass
        return False

    def _read_rpicam_pipe_frame(self):
        if self._pipe_proc is None or self._pipe_proc.stdout is None:
            return None
        while True:
            start = self._pipe_buf.find(b"\xff\xd8")
            end = self._pipe_buf.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = self._pipe_buf[start:end + 2]
                self._pipe_buf = self._pipe_buf[end + 2:]
                arr = np.frombuffer(jpg, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            chunk = self._pipe_proc.stdout.read(4096)
            if not chunk:
                return None
            self._pipe_buf += chunk
            if len(self._pipe_buf) > 512000:
                self._pipe_buf = self._pipe_buf[-128000:]

    def _try_rpicam_pipe(self, w, h):
        cmd_bin = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
        if not cmd_bin:
            return False

        cap_w, cap_h = (640, 480) if (w * 3 > h * 4) else (w, h)
        cmd = [
            cmd_bin,
            "-t", "0",
            "--nopreview",
            "--roi", "0,0,1,1",
            "--width", str(cap_w),
            "--height", str(cap_h),
            "--framerate", "30",
            "--codec", "mjpeg",
            "-o", "-",
        ]
        try:
            self._pipe_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=65536)
            self._pipe_buf = b""
            deadline = time.time() + 2.0
            while time.time() < deadline:
                frame = self._read_rpicam_pipe_frame()
                if frame is not None and frame.size > 0:
                    self._active_backend = "rpicam_pipe"
                    print(f"[CAMERA OK] Da khoi dong Pi Camera Full-FOV qua {cmd_bin} MJPEG!")
                    return True
            self.close()
        except Exception:
            self.close()
        return False

    def _open_usb_capture(self, idx):
        if os.name == "nt":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            return None, None
        configure_wide_fov_webcam(cap)
        for _ in range(6):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return cap, frame
            time.sleep(0.08)
        cap.release()
        return None, None

    def get_label(self):
        if self._active_backend == "opencv" and self._usb_idx is not None:
            return f"USB-{self._usb_idx}"
        if self._active_backend == "picamera2":
            return "CSI-PiCam"
        if self._active_backend == "rpicam_pipe":
            return "CSI-Pipe"
        return self._active_backend or "NONE"

    def get_usb_index(self):
        return self._usb_idx if self._active_backend == "opencv" else None

    def _try_usb_or_v4l2(self, w, h, preferred_idx=None, exclude_indices=None):
        excluded = set(exclude_indices or ())
        indices = [i for i in (0, 1, 2, 3) if i not in excluded]
        if preferred_idx is not None and preferred_idx in indices:
            indices = [preferred_idx] + [i for i in indices if i != preferred_idx]
        elif preferred_idx is not None and preferred_idx not in excluded:
            indices = [preferred_idx] + indices
        for idx in indices:
            try:
                cap, frame = self._open_usb_capture(idx)
                if cap is not None:
                    self._cap = cap
                    self._active_backend = "opencv"
                    self._usb_idx = idx
                    print(f"[CAMERA OK] Da ket noi Webcam Full-FOV tai index {idx} ({frame.shape[1]}x{frame.shape[0]})!")
                    return True
            except Exception:
                continue
        return False

    def boot(self, size=(frameWidth, frameHeight), max_retries=5, retry_delay=1.5,
             preferred_idx=None, exclude_indices=None, allow_csi=True):
        w, h = size[0], size[1]
        with self._io_lock:
            self.close()

            for attempt in range(1, max_retries + 1):
                print(f"[CAMERA] Dang do tim va khoi dong Camera (Lan thu {attempt}/{max_retries})...")
                if preferred_idx is None and allow_csi:
                    if self._try_picamera2():
                        return True
                    if self._try_gstreamer(w, h):
                        return True
                    if self._try_rpicam_pipe(w, h):
                        return True
                if self._try_usb_or_v4l2(w, h, preferred_idx=preferred_idx, exclude_indices=exclude_indices):
                    return True

                if attempt < max_retries:
                    print(f"[WARNING] Chua tim thay Camera, doi {retry_delay}s de thu lai...")
                    time.sleep(retry_delay)

            print("[ERROR] Khong tim thay bat ky Camera nao sau khi thu tat ca che do!")
            return False

    def switch_next_usb_camera(self, size=(frameWidth, frameHeight), exclude_indices=None):
        """Chuyen sang camera USB tiep theo (0 -> 1 -> 2 -> 3) khi dang chay."""
        excluded = set(exclude_indices or ())
        with self._io_lock:
            cur = self._usb_idx if self._usb_idx is not None else -1
            candidates = [(cur + offset) % 4 for offset in range(1, 5)]
            for idx in candidates:
                if idx == self._usb_idx or idx in excluded:
                    continue
                cap, frame = self._open_usb_capture(idx)
                if cap is not None:
                    self.close()
                    self._cap = cap
                    self._active_backend = "opencv"
                    self._usb_idx = idx
                    self._fail_count = 0
                    print(f"[CAMERA SWITCH] Da chuyen sang Webcam index {idx} ({frame.shape[1]}x{frame.shape[0]})!")
                    return idx
        print("[CAMERA] Khong tim thay camera nao khac de chuyen.")
        return self._usb_idx

    def read_frame(self, display=False, size=(frameWidth, frameHeight), fov_pct=100):
        if self._active_backend is None:
            if not self.boot(size=size, max_retries=3, retry_delay=1.0):
                return None

        img = None
        with self._io_lock:
            try:
                if self._active_backend == "picamera2" and self._picam2 is not None:
                    frame = _capture_array_with_timeout(self._picam2, timeout=1.5)
                    if frame is not None and frame.size > 0:
                        img = frame
                elif self._active_backend == "opencv" and self._cap is not None:
                    ret, frame = self._cap.read()
                    if ret and frame is not None and frame.size > 0:
                        img = frame
                elif self._active_backend == "rpicam_pipe":
                    img = self._read_rpicam_pipe_frame()
            except Exception as exc:
                print(f"[WARNING] Loi doc frame Camera ({exc}), dang tu dong khoi dong lai...")
                img = None

        if img is None:
            self._fail_count += 1
            if self._fail_count >= 5:
                print("[CAMERA] Mat tin hieu Camera! Dang tu dong ket noi lai...")
                self._fail_count = 0
                self.boot(size=size, max_retries=2, retry_delay=0.5)
            return None

        self._fail_count = 0
        img = apply_fov_width(img, fov_pct=fov_pct)
        if img.shape[1] != size[0] or img.shape[0] != size[1]:
            img = cv2.resize(img, (size[0], size[1]))

        if display:
            try:
                cv2.imshow("PiCamera", img)
                cv2.waitKey(1)
            except Exception:
                pass

        return img


_default_manager = CameraManager()


def boot_camera(size=(frameWidth, frameHeight), max_retries=5, retry_delay=1.5, preferred_idx=None):
    return _default_manager.boot(size=size, max_retries=max_retries, retry_delay=retry_delay, preferred_idx=preferred_idx)


def switch_camera(size=(frameWidth, frameHeight)):
    return _default_manager.switch_next_usb_camera(size=size)


def get_active_camera_label():
    if _default_manager._active_backend == "opencv" and _default_manager._usb_idx is not None:
        return f"USB-{_default_manager._usb_idx}"
    return _default_manager._active_backend or "NONE"


def getImg(display=False, size=(frameWidth, frameHeight), fov_pct=100):
    return _default_manager.read_frame(display=display, size=size, fov_pct=fov_pct)


def release_camera():
    _default_manager.close()


class DualCameraRig:
    """
    Quan ly dong thoi 2 Camera doc lap cho Xe Tu Hanh:
      - lane_cam: Camera 1 (nhin xuong mat duong -> Bam lan duong)
      - sign_cam: Camera 2 (nhin thang phia truoc -> Nhan dien bien bao YOLO11n)
    Tu dong fallback che do shared_mode neu may chi cam 1 camera, va ho tro hoan doi 2 cam tuc thi (swap).
    """

    def __init__(self):
        self.lane_cam = CameraManager()
        self.sign_cam = CameraManager()
        self.shared_mode = False
        self._rig_lock = threading.Lock()
        self._lane_size = (frameWidth, frameHeight)
        self._sign_size = (480, 320)

    def boot(self, lane_size=(480, 240), sign_size=(480, 320),
             lane_idx=None, sign_idx=None, max_retries=5, retry_delay=1.5):
        self._lane_size = tuple(lane_size)
        self._sign_size = tuple(sign_size)

        ok_lane = self.lane_cam.boot(
            size=self._lane_size,
            max_retries=max_retries,
            retry_delay=retry_delay,
            preferred_idx=lane_idx,
            allow_csi=(lane_idx is None),
        )
        if not ok_lane:
            return False

        lane_usb = self.lane_cam.get_usb_index()
        lane_is_csi = self.lane_cam._active_backend in ("picamera2", "rpicam_pipe")
        exclude = [lane_usb] if lane_usb is not None else []

        # Neu nguoi dung ep --sign-cam trung voi --lane-cam -> dung chung 1 cam
        if sign_idx is not None and lane_usb is not None and sign_idx == lane_usb:
            self.shared_mode = True
            print(f"[DUAL-CAM] Cam Bien bao dung chung Cam Lan ({self.lane_cam.get_label()}).")
            return True

        print("[DUAL-CAM] Dang do tim Camera thu 2 cho Nhan dien Bien bao (YOLO11n)...")
        ok_sign = self.sign_cam.boot(
            size=self._sign_size,
            max_retries=1,
            retry_delay=0.3,
            preferred_idx=sign_idx,
            exclude_indices=exclude,
            allow_csi=(not lane_is_csi and sign_idx is None),
        )
        if ok_sign:
            self.shared_mode = False
            print(
                f"[DUAL-CAM OK] Da ket noi 2 Camera doc lap: "
                f"Lan={self.lane_cam.get_label()} | Bien={self.sign_cam.get_label()}"
            )
        else:
            self.shared_mode = True
            print(
                f"[DUAL-CAM] Chi phat hien 1 Camera ({self.lane_cam.get_label()}). "
                "Tam dung chung cho Bien bao (bam [V] sau khi cam them cam thu 2)."
            )
        return True

    def read_lane_frame(self, fov_pct=100):
        with self._rig_lock:
            cam = self.lane_cam
            size = self._lane_size
        return cam.read_frame(display=False, size=size, fov_pct=fov_pct)

    def read_sign_frame(self, fov_pct=100):
        with self._rig_lock:
            if self.shared_mode:
                return None
            cam = self.sign_cam
            size = self._sign_size
        return cam.read_frame(display=False, size=size, fov_pct=fov_pct)

    def switch_lane_camera(self):
        with self._rig_lock:
            exclude = []
            if not self.shared_mode:
                s_idx = self.sign_cam.get_usb_index()
                if s_idx is not None:
                    exclude.append(s_idx)
            self.lane_cam.switch_next_usb_camera(size=self._lane_size, exclude_indices=exclude)

    def switch_sign_camera(self):
        with self._rig_lock:
            l_idx = self.lane_cam.get_usb_index()
            exclude = [l_idx] if l_idx is not None else []
            if self.shared_mode:
                ok = self.sign_cam.boot(
                    size=self._sign_size,
                    max_retries=1,
                    retry_delay=0.2,
                    exclude_indices=exclude,
                    allow_csi=False,
                )
                if ok:
                    self.shared_mode = False
                    print(f"[DUAL-CAM] Da tach rieng Cam Bien bao sang {self.sign_cam.get_label()}!")
                else:
                    print("[DUAL-CAM] Khong tim thay camera thu 2. Van dang dung chung Cam Lan.")
            else:
                prev_idx = self.sign_cam.get_usb_index()
                new_idx = self.sign_cam.switch_next_usb_camera(size=self._sign_size, exclude_indices=exclude)
                if new_idx == prev_idx:
                    # Cho phep quay lai che do dung chung neu khong con cam nao khac
                    print("[DUAL-CAM] Khong con camera khac cho Bien bao.")

    def swap_cameras(self):
        """Hoan doi tuc thi Camera Lan <-> Camera Bien bao (khong can dong/mo lai USB)."""
        with self._rig_lock:
            if self.shared_mode:
                print("[DUAL-CAM] Chi co 1 camera nen khong the hoan doi.")
                return False
            self.lane_cam, self.sign_cam = self.sign_cam, self.lane_cam
            print(
                f"[DUAL-CAM SWAP] Da hoan doi 2 Camera: "
                f"Lan={self.lane_cam.get_label()} | Bien={self.sign_cam.get_label()}"
            )
            return True

    def get_lane_label(self):
        with self._rig_lock:
            return self.lane_cam.get_label()

    def get_sign_label(self):
        with self._rig_lock:
            if self.shared_mode:
                return f"CHUNG({self.lane_cam.get_label()})"
            return self.sign_cam.get_label()

    def release_all(self):
        with self._rig_lock:
            self.lane_cam.close()
            self.sign_cam.close()


if __name__ == "__main__":
    if boot_camera():
        while True:
            frame = getImg(display=True)
            if frame is None:
                break
