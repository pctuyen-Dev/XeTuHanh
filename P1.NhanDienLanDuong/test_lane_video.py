

import argparse
import os
import sys
import time

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from LaneDetectionModule import LaneDetector, PRESET_ROBOT_TOY  # noqa: E402
import utils  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Test nhan dien lan duong AI tren video")
    parser.add_argument("--video", type=str, default=os.path.join(ROOT_DIR, "xetuhanh.mp4"))
    parser.add_argument("--save", type=str, default="", help="Luu video ket qua neu can (vd: out_test.mp4)")
    parser.add_argument("--no-gui", action="store_true", help="Chay khong mo cua so (chi in thong so)")
    parser.add_argument("--max-frames", type=int, default=0, help="Gioi han so frame (0 = het video)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Khong tim thay video: {args.video}")

    wT, hT = 480, 240
    roi_cfg = utils.load_roi_config(wT=wT, hT=hT)
    pts = utils.roi_cfg_to_points(roi_cfg, wT=wT, hT=hT)
    hsv_range = PRESET_ROBOT_TOY["hsv_range"]

    det_ai = LaneDetector(avg_window=6, use_ai=True)
    det_cv = LaneDetector(avg_window=6, use_ai=False)
    use_ai_mode = det_ai._ai_segmenter is not None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {args.video}")

    writer = None
    if args.save:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (wT * 2, hT))

    win = "Test Lan Duong AI (Space: Pause | m: Doi AI/CV | s: Luu ROI | r: Lai tu dau | q: Thoat)"
    if not args.no_gui:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, wT * 2, hT * 2)
        cv2.createTrackbar("W_Top", win, roi_cfg["w_top"], wT // 2 - 10, lambda _v: None)
        cv2.createTrackbar("H_Top", win, roi_cfg["h_top"], hT - 30, lambda _v: None)
        cv2.createTrackbar("W_Bot", win, roi_cfg["w_bot"], wT // 2 - 10, lambda _v: None)
        cv2.createTrackbar("H_Bot", win, roi_cfg["h_bot"], hT - 1, lambda _v: None)

    paused = False
    idx = 0
    last_panel = None

    while True:
        if not args.no_gui:
            roi_cfg = utils.clamp_roi_cfg({
                "w_top": cv2.getTrackbarPos("W_Top", win),
                "h_top": cv2.getTrackbarPos("H_Top", win),
                "w_bot": cv2.getTrackbarPos("W_Bot", win),
                "h_bot": cv2.getTrackbarPos("H_Bot", win),
            }, wT=wT, hT=hT)
            pts = utils.roi_cfg_to_points(roi_cfg, wT=wT, hT=hT)

        if not paused:
            ok, frame = cap.read()
            if not ok:
                if args.no_gui or args.max_frames > 0:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            img = cv2.resize(frame, (wT, hT), interpolation=cv2.INTER_AREA)
            det = det_ai if (use_ai_mode and det_ai._ai_segmenter is not None) else det_cv

            t0 = time.perf_counter()
            curve, _, img_thres, _, _, img_result = det.process_frame(
                img, pts, hsv_range=hsv_range, use_dual_edge=True, profile="robot",
                fresh_observation=True, restrict_roi=True
            )
            ms = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / max(ms, 0.1)

            mask_bgr = cv2.cvtColor(img_thres, cv2.COLOR_GRAY2BGR)
            pts_roi = np.asarray(pts, dtype=np.int32)
            poly_roi = np.array([pts_roi[0], pts_roi[1], pts_roi[3], pts_roi[2]], dtype=np.int32)
            cv2.polylines(mask_bgr, [poly_roi], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            mode_tag = "AI: MobileNetV2-UNet" if det is det_ai and det_ai._ai_segmenter is not None else "CV: OpenCV"
            cv2.putText(mask_bgr, f"{mode_tag} | {ms:.1f}ms ({fps:.0f} FPS)", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(mask_bgr, f"Curve: {curve:+.3f} | Servo: {det.servo_angle_deg} deg", (8, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            last_panel = np.hstack([img_result, mask_bgr])
            if writer is not None:
                writer.write(last_panel)

            idx += 1
            if args.no_gui and idx % 30 == 1:
                print(f"[{idx}] {mode_tag} | {ms:.1f}ms | curve={curve:+.3f} | servo={det.servo_angle_deg}")
            if args.max_frames > 0 and idx >= args.max_frames:
                break

        if not args.no_gui and last_panel is not None:
            cv2.imshow(win, last_panel)
            key = cv2.waitKey(25 if not paused else 40) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("m"):
                use_ai_mode = not use_ai_mode
            elif key == ord("s"):
                saved_cfg = utils.save_roi_config(roi_cfg, wT=wT, hT=hT)
                print(f"[ROI SAVED] {saved_cfg} -> {utils.ROI_CONFIG_PATH}")
            elif key == ord("r"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                det_ai.reset()
                det_cv.reset()

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_gui:
        cv2.destroyAllWindows()
    print(f"[DONE] Da xu ly {idx} frame.")


if __name__ == "__main__":
    main()
