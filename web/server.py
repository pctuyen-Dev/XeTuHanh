from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import unquote

WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import BlackBoxModule

BLACKBOX_DIR = ROOT_DIR / "blackbox_logs"
ROI_CONFIG_PATH = ROOT_DIR / "P1.NhanDienLanDuong" / "roi_config.json"
DEFAULT_ROI = {"w_top": 123, "h_top": 105, "w_bot": 5, "h_bot": 222}
PORT = 8080
START_TIME = time.time()
CAR_HEARTBEAT_TIMEOUT_S = 1.5


def _clamp_roi(cfg):
    w_top = max(0, min(230, int(cfg.get("w_top", DEFAULT_ROI["w_top"]))))
    h_top = max(0, min(210, int(cfg.get("h_top", DEFAULT_ROI["h_top"]))))
    w_bot = max(0, min(230, int(cfg.get("w_bot", DEFAULT_ROI["w_bot"]))))
    h_bot = max(h_top + 20, min(239, int(cfg.get("h_bot", DEFAULT_ROI["h_bot"]))))
    return {"w_top": w_top, "h_top": h_top, "w_bot": w_bot, "h_bot": h_bot}


def _load_roi():
    cfg = dict(DEFAULT_ROI)
    if ROI_CONFIG_PATH.exists():
        try:
            data = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update(data)
        except Exception:
            pass
    return _clamp_roi(cfg)


def _save_roi(cfg):
    clean = _clamp_roi(cfg)
    try:
        ROI_CONFIG_PATH.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    except Exception:
        pass
    return clean


class WebBridgeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_car_heartbeat = 0.0

        # Lenh tu Web -> Xe
        self.car_running = False
        self.drive_mode = "AUTO"       # "AUTO" | "MANUAL"
        self.manual_speed = 0          # -100 .. 100
        self.manual_steer = 0.0        # -1.0 .. +1.0
        self.max_speed = 100           # 20 .. 100
        self.fov_pct = 100             # 40 .. 100
        self.roi = _load_roi()
        self.pending_actions = []      # ["switch_cam1", "switch_cam2", "swap_cams"]
        self.blackbox_auto = True
        self.blackbox_manual = False

        # Telemetry tu Xe -> Web
        self.esp32_connected = False
        self.speed = 0
        self.steer = 0.0
        self.servo_deg = 90
        self.cmd = "STOP"
        self.packet = "<0,+0.00,STOP>"
        self.cam1_src = "CSI"
        self.cam2_src = "USB-0"
        self.signs = []
        self.blackbox = {
            "recording": False,
            "session_id": None,
            "duration_s": 0.0,
            "ticks": 0,
            "frames": 0,
            "anomalies": 0,
        }

        # Frame JPEG moi nhat cua 2 Camera
        self.cam1_jpeg = None
        self.cam2_jpeg = None
        self.cam1_seq = 0
        self.cam2_seq = 0
        self.cam_cond = threading.Condition()

    def is_car_connected(self):
        with self.lock:
            return (time.time() - self.last_car_heartbeat) <= CAR_HEARTBEAT_TIMEOUT_S

    def get_control_snapshot(self):
        with self.lock:
            actions = list(self.pending_actions)
            self.pending_actions.clear()
            return {
                "car_running": self.car_running,
                "drive_mode": self.drive_mode,
                "manual_speed": self.manual_speed,
                "manual_steer": self.manual_steer,
                "max_speed": self.max_speed,
                "fov_pct": self.fov_pct,
                "roi": dict(self.roi),
                "actions": actions,
                "blackbox_auto": self.blackbox_auto,
                "blackbox_manual": self.blackbox_manual,
            }

    def get_blackbox_status(self):
        now = time.time()
        with self.lock:
            car_connected = (now - self.last_car_heartbeat) <= CAR_HEARTBEAT_TIMEOUT_S
            bb = dict(self.blackbox) if car_connected else {
                "recording": False,
                "session_id": None,
                "duration_s": 0.0,
                "ticks": 0,
                "frames": 0,
                "anomalies": 0,
            }
            bb["blackbox_auto"] = self.blackbox_auto
            bb["blackbox_manual"] = self.blackbox_manual
            return bb

    def get_public_status(self):
        now = time.time()
        with self.lock:
            car_connected = (now - self.last_car_heartbeat) <= CAR_HEARTBEAT_TIMEOUT_S
            return {
                "server_online": True,
                "car_connected": car_connected,
                "esp32_connected": self.esp32_connected if car_connected else False,
                "car_running": self.car_running,
                "drive_mode": self.drive_mode,
                "max_speed": self.max_speed,
                "fov_pct": self.fov_pct,
                "roi": dict(self.roi),
                "speed": self.speed if car_connected else 0,
                "steer": self.steer if car_connected else 0.0,
                "servo_deg": self.servo_deg if car_connected else 90,
                "cmd": self.cmd if car_connected else "STOP",
                "packet": self.packet if car_connected else "<0,+0.00,STOP>",
                "cam1_src": self.cam1_src,
                "cam2_src": self.cam2_src,
                "signs": list(self.signs) if car_connected else [],
                "blackbox_auto": self.blackbox_auto,
                "blackbox_manual": self.blackbox_manual,
                "blackbox": dict(self.blackbox) if car_connected else {
                    "recording": False,
                    "session_id": None,
                    "duration_s": 0.0,
                    "ticks": 0,
                    "frames": 0,
                    "anomalies": 0,
                },
                "uptime_s": round(now - START_TIME, 1),
            }

    def update_from_web(self, data):
        with self.lock:
            if "car_running" in data:
                self.car_running = bool(data["car_running"])
                if not self.car_running:
                    self.manual_speed = 0
                    self.manual_steer = 0.0
            if "drive_mode" in data and data["drive_mode"] in ("AUTO", "MANUAL"):
                self.drive_mode = data["drive_mode"]
            if "manual_speed" in data:
                self.manual_speed = max(-100, min(100, int(round(float(data["manual_speed"])))))
            if "manual_steer" in data:
                self.manual_steer = max(-1.0, min(1.0, float(data["manual_steer"])))
            if "max_speed" in data:
                self.max_speed = max(20, min(100, int(round(float(data["max_speed"])))))
            if "fov_pct" in data:
                self.fov_pct = max(40, min(100, int(round(float(data["fov_pct"])))))
            if "roi" in data and isinstance(data["roi"], dict):
                merged = dict(self.roi)
                for k in ("w_top", "h_top", "w_bot", "h_bot"):
                    if k in data["roi"]:
                        merged[k] = int(round(float(data["roi"][k])))
                self.roi = _save_roi(merged)
            if "blackbox_auto" in data:
                self.blackbox_auto = bool(data["blackbox_auto"])
            if "blackbox_manual" in data:
                self.blackbox_manual = bool(data["blackbox_manual"])
            action = data.get("action")
            if action in ("switch_cam1", "switch_cam2", "swap_cams"):
                self.pending_actions.append(action)

    def update_from_car(self, telemetry, cam1_jpeg=None, cam2_jpeg=None):
        with self.lock:
            self.last_car_heartbeat = time.time()
            self.esp32_connected = bool(telemetry.get("esp32_connected", False))
            self.speed = int(telemetry.get("speed", 0))
            self.steer = float(telemetry.get("steer", 0.0))
            self.servo_deg = int(telemetry.get("servo_deg", 90))
            self.cmd = str(telemetry.get("cmd", "STOP"))
            self.packet = str(telemetry.get("packet", "<0,+0.00,STOP>"))
            self.cam1_src = str(telemetry.get("cam1_src", self.cam1_src))
            self.cam2_src = str(telemetry.get("cam2_src", self.cam2_src))
            self.signs = list(telemetry.get("signs", []))
            if isinstance(telemetry.get("blackbox"), dict):
                self.blackbox = dict(telemetry["blackbox"])
            if "blackbox_auto" in telemetry:
                self.blackbox_auto = bool(telemetry["blackbox_auto"])
            if "blackbox_manual" in telemetry:
                self.blackbox_manual = bool(telemetry["blackbox_manual"])

        if cam1_jpeg is not None or cam2_jpeg is not None:
            with self.cam_cond:
                if cam1_jpeg is not None:
                    self.cam1_jpeg = cam1_jpeg
                    self.cam1_seq += 1
                if cam2_jpeg is not None:
                    self.cam2_jpeg = cam2_jpeg
                    self.cam2_seq += 1
                self.cam_cond.notify_all()


BRIDGE = WebBridgeState()


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class RobotWebHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, code, content_type, data_bytes, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data_bytes)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data_bytes)

    def _stream_mjpeg(self, cam_idx):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_seq = -1
        try:
            while True:
                with BRIDGE.cam_cond:
                    BRIDGE.cam_cond.wait(timeout=0.5)
                    seq = BRIDGE.cam1_seq if cam_idx == 1 else BRIDGE.cam2_seq
                    jpg = BRIDGE.cam1_jpeg if cam_idx == 1 else BRIDGE.cam2_jpeg
                if jpg is None or seq == last_seq:
                    if not BRIDGE.is_car_connected():
                        break
                    continue
                last_seq = seq
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(header + jpg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            self._send_json(200, BRIDGE.get_public_status())
            return
        if path == "/api/blackbox/sessions":
            self._send_json(200, {
                "ok": True,
                "status": BRIDGE.get_blackbox_status(),
                "sessions": BlackBoxModule.list_sessions(BLACKBOX_DIR),
            })
            return
        if path.startswith("/api/blackbox/session/"):
            rest = unquote(path[len("/api/blackbox/session/"):]).strip("/")
            parts = [p for p in rest.split("/") if p]
            if len(parts) == 1:
                sid = parts[0]
                try:
                    data = BlackBoxModule.load_session(sid, BLACKBOX_DIR)
                    self._send_json(200, {"ok": True, **data})
                except FileNotFoundError:
                    self._send_json(404, {"ok": False, "error": "Khong tim thay phien"})
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return
            if len(parts) == 2 and parts[1] == "export":
                sid = parts[0]
                try:
                    zbytes = BlackBoxModule.build_session_zip_bytes(sid, BLACKBOX_DIR)
                    self._send_bytes(
                        200, "application/zip", zbytes,
                        extra_headers={"Content-Disposition": f'attachment; filename="{sid}.zip"'},
                    )
                except FileNotFoundError:
                    self._send_json(404, {"ok": False, "error": "Khong tim thay phien"})
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return
            if len(parts) == 3 and parts[1] == "frame":
                sid, fname = parts[0], parts[2]
                if not (BlackBoxModule.is_safe_name(sid) and BlackBoxModule.is_safe_name(fname)):
                    self._send_json(400, {"ok": False, "error": "Invalid path"})
                    return
                fpath = BLACKBOX_DIR / sid / "frames" / fname
                if not fpath.is_file():
                    self._send_json(404, {"ok": False, "error": "Frame not found"})
                    return
                self._send_bytes(200, "image/jpeg", fpath.read_bytes())
                return
        if path == "/stream/cam1.mjpg":
            self._stream_mjpeg(1)
            return
        if path == "/stream/cam2.mjpg":
            self._stream_mjpeg(2)
            return
        super().do_GET()

    def do_DELETE(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/blackbox/session/"):
            sid = unquote(path[len("/api/blackbox/session/"):]).strip("/")
            active_sid = BRIDGE.get_blackbox_status().get("session_id")
            try:
                BlackBoxModule.delete_session(sid, active_session_id=active_sid, base_dir=BLACKBOX_DIR)
                self._send_json(200, {"ok": True})
            except RuntimeError as exc:
                self._send_json(409, {"ok": False, "error": str(exc)})
            except FileNotFoundError:
                self._send_json(404, {"ok": False, "error": "Khong tim thay phien"})
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"

        if path == "/api/control":
            try:
                data = json.loads(raw.decode("utf-8"))
                BRIDGE.update_from_web(data)
                self._send_json(200, {"ok": True, **BRIDGE.get_control_snapshot()})
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            return

        if path.startswith("/api/blackbox/session/"):
            rest = unquote(path[len("/api/blackbox/session/"):]).strip("/")
            parts = [p for p in rest.split("/") if p]
            if len(parts) == 2 and parts[1] == "note":
                try:
                    data = json.loads(raw.decode("utf-8"))
                    meta = BlackBoxModule.save_session_note(parts[0], data.get("notes", ""), BLACKBOX_DIR)
                    self._send_json(200, {"ok": True, "meta": meta})
                except FileNotFoundError:
                    self._send_json(404, {"ok": False, "error": "Khong tim thay phien"})
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return
            if len(parts) == 2 and parts[1] == "delete":
                active_sid = BRIDGE.get_blackbox_status().get("session_id")
                try:
                    BlackBoxModule.delete_session(parts[0], active_session_id=active_sid, base_dir=BLACKBOX_DIR)
                    self._send_json(200, {"ok": True})
                except RuntimeError as exc:
                    self._send_json(409, {"ok": False, "error": str(exc)})
                except FileNotFoundError:
                    self._send_json(404, {"ok": False, "error": "Khong tim thay phien"})
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return

        if path == "/internal/sync":
            try:
                import base64
                data = json.loads(raw.decode("utf-8"))
                c1 = base64.b64decode(data["cam1_b64"]) if data.get("cam1_b64") else None
                c2 = base64.b64decode(data["cam2_b64"]) if data.get("cam2_b64") else None
                BRIDGE.update_from_car(data.get("telemetry", {}), cam1_jpeg=c1, cam2_jpeg=c2)
                self._send_json(200, BRIDGE.get_control_snapshot())
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            return

        self._send_json(404, {"error": "Not found"})

    def log_message(self, format, *args):
        if args:
            msg = str(args[0])
            if any(k in msg for k in ("/api/status", "/api/control", "/internal/sync", "/stream/", "/api/blackbox/")):
                return
        super().log_message(format, *args)


def get_lan_ips():
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    if not ips:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ips.append(s.getsockname()[0])
        except Exception:
            pass
    return ips


def is_port_open(port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_background_server(port=PORT):
    """Khoi dong Web Server o che do nen neu cong 8080 chua co ai chay."""
    if is_port_open(port):
        return None
    handler = partial(RobotWebHandler, directory=str(WEB_DIR))
    server = ReusableThreadingHTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=server.serve_forever, name="WebServer", daemon=True)
    t.start()
    return server


def main():
    handler = partial(RobotWebHandler, directory=str(WEB_DIR))
    server = ReusableThreadingHTTPServer(("0.0.0.0", PORT), handler)
    lan_ips = get_lan_ips()

    print("=" * 64)
    print("  WEB DIEU KHIEN XE TU HANH (LUON CHAY - LAN / WIFI)")
    print("=" * 64)
    print(f"  * Tren May tinh nay    : http://localhost:{PORT}")
    for ip in lan_ips:
        print(f"  * Tren Dien thoai / LAN: http://{ip}:{PORT}")
    print("  Nhan Ctrl+C de dung may chu.")
    print("=" * 64)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Da dung may chu Web.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
