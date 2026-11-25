"""
server.py - May chu Web luon chay (0.0.0.0:8080), phuc vu giao dien,
luong MJPEG 2 camera va dong bo trang thai/lenh dieu khien voi RobotMain.py.
Chay doc lap: python web/server.py
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
import time

WEB_DIR = Path(__file__).resolve().parent
PORT = 8080
START_TIME = time.time()
CAR_HEARTBEAT_TIMEOUT_S = 1.5


class WebBridgeState:
    """Bo nho chia se giua Web UI va RobotMain.py."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last_car_heartbeat = 0.0

        # Lenh tu Web -> Xe
        self.car_running = False
        self.drive_mode = "AUTO"       # "AUTO" | "MANUAL"
        self.manual_speed = 0          # -100 .. 100
        self.manual_steer = 0.0        # -1.0 .. +1.0
        self.max_speed = 85            # 20 .. 100
        self.fov_pct = 100             # 40 .. 100
        self.pending_actions = []      # ["switch_cam1", "switch_cam2", "swap_cams"]

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
                "actions": actions,
            }

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
                "speed": self.speed if car_connected else 0,
                "steer": self.steer if car_connected else 0.0,
                "servo_deg": self.servo_deg if car_connected else 90,
                "cmd": self.cmd if car_connected else "STOP",
                "packet": self.packet if car_connected else "<0,+0.00,STOP>",
                "cam1_src": self.cam1_src,
                "cam2_src": self.cam2_src,
                "signs": list(self.signs) if car_connected else [],
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
        if path == "/stream/cam1.mjpg":
            self._stream_mjpeg(1)
            return
        if path == "/stream/cam2.mjpg":
            self._stream_mjpeg(2)
            return
        super().do_GET()

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
            if any(k in msg for k in ("/api/status", "/api/control", "/internal/sync", "/stream/")):
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
