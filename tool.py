#!/usr/bin/env python3
"""Cong cu 1 file duy nhat: mo terminal tuong tac de chay, day len (push), tai ve (pull/clone) du an XeTuHanh."""

from datetime import datetime
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/pctuyen-Dev/XeTuHanh.git"
PYTHON_EXE = sys.executable


def ensure_terminal():
    """Neu file duoc chay bang pythonw (khong co console), tu dong mo cua so terminal moi."""
    if os.name == "nt" and len(sys.argv) == 1 and (sys.stdin is None or sys.stdout is None):
        py_cmd = PYTHON_EXE
        if py_cmd.lower().endswith("pythonw.exe"):
            py_cmd = py_cmd[:-5] + ".exe"
        subprocess.Popen(
            ["cmd", "/c", "start", "XeTuHanh Tool", "cmd", "/k", py_cmd, os.path.abspath(__file__)]
        )
        sys.exit(0)


def run_cmd(cmd, cwd=BASE_DIR):
    """Chay lenh he thong va hien thi truc tiep ra terminal."""
    print(f"\n>>> {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=cwd)
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[INFO] Da dung lenh theo yeu cau (Ctrl+C).")
        return 130
    except FileNotFoundError:
        print(f"[ERROR] Khong tim thay lenh: {cmd[0]}")
        return 127


def run_capture(cmd, cwd=BASE_DIR):
    """Chay lenh va tra ve (returncode, stdout, stderr)."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def check_git_ready():
    if shutil.which("git") is None:
        print("[ERROR] May chua cai dat Git!")
        return False
    if not os.path.isdir(os.path.join(BASE_DIR, ".git")):
        print("[WARNING] Thu muc chua khoi tao Git. Dang tu dong ket noi voi repo...")
        run_cmd(["git", "init", "-b", "main"])
        run_cmd(["git", "remote", "add", "origin", REPO_URL])
    return True


def action_run_robot(headless=False):
    cmd = [PYTHON_EXE, "RobotMain.py"]
    if headless:
        cmd.append("--headless")
    run_cmd(cmd)


def action_run_test():
    run_cmd([PYTHON_EXE, "test.py"])


def action_git_push():
    if not check_git_ready():
        return

    rc, status_out, _ = run_capture(["git", "status", "--short"])
    if rc != 0:
        print("[ERROR] Khong kiem tra duoc trang thai Git.")
        return

    if not status_out:
        print("\n[INFO] Khong co thay doi moi de commit.")
        ans = input("Ban co muon thu day cac commit chua push len GitHub khong? (y/N): ").strip().lower()
        if ans == "y":
            run_cmd(["git", "push"])
        return

    print("\n--- Cac file thay doi ---")
    print(status_out)
    default_msg = f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    msg = input(f"\nNhap noi dung commit (Enter de dung '{default_msg}'): ").strip()
    if not msg:
        msg = default_msg

    if run_cmd(["git", "add", "-A"]) != 0:
        return
    if run_cmd(["git", "commit", "-m", msg]) != 0:
        return

    rc_push = run_cmd(["git", "push", "-u", "origin", "main"])
    if rc_push == 0:
        print("\n[OK] Da day code len GitHub thanh cong!")
    else:
        print("\n[ERROR] Day code that bai. Kiem tra ket noi mang hoac quyen truy cap GitHub.")


def action_git_pull():
    if not check_git_ready():
        return

    _, status_out, _ = run_capture(["git", "status", "--porcelain"])
    stashed = False
    if status_out:
        print("\n[WARNING] Phat hien co thay doi chua commit o may nay:")
        print(status_out)
        ans = input("Tu dong cat tam (git stash) truoc khi tai ve va khoi phuc lai? (Y/n): ").strip().lower()
        if ans in ("", "y"):
            if run_cmd(["git", "stash", "push", "-m", "auto-stash-before-pull"]) == 0:
                stashed = True
        else:
            print("[INFO] Bo qua stash, tien hanh git pull truc tiep...")

    rc = run_cmd(["git", "pull", "--rebase", "origin", "main"])
    if stashed:
        run_cmd(["git", "stash", "pop"])

    if rc == 0:
        print("\n[OK] Da tai ban moi nhat tu GitHub ve thanh cong!")
    else:
        print("\n[ERROR] Tai ve that bai. Neu bi xung dot (conflict), hay kiem tra lai cac file.")


def action_git_clone():
    if shutil.which("git") is None:
        print("[ERROR] May chua cai dat Git!")
        return
    target = input("\nNhap ten hoac duong dan thu muc muon tai ve (VD: XeTuHanh_new): ").strip()
    if not target:
        print("[INFO] Da huy thao tac clone.")
        return
    run_cmd(["git", "clone", REPO_URL, target], cwd=os.path.dirname(BASE_DIR))


def action_git_status():
    if not check_git_ready():
        return
    print("\n--- TRANG THAI GIT ---")
    run_cmd(["git", "status", "-sb"])
    print("\n--- 5 COMMIT GAN NHAT ---")
    run_cmd(["git", "log", "-n", "5", "--oneline", "--decorate"])


def action_install_deps():
    req_file = os.path.join(BASE_DIR, "requirements.txt")
    if not os.path.isfile(req_file):
        print("[ERROR] Khong tim thay requirements.txt")
        return
    rc = run_cmd([PYTHON_EXE, "-m", "pip", "install", "-r", "requirements.txt"])
    if rc != 0 and os.name != "nt":
        print("\n[INFO] Thu cai lai voi co --break-system-packages (Raspberry Pi OS)...")
        run_cmd([PYTHON_EXE, "-m", "pip", "install", "-r", "requirements.txt", "--break-system-packages"])


def action_syntax_check():
    files = [
        "RobotMain.py",
        "RobotUI.py",
        "MotorModule.py",
        "cameraModule.py",
        "test.py",
        "tool.py",
        os.path.join("P1.NhanDienLanDuong", "LaneDetectionModule.py"),
        os.path.join("P1.NhanDienLanDuong", "utils.py"),
        os.path.join("P2.NhanDienBienBao", "TrafficSignDetectionModule.py"),
    ]
    existing = [f for f in files if os.path.isfile(os.path.join(BASE_DIR, f))]
    rc = run_cmd([PYTHON_EXE, "-m", "py_compile"] + existing)
    if rc == 0:
        print(f"\n[OK] Tat ca {len(existing)} file Python deu hop le, khong co loi cu phap!")


def print_menu():
    print("\n" + "=" * 62)
    print("   XE TU HANH - CONG CU QUAN LY & CHAY DU AN")
    print("=" * 62)
    print("  [1] Chay Xe Tu Hanh (RobotMain.py - Co giao dien GUI)")
    print("  [2] Chay Xe Tu Hanh (RobotMain.py --headless - Khong man hinh)")
    print("  [3] Chay Trinh Mo Phong / Test (test.py)")
    print("  " + "-" * 58)
    print("  [4] Day code len GitHub (Git Add + Commit + Push)")
    print("  [5] Tai code moi tu GitHub ve (Git Pull)")
    print("  [6] Tai ve thu muc moi (Git Clone)")
    print("  [7] Xem trang thai & Lich su Git (Status + Log)")
    print("  " + "-" * 58)
    print("  [8] Cai dat thu vien (pip install -r requirements.txt)")
    print("  [9] Kiem tra cu phap Python (py_compile)")
    print("  [0] Thoat")
    print("=" * 62)


def main():
    ensure_terminal()

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        dispatch = {
            "run": lambda: action_run_robot(False),
            "headless": lambda: action_run_robot(True),
            "test": action_run_test,
            "push": action_git_push,
            "pull": action_git_pull,
            "clone": action_git_clone,
            "status": action_git_status,
            "install": action_install_deps,
            "check": action_syntax_check,
        }
        if arg in dispatch:
            dispatch[arg]()
            return
        print(f"[WARNING] Tham so khong hop le: {arg}. Mo menu tuong tac...")

    actions = {
        "1": lambda: action_run_robot(False),
        "2": lambda: action_run_robot(True),
        "3": action_run_test,
        "4": action_git_push,
        "5": action_git_pull,
        "6": action_git_clone,
        "7": action_git_status,
        "8": action_install_deps,
        "9": action_syntax_check,
    }

    while True:
        print_menu()
        try:
            choice = input("Chon chuc nang [0-9]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTam biet!")
            break
        if choice == "0":
            print("Tam biet!")
            break
        fn = actions.get(choice)
        if fn:
            fn()
        else:
            print("[WARNING] Lua chon khong hop le, vui long nhap tu 0 den 9.")


if __name__ == "__main__":
    main()
