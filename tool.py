from datetime import datetime
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys

SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent
REPO_URL = "https://github.com/pctuyen-Dev/XeTuHanh.git"
PYTHON_EXE = sys.executable
DEFAULT_BRANCH = "main"
IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "build", "dist", ".agents"}


def ensure_terminal():
    if os.name == "nt" and len(sys.argv) == 1 and (sys.stdin is None or not sys.stdin.isatty()):
        py_path = Path(PYTHON_EXE)
        if py_path.name.lower() == "pythonw.exe":
            candidate = py_path.with_name("python.exe")
            if candidate.is_file():
                py_path = candidate
        subprocess.Popen(
            [str(py_path), str(SCRIPT_PATH)],
            cwd=str(BASE_DIR),
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
        sys.exit(0)


def format_cmd(cmd):
    parts = [str(part) for part in cmd]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def read_input(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def run_cmd(cmd, cwd=BASE_DIR):
    parts = [str(part) for part in cmd]
    print(f"\n>>> {format_cmd(parts)}", flush=True)
    try:
        return subprocess.run(parts, cwd=str(cwd)).returncode
    except KeyboardInterrupt:
        print("\n[INFO] Da dung lenh theo yeu cau (Ctrl+C).")
        return 130
    except FileNotFoundError:
        print(f"[ERROR] Khong tim thay lenh: {parts[0]}")
        return 127


def run_capture(cmd, cwd=BASE_DIR):
    parts = [str(part) for part in cmd]
    try:
        proc = subprocess.run(
            parts,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, proc.stdout.rstrip(), proc.stderr.rstrip()
    except FileNotFoundError:
        return 127, "", f"Khong tim thay lenh: {parts[0]}"


def has_git():
    if shutil.which("git") is None:
        print("[ERROR] May chua cai dat Git!")
        return False
    return True


def check_git_ready():
    if not has_git():
        return False
    if not (BASE_DIR / ".git").exists():
        print("[WARNING] Thu muc chua khoi tao Git. Dang tu dong ket noi voi repo...")
        if run_cmd(["git", "init", "-b", DEFAULT_BRANCH]) != 0:
            print("[ERROR] Khoi tao Git that bai.")
            return False
        if run_cmd(["git", "remote", "add", "origin", REPO_URL]) != 0:
            print("[ERROR] Khong the them remote origin.")
            return False
        return True
    rc, _, _ = run_capture(["git", "remote", "get-url", "origin"])
    if rc != 0:
        if run_cmd(["git", "remote", "add", "origin", REPO_URL]) != 0:
            print("[ERROR] Khong the them remote origin.")
            return False
    return True


def get_current_branch():
    rc, branch, _ = run_capture(["git", "branch", "--show-current"])
    branch = branch.strip()
    return branch if rc == 0 and branch else DEFAULT_BRANCH


def get_stash_ref():
    rc, out, _ = run_capture(["git", "rev-parse", "--verify", "-q", "refs/stash"])
    return out.strip() if rc == 0 else ""


def has_commits():
    rc, _, _ = run_capture(["git", "rev-parse", "--verify", "-q", "HEAD"])
    return rc == 0


def run_python_script(filename, extra_args=None):
    script = BASE_DIR / filename
    if not script.is_file():
        print(f"[ERROR] Khong tim thay file: {filename}")
        return 1
    cmd = [PYTHON_EXE, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    return run_cmd(cmd)


def action_run_robot(headless=False):
    args = ["--headless"] if headless else None
    return run_python_script("RobotMain.py", args)


def action_run_test():
    return run_python_script("test.py")


def action_git_push():
    if not check_git_ready():
        return 1

    branch = get_current_branch()
    rc, status_out, err_out = run_capture(["git", "status", "--porcelain"])
    if rc != 0:
        print(f"[ERROR] Khong kiem tra duoc trang thai Git. {err_out}".strip())
        return rc

    if not status_out:
        print("\n[INFO] Khong co thay doi moi de commit.")
        ans = read_input(f"Ban co muon thu day cac commit chua push len nhanh '{branch}' khong? (y/N): ")
        if ans and ans.lower() == "y":
            rc_push = run_cmd(["git", "push", "-u", "origin", branch])
            if rc_push == 0:
                print("\n[OK] Da day code len GitHub thanh cong!")
            else:
                print("\n[ERROR] Day code that bai. Kiem tra ket noi mang hoac quyen truy cap GitHub.")
            return rc_push
        return 0

    print("\n--- Cac file thay doi ---")
    print(status_out)
    default_msg = f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    msg = read_input(f"\nNoi dung commit{default_msg}'): ")
    if msg is None:
        print("[INFO] Da huy thao tac push.")
        return 130
    if not msg:
        msg = default_msg

    rc_add = run_cmd(["git", "add", "-A"])
    if rc_add != 0:
        return rc_add
    rc_commit = run_cmd(["git", "commit", "-m", msg])
    if rc_commit != 0:
        return rc_commit

    rc_push = run_cmd(["git", "push", "-u", "origin", branch])
    if rc_push == 0:
        print("\n[DONE] Da day code len GitHub thanh cong!")
    else:
        print("\n[ERROR] Kiem tra ket noi mang hoac quyen truy cap GitHub.")
    return rc_push


def action_git_pull():
    if not check_git_ready():
        return 1

    branch = get_current_branch()
    rc_status, status_out, err_out = run_capture(["git", "status", "--porcelain"])
    if rc_status != 0:
        print(f"[ERROR] Khong kiem tra duoc trang thai Git. {err_out}".strip())
        return rc_status

    stashed = False
    if status_out:
        print("\n[WARNING] Phat hien co thay doi chua commit o may nay:")
        print(status_out)
        ans = read_input("Tu dong cat tam truoc khi tai ve va khoi phuc lai? (Y/n): ")
        if ans is None:
            print("[INFO] Da huy thao tac pull.")
            return 130
        if ans.lower() in ("", "y"):
            stash_before = get_stash_ref()
            rc_stash = run_cmd(["git", "stash", "push", "-u", "-m", "auto-stash-before-pull"])
            if rc_stash != 0:
                print("[ERROR] Khong the cat tam thay doi. Da dung thao tac pull.")
                return rc_stash
            stashed = get_stash_ref() != stash_before
        else:
            print("[INFO] Bo qua stash, tien hanh git pull truc tiep...")

    rc_pull = run_cmd(["git", "pull", "--rebase", "origin", branch])
    if rc_pull != 0:
        print("\n[ERROR] Tai ve that bai. Neu bi xung dot (conflict), hay kiem tra lai cac file.")
        if stashed:
            print("[INFO] Thay doi cuc bo van duoc giu an toan trong git stash.")
        return rc_pull

    if stashed:
        rc_pop = run_cmd(["git", "stash", "pop"])
        if rc_pop != 0:
            print("\n[WARNING] Da tai code moi nhung khoi phuc stash bi xung dot. Hay kiem tra lai.")
            return rc_pop

    print("\n[DONE] Da tai ban moi nhat tu GitHub ve thanh cong!")
    return 0


def action_git_clone():
    if not has_git():
        return 1

    target = read_input("\nNhap ten hoac duong dan thu muc muon tai ve: ")
    if not target:
        print("[INFO] Da huy thao tac clone.")
        return 0

    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = BASE_DIR.parent / target_path

    if target_path.exists() and (target_path.is_file() or any(target_path.iterdir())):
        print(f"[ERROR] Duong dan dich da ton tai va khong rong: {target_path}")
        return 1

    rc = run_cmd(["git", "clone", REPO_URL, str(target_path)], cwd=BASE_DIR.parent)
    if rc == 0:
        print(f"\n[DONE] Da clone du an vao: {target_path}")
    else:
        print("\n[ERROR] Clone du an that bai.")
    return rc


def action_git_status():
    if not check_git_ready():
        return 1

    print("\n--- TRANG THAI GIT ------")
    rc_status = run_cmd(["git", "status", "-sb"])
    print("\n--- 5 COMMIT GAN NHAT ---")
    if has_commits():
        rc_log = run_cmd(["git", "log", "-n", "5", "--oneline", "--decorate"])
        return rc_status or rc_log
    print("[INFO] Chua co commit nao.")
    return rc_status


def action_install_deps():
    req_file = BASE_DIR / "requirements.txt"
    if not req_file.is_file():
        print("[ERROR] Khong tim thay requirements.txt")
        return 1

    rc = run_cmd([PYTHON_EXE, "-m", "pip", "install", "-r", str(req_file)])
    if rc != 0 and os.name != "nt":
        print("\n[INFO] Thu cai lai voi co --break-system-packages (Raspberry Pi OS)...")
        rc = run_cmd([PYTHON_EXE, "-m", "pip", "install", "-r", str(req_file), "--break-system-packages"])

    if rc == 0:
        print("\n[DONE] Da cai dat thu vien thanh cong!")
    else:
        print("\n[ERROR] Cai dat thu vien that bai.")
    return rc


def collect_python_files():
    py_files = []
    for path in BASE_DIR.rglob("*.py"):
        rel = path.relative_to(BASE_DIR)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        py_files.append(rel.as_posix())
    return sorted(py_files)


def action_syntax_check():
    py_files = collect_python_files()
    if not py_files:
        print("[WARNING] Khong tim thay file Python nao de kiem tra.")
        return 1

    rc = run_cmd([PYTHON_EXE, "-m", "py_compile", *py_files])
    if rc == 0:
        print(f"\n[OK] Tat ca {len(py_files)} file Python deu hop le, khong co loi cu phap!")
    else:
        print("\n[ERROR] Phat hien loi cu phap trong du an.")
    return rc


MENU_ITEMS = [
    ("1", "run", "Chay RobotMain.py - Co giao dien", lambda: action_run_robot(False)),
    ("2", "headless", "Chay RobotMain.py - không giao diện", lambda: action_run_robot(True)),
    ("3", "test", "Chay Test", action_run_test),
    None,
    ("4", "push", "Day code len GitHub (Git Add + Commit + Push)", action_git_push),
    ("5", "pull", "Tai code moi tu GitHub ve (Git Pull)", action_git_pull),
    ("6", "clone", "Tai ve thu muc moi (Git Clone)", action_git_clone),
    ("7", "status", "Xem trang thai & Lich su Git (Status + Log)", action_git_status),
    None,
    ("8", "install", "Cai dat thu vien (pip install -r requirements.txt)", action_install_deps),
    ("9", "check", "Kiem tra cu phap Python", action_syntax_check),
]


def print_menu():
    print("\n" + "=" * 62)
    print("   XE ROBOT - TOOL")
    print("=" * 62)
    for item in MENU_ITEMS:
        if item is None:
            print("  " + "-" * 58)
        else:
            key, _, label, _ = item
            print(f"  [{key}] {label}")
    print("  [0] Thoat")
    print("=" * 62)


def print_help():
    print("Cach dung: python tool.py [lenh]")
    print("Cac lenh ho tro:")
    for item in MENU_ITEMS:
        if item is not None:
            _, cli_name, label, _ = item
            print(f"  {cli_name:<10} {label}")


def main():
    ensure_terminal()

    valid_items = [item for item in MENU_ITEMS if item is not None]
    cli_dispatch = {cli_name: handler for _, cli_name, _, handler in valid_items}
    menu_dispatch = {key: handler for key, _, _, handler in valid_items}

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("-h", "--help", "help"):
            print_help()
            return
        handler = cli_dispatch.get(arg)
        if handler is not None:
            sys.exit(handler())
        print(f"[WARNING] Tham so khong hop le: {arg}. Mo menu tuong tac...")

    while True:
        print_menu()
        choice = read_input("Chon chuc nang [0-9]: ")
        if choice is None or choice == "0":
            print("Tam biet!")
            break
        handler = menu_dispatch.get(choice)
        if handler is not None:
            handler()
        else:
            print("[WARNING] Lua chon khong hop le, vui long nhap tu 0 den 9.")


if __name__ == "__main__":
    main()
