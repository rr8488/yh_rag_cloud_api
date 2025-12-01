import os, shlex, socket, subprocess, time
import psutil
import uvicorn
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "8088"))

APP_IMPORT = "yh_rag_cloud_api.app:app"
RELOAD_DIRS = ["yh_rag_cloud_api"]
RELOAD_EXCLUDES = [".venv", ".git", "__pycache__", "functions"]


def p(cmd: str) -> str:
    try:
        out = subprocess.check_output(shlex.split(cmd), text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()


def pids_on_port(port: int) -> set[int]:
    """Use lsof (macOS-friendly) to find PIDs listening on a port."""
    out = p(f"lsof -ti tcp:{port}")
    if not out:
        return set()
    return {int(x) for x in out.splitlines() if x.strip().isdigit()}


def looks_like_our_server(proc: psutil.Process) -> bool:
    try:
        cl = " ".join(proc.cmdline() or [])
        return ("uvicorn" in cl or "python" in cl) and ("yh_rag_cloud_api" in cl or "uvicorn" in cl)
    except Exception:
        return False


def kill_pid(pid: int):
    try:
        proc = psutil.Process(pid)
        if not looks_like_our_server(proc):
            # Safety: don't murder unrelated apps
            print(f"⚠️  PID {pid} is not uvicorn/yh_rag_cloud_api (cmd: {' '.join(proc.cmdline() or [])}); skipping")
            return
        print(f"🧹 Killing PID {pid} ({proc.name()}) …")
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except psutil.TimeoutExpired:
            proc.kill()
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"⚠️  Could not kill {pid}: {e}")


def free_port(port: int):
    """Find and kill any process that looks like our server listening on the given TCP port."""
    pids = pids_on_port(port)
    if not pids:
        return

    current_pid = os.getpid()
    for pid in pids:
        if pid == current_pid:
            continue
        # kill_pid already contains the logic to safely terminate the process
        kill_pid(pid)

    # A short delay to allow the OS to release the port, then check again.
    try:
        time.sleep(0.5)
        if port_in_use(HOST, port):
            print(f"⚠️  Port {port} is still in use after attempting to free it.")
    except Exception:
        pass


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


if __name__ == "__main__":
    free_port(PORT)
    print(f"🚀 Starting FastAPI dev server on http://{HOST}:{PORT}")
    uvicorn.run(
        APP_IMPORT,
        host=HOST,
        port=PORT,
        reload=True,
        reload_excludes=RELOAD_EXCLUDES,
        reload_dirs=RELOAD_DIRS, # Your existing code directories
        env_file=".env" # <-- ADD THIS LINE
    )