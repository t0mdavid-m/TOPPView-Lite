#!/usr/bin/env python3
"""TOPPView-Lite Command Line Interface.

This CLI provides a fast way to open mzML files by maintaining a background
Streamlit server. Instead of starting fresh each time (slow), it keeps the
server running and feeds new files to it.

Usage:
    toppview-lite                    # Start server and open browser
    toppview-lite file.mzML          # Open file in running server
    toppview-lite --stop             # Stop the background server
    toppview-lite --status           # Check server status
    toppview-lite --help             # Show help
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Default configuration
DEFAULT_PORT = 8501
LOCK_FILE_NAME = "server.lock"
SERVER_STARTUP_TIMEOUT = 30  # seconds


def get_app_data_dir() -> Path:
    """Get the application data directory for storing lock files etc."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    app_dir = base / "TOPPView-Lite"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_lock_file_path() -> Path:
    """Get the path to the server lock file."""
    return get_app_data_dir() / LOCK_FILE_NAME


def get_workspace_dir() -> Path:
    """Get the workspace directory for mzML files."""
    # Load settings to get workspaces_dir
    settings_path = Path(__file__).parent / "settings.json"
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
        workspaces_dir = settings.get("workspaces_dir", "..")
        repo_name = settings.get("repository-name", "toppview-lite")
        workspace_base = Path(workspaces_dir) / f"workspaces-{repo_name}"
    else:
        workspace_base = Path("..") / "workspaces-toppview-lite"

    # Use 'default' workspace for CLI
    workspace = workspace_base / "default"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def read_lock_file() -> dict | None:
    """Read the lock file and return server info, or None if not found."""
    lock_file = get_lock_file_path()
    if not lock_file.exists():
        return None

    try:
        with open(lock_file) as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError):
        return None


def write_lock_file(pid: int, port: int) -> None:
    """Write server info to the lock file."""
    lock_file = get_lock_file_path()
    with open(lock_file, "w") as f:
        json.dump({"pid": pid, "port": port, "started": time.time()}, f)


def remove_lock_file() -> None:
    """Remove the lock file."""
    lock_file = get_lock_file_path()
    if lock_file.exists():
        lock_file.unlink()


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def server_is_running() -> tuple[bool, dict | None]:
    """Check if the server is running.

    Returns:
        Tuple of (is_running, server_info)
    """
    lock_info = read_lock_file()
    if not lock_info:
        return False, None

    pid = lock_info.get("pid")
    port = lock_info.get("port", DEFAULT_PORT)

    # Check if process is running AND port is in use
    if pid and is_process_running(pid) and is_port_in_use(port):
        return True, lock_info

    # Stale lock file - clean up
    remove_lock_file()
    return False, None


def start_server(port: int = DEFAULT_PORT, wait: bool = True) -> bool:
    """Start the Streamlit server in the background.

    Args:
        port: Port to run the server on
        wait: Whether to wait for the server to be ready

    Returns:
        True if server started successfully
    """
    running, info = server_is_running()
    if running:
        print(f"Server already running on port {info['port']} (PID: {info['pid']})")
        return True

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    app_py = script_dir / "app.py"

    if not app_py.exists():
        print(f"Error: app.py not found at {app_py}")
        return False

    print(f"Starting TOPPView-Lite server on port {port}...")

    # Build the command
    # Use the Python executable that's running this script
    python_exe = sys.executable

    # Determine how to run streamlit
    cmd = [
        python_exe, "-m", "streamlit", "run",
        str(app_py),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--",  # Separator for app arguments
        "local"  # Our custom arg indicating local mode
    ]

    # Start the process
    if sys.platform == "win32":
        # On Windows, use CREATE_NO_WINDOW to hide the console
        CREATE_NO_WINDOW = 0x08000000
        DETACHED_PROCESS = 0x00000008
        process = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS,
        )
    else:
        # On Unix, use nohup-like behavior
        process = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Write lock file with PID
    write_lock_file(process.pid, port)

    if wait:
        # Wait for server to be ready
        print("Waiting for server to start...", end="", flush=True)
        for i in range(SERVER_STARTUP_TIMEOUT):
            if is_port_in_use(port):
                print(" Ready!")
                return True
            print(".", end="", flush=True)
            time.sleep(1)

        print("\nWarning: Server may not have started properly")
        return False

    return True


def stop_server() -> bool:
    """Stop the background server.

    Returns:
        True if server was stopped
    """
    running, info = server_is_running()
    if not running:
        print("Server is not running")
        remove_lock_file()  # Clean up any stale lock
        return True

    pid = info["pid"]
    print(f"Stopping server (PID: {pid})...")

    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                         capture_output=True)
        else:
            os.kill(pid, 15)  # SIGTERM
            time.sleep(1)
            if is_process_running(pid):
                os.kill(pid, 9)  # SIGKILL
    except Exception as e:
        print(f"Error stopping server: {e}")

    remove_lock_file()
    print("Server stopped")
    return True


def get_server_status() -> None:
    """Print the server status."""
    running, info = server_is_running()
    if running:
        uptime = time.time() - info.get("started", 0)
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)

        print("TOPPView-Lite Server Status:")
        print(f"  Status: Running")
        print(f"  Port: {info['port']}")
        print(f"  PID: {info['pid']}")
        print(f"  Uptime: {hours}h {minutes}m {seconds}s")
        print(f"  URL: http://localhost:{info['port']}")
    else:
        print("TOPPView-Lite Server Status:")
        print("  Status: Not running")


def copy_file_to_workspace(file_path: Path) -> Path | None:
    """Copy an mzML file to the workspace.

    Args:
        file_path: Path to the mzML file

    Returns:
        Path to the file in the workspace, or None on error
    """
    workspace = get_workspace_dir()
    mzml_dir = workspace / "mzML-files"
    mzml_dir.mkdir(parents=True, exist_ok=True)

    dest_path = mzml_dir / file_path.name

    # If source is already in workspace, don't copy
    if file_path.resolve() == dest_path.resolve():
        return dest_path

    try:
        shutil.copy2(file_path, dest_path)
        print(f"Copied {file_path.name} to workspace")
        return dest_path
    except Exception as e:
        print(f"Error copying file: {e}")
        return None


def open_file(file_path: Path, port: int = DEFAULT_PORT) -> bool:
    """Open an mzML file in TOPPView-Lite.

    This will:
    1. Start the server if not running
    2. Copy the file to the workspace
    3. Open the browser with the file loaded

    Args:
        file_path: Path to the mzML file
        port: Server port

    Returns:
        True if successful
    """
    # Validate file
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return False

    if file_path.suffix.lower() not in [".mzml"]:
        print(f"Warning: File may not be an mzML file: {file_path}")

    # Ensure server is running
    running, info = server_is_running()
    if not running:
        if not start_server(port):
            return False
        info = {"port": port}

    port = info.get("port", DEFAULT_PORT)

    # Copy file to workspace
    workspace_file = copy_file_to_workspace(file_path)
    if not workspace_file:
        return False

    # Build URL with load_file parameter
    # The file name (without path) is passed as the parameter
    url = f"http://localhost:{port}/?load_file={file_path.name}"

    print(f"Opening {file_path.name} in browser...")
    webbrowser.open(url)

    return True


def open_browser(port: int = DEFAULT_PORT) -> None:
    """Open the browser to the running server."""
    running, info = server_is_running()
    if running:
        port = info.get("port", port)

    url = f"http://localhost:{port}"
    print(f"Opening {url}...")
    webbrowser.open(url)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TOPPView-Lite - Lightweight mzML viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  toppview-lite                    Start server and open browser
  toppview-lite sample.mzML        Open mzML file in viewer
  toppview-lite --status           Check if server is running
  toppview-lite --stop             Stop the background server
  toppview-lite --port 8502        Use a different port
        """
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="mzML file to open"
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})"
    )

    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the server (default action if no file given)"
    )

    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the background server"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show server status"
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )

    args = parser.parse_args()

    # Handle commands
    if args.status:
        get_server_status()
        return 0

    if args.stop:
        stop_server()
        return 0

    if args.file:
        # Open a file
        file_path = Path(args.file).resolve()
        if open_file(file_path, args.port):
            return 0
        return 1

    # Default: start server and open browser
    if not start_server(args.port):
        return 1

    if not args.no_browser:
        open_browser(args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
