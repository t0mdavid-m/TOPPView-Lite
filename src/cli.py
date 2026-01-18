"""Command-line interface for TOPPView-Lite.

Allows launching the viewer with files pre-loaded:
    toppview-lite file.mzML
    toppview-lite file1.mzML file2.mzML
    toppview-lite --port 8502 file.mzML
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def get_app_dir() -> Path:
    """Get the directory containing the app.py file."""
    return Path(__file__).parent.parent


def setup_workspace(workspace_dir: Path) -> Path:
    """Set up the workspace directory structure.

    Args:
        workspace_dir: Base directory for workspaces

    Returns:
        Path to the workspace
    """
    workspace = workspace_dir / "cli-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "mzML-files").mkdir(parents=True, exist_ok=True)
    (workspace / "idXML-files").mkdir(parents=True, exist_ok=True)
    return workspace


def copy_files_to_workspace(files: list[Path], workspace: Path) -> list[Path]:
    """Copy or link mzML files to the workspace.

    Args:
        files: List of mzML file paths
        workspace: Workspace path

    Returns:
        List of paths to files in workspace
    """
    mzml_dir = workspace / "mzML-files"
    workspace_files = []

    for f in files:
        dest = mzml_dir / f.name
        if not dest.exists() or dest.stat().st_mtime < f.stat().st_mtime:
            # Use external_files.txt to reference without copying (faster for large files)
            external_files = mzml_dir / "external_files.txt"
            # Read existing paths if file exists
            existing_paths = set()
            if external_files.exists():
                with open(external_files, "r") as fh:
                    existing_paths = {line.strip() for line in fh if line.strip()}

            # Add this file's path if not already present
            abs_path = str(f.resolve())
            if abs_path not in existing_paths:
                with open(external_files, "a") as fh:
                    fh.write(f"{abs_path}\n")
            workspace_files.append(f.resolve())
        else:
            workspace_files.append(dest)

    return workspace_files


def write_startup_config(workspace: Path, files: list[Path], auto_process: bool = True) -> None:
    """Write startup configuration for the Streamlit app.

    Args:
        workspace: Workspace path
        files: List of mzML files to load
        auto_process: Whether to automatically process files
    """
    config = {
        "files": [str(f) for f in files],
        "auto_process": auto_process,
        "workspace": str(workspace),
    }
    config_path = get_app_dir() / ".cli_startup.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def run_streamlit(port: int = 8501, browser: bool = True) -> None:
    """Launch the Streamlit app.

    Args:
        port: Port to run on
        browser: Whether to open browser automatically
    """
    import subprocess

    app_dir = get_app_dir()
    app_py = app_dir / "app.py"

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_py),
        "--server.port", str(port),
    ]

    if not browser:
        cmd.extend(["--server.headless", "true"])

    # Add query param to use CLI workspace
    cmd.extend(["--", "--workspace=cli-workspace"])

    subprocess.run(cmd, cwd=str(app_dir))


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="toppview-lite",
        description="TOPPView-Lite: Interactive mass spectrometry data viewer",
        epilog="Example: toppview-lite sample.mzML",
    )

    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="mzML files to open",
    )

    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8501,
        help="Port to run the server on (default: 8501)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    parser.add_argument(
        "--no-process",
        action="store_true",
        help="Don't automatically preprocess files",
    )

    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    # Show version
    if args.version:
        settings_path = get_app_dir() / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
            print(f"TOPPView-Lite {settings.get('version', 'unknown')}")
        else:
            print("TOPPView-Lite (version unknown)")
        return 0

    # Validate files if provided
    if args.files:
        for f in args.files:
            if not f.exists():
                print(f"Error: File not found: {f}", file=sys.stderr)
                return 1
            if not f.suffix.lower() == ".mzml":
                print(f"Warning: {f} may not be an mzML file", file=sys.stderr)

    # Load settings to get workspace directory
    app_dir = get_app_dir()
    settings_path = app_dir / "settings.json"

    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
        workspaces_dir = Path(settings.get("workspaces_dir", ".."))
        if not workspaces_dir.is_absolute():
            workspaces_dir = app_dir / workspaces_dir
        workspaces_dir = workspaces_dir / f"workspaces-{settings.get('repository-name', 'toppview-lite')}"
    else:
        workspaces_dir = app_dir.parent / "workspaces-toppview-lite"

    # Set up workspace and files
    if args.files:
        workspace = setup_workspace(workspaces_dir)
        workspace_files = copy_files_to_workspace(args.files, workspace)
        write_startup_config(workspace, workspace_files, auto_process=not args.no_process)
        print(f"Added {len(args.files)} file(s) to workspace")
    else:
        # No files - just launch the app
        # Clear any existing startup config
        config_path = app_dir / ".cli_startup.json"
        if config_path.exists():
            config_path.unlink()

    # Launch Streamlit
    print(f"Starting TOPPView-Lite on port {args.port}...")
    run_streamlit(port=args.port, browser=not args.no_browser)

    return 0


if __name__ == "__main__":
    sys.exit(main())
