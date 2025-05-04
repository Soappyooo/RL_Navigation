import os
import platform
from pathlib import Path


def get_unity_executable_path() -> str:
    """
    Get the platform-specific Unity executable path.

    Returns:
        str: Path to the Unity executable for the current platform
    """
    # Get the base directory containing the Unity build
    base_dir = Path(__file__).parent / "MineField"

    # Define platform-specific executable names
    platform_executables = {"Linux": "drl.x86_64", "Windows": "drl.exe"}

    # Get current platform's executable name
    current_platform = platform.system()
    executable = platform_executables.get(current_platform)

    if executable is None:
        raise RuntimeError(f"Unsupported platform: {current_platform}")

    # Check if executable exists
    executable_path = base_dir / executable
    if not executable_path.exists():
        raise FileNotFoundError(f"Unity executable not found at: {executable_path}")

    return str(executable_path)


# Environment variables can override the default path
# default to "EpMineEnv-main/envs/SingleAgent/MineField/drl.x86_64 or drl.exe"
UNITY_ENV_PATH = os.environ.get("UNITY_ENV_PATH", get_unity_executable_path())
