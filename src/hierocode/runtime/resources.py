import os
import platform
import subprocess

def get_cpu_count() -> int:
    return os.cpu_count() or 4

def get_total_ram_gb() -> float:
    """Best effort retrieval of total system RAM in GB for heuristics."""
    sys_name = platform.system()
    try:
        if sys_name == "Darwin":
            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode("utf-8").strip()
            return int(output) / (1024**3)
        elif sys_name == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        return kb / (1024**2)
    except Exception:
        pass
        
    return 16.0 # Arbitrary fallback assumption

def get_available_ram_gb() -> float:
    """Best effort retrieval of available system RAM in GB."""
    sys_name = platform.system()
    try:
        if sys_name == "Darwin":
            # Very rough estimate on mac using standard tools.
            # Accurately reading macOS available RAM without external libs like psutil is notoriously hard.
            # As a heuristic fallback, we will just say 'half of total'.
            return get_total_ram_gb() * 0.5
        elif sys_name == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemAvailable" in line:
                        kb = int(line.split()[1])
                        return kb / (1024**2)
    except Exception:
        pass
        
    return 8.0 # Arbitrary fallback assumption
