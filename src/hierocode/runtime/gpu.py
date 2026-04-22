"""Best-effort GPU detection across NVIDIA, Apple Silicon, and AMD platforms."""

import platform
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional

from hierocode.runtime.resources import get_total_ram_gb

Backend = Literal["nvidia", "apple", "amd", "none", "unknown"]


@dataclass
class GPUInfo:
    has_gpu: bool
    vram_gb: float
    gpu_name: Optional[str]
    backend: Backend


def probe_gpu() -> GPUInfo:
    """Best-effort GPU detection across platforms. Never raises — degrades gracefully."""
    # 1. NVIDIA (any OS)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            first_line = result.stdout.strip().splitlines()[0]
            parts = first_line.split(",", 1)
            if len(parts) == 2:
                vram_mib = float(parts[0].strip())
                gpu_name = parts[1].strip()
                return GPUInfo(
                    has_gpu=True,
                    vram_gb=vram_mib / 1024,
                    gpu_name=gpu_name,
                    backend="nvidia",
                )
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    # 2. Apple Silicon (macOS arm64)
    try:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return GPUInfo(
                has_gpu=True,
                vram_gb=get_total_ram_gb(),
                gpu_name="Apple Silicon (unified memory)",
                backend="apple",
            )
    except Exception:
        pass

    # 3. AMD via ROCm (Linux, best-effort)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            import json

            data = json.loads(result.stdout)
            # Iterate keys to find first GPU VRAM entry
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                for field, field_val in value.items():
                    if "vram" in field.lower() and "total" in field.lower() and "b" in field.lower():
                        vram_bytes = float(field_val)
                        return GPUInfo(
                            has_gpu=True,
                            vram_gb=vram_bytes / (1024**3),
                            gpu_name=key,
                            backend="amd",
                        )
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    # 4. Fallback
    return GPUInfo(has_gpu=False, vram_gb=0.0, gpu_name=None, backend="none")
