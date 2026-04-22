"""Unit tests for hierocode.runtime.gpu — all subprocess/platform calls are mocked."""

import json
import subprocess
import unittest
from unittest.mock import patch

from hierocode.runtime.gpu import GPUInfo, probe_gpu


def _completed(returncode: int, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class TestNvidiaDetection(unittest.TestCase):
    def test_nvidia_detected(self):
        with patch("hierocode.runtime.gpu.subprocess.run") as mock_run:
            mock_run.return_value = _completed(0, "12288, NVIDIA GeForce RTX 3060\n")
            info = probe_gpu()
        self.assertTrue(info.has_gpu)
        self.assertAlmostEqual(info.vram_gb, 12.0)
        self.assertEqual(info.gpu_name, "NVIDIA GeForce RTX 3060")
        self.assertEqual(info.backend, "nvidia")

    def test_nvidia_multi_gpu_picks_first(self):
        stdout = "8192, NVIDIA RTX 4070\n16384, NVIDIA RTX 4090\n"
        with patch("hierocode.runtime.gpu.subprocess.run") as mock_run:
            mock_run.return_value = _completed(0, stdout)
            info = probe_gpu()
        self.assertEqual(info.gpu_name, "NVIDIA RTX 4070")
        self.assertAlmostEqual(info.vram_gb, 8.0)

    def test_nvidia_not_installed(self):
        with (
            patch("hierocode.runtime.gpu.subprocess.run", side_effect=FileNotFoundError),
            patch("hierocode.runtime.gpu.platform.system", return_value="Linux"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="x86_64"),
        ):
            info = probe_gpu()
        self.assertFalse(info.has_gpu)
        self.assertEqual(info.backend, "none")

    def test_nvidia_nonzero_exit(self):
        with (
            patch("hierocode.runtime.gpu.subprocess.run", return_value=_completed(1, "")),
            patch("hierocode.runtime.gpu.platform.system", return_value="Linux"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="x86_64"),
        ):
            info = probe_gpu()
        self.assertFalse(info.has_gpu)
        self.assertEqual(info.backend, "none")

    def test_nvidia_timeout(self):
        with (
            patch(
                "hierocode.runtime.gpu.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
            ),
            patch("hierocode.runtime.gpu.platform.system", return_value="Linux"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="x86_64"),
        ):
            info = probe_gpu()
        self.assertFalse(info.has_gpu)
        self.assertEqual(info.backend, "none")

    def test_nvidia_smi_malformed_stdout(self):
        with (
            patch("hierocode.runtime.gpu.subprocess.run", return_value=_completed(0, "garbage")),
            patch("hierocode.runtime.gpu.platform.system", return_value="Linux"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="x86_64"),
        ):
            info = probe_gpu()
        # malformed output: no comma → falls through to fallback
        self.assertFalse(info.has_gpu)
        self.assertEqual(info.backend, "none")


class TestAppleSiliconDetection(unittest.TestCase):
    def test_apple_silicon_detected(self):
        with (
            patch(
                "hierocode.runtime.gpu.subprocess.run",
                side_effect=FileNotFoundError,
            ),
            patch("hierocode.runtime.gpu.platform.system", return_value="Darwin"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="arm64"),
            patch("hierocode.runtime.gpu.get_total_ram_gb", return_value=32.0),
        ):
            info = probe_gpu()
        self.assertTrue(info.has_gpu)
        self.assertAlmostEqual(info.vram_gb, 32.0)
        self.assertEqual(info.backend, "apple")
        self.assertIn("Apple", info.gpu_name)

    def test_apple_silicon_but_nvidia_wins(self):
        with (
            patch(
                "hierocode.runtime.gpu.subprocess.run",
                return_value=_completed(0, "6144, NVIDIA Quadro\n"),
            ),
            patch("hierocode.runtime.gpu.platform.system", return_value="Darwin"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="arm64"),
            patch("hierocode.runtime.gpu.get_total_ram_gb", return_value=32.0),
        ):
            info = probe_gpu()
        self.assertEqual(info.backend, "nvidia")
        self.assertEqual(info.gpu_name, "NVIDIA Quadro")

    def test_darwin_intel_is_none(self):
        with (
            patch(
                "hierocode.runtime.gpu.subprocess.run",
                side_effect=FileNotFoundError,
            ),
            patch("hierocode.runtime.gpu.platform.system", return_value="Darwin"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="x86_64"),
        ):
            info = probe_gpu()
        self.assertFalse(info.has_gpu)
        self.assertEqual(info.backend, "none")


class TestAMDDetection(unittest.TestCase):
    def _rocm_side_effect(self, cmd, **kwargs):
        """Return nvidia-smi FileNotFoundError, then valid rocm-smi JSON."""
        if "nvidia-smi" in cmd:
            raise FileNotFoundError
        # rocm-smi call
        payload = {"GPU[0]": {"VRAM Total Memory (B)": "8589934592"}}
        return _completed(0, json.dumps(payload))

    def test_rocm_parses_vram(self):
        with (
            patch("hierocode.runtime.gpu.subprocess.run", side_effect=self._rocm_side_effect),
            patch("hierocode.runtime.gpu.platform.system", return_value="Linux"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="x86_64"),
        ):
            info = probe_gpu()
        self.assertTrue(info.has_gpu)
        self.assertAlmostEqual(info.vram_gb, 8.0, places=3)
        self.assertEqual(info.backend, "amd")

    def test_linux_no_gpu_falls_back(self):
        with (
            patch(
                "hierocode.runtime.gpu.subprocess.run",
                side_effect=FileNotFoundError,
            ),
            patch("hierocode.runtime.gpu.platform.system", return_value="Linux"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="x86_64"),
        ):
            info = probe_gpu()
        self.assertFalse(info.has_gpu)
        self.assertEqual(info.backend, "none")


class TestRobustness(unittest.TestCase):
    def test_never_raises(self):
        with (
            patch(
                "hierocode.runtime.gpu.subprocess.run",
                side_effect=OSError("disk full"),
            ),
            patch("hierocode.runtime.gpu.platform.system", return_value="Linux"),
            patch("hierocode.runtime.gpu.platform.machine", return_value="x86_64"),
        ):
            info = probe_gpu()
        self.assertIsInstance(info, GPUInfo)
        self.assertFalse(info.has_gpu)
        self.assertEqual(info.backend, "none")


if __name__ == "__main__":
    unittest.main()
