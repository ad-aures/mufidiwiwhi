# Mufidiwiwhi - multi-file diarisation transcription with Whisper.
# (C) 2026 Ad Aures · Benjamin Bellamy <benjamin@podlibre.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Best-effort system-metric probes for the Run-tab status strip.

Every probe is wrapped in try/except so a single missing sensor (no
psutil temperature support, no NVIDIA GPU, etc.) doesn't crash the
GUI: the metric is simply absent from the dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SysMetrics:
    cpu_percent: Optional[float] = None
    cpu_temp_c: Optional[float] = None
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    gpu_name: Optional[str] = None
    gpu_percent: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None

    def has_any_cpu(self) -> bool:
        return any(
            v is not None
            for v in (self.cpu_percent, self.cpu_temp_c, self.ram_used_gb)
        )

    def has_any_gpu(self) -> bool:
        return any(
            v is not None
            for v in (self.gpu_percent, self.gpu_temp_c, self.vram_used_gb)
        )


# Module-level NVML handle: initialised lazily once and reused so we
# don't spend every tick on nvmlInit().
_nvml_handle = None
_nvml_init_attempted = False


def _ensure_nvml():
    global _nvml_handle, _nvml_init_attempted
    if _nvml_init_attempted:
        return _nvml_handle
    _nvml_init_attempted = True
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        _nvml_handle = None
    return _nvml_handle


def read_metrics() -> SysMetrics:
    m = SysMetrics()
    _read_cpu_ram(m)
    _read_cpu_temp(m)
    _read_gpu(m)
    return m


def _read_cpu_ram(m: SysMetrics) -> None:
    try:
        import psutil  # type: ignore

        # cpu_percent(interval=None) is non-blocking and uses the
        # delta since the previous call. The first call returns 0,
        # subsequent calls return the real average.
        m.cpu_percent = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        m.ram_used_gb = vm.used / (1024 ** 3)
        m.ram_total_gb = vm.total / (1024 ** 3)
    except Exception:
        pass


def _read_cpu_temp(m: SysMetrics) -> None:
    try:
        import psutil  # type: ignore

        if not hasattr(psutil, "sensors_temperatures"):
            return
        temps = psutil.sensors_temperatures()
        if not temps:
            return
        # Pick the first plausible CPU sensor: coretemp / k10temp /
        # cpu_thermal / acpitz, in that order.
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps and temps[key]:
                m.cpu_temp_c = float(temps[key][0].current)
                return
        # Fallback: take the first sensor reading we find.
        for entries in temps.values():
            if entries:
                m.cpu_temp_c = float(entries[0].current)
                return
    except Exception:
        pass


def _read_gpu(m: SysMetrics) -> None:
    handle = _ensure_nvml()
    if handle is None:
        return
    try:
        import pynvml  # type: ignore

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        m.gpu_name = name
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        m.gpu_percent = float(util.gpu)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        m.vram_used_gb = mem.used / (1024 ** 3)
        m.vram_total_gb = mem.total / (1024 ** 3)
        try:
            m.gpu_temp_c = float(
                pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        except Exception:
            pass
    except Exception:
        pass


def format_metrics(m: SysMetrics) -> str:
    """Single-line monospace-friendly summary."""
    parts: list[str] = []
    if m.cpu_percent is not None:
        parts.append(f"CPU {m.cpu_percent:5.1f}%")
    if m.ram_used_gb is not None and m.ram_total_gb is not None:
        parts.append(
            f"RAM {m.ram_used_gb:5.1f}/{m.ram_total_gb:5.1f} GB"
        )
    if m.cpu_temp_c is not None:
        parts.append(f"CPU {m.cpu_temp_c:4.0f}°C")
    if m.gpu_percent is not None or m.vram_used_gb is not None:
        parts.append("|")
    if m.gpu_name:
        parts.append(f"GPU ({m.gpu_name})")
    if m.gpu_percent is not None:
        parts.append(f"{m.gpu_percent:5.1f}%")
    if m.vram_used_gb is not None and m.vram_total_gb is not None:
        parts.append(
            f"VRAM {m.vram_used_gb:5.1f}/{m.vram_total_gb:5.1f} GB"
        )
    if m.gpu_temp_c is not None:
        parts.append(f"GPU {m.gpu_temp_c:4.0f}°C")
    return "  ".join(parts) if parts else ""
