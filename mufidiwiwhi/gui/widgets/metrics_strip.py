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

"""Metrics strip widget shown above the Run-tab log.

Renders system metrics as small "stat bubble" pills, one per
metric. Each bubble gets a usage-based border color: default below
25 %, accent blue below 50 %, accent green below 75 %, red at and
above 75 %. CPU and GPU bubble groups are kept together as units;
when both groups fit on one row they share it, else they stack on
two rows.

Temperature percentages are computed from an observed min / max
range that's persisted to the user's config so the colors keep
adapting across launches.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..settings import SettingsManager
from ..sysmon import SysMetrics, read_metrics


def _short_gpu_name(full_name: str) -> str:
    """Trim long GPU model strings down to a recognisable short form."""
    if not full_name:
        return ""
    m = re.search(r"(GTX|RTX|GT|MX|A[0-9]+)\s*([A-Za-z0-9 ]+)", full_name)
    if m:
        suffix = m.group(2).strip()
        suffix = re.sub(r"\b(Laptop|Mobile|GPU|Super|TI|Ti)\b", "", suffix)
        suffix = " ".join(suffix.split())
        return f"{m.group(1)} {suffix}".strip()
    return full_name


def _tone_for_pct(pct: Optional[float]) -> str:
    """Return the QSS tone class for a 0..1 usage fraction.
    `None` (uninitialised range, missing reading) -> default tone.
    """
    if pct is None:
        return ""
    if pct < 0.25:
        return ""
    if pct < 0.50:
        return "info"
    if pct < 0.75:
        return "ok"
    return "hot"


def _bubble(
    label: str,
    value: str,
    tone: str = "",
    parent: Optional[QWidget] = None,
) -> QFrame:
    """A pill QFrame: muted small label + bolder value."""
    f = QFrame(parent)
    f.setProperty("class", "statBubble")
    if tone:
        f.setProperty("tone", tone)
    f.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    h = QHBoxLayout(f)
    h.setContentsMargins(12, 6, 12, 6)
    h.setSpacing(6)
    if label:
        label_w = QLabel(label)
        label_w.setProperty("class", "statBubbleLabel")
        h.addWidget(label_w)
    value_w = QLabel(value)
    value_w.setProperty("class", "statBubbleValue")
    h.addWidget(value_w)
    return f


class MetricsStrip(QFrame):
    """Adaptive pill strip: one row when CPU + GPU groups fit,
    two rows otherwise. Each bubble's border encodes usage."""

    def __init__(
        self, settings: SettingsManager, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        # In-memory snapshot of the observed temp range. Loaded from
        # settings so the very first reading on a fresh launch can
        # already be coloured against past observations.
        gs = self._settings.load_global()
        self._cpu_temp_min = gs.cpu_temp_min
        self._cpu_temp_max = gs.cpu_temp_max
        self._gpu_temp_min = gs.gpu_temp_min
        self._gpu_temp_max = gs.gpu_temp_max

        self.setObjectName("metricsStrip")
        self.setFrameShape(QFrame.Shape.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        self._outer = outer

        # Stable two-row container. We never replace these widgets;
        # we just clear / refill their inner layouts and toggle row2
        # visibility. Stable parents avoid layout-invalidation
        # pitfalls when QVBoxLayout children get swapped on the fly.
        self._row1 = QWidget(self)
        self._row1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._row1_layout = QHBoxLayout(self._row1)
        self._row1_layout.setContentsMargins(0, 0, 0, 0)
        self._row1_layout.setSpacing(self._BUBBLE_GAP_PX)
        outer.addWidget(self._row1)

        self._row2 = QWidget(self)
        self._row2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._row2_layout = QHBoxLayout(self._row2)
        self._row2_layout.setContentsMargins(0, 0, 0, 0)
        self._row2_layout.setSpacing(self._BUBBLE_GAP_PX)
        outer.addWidget(self._row2)

        # Cache of the last batch of bubble widgets so resize events
        # can re-arrange without re-reading metrics.
        self._cpu_bubbles: list[QFrame] = []
        self._gpu_bubbles: list[QFrame] = []

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    # ----- start / stop kept for backward compatibility ---------------
    def start(self) -> None:
        # The timer is always running so observed-temp ranges keep
        # accumulating; calling start()/stop() is a no-op.
        pass

    def stop(self) -> None:
        pass

    # ----- update -----------------------------------------------------
    def refresh(self) -> None:
        m = read_metrics()
        self._update_temp_bounds(m)
        self._cpu_bubbles = list(self._build_cpu_bubbles(m))
        self._gpu_bubbles = list(self._build_gpu_bubbles(m))
        self._rearrange()
        self.setVisible(bool(self._cpu_bubbles or self._gpu_bubbles))

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Re-arrange existing bubbles when the strip's width changes;
        # we don't re-read metrics here.
        self._rearrange()

    # ----- layout arrangement ----------------------------------------
    _GROUP_GAP_PX = 16  # px between CPU group and GPU group on one row
    _BUBBLE_GAP_PX = 8

    def _rearrange(self) -> None:
        # Empty both row layouts. The bubbles in `self._cpu_bubbles`
        # / `self._gpu_bubbles` are fresh objects from `refresh`;
        # the widgets we're removing here are stale ones from the
        # previous refresh. Detach them so they don't paint at old
        # positions, and schedule deletion for the next event loop
        # tick (so we don't accidentally delete a bubble we're
        # still using to compute layout).
        stale: list[QWidget] = []
        for layout in (self._row1_layout, self._row2_layout):
            while layout.count() > 0:
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None and w not in self._cpu_bubbles \
                        and w not in self._gpu_bubbles:
                    w.setParent(None)
                    stale.append(w)
        for w in stale:
            w.deleteLater()
        if not self._cpu_bubbles and not self._gpu_bubbles:
            self._row1.setVisible(False)
            self._row2.setVisible(False)
            return
        cpu_w = self._group_width(self._cpu_bubbles)
        gpu_w = self._group_width(self._gpu_bubbles)
        mid_gap = self._GROUP_GAP_PX if (cpu_w and gpu_w) else 0
        avail = self.width()
        one_row = avail > 0 and (cpu_w + mid_gap + gpu_w) <= avail
        if one_row:
            self._fill_row(
                self._row1_layout,
                self._cpu_bubbles + self._gpu_bubbles,
                cpu_count=len(self._cpu_bubbles),
                mid_gap=mid_gap,
            )
            self._row1.setVisible(True)
            self._row2.setVisible(False)
        else:
            self._fill_row(self._row1_layout, self._cpu_bubbles)
            self._fill_row(self._row2_layout, self._gpu_bubbles)
            self._row1.setVisible(bool(self._cpu_bubbles))
            self._row2.setVisible(bool(self._gpu_bubbles))
        # Force immediate layout so freshly added bubbles get their
        # final geometry on this paint pass instead of after the
        # next event-loop tick.
        self._row1_layout.activate()
        self._row2_layout.activate()

    def _fill_row(
        self,
        layout: QHBoxLayout,
        bubbles: list[QFrame],
        cpu_count: int = 0,
        mid_gap: int = 0,
    ) -> None:
        for i, b in enumerate(bubbles):
            if mid_gap and i == cpu_count and cpu_count > 0:
                layout.addSpacing(mid_gap - self._BUBBLE_GAP_PX)
            layout.addWidget(b, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)

    def _group_width(self, bubbles: list[QFrame]) -> int:
        if not bubbles:
            return 0
        widths = [b.sizeHint().width() for b in bubbles]
        return sum(widths) + self._BUBBLE_GAP_PX * (len(bubbles) - 1)

    # ----- temp bounds + persistence ---------------------------------
    def _update_temp_bounds(self, m: SysMetrics) -> None:
        changed: dict[str, float] = {}
        if m.cpu_temp_c is not None:
            v = float(m.cpu_temp_c)
            if self._cpu_temp_min < 0 or v < self._cpu_temp_min:
                self._cpu_temp_min = v
                changed["cpu_temp_min"] = v
            if v > self._cpu_temp_max:
                self._cpu_temp_max = v
                changed["cpu_temp_max"] = v
        if m.gpu_temp_c is not None:
            v = float(m.gpu_temp_c)
            if self._gpu_temp_min < 0 or v < self._gpu_temp_min:
                self._gpu_temp_min = v
                changed["gpu_temp_min"] = v
            if v > self._gpu_temp_max:
                self._gpu_temp_max = v
                changed["gpu_temp_max"] = v
        if changed:
            self._persist(changed)

    def _persist(self, changed: dict[str, float]) -> None:
        try:
            cp = self._settings._read_parser()
        except Exception:
            return
        if not cp.has_section("global"):
            cp.add_section("global")
        for key, value in changed.items():
            cp.set("global", key, str(value))
        try:
            self._settings._write_parser(cp)
        except Exception:
            pass

    @staticmethod
    def _temp_pct(current: float, lo: float, hi: float) -> Optional[float]:
        if lo < 0 or hi < 0:
            return None
        span = hi - lo
        if span <= 0.0:
            return 0.0
        return max(0.0, min(1.0, (current - lo) / span))

    # ----- builders --------------------------------------------------
    # Space-padded fixed widths so each bubble's text never changes
    # length: percentages render as 3 chars wide ("  4%"..."100%"),
    # GB values as XX.X ("4.4 / 30.0 GB", padded to 4 chars each
    # with leading spaces), temperatures as 3 chars (" 70°C").
    # Combined with the monospace value font this gives stable
    # pixel widths without the visual weight of leading zeros.
    def _build_cpu_bubbles(self, m: SysMetrics) -> Iterable[QFrame]:
        if m.cpu_percent is not None:
            tone = _tone_for_pct(max(0.0, min(1.0, m.cpu_percent / 100.0)))
            yield _bubble(
                self.tr("CPU"), f"{m.cpu_percent:3.0f}%", tone, self
            )
        if m.ram_used_gb is not None and m.ram_total_gb is not None:
            tone = _tone_for_pct(
                m.ram_used_gb / m.ram_total_gb if m.ram_total_gb else None
            )
            yield _bubble(
                self.tr("RAM"),
                f"{m.ram_used_gb:4.1f} / {m.ram_total_gb:4.1f} GB",
                tone,
                self,
            )
        if m.cpu_temp_c is not None:
            pct = self._temp_pct(
                m.cpu_temp_c, self._cpu_temp_min, self._cpu_temp_max
            )
            yield _bubble(
                self.tr("CPU temp"),
                f"{m.cpu_temp_c:3.0f}°C",
                _tone_for_pct(pct),
                self,
            )

    def _build_gpu_bubbles(self, m: SysMetrics) -> Iterable[QFrame]:
        if m.gpu_percent is not None:
            tone = _tone_for_pct(max(0.0, min(1.0, m.gpu_percent / 100.0)))
            yield _bubble(
                self.tr("GPU load"), f"{m.gpu_percent:3.0f}%", tone, self
            )
        if m.vram_used_gb is not None and m.vram_total_gb is not None:
            tone = _tone_for_pct(
                m.vram_used_gb / m.vram_total_gb if m.vram_total_gb else None
            )
            yield _bubble(
                self.tr("VRAM"),
                f"{m.vram_used_gb:4.1f} / {m.vram_total_gb:4.1f} GB",
                tone,
                self,
            )
        if m.gpu_temp_c is not None:
            pct = self._temp_pct(
                m.gpu_temp_c, self._gpu_temp_min, self._gpu_temp_max
            )
            yield _bubble(
                self.tr("GPU temp"),
                f"{m.gpu_temp_c:3.0f}°C",
                _tone_for_pct(pct),
                self,
            )
