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

"""Helpers to load bundled SVG resources tinted to a target color.

The SVGs we ship (sidebar nav icons, folder, pencil, chevron) all
have hardcoded fill / stroke colors. To keep them in step with the
text palette we read the file, do a small string substitution on
the known default colors, and render to a `QPixmap` via Qt's SVG
module. The result is wrapped in a `QIcon` for use anywhere Qt
expects one.

Also exposes `IconButton`: a flat, no-border `QToolButton` carrying
a tinted SVG icon. Used in `EntryRow` extras instead of the bare
"Browse..." text buttons for a libadwaita feel.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication, QPushButton, QToolButton

from ...resources import tinted_svg_bytes


def _device_pixel_ratio() -> float:
    """Best-effort current display DPR. Falls back to 1.0 when no
    QApplication exists yet (e.g. unit tests)."""
    app = QApplication.instance()
    if app is None:
        return 1.0
    try:
        return float(app.devicePixelRatio())
    except Exception:
        return 1.0


def tinted_icon(name: str, color: str, size: int = 16) -> QIcon:
    """Return a `QIcon` rendered from the SVG `name` at the given
    pixel size, with all hardcoded fill / stroke colors replaced
    with `color` (e.g. '#1e1e1e').
    """
    pix = tinted_pixmap(name, color, size)
    if pix is None:
        return QIcon()
    return QIcon(pix)


def render_svg_to_pixmap(renderer: QSvgRenderer, size: int) -> QPixmap:
    """Rasterise `renderer` into a square pixmap of logical `size`,
    tagged with the current device-pixel-ratio so it stays sharp on
    HiDPI screens. The explicit `QRectF` target is required: without
    it, `QSvgRenderer.render` falls back to `painter.viewport()`
    (physical pixels) interpreted as logical, blowing the SVG past
    the pixmap bounds and clipping the edges.
    """
    dpr = _device_pixel_ratio()
    phys = max(1, int(round(size * dpr)))
    pix = QPixmap(QSize(phys, phys))
    pix.setDevicePixelRatio(dpr)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter, QRectF(0.0, 0.0, float(size), float(size)))
    painter.end()
    return pix


def tinted_pixmap(name: str, color: str, size: int = 16) -> Optional[QPixmap]:
    data = tinted_svg_bytes(name, color)
    if data is None:
        return None
    renderer = QSvgRenderer(QByteArray(data))
    if not renderer.isValid():
        return None
    return render_svg_to_pixmap(renderer, size)


def labeled_icon_button(
    text: str,
    icon_svg: str,
    primary: bool = False,
    destructive: bool = False,
    tooltip: str = "",
    icon_size: int = 16,
    icon_right_pad: int = 8,
    parent=None,
) -> QPushButton:
    """Build a QPushButton with a tinted SVG icon + label.

    Icon color is automatically chosen so it matches the button
    label color: white for primary / destructive, near-black for
    secondary buttons. `icon_right_pad` is baked into the rendered
    pixmap as transparent space, so the visual gap between the
    icon and the label is wider than Fusion's default ~4 px. The
    pixmap also caches a hover variant so secondary buttons can
    flip to white when the user hovers (like primary / destructive
    buttons already do via QSS).
    """
    btn = _LabeledIconButton(
        text,
        icon_svg=icon_svg,
        primary=primary,
        destructive=destructive,
        icon_size=icon_size,
        icon_right_pad=icon_right_pad,
        parent=parent,
    )
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


class _LabeledIconButton(QPushButton):
    """QPushButton with extra right-pad in the icon and a hover-color
    swap (dark icon -> white) so secondary buttons match the colour
    treatment of primary / destructive ones.
    """

    def __init__(
        self,
        text: str,
        icon_svg: str,
        primary: bool,
        destructive: bool,
        icon_size: int,
        icon_right_pad: int,
        parent=None,
    ) -> None:
        super().__init__(text, parent)
        if primary:
            self.setProperty("primary", True)
            normal_color = "#ffffff"
        elif destructive:
            self.setProperty("destructive", True)
            normal_color = "#ffffff"
        else:
            normal_color = "#1e1e1e"
        self._icon_size = icon_size
        self._icon_right_pad = icon_right_pad
        self._icon_svg = icon_svg
        self._normal_icon = self._build_icon(icon_svg, normal_color)
        # Disabled icon matches the disabled label color (#a8a8a8)
        # from style.py so the glyph fades along with the text on
        # an inactive button.
        self._disabled_icon = self._build_icon(icon_svg, "#a8a8a8")
        # Secondary buttons swap to white on hover (their QSS bg
        # turns accent green so the icon needs to flip too). Primary
        # / destructive icons stay white at all states.
        if primary or destructive:
            self._hover_icon = self._normal_icon
        else:
            self._hover_icon = self._build_icon(icon_svg, "#ffffff")
        if not self._normal_icon.isNull():
            self.setIcon(self._normal_icon)
            self.setIconSize(
                QSize(icon_size + icon_right_pad, icon_size)
            )

    def _build_icon(self, name: str, color: str) -> QIcon:
        base = tinted_pixmap(name, color, self._icon_size)
        if base is None:
            return QIcon()
        # Match the base pixmap's DPR so the padded canvas is also
        # rendered at physical pixel size on HiDPI screens.
        dpr = base.devicePixelRatio() or 1.0
        logical_w = self._icon_size + self._icon_right_pad
        logical_h = self._icon_size
        padded = QPixmap(
            QSize(
                max(1, int(round(logical_w * dpr))),
                max(1, int(round(logical_h * dpr))),
            )
        )
        padded.setDevicePixelRatio(dpr)
        padded.fill(Qt.GlobalColor.transparent)
        p = QPainter(padded)
        p.drawPixmap(0, 0, base)
        p.end()
        return QIcon(padded)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        # Only swap to the hover icon when the button is actually
        # interactive. A disabled button has no hover background
        # change, so flipping its icon to white would leave a white
        # glyph floating on the disabled-grey pill.
        if self.isEnabled() and not self._hover_icon.isNull():
            self.setIcon(self._hover_icon)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._restore_resting_icon()
        super().leaveEvent(event)

    def changeEvent(self, event) -> None:  # type: ignore[override]
        # When the button's enabled-state flips, swap between the
        # normal and disabled icon variants so the glyph greys-out
        # with its label.
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.EnabledChange:
            self._restore_resting_icon()
        super().changeEvent(event)

    def _restore_resting_icon(self) -> None:
        """Set the icon to its non-hover state, picking the disabled
        variant when the button is disabled."""
        target = (
            self._normal_icon if self.isEnabled() else self._disabled_icon
        )
        if not target.isNull():
            self.setIcon(target)


class IconButton(QToolButton):
    """Flat icon-only button: no border, transparent background,
    subtle hover tint. The icon is loaded from a bundled SVG and
    tinted to the body-text color.
    """

    def __init__(
        self,
        icon_svg: str,
        tooltip: str = "",
        size: int = 28,
        icon_size: int = 16,
        color: str = "#1e1e1e",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("class", "iconButton")
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(QSize(size, size))
        self.setIcon(tinted_icon(icon_svg, color, icon_size))
        self.setIconSize(QSize(icon_size, icon_size))
        if tooltip:
            self.setToolTip(tooltip)
