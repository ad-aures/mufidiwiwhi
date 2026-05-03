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

"""Bundled non-code resources (icon, examples, tinted SVG helpers)."""

from __future__ import annotations

import os
import re
from importlib.resources import files
from typing import Optional


def icon_path() -> str:
    """Return the absolute path to the bundled application icon.

    Empty string if the resource cannot be located (e.g. when running
    from a stripped distribution).
    """
    return _resource_path("mufidiwiwhi.svg")


def chevron_down_path() -> str:
    """Path to the small chevron used as the QComboBox dropdown indicator."""
    return _resource_path("chevron-down.svg")


def check_white_path() -> str:
    """Path to the white-stroke checkmark used inside checked QCheckBox indicators."""
    return _resource_path("check-white.svg")


def resource_path(name: str) -> str:
    """Public accessor for arbitrary bundled resource files."""
    return _resource_path(name)


def _resource_path(name: str) -> str:
    try:
        path = files(__package__).joinpath(name)
        as_str = str(path)
        return as_str if os.path.isfile(as_str) else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Tinted icon loading
# ---------------------------------------------------------------------------

# Colors hardcoded in the bundled SVGs (svgrepo + our own line-art icons).
_DEFAULT_SVG_COLORS = ("#1C274C", "#1c274c", "#5e5e5e", "#5E5E5E")


def tinted_svg_bytes(name: str, color: str) -> Optional[bytes]:
    """Read the SVG resource `name` and replace any of the well-known
    fill / stroke colors with `color` (a hex string like '#1e1e1e').
    Returns the raw bytes ready for QSvgRenderer, or None if the
    resource is missing.
    """
    path = _resource_path(name)
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    pattern = re.compile("|".join(re.escape(c) for c in _DEFAULT_SVG_COLORS))
    content = pattern.sub(color, content)
    return content.encode("utf-8")
