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

"""GUI application entry point.

`mufidiwiwhi-gui` console script -> `main()`. Without arguments, opens
an empty project. With CLI-style arguments, reuses the CLI parser to
prefill the speaker table and any other compatible fields.
"""

from __future__ import annotations

import sys
from typing import Optional

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from ..cli import _build_parser, _audio_dicts
from ..core import set_verbose
from ..resources import icon_path
from .main_window import MainWindow
from .style import apply_app_style


def _parse_passthrough(argv: list[str]) -> tuple[list, Optional[str], Optional[str]]:
    """Parse argv with the CLI parser and return GUI-relevant overrides.

    Returns (speakers, dictionary_path, output_dir). Errors are silently
    ignored: the GUI is still useful even with bad CLI args (e.g. just
    `--verbose` with no audio_args).
    """
    if not argv:
        return [], None, None
    import argparse as _argparse

    parser = _build_parser()
    parser.exit_on_error = False  # type: ignore[attr-defined]
    try:
        ns = parser.parse_args(argv)
    except (SystemExit, _argparse.ArgumentError):
        return [], None, None
    try:
        speakers = _audio_dicts(ns.audio_args) if ns.audio_args else []
    except (SystemExit, OSError, ValueError):
        speakers = []
    return speakers, ns.dictionary, ns.output_dir if ns.output_dir != "." else None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `--verbose` toggles every stderr breadcrumb (chunker timing,
    # phonetic-fix, hunspell-suggest pathology, safety-net veto).
    # Detected before Qt boots so even early diagnostics respect
    # the flag.
    set_verbose("--verbose" in argv or "-v" in argv)
    app = QApplication(sys.argv[:1])
    app.setOrganizationName("Ad Aures")
    app.setApplicationName("Mufidiwiwhi")
    apply_app_style(app)
    icon_file = icon_path()
    if icon_file:
        app.setWindowIcon(QIcon(icon_file))
    window = MainWindow()
    if icon_file:
        window.setWindowIcon(QIcon(icon_file))
    speakers, dictionary, output_dir = _parse_passthrough(argv)
    if speakers or dictionary or output_dir:
        window.apply_cli_overrides(speakers, dictionary, output_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
