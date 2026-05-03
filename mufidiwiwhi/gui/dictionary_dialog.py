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

"""Modal editor for the post-correction dictionary.

A `QDialog` with a `QPlainTextEdit`, an explanatory header, and a
file-picker so the user can save to disk. The dictionary format is
the one consumed by `mufidiwiwhi.correct.load_dictionary`: one entry
per line, `#` for comments, blank lines ignored, multi-word entries
allowed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .widgets.icons import labeled_icon_button


_EXAMPLE_TEMPLATE = """\
# Mufidiwiwhi correction dictionary
# - One entry per line.
# - Lines starting with # are comments.
# - Multi-word entries are allowed (max 4 words per entry).
# - Entries are matched phonetically: spelling does not have to be exact.
# - The line you write here is the EXACT replacement text used in the
#   transcript, so use the casing and punctuation you want.

# --- Examples (delete or edit as needed) ---
Castopod
OpenRAG
Mufidiwiwhi
Podcasting 2.0
Free Software Foundation
Saint-Étienne
"""


def default_dictionary_path() -> str:
    """Where to save a fresh dictionary by default."""
    base = Path.home() / ".config" / "mufidiwiwhi"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "dictionary.txt")


def sort_dictionary_text(text: str) -> str:
    """Return `text` with comment lines preserved at the top and
    entries sorted alphabetically (case-insensitive), de-duplicated.

    Behaviour:
      * Leading comment lines (those before the first entry) are kept
        verbatim, in their original order, at the top of the file.
      * Blank lines are dropped except for a single blank line that
        separates the header comments from the sorted entries.
      * Comments that appear later in the file (between or after
        entries) are kept too, but sorted along with the entries so
        they end up grouped at the end. This avoids losing them while
        keeping the result deterministic.
      * Entries are sorted case-insensitively, accent-aware via the
        Unicode normalised form, with duplicates collapsed.
    """
    import unicodedata

    if not text.strip():
        return ""

    leading_comments: list[str] = []
    seen_entry = False
    later_lines: list[str] = []
    entries: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if not seen_entry:
                leading_comments.append(line)
            else:
                later_lines.append(line)
            continue
        seen_entry = True
        entries.append(line)
    # Append any later (post-entry) comments so we do not lose them.
    entries.extend(later_lines)

    def sort_key(s: str) -> tuple[str, str]:
        norm = unicodedata.normalize("NFKD", s)
        return (norm.casefold(), s)

    seen: set[str] = set()
    deduped: list[str] = []
    for entry in sorted(entries, key=sort_key):
        if entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)

    parts: list[str] = []
    parts.extend(leading_comments)
    if leading_comments and deduped:
        parts.append("")
    parts.extend(deduped)
    return "\n".join(parts) + "\n"


class DictionaryEditorDialog(QDialog):
    """Edit (or create) a dictionary file in place."""

    def __init__(
        self,
        path: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Edit dictionary"))
        self.resize(640, 480)
        self._path = path or ""
        self._build_ui()
        self._load()

    # ----- UI ------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        header = QLabel(
            self.tr(
                "One entry per line. Use '#' for comments. Multi-word "
                "entries (up to 4 words) are allowed. Matching is "
                "phonetic, so spelling does not have to be exact, but "
                "the line below is the exact replacement that will "
                "appear in the transcript."
            )
        )
        header.setWordWrap(True)
        outer.addWidget(header)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel(self.tr("File:")))
        self.path_edit = QLineEdit(self._path or default_dictionary_path())
        self.path_edit.setToolTip(
            self.tr("Where the dictionary will be saved when you click Save.")
        )
        path_row.addWidget(self.path_edit, stretch=1)
        self.browse_btn = labeled_icon_button(
            self.tr("Browse..."),
            "folder-open-svgrepo-com.svg",
            tooltip=self.tr("Choose a different file."),
        )
        self.browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(self.browse_btn)
        outer.addLayout(path_row)

        self.editor = QPlainTextEdit()
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.editor.setFont(font)
        self.editor.setTabChangesFocus(True)
        self.editor.setPlaceholderText(_EXAMPLE_TEMPLATE)
        outer.addWidget(self.editor, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_btn = labeled_icon_button(
            self.tr("Cancel"),
            "close-square-svgrepo-com.svg",
            tooltip=self.tr("Discard changes and close."),
        )
        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn = labeled_icon_button(
            self.tr("Save"),
            "file-download-svgrepo-com.svg",
            primary=True,
            tooltip=self.tr("Write the contents to the file above and close."),
        )
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._on_save)
        button_row.addWidget(self.cancel_btn)
        button_row.addWidget(self.save_btn)
        outer.addLayout(button_row)

    # ----- I/O -----------------------------------------------------------
    def _load(self) -> None:
        self._load_path(self._path)

    def _load_path(self, path: Optional[str]) -> None:
        """Read `path`, sort and de-duplicate its contents, write the
        sorted version back to disk if it differs from what was on
        disk, and display it in the editor.

        If the path does not exist or cannot be read, fall back to the
        example template.
        """
        if not path or not os.path.isfile(path):
            self.editor.setPlainText(_EXAMPLE_TEMPLATE)
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            QMessageBox.warning(
                self,
                self.tr("Cannot read"),
                self.tr("Could not read {0}: {1}").format(path, exc),
            )
            self.editor.setPlainText(_EXAMPLE_TEMPLATE)
            return
        sorted_text = sort_dictionary_text(raw)
        if sorted_text and sorted_text != raw:
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(sorted_text)
            except OSError:
                # If writing the sorted version back fails (read-only
                # filesystem, permissions, etc.) we still display the
                # sorted text in the editor; saving from the editor
                # later will retry the write.
                pass
        self.editor.setPlainText(sorted_text or raw)

    def _on_browse(self) -> None:
        start = self.path_edit.text() or default_dictionary_path()
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Dictionary file"),
            start,
            self.tr("Text files (*.txt);;All files (*.*)"),
        )
        if path:
            self.path_edit.setText(path)
            self._load_path(path)

    def _on_save(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(
                self,
                self.tr("No path"),
                self.tr("Please choose a file path before saving."),
            )
            return
        sorted_text = sort_dictionary_text(self.editor.toPlainText())
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(sorted_text)
        except OSError as exc:
            QMessageBox.warning(
                self,
                self.tr("Cannot save"),
                self.tr("Could not write {0}: {1}").format(path, exc),
            )
            return
        # reflect the sorted text back into the editor so the user
        # sees what was written
        if sorted_text != self.editor.toPlainText():
            self.editor.setPlainText(sorted_text)
        self._path = path
        self.accept()

    # ----- public --------------------------------------------------------
    def saved_path(self) -> str:
        return self._path
