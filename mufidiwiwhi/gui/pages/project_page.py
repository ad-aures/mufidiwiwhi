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

"""Project tab, GNOME-49 / libadwaita styled.

Per-run inputs only: speaker table, output settings, and a small
hint about the corrector. Three cards, in order: Speakers, Output,
Correction. Run + Copy CLI buttons sit in a bottom row outside any
card.
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ... import aup3
from ...core import (
    CorrectionConfig,
    RunConfig,
    SpeakerInput,
    _suggest_output_filename,  # type: ignore
)
from ..settings import GlobalSettings, ProjectSnapshot, SettingsManager
from ..widgets.adwaita import (
    ActionRow,
    CardFrame,
    EntryRow,
    SectionHeader,
)
from ..widgets.audio_drop import AUDIO_EXTENSIONS, filter_audio_paths
from ..widgets.icons import IconButton, labeled_icon_button
from ..widgets.speaker_list import SpeakerListView
from ..widgets.speaker_table import SpeakerTableModel


def _set_tone(widget: QWidget, tone: str) -> None:
    widget.setProperty("tone", tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class ProjectPage(QWidget):
    """Per-run inputs: speakers, output, and correction hint."""

    runRequested = pyqtSignal(object)  # emits a RunConfig

    def __init__(self, settings: SettingsManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._build_ui()
        self._connect()
        self._restore_last_session()
        self._refresh_run_button()
        self._refresh_correction_hints()

    # ----- UI ------------------------------------------------------------
    def _build_ui(self) -> None:
        from PyQt6.QtWidgets import QScrollArea

        # Scrollable content: only the configuration cards live
        # here. The action button row stays fixed at the bottom of
        # the page so Copy CLI / Start transcription are always
        # visible regardless of how far the user has scrolled.
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(0, 0, 12, 0)
        outer.setSpacing(0)

        self._add_section(outer, self.tr("Speakers"), self._speakers_card())
        self._add_section(outer, self.tr("Output"), self._output_card())
        self._add_section(outer, self.tr("Correction"), self._correction_card())
        outer.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        page_layout.addWidget(scroll, stretch=1)
        page_layout.addSpacing(16)
        page_layout.addLayout(self._bottom_row())

    def _add_section(
        self,
        outer: QVBoxLayout,
        title: str,
        card: QWidget,
        stretch: int = 0,
    ) -> None:
        if outer.count() > 0:
            outer.addSpacing(32)
        outer.addWidget(SectionHeader(title))
        outer.addSpacing(8)
        outer.addWidget(card, stretch=stretch)

    # ----- Speakers card -------------------------------------------------
    def _speakers_card(self) -> QWidget:
        self.speaker_model = SpeakerTableModel(self)
        self.speaker_view = SpeakerListView(self.speaker_model, self)
        self.speaker_view.setToolTip(
            self.tr(
                "Drop audio files here, or use the 'Add speaker' row. "
                "Each row has Edit and Remove buttons on the right."
            )
        )
        self.speaker_view.addRequested.connect(self._add_speaker)
        return self.speaker_view

    # ----- Output card ---------------------------------------------------
    def _output_card(self) -> CardFrame:
        card = CardFrame()

        self.output_dir_edit = QLineEdit()
        out_btn = IconButton(
            "folder.svg", tooltip=self.tr("Choose output directory...")
        )
        out_btn.clicked.connect(self._pick_output_dir)
        card.add_row(
            EntryRow(
                self.tr("Output directory"),
                self.output_dir_edit,
                extra_buttons=[out_btn],
            )
        )

        self.output_filename_edit = QLineEdit()
        card.add_row(
            EntryRow(
                self.tr("Output filename"),
                self.output_filename_edit,
            )
        )

        # Chapters combo: disabled by default. Populated with one
        # entry per Audacity labeltrack on `.aup3` import.
        self.chapters_combo = QComboBox()
        self.chapters_combo.setEnabled(False)
        self.chapters_combo.addItem(self.tr("(disabled)"), userData=None)
        # Stash extracted labeltracks (full label list) keyed by
        # combo index so we can write the chapter JSON without
        # re-opening the .aup3 at run time.
        self._chapter_tracks: list[list[dict]] = []
        card.add_row(
            EntryRow(
                self.tr("Chapters"),
                self.chapters_combo,
            )
        )

        # Format pill toggles, GNOME-style.
        self.fmt_srt = self._make_pill_toggle("SRT", self.tr("SubRip subtitles."))
        self.fmt_vtt = self._make_pill_toggle("VTT", self.tr("WebVTT subtitles."))
        self.fmt_txt = self._make_pill_toggle(
            "TXT", self.tr("Plain text, no timestamps.")
        )
        self.fmt_json = self._make_pill_toggle(
            "JSON",
            self.tr("Full result with timestamps and confidences."),
        )
        self.fmt_tsv = self._make_pill_toggle(
            "TSV", self.tr("Tab-separated values.")
        )
        formats_row = QFrame()
        formats_row.setProperty("class", "row")
        fr_layout = QVBoxLayout(formats_row)
        fr_layout.setContentsMargins(16, 12, 16, 12)
        fr_layout.setSpacing(4)
        title = QLabel(self.tr("Formats"))
        title.setProperty("class", "rowTitle")
        fr_layout.addWidget(title)
        subtitle = QLabel(
            self.tr(
                "Which transcript files are produced. At least one must be selected."
            )
        )
        subtitle.setProperty("class", "rowSubtitle")
        subtitle.setWordWrap(True)
        fr_layout.addWidget(subtitle)
        pill_row = QHBoxLayout()
        pill_row.setContentsMargins(0, 6, 0, 0)
        pill_row.setSpacing(8)
        for pill in (
            self.fmt_srt,
            self.fmt_vtt,
            self.fmt_txt,
            self.fmt_json,
            self.fmt_tsv,
        ):
            pill_row.addWidget(pill)
        pill_row.addStretch(1)
        fr_layout.addLayout(pill_row)
        card.add_row(formats_row)
        return card

    @staticmethod
    def _make_pill_toggle(label: str, tooltip: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setProperty("class", "pillToggle")
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            btn.setToolTip(tooltip)
        return btn

    # ----- Correction card -----------------------------------------------
    def _correction_card(self) -> CardFrame:
        card = CardFrame()

        info_row = QFrame()
        info_row.setProperty("class", "row")
        info_layout = QHBoxLayout(info_row)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info = QLabel(
            self.tr(
                "Phonetic and confidence-based correction runs "
                "automatically when a dictionary is configured on the "
                "Settings tab."
            )
        )
        info.setWordWrap(True)
        info_layout.addWidget(info, stretch=1)
        card.add_row(info_row)

        hint_row = QFrame()
        hint_row.setProperty("class", "row")
        hint_layout = QHBoxLayout(hint_row)
        hint_layout.setContentsMargins(16, 12, 16, 12)
        self.phonetic_hint = QLabel()
        self.phonetic_hint.setWordWrap(True)
        self.phonetic_hint.setIndent(0)
        _set_tone(self.phonetic_hint, "muted")
        hint_layout.addWidget(self.phonetic_hint, stretch=1)
        card.add_row(hint_row)

        return card

    def _bottom_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 8, 0, 0)
        row.setSpacing(8)
        row.addStretch(1)
        self.copy_cli_btn = labeled_icon_button(
            self.tr("Copy CLI command"),
            "clipboard-text-svgrepo-com.svg",
            tooltip=self.tr(
                "Copy the equivalent `mufidiwiwhi` shell command (with "
                "every option matching the current GUI state) to the "
                "clipboard. Useful for scripting and batch runs."
            ),
        )
        row.addWidget(self.copy_cli_btn)
        self.run_btn = labeled_icon_button(
            self.tr("Start transcription"),
            "play-circle-svgrepo-com.svg",
            primary=True,
            tooltip=self.tr("Begin processing. The Run tab will show progress."),
        )
        self.run_btn.setEnabled(False)
        row.addWidget(self.run_btn)
        return row

    # ----- wiring --------------------------------------------------------
    def _connect(self) -> None:
        self.speaker_model.dataChanged.connect(self._on_table_changed)
        self.speaker_model.rowsInserted.connect(self._on_table_changed)
        self.speaker_model.rowsRemoved.connect(self._on_table_changed)
        self.output_dir_edit.textChanged.connect(self._refresh_run_button)
        self.output_filename_edit.textChanged.connect(self._snapshot)
        for cb in (
            self.fmt_srt, self.fmt_vtt, self.fmt_txt, self.fmt_json, self.fmt_tsv,
        ):
            cb.toggled.connect(self._snapshot)
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.copy_cli_btn.clicked.connect(self._on_copy_cli_clicked)

    # ----- actions -------------------------------------------------------
    def _add_speaker(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("Select audio file(s)"),
            "",
            self.tr("Audio files (")
            + " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)
            + ");;"
            + self.tr("All files (*)"),
        )
        for path in filter_audio_paths(paths):
            if aup3.is_aup3(path):
                self._import_aup3(path)
                continue
            default_name = os.path.splitext(os.path.basename(path))[0]
            speaker, ok = QInputDialog.getText(
                self,
                self.tr("Speaker name"),
                self.tr("Speaker name for {0}:").format(os.path.basename(path)),
                text=default_name,
            )
            if ok and speaker.strip():
                self.speaker_model.add_row(speaker.strip(), path)

    def _import_aup3(self, path: str) -> None:
        """Extract every Audacity track to a sibling WAV file and
        add one speaker row per track. Shows a modal progress
        dialog with a per-track status line, since extraction can
        take a while on large projects."""
        basename = os.path.basename(path)
        progress = QProgressDialog(
            self.tr(
                "Reading audio blocks from {0}.\nThis may take a while..."
            ).format(basename),
            "",  # cancel button text (suppressed below)
            0,
            0,  # range 0..0 = indeterminate / busy spinner
            self,
        )
        progress.setWindowTitle(self.tr("Importing Audacity project"))
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        QApplication.processEvents()

        def _on_log(msg: str) -> None:
            progress.setLabelText(msg)
            QApplication.processEvents()

        try:
            speakers = aup3.extract_aup3(path, log=_on_log)
        except Exception as exc:
            progress.close()
            QMessageBox.warning(
                self,
                self.tr("Cannot read .aup3"),
                self.tr("Could not extract {0}: {1}").format(basename, exc),
            )
            return
        progress.close()
        for s in speakers:
            self.speaker_model.add_row(s.speaker, s.file_path)
        # Override the auto-derived output filename and dir so the
        # transcript is named after the source `.aup3` (not after a
        # longest-common-prefix of the per-track WAV names) and
        # lands next to the project, not inside the `_tracks/`
        # subfolder.
        aup3_dir = os.path.dirname(os.path.abspath(path)) or "."
        aup3_stem = os.path.splitext(os.path.basename(path))[0]
        self.output_dir_edit.setText(aup3_dir)
        self.output_filename_edit.setText(aup3_stem)
        # Populate chapters from any labeltracks in the project.
        try:
            label_tracks = aup3.extract_label_tracks(path)
        except Exception:
            label_tracks = []
        self._populate_chapters(label_tracks)

    def _populate_chapters(self, label_tracks: list[dict]) -> None:
        """Fill the Chapters combo from a list of `{"name", "labels"}`
        dicts. Empty list resets the combo to its default disabled
        state. First labeltrack is auto-selected."""
        self.chapters_combo.blockSignals(True)
        try:
            self.chapters_combo.clear()
            self._chapter_tracks = []
            self.chapters_combo.addItem(self.tr("(disabled)"), userData=None)
            for i, lt in enumerate(label_tracks):
                if not lt.get("labels"):
                    continue
                name = lt.get("name") or self.tr("(unnamed)")
                count = len(lt["labels"])
                label = self.tr("{0} ({1} markers)").format(name, count)
                self.chapters_combo.addItem(label, userData=i)
                self._chapter_tracks.append(lt["labels"])
            if self._chapter_tracks:
                self.chapters_combo.setEnabled(True)
                # Index 0 is "(disabled)", so first labeltrack is at 1.
                self.chapters_combo.setCurrentIndex(1)
            else:
                self.chapters_combo.setEnabled(False)
                self.chapters_combo.setCurrentIndex(0)
        finally:
            self.chapters_combo.blockSignals(False)

    def _selected_chapters(self) -> Optional[list[dict]]:
        """Return the labels of the currently-selected chapter
        track, or None when chapters are disabled."""
        idx = self.chapters_combo.currentData()
        if idx is None:
            return None
        if 0 <= idx < len(self._chapter_tracks):
            return list(self._chapter_tracks[idx])
        return None

    def _pick_output_dir(self) -> None:
        start = self.output_dir_edit.text() or ""
        path = QFileDialog.getExistingDirectory(
            self, self.tr("Output directory"), start
        )
        if path:
            self.output_dir_edit.setText(path)

    def _on_table_changed(self, *_args) -> None:
        self._auto_outdir()
        self._auto_filename()
        self._refresh_run_button()
        self._snapshot()

    def _auto_filename(self) -> None:
        """Always overwrite the filename field from the current input list."""
        paths = self.speaker_model.file_paths()
        if not paths:
            return
        self.output_filename_edit.setText(_suggest_output_filename(paths))

    def _auto_outdir(self) -> None:
        """Always overwrite the output dir from the first input file."""
        paths = self.speaker_model.file_paths()
        if not paths:
            return
        first = paths[0]
        out_dir = os.path.dirname(os.path.abspath(first)) or "."
        # When all inputs sit in a `<name>_tracks/` subfolder
        # produced by aup3 extraction, step up so the transcript
        # lands next to the original .aup3 file.
        if out_dir.endswith("_tracks"):
            out_dir = os.path.dirname(out_dir) or "."
        self.output_dir_edit.setText(out_dir)

    def _refresh_run_button(self) -> None:
        ok = (
            self.speaker_model.rowCount() >= 1
            and bool(self.output_dir_edit.text().strip())
        )
        self.run_btn.setEnabled(ok)

    def _selected_formats(self) -> list[str]:
        out: list[str] = []
        if self.fmt_srt.isChecked():
            out.append("srt")
        if self.fmt_vtt.isChecked():
            out.append("vtt")
        if self.fmt_txt.isChecked():
            out.append("txt")
        if self.fmt_json.isChecked():
            out.append("json")
        if self.fmt_tsv.isChecked():
            out.append("tsv")
        return out or ["srt"]

    # ----- correction hints (refreshed when the page becomes visible) ---
    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        self._refresh_correction_hints()

    def _refresh_correction_hints(self) -> None:
        gs = self._settings.load_global()
        if gs.dictionary_path:
            self.phonetic_hint.setText(
                self.tr("Using dictionary: {0}").format(gs.dictionary_path)
            )
            _set_tone(self.phonetic_hint, "muted")
        else:
            self.phonetic_hint.setText(
                self.tr(
                    "No dictionary set. Open the Settings tab and "
                    "click 'Edit...' to create one (correction will be "
                    "skipped until then)."
                )
            )
            _set_tone(self.phonetic_hint, "warn")

    # ----- Run -----------------------------------------------------------
    def _on_run_clicked(self) -> None:
        cfg = self.build_run_config()
        if cfg is None:
            return
        self._snapshot()
        self.runRequested.emit(cfg)

    def _on_copy_cli_clicked(self) -> None:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        from ..cli_export import runconfig_to_cli

        cfg = self.build_run_config()
        if cfg is None:
            QMessageBox.information(
                self,
                self.tr("Nothing to copy"),
                self.tr(
                    "Add at least one speaker file and an output "
                    "directory before copying the CLI command."
                ),
            )
            return
        cmd = runconfig_to_cli(cfg)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(cmd)
        QMessageBox.information(
            self,
            self.tr("CLI command copied"),
            self.tr(
                "The equivalent shell command has been copied to your "
                "clipboard. You can paste it into a terminal to "
                "reproduce this run without the GUI:\n\n{0}"
            ).format(cmd),
        )

    def build_run_config(self) -> Optional[RunConfig]:
        gs = self._settings.load_global()
        speakers = [
            SpeakerInput(speaker=r["speaker"], file_path=r["file_path"])
            for r in self.speaker_model.rows()
            if r["speaker"].strip() and r["file_path"].strip()
        ]
        if not speakers:
            return None
        out_dir = self.output_dir_edit.text().strip()
        if not out_dir:
            out_dir = os.path.dirname(os.path.abspath(speakers[0].file_path)) or "."
        out_filename = self.output_filename_edit.text().strip() or None
        formats = self._selected_formats()

        # Phonetic correction is always on when a dictionary is set;
        # there is no toggle anymore.
        correction = None
        if gs.dictionary_path:
            correction = CorrectionConfig(
                dictionary_path=gs.dictionary_path or None,
                phonetic_lang=gs.phonetic_lang,
                phonetic_lang_secondary=gs.phonetic_lang_secondary or None,
                low_conf=gs.correct_low_conf,
                high_conf=gs.correct_high_conf,
                edit_distance_threshold=gs.correct_edit_distance,
                hunspell_primary=gs.hunspell_primary or None,
                hunspell_secondary=gs.hunspell_secondary or None,
                use_hunspell=gs.use_hunspell,
                hunspell_threshold=gs.hunspell_threshold,
            )

        return RunConfig(
            speakers=speakers,
            model_name=gs.model_name,
            model_dir=gs.model_dir or None,
            device=gs.device,
            compute_type=gs.compute_type,
            language=gs.default_language or None,
            # Always ask for word timestamps in the GUI so the live log
            # can colour each word by Whisper confidence.
            word_timestamps=True,
            output_dir=out_dir,
            output_formats=formats,
            output_filename=out_filename,
            correction=correction,
            conf_thresholds=(
                gs.conf_threshold_excellent,
                gs.conf_threshold_high,
                gs.conf_threshold_mid,
                gs.conf_threshold_low,
            ),
            chapters=self._selected_chapters(),
        )

    # ----- session snapshot ---------------------------------------------
    def _snapshot(self) -> None:
        snap = ProjectSnapshot(
            speakers=self.speaker_model.rows(),
            output_dir=self.output_dir_edit.text().strip(),
            output_filename=self.output_filename_edit.text().strip(),
            output_formats=self._selected_formats(),
        )
        self._settings.save_last_session(snap)

    def _restore_last_session(self) -> None:
        """Restore the non-input parts of the previous session.

        Input files are intentionally NOT restored: every launch
        starts with an empty speaker list. Output dir / filename are
        also left empty, since they will be auto-filled the moment
        the user adds an input file.
        """
        snap = self._settings.load_last_session()
        if snap is None:
            self._formats_to_ui(("srt",))
            return
        self._formats_to_ui(tuple(snap.output_formats or ("srt",)))

    def _formats_to_ui(self, formats: tuple[str, ...]) -> None:
        s = set(formats) if formats else {"srt"}
        self.fmt_srt.setChecked("srt" in s)
        self.fmt_vtt.setChecked("vtt" in s)
        self.fmt_txt.setChecked("txt" in s)
        self.fmt_json.setChecked("json" in s)
        self.fmt_tsv.setChecked("tsv" in s)

    # ----- CLI passthrough ----------------------------------------------
    def apply_cli_overrides(
        self,
        speakers: list[SpeakerInput],
        dictionary: Optional[str],
        output_dir: Optional[str],
    ) -> None:
        if speakers:
            self.speaker_model.set_rows(
                [{"speaker": s.speaker, "file_path": s.file_path} for s in speakers]
            )
        if dictionary:
            # Persist only the dictionary path; do NOT bulk-save the
            # whole GlobalSettings (that path could overwrite freshly
            # set fields with stale ones).
            cp = self._settings._read_parser()
            if not cp.has_section("global"):
                cp.add_section("global")
            cp.set("global", "dictionary_path", dictionary)
            self._settings._write_parser(cp)
        if output_dir:
            self.output_dir_edit.setText(output_dir)
        else:
            self._auto_outdir()
        self._auto_filename()
        self._refresh_run_button()
        self._refresh_correction_hints()
