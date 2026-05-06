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

"""Settings tab, GNOME-49 / libadwaita styled.

Each "group" is a CardFrame containing rows separated by hairlines.
Two row variants are used:
  * EntryRow for editable fields (line edit, combo, spin box).
  * ActionRow for title + control widgets (e.g. format checkboxes).
No QFormLayout, no QGridLayout, no QGroupBox: the previous overlap
bug went away with the abstraction.
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..dictionary_dialog import DictionaryEditorDialog, default_dictionary_path
from ..helpers import (
    compute_types_for_device,
    default_model_dir,
    detect_cuda,
    detect_vram_gb,
    list_downloaded_models,
    model_fits_on_gpu,
    model_vram_gb,
    recommend_compute_type,
    recommend_whisper_model,
)
from ..settings import GlobalSettings, SettingsManager
from ..widgets.adwaita import (
    ActionRow,
    CardFrame,
    EntryRow,
    SectionHeader,
)
from ..widgets.icons import IconButton
from ..widgets.switch import Switch


_MODEL_CHOICES = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v2",
    "large-v3",
    "distil-large-v3",
)

_LANG_CHOICES = ("", "en", "fr", "de", "es", "it", "pt", "nl", "ja", "zh")


def _set_tone(widget: QWidget, tone: str) -> None:
    widget.setProperty("tone", tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class SetupPage(QWidget):
    """Persistent settings, saved to QSettings on every change."""

    def __init__(self, settings: SettingsManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._build_ui()
        self._load()
        self._connect()

        QTimer.singleShot(0, self._refresh_downloaded_models)
        QTimer.singleShot(0, self._refresh_model_recommendation)
        QTimer.singleShot(0, self._refresh_compute_choices)

    # ----- UI ------------------------------------------------------------
    def _build_ui(self) -> None:
        from PyQt6.QtWidgets import QScrollArea

        # Build the content widget. Sections are added via the
        # `_add_section` helper so the spec for "header sits 8-12 px
        # above its card, 32 px between groups" is enforced in one
        # place. The 12 px right padding gives the cards breathing
        # room from the scrollbar.
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(0, 0, 12, 0)
        outer.setSpacing(0)
        self._outer_layout = outer

        self._add_section(self.tr("Transcription engine"), self._whisper_card())
        self._add_section(self.tr("User correction"), self._correction_card())
        self._add_section(
            self.tr("Hunspell dictionary (second opinion)"),
            self._hunspell_card(),
        )
        self._add_section(
            self.tr("Confidence colours"), self._confidence_colours_card()
        )

        from ..widgets.icons import labeled_icon_button

        outer.addSpacing(24)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addStretch(1)
        self.reset_btn = labeled_icon_button(
            self.tr("Reset to defaults"),
            "eraser-svgrepo-com.svg",
            tooltip=self.tr(
                "Restore the canned defaults for every field on this tab."
            ),
        )
        bottom.addWidget(self.reset_btn)
        outer.addLayout(bottom)

        self._config_path_label = QLabel()
        self._config_path_label.setCursor(Qt.CursorShape.PointingHandCursor)
        # Single-click anywhere on the label copies the path. Replace
        # the default mouseReleaseEvent with our copy-to-clipboard
        # handler.
        self._config_path_label.mouseReleaseEvent = (  # type: ignore[assignment]
            lambda _e: self._copy_config_path()
        )
        self._config_path_label.setToolTip(
            self.tr("Click to copy this path to the clipboard.")
        )
        _set_tone(self._config_path_label, "muted")
        self._config_path_label.setText(
            self.tr("Settings are saved to: {0} (click to copy)").format(
                self._settings.config_path()
            )
        )
        outer.addWidget(self._config_path_label)
        outer.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

    def _add_section(self, title: str, card: QWidget) -> None:
        """Insert a section: 32 px gap before the title (except for the
        first section), 8 px between title and its card.
        """
        if self._outer_layout.count() > 0:
            self._outer_layout.addSpacing(32)
        self._outer_layout.addWidget(SectionHeader(title))
        self._outer_layout.addSpacing(8)
        self._outer_layout.addWidget(card)

    # ----- Whisper card --------------------------------------------------
    def _whisper_card(self) -> CardFrame:
        card = CardFrame()

        # Whisper model
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(_MODEL_CHOICES)
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setMaxVisibleItems(16)
        self.model_recommendation_label = QLabel()
        self.model_recommendation_label.setWordWrap(True)
        self.downloaded_label = QLabel()
        self.downloaded_label.setWordWrap(True)
        _set_tone(self.downloaded_label, "muted")
        # Combine recommendation + downloaded into a single subtitle
        # block on the model row so we don't burn rows on hints.
        model_subtitle = QWidget()
        ms_l = QVBoxLayout(model_subtitle)
        ms_l.setContentsMargins(0, 0, 0, 0)
        ms_l.setSpacing(2)
        ms_l.addWidget(self.model_recommendation_label)
        ms_l.addWidget(self.downloaded_label)
        # ActionRow with subtitle widget: emulate by building manually.
        card.add_row(
            self._action_row_with_widget_subtitle(
                self.tr("Whisper model"),
                model_subtitle,
                [self.model_combo],
            )
        )

        # Device
        cuda_available, cuda_label = detect_cuda()
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"] if cuda_available else ["cpu"])
        self.device_combo.setMaxVisibleItems(16)
        card.add_row(
            ActionRow(
                self.tr("Device"),
                subtitle=cuda_label,
                controls=[self.device_combo],
            )
        )

        # Compute type
        self.compute_combo = QComboBox()
        self.compute_combo.setEditable(False)
        self.compute_combo.setMaxVisibleItems(16)
        self.compute_recommendation_label = QLabel()
        self.compute_recommendation_label.setWordWrap(True)
        compute_row = self._action_row_with_widget_subtitle(
            self.tr("Compute type"),
            self.compute_recommendation_label,
            [self.compute_combo],
        )
        card.add_row(compute_row)

        # Language
        self.lang_combo = QComboBox()
        self.lang_combo.setEditable(True)
        self.lang_combo.addItems(_LANG_CHOICES)
        self.lang_combo.setMinimumWidth(140)
        self.lang_combo.setMaxVisibleItems(16)
        card.add_row(
            ActionRow(
                self.tr("Language"),
                subtitle=self.tr(
                    "ISO code (en, fr...). Leave empty to auto-detect."
                ),
                controls=[self.lang_combo],
            )
        )

        # Model cache dir (text field => EntryRow)
        self.model_dir_edit = QLineEdit()
        model_dir_btn = IconButton(
            "add-folder-svgrepo-com.svg", tooltip=self.tr("Choose model cache directory...")
        )
        model_dir_btn.clicked.connect(
            lambda: self._pick_directory(
                self.model_dir_edit, self.tr("Model cache directory")
            )
        )
        card.add_row(
            EntryRow(
                self.tr("Model cache dir"),
                self.model_dir_edit,
                extra_buttons=[model_dir_btn],
            )
        )
        return card

    def _action_row_with_widget_subtitle(
        self, title: str, subtitle_widget: QWidget, controls: list[QWidget]
    ) -> "ActionRow":
        """ActionRow variant that takes a QWidget as its subtitle so we
        can place a multi-line label / multi-paragraph hint where a
        bare string would not work. Built by hand to match the
        adwaita.py rendering rules.
        """
        from PyQt6.QtWidgets import QFrame

        row = QFrame()
        row.setProperty("class", "row")
        h = QHBoxLayout(row)
        h.setContentsMargins(16, 12, 16, 12)
        h.setSpacing(12)
        block = QVBoxLayout()
        block.setContentsMargins(0, 0, 0, 0)
        block.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setProperty("class", "rowTitle")
        block.addWidget(title_lbl)
        # Decorate the subtitle widget so it picks up muted styling
        # by walking its QLabel descendants.
        for lbl in subtitle_widget.findChildren(QLabel):
            if not lbl.property("class"):
                lbl.setProperty("class", "rowSubtitle")
        block.addWidget(subtitle_widget)
        h.addLayout(block, stretch=1)
        for c in controls:
            h.addWidget(c)
        return row

    # ----- Correction card -----------------------------------------------
    def _correction_card(self) -> CardFrame:
        card = CardFrame()

        self.dictionary_edit = QLineEdit()
        dict_browse_btn = IconButton(
            "add-folder-svgrepo-com.svg", tooltip=self.tr("Choose dictionary file...")
        )
        dict_browse_btn.clicked.connect(self._pick_dictionary)
        self.dict_edit_btn = IconButton(
            "pen-new-square-svgrepo-com.svg",
            tooltip=self.tr("Edit dictionary in built-in editor"),
        )
        self.dict_edit_btn.clicked.connect(self._open_dictionary_editor)
        card.add_row(
            EntryRow(
                self.tr("User Dictionary"),
                self.dictionary_edit,
                extra_buttons=[dict_browse_btn, self.dict_edit_btn],
            )
        )

        self.phon_lang_combo = QComboBox()
        self.phon_lang_combo.addItems(["auto", "fr", "en"])
        self.phon_lang_combo.setMaxVisibleItems(16)
        card.add_row(
            ActionRow(
                self.tr("Phonetic language"),
                subtitle=self.tr("Algorithm used to compare how words sound."),
                controls=[self.phon_lang_combo],
            )
        )

        self.phon_lang2_combo = QComboBox()
        self.phon_lang2_combo.addItems(["", "fr", "en"])
        self.phon_lang2_combo.setMaxVisibleItems(16)
        card.add_row(
            ActionRow(
                self.tr("Secondary phonetic language"),
                subtitle=self.tr("Optional fallback for mixed-language podcasts."),
                controls=[self.phon_lang2_combo],
            )
        )

        self.low_conf_spin = self._make_threshold_spin(0.50)
        card.add_row(
            ActionRow(
                self.tr("Low confidence threshold"),
                subtitle=self.tr(
                    "Below this Whisper confidence, replace with any "
                    "phonetic match in the dictionary."
                ),
                controls=[self.low_conf_spin],
            )
        )
        self.high_conf_spin = self._make_threshold_spin(0.95)
        card.add_row(
            ActionRow(
                self.tr("High confidence threshold"),
                subtitle=self.tr(
                    "Above this confidence, never replace."
                ),
                controls=[self.high_conf_spin],
            )
        )
        self.edit_dist_spin = QSpinBox()
        self.edit_dist_spin.setRange(0, 10)
        card.add_row(
            ActionRow(
                self.tr("Edit distance threshold"),
                subtitle=self.tr(
                    "Max letter edits for a mid-confidence replacement."
                ),
                controls=[self.edit_dist_spin],
            )
        )
        return card

    # ----- Hunspell card -------------------------------------------------
    def _hunspell_card(self) -> CardFrame:
        from ...correct_hunspell import (
            list_installed_dictionaries,
        )

        card = CardFrame()

        self.hunspell_primary_edit = QLineEdit()
        hs_browse_btn = IconButton(
            "add-folder-svgrepo-com.svg", tooltip=self.tr("Choose primary Hunspell dictionary...")
        )
        hs_browse_btn.clicked.connect(
            lambda: self._pick_hunspell(self.hunspell_primary_edit)
        )
        card.add_row(
            EntryRow(
                self.tr("Hunspell primary"),
                self.hunspell_primary_edit,
                extra_buttons=[hs_browse_btn],
            )
        )

        self.hunspell_secondary_edit = QLineEdit()
        hs2_browse_btn = IconButton(
            "add-folder-svgrepo-com.svg", tooltip=self.tr("Choose secondary Hunspell dictionary...")
        )
        hs2_browse_btn.clicked.connect(
            lambda: self._pick_hunspell(self.hunspell_secondary_edit)
        )
        card.add_row(
            EntryRow(
                self.tr("Hunspell secondary"),
                self.hunspell_secondary_edit,
                extra_buttons=[hs2_browse_btn],
            )
        )

        self.use_hunspell_check = Switch()
        self.use_hunspell_check.setToolTip(
            self.tr(
                "When on (default), the corrector consults Hunspell "
                "to decide which words look like real language. "
                "Turn off to fall back to phonetic-only."
            )
        )
        card.add_row(
            ActionRow(
                self.tr("Enabled"),
                subtitle=self.tr(
                    "Apply when a dictionary is loaded."
                ),
                controls=[self.use_hunspell_check],
            )
        )

        self.hunspell_threshold_spin = QDoubleSpinBox()
        self.hunspell_threshold_spin.setRange(0.0, 1.0)
        self.hunspell_threshold_spin.setSingleStep(0.05)
        self.hunspell_threshold_spin.setDecimals(2)
        self.hunspell_threshold_spin.setFixedWidth(90)
        card.add_row(
            ActionRow(
                self.tr("Correction threshold"),
                subtitle=self.tr(
                    "Words below this Whisper confidence are auto-"
                    "corrected with Hunspell's first suggestion."
                ),
                controls=[self.hunspell_threshold_spin],
            )
        )

        installed = list_installed_dictionaries()
        hint = self._hint_label()
        if installed:
            hint.setText(
                self.tr("Detected on this system: <b>{0}</b>").format(
                    ", ".join(os.path.basename(p) for p in installed)
                )
            )
            _set_tone(hint, "good")
        else:
            hint.setText(
                self.tr(
                    "No Hunspell dictionaries detected. "
                    "Install <code>hunspell-fr</code> / "
                    "<code>hunspell-en-us</code> (or your language)."
                )
            )
            _set_tone(hint, "warn")
        self.hunspell_detected_label = hint
        # Plain hint as a row: an empty-titled row containing just the
        # label, so it picks up the card's hairline divider.
        from PyQt6.QtWidgets import QFrame as _QFrame

        hint_row = _QFrame()
        hint_row.setProperty("class", "row")
        hint_layout = QHBoxLayout(hint_row)
        hint_layout.setContentsMargins(16, 8, 16, 8)
        hint_layout.addWidget(hint, stretch=1)
        card.add_row(hint_row)
        return card

    # ----- Confidence colours card --------------------------------------
    def _confidence_colours_card(self) -> CardFrame:
        defaults = GlobalSettings()
        card = CardFrame()
        self.conf_excellent_spin = self._make_threshold_spin(
            defaults.conf_threshold_excellent
        )
        self.conf_high_spin = self._make_threshold_spin(
            defaults.conf_threshold_high
        )
        self.conf_mid_spin = self._make_threshold_spin(
            defaults.conf_threshold_mid
        )
        self.conf_low_spin = self._make_threshold_spin(
            defaults.conf_threshold_low
        )
        rows = (
            (self.tr("Excellent (>=)"), self.conf_excellent_spin,
             "#e5ffd5", self.tr("green tint")),
            (self.tr("High (>=)"), self.conf_high_spin,
             None, self.tr("no background")),
            (self.tr("Mid (>=)"), self.conf_mid_spin,
             "#fff6d5", self.tr("pale yellow")),
            (self.tr("Low (>=)"), self.conf_low_spin,
             "#ffe6d5", self.tr("light orange (below this: pink #ffd5d5)")),
        )
        for label_text, spin, swatch_color, swatch_text in rows:
            swatch = QLabel(" " * 4)
            swatch.setFixedSize(28, 22)
            if swatch_color:
                swatch.setStyleSheet(
                    f"background-color: {swatch_color}; "
                    f"border: 1px solid #cdcdd6; border-radius: 4px;"
                )
            else:
                swatch.setStyleSheet(
                    "background-color: #ffffff; "
                    "border: 1px dashed #cdcdd6; border-radius: 4px;"
                )
            card.add_row(
                ActionRow(
                    label_text,
                    subtitle=swatch_text,
                    controls=[spin, swatch],
                )
            )
        return card

    # ----- helpers -------------------------------------------------------
    @staticmethod
    def _hint_label() -> QLabel:
        lbl = QLabel()
        lbl.setWordWrap(True)
        lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        return lbl

    @staticmethod
    def _make_threshold_spin(default: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(2)
        spin.setValue(default)
        spin.setFixedWidth(90)
        return spin

    # ----- wiring --------------------------------------------------------
    def _connect(self) -> None:
        self.model_combo.currentTextChanged.connect(
            lambda txt: self._persist_model(txt)
        )
        self.model_combo.activated.connect(
            lambda _: self._persist_model(self.model_combo.currentText())
        )
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        self.device_combo.activated.connect(
            lambda _: self._on_device_changed(self.device_combo.currentText())
        )
        self.compute_combo.currentTextChanged.connect(self._on_compute_changed)
        self.compute_combo.activated.connect(
            lambda _: self._on_compute_changed(self.compute_combo.currentText())
        )
        self.lang_combo.currentTextChanged.connect(
            lambda txt: self._persist_field("default_language", txt.strip())
        )
        self.model_dir_edit.textChanged.connect(self._on_model_dir_changed)
        self.dictionary_edit.textChanged.connect(
            lambda txt: self._persist_field("dictionary_path", txt.strip())
        )
        self.phon_lang_combo.currentTextChanged.connect(
            lambda txt: self._persist_field(
                "phonetic_lang", txt.strip() or "auto"
            )
        )
        self.phon_lang2_combo.currentTextChanged.connect(
            lambda txt: self._persist_field("phonetic_lang_secondary", txt.strip())
        )
        self.low_conf_spin.valueChanged.connect(
            lambda v: self._persist_field("correct_low_conf", float(v))
        )
        self.high_conf_spin.valueChanged.connect(
            lambda v: self._persist_field("correct_high_conf", float(v))
        )
        self.edit_dist_spin.valueChanged.connect(
            lambda v: self._persist_field("correct_edit_distance", int(v))
        )
        self.hunspell_primary_edit.textChanged.connect(
            lambda txt: self._persist_field("hunspell_primary", txt.strip())
        )
        self.hunspell_secondary_edit.textChanged.connect(
            lambda txt: self._persist_field("hunspell_secondary", txt.strip())
        )
        self.use_hunspell_check.toggled.connect(
            lambda v: self._persist_field("use_hunspell", bool(v))
        )
        self.hunspell_threshold_spin.valueChanged.connect(
            lambda v: self._persist_field("hunspell_threshold", float(v))
        )
        self.conf_excellent_spin.valueChanged.connect(
            lambda v: self._persist_field("conf_threshold_excellent", float(v))
        )
        self.conf_high_spin.valueChanged.connect(
            lambda v: self._persist_field("conf_threshold_high", float(v))
        )
        self.conf_mid_spin.valueChanged.connect(
            lambda v: self._persist_field("conf_threshold_mid", float(v))
        )
        self.conf_low_spin.valueChanged.connect(
            lambda v: self._persist_field("conf_threshold_low", float(v))
        )
        self.reset_btn.clicked.connect(self._reset)

    # ----- per-field direct persistence ---------------------------------
    def _persist_field(self, key: str, value) -> None:
        cp = self._settings._read_parser()
        if not cp.has_section("global"):
            cp.add_section("global")
        cp.set("global", key, "" if value is None else str(value))
        self._settings._write_parser(cp)

    def _persist_model(self, txt: str) -> None:
        value = (txt or "").strip() or "medium"
        self._persist_field("model_name", value)
        self._refresh_model_recommendation()

    # ----- pickers -------------------------------------------------------
    def _pick_directory(self, target: QLineEdit, title: str) -> None:
        start = target.text() or default_model_dir()
        path = QFileDialog.getExistingDirectory(self, title, start)
        if path:
            target.setText(path)

    def _pick_hunspell(self, target: QLineEdit) -> None:
        from ...correct_hunspell import _HUNSPELL_DIRS

        start = target.text().strip() or next(
            (d for d in _HUNSPELL_DIRS if os.path.isdir(d)), ""
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Choose a Hunspell .aff file"),
            start,
            "Hunspell affix (*.aff);;All files (*.*)",
        )
        if path:
            if path.lower().endswith(".aff"):
                path = path[:-4]
            target.setText(path)

    def _copy_config_path(self) -> None:
        from PyQt6.QtWidgets import QApplication, QToolTip
        from PyQt6.QtGui import QCursor

        path = self._settings.config_path()
        clip = QApplication.clipboard()
        if clip is not None:
            clip.setText(path)
        # Brief visual confirmation: a tooltip that fades on its own.
        QToolTip.showText(QCursor.pos(), self.tr("Path copied"), self._config_path_label)

    def _pick_dictionary(self) -> None:
        start = self.dictionary_edit.text() or default_dictionary_path()
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Dictionary file"),
            start,
            "Text files (*.txt);;All files (*.*)",
        )
        if path:
            self.dictionary_edit.setText(path)

    def _open_dictionary_editor(self) -> None:
        current = self.dictionary_edit.text().strip() or default_dictionary_path()
        dlg = DictionaryEditorDialog(current, self)
        if dlg.exec() == DictionaryEditorDialog.DialogCode.Accepted:
            saved = dlg.saved_path()
            if saved:
                self.dictionary_edit.setText(saved)

    # ----- load / save / reset ------------------------------------------
    def _load(self) -> None:
        gs = self._settings.load_global()
        self._set_combo_value(self.model_combo, gs.model_name)
        self._set_combo_value(self.device_combo, gs.device)
        self._refresh_compute_choices(initial_value=gs.compute_type)
        self._set_combo_value(self.lang_combo, gs.default_language)
        self.model_dir_edit.setText(gs.model_dir or default_model_dir())
        self.dictionary_edit.setText(gs.dictionary_path)
        self._set_combo_value(self.phon_lang_combo, gs.phonetic_lang)
        self._set_combo_value(self.phon_lang2_combo, gs.phonetic_lang_secondary)
        self.low_conf_spin.setValue(gs.correct_low_conf)
        self.high_conf_spin.setValue(gs.correct_high_conf)
        self.edit_dist_spin.setValue(gs.correct_edit_distance)
        self.hunspell_primary_edit.setText(gs.hunspell_primary)
        self.hunspell_secondary_edit.setText(gs.hunspell_secondary)
        self.use_hunspell_check.setChecked(gs.use_hunspell)
        self.hunspell_threshold_spin.setValue(gs.hunspell_threshold)
        self.conf_excellent_spin.setValue(gs.conf_threshold_excellent)
        self.conf_high_spin.setValue(gs.conf_threshold_high)
        self.conf_mid_spin.setValue(gs.conf_threshold_mid)
        self.conf_low_spin.setValue(gs.conf_threshold_low)

    @staticmethod
    def _set_combo_value(combo, value: str) -> None:
        idx = combo.findText(value) if value else -1
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setEditText(value or "")

    def _reset(self) -> None:
        self._settings.save_global(GlobalSettings())
        self._load()
        self._refresh_downloaded_models()
        self._refresh_model_recommendation()
        self._refresh_compute_choices()

    # ----- compute type --------------------------------------------------
    def _refresh_compute_choices(self, initial_value: Optional[str] = None) -> None:
        device = self.device_combo.currentText().strip() or "cpu"
        choices = compute_types_for_device(device)
        current = (
            initial_value
            if initial_value is not None
            else self.compute_combo.currentText().strip()
        )
        self.compute_combo.blockSignals(True)
        self.compute_combo.clear()
        self.compute_combo.addItems(choices)
        if current in choices:
            self.compute_combo.setCurrentText(current)
        else:
            self.compute_combo.setCurrentText("auto")
        self.compute_combo.blockSignals(False)
        recommended = recommend_compute_type(device)
        self.compute_recommendation_label.setText(
            self.tr("Recommended for {0}: <b>{1}</b>").format(
                device.upper(), recommended
            )
        )
        _set_tone(self.compute_recommendation_label, "good")

    def _on_compute_changed(self, text: str = "") -> None:
        if not text:
            text = self.compute_combo.currentText()
        self._persist_field("compute_type", text.strip() or "auto")

    def _on_device_changed(self, text: str = "") -> None:
        if not text:
            text = self.device_combo.currentText()
        self._persist_field("device", text.strip() or "cpu")
        self._refresh_compute_choices()
        self._refresh_model_recommendation()
        self._refresh_downloaded_models()

    # ----- model decoration ---------------------------------------------
    def _on_model_dir_changed(self) -> None:
        self._persist_field(
            "model_dir", self.model_dir_edit.text().strip() or default_model_dir()
        )
        self._refresh_downloaded_models()

    def _refresh_downloaded_models(self) -> None:
        downloaded = list_downloaded_models(self.model_dir_edit.text().strip())
        if downloaded:
            self.downloaded_label.setText(
                self.tr("Downloaded on disk: <b>{0}</b>").format(
                    ", ".join(downloaded)
                )
            )
            _set_tone(self.downloaded_label, "good")
        else:
            self.downloaded_label.setText(
                self.tr("No models on disk yet (will download on first run).")
            )
            _set_tone(self.downloaded_label, "muted")
        self._decorate_model_combo(downloaded)

    def _decorate_model_combo(self, downloaded: list[str]) -> None:
        device = self.device_combo.currentText().strip() or "cpu"
        vram_gb = detect_vram_gb() if device == "cuda" else 0.0
        bold = QFont()
        bold.setBold(True)
        plain = QFont()

        # Block signals to avoid setItemData re-emitting currentTextChanged
        # (a Qt6 quirk that would otherwise overwrite the saved model_name
        # with item-0's text).
        was_blocked = self.model_combo.signalsBlocked()
        self.model_combo.blockSignals(True)
        try:
            for i in range(self.model_combo.count()):
                name = self.model_combo.itemText(i)
                font = bold if name in downloaded else plain
                self.model_combo.setItemData(i, font, Qt.ItemDataRole.FontRole)
                # Clear any previous foreground color so QSS controls
                # the item color uniformly.
                self.model_combo.setItemData(i, None, Qt.ItemDataRole.ForegroundRole)
                tooltip_parts: list[str] = []
                need = model_vram_gb(name)
                if need > 0:
                    tooltip_parts.append(
                        self.tr("Approximate VRAM needed: {0:.1f} GB").format(need)
                    )
                if name in downloaded:
                    tooltip_parts.append(self.tr("Already downloaded."))
                if device == "cuda" and not model_fits_on_gpu(name, vram_gb):
                    tooltip_parts.append(
                        self.tr(
                            "Will not fit on the current GPU "
                            "({0:.1f} GB VRAM). Use a smaller model "
                            "or switch device to CPU."
                        ).format(vram_gb)
                    )
                self.model_combo.setItemData(
                    i, "\n".join(tooltip_parts), Qt.ItemDataRole.ToolTipRole
                )
        finally:
            self.model_combo.blockSignals(was_blocked)

    def _refresh_model_recommendation(self) -> None:
        device = self.device_combo.currentText().strip() or "cpu"
        vram_gb = detect_vram_gb() if device == "cuda" else 0.0
        recommended = recommend_whisper_model(device, vram_gb)
        current = self.model_combo.currentText().strip()
        downloaded = list_downloaded_models(self.model_dir_edit.text().strip())
        self._decorate_model_combo(downloaded)

        msgs: list[str] = [
            self.tr("Best pick for your setup: <b>{0}</b>").format(recommended)
        ]
        if device == "cuda" and current and not model_fits_on_gpu(current, vram_gb):
            msgs.append(
                self.tr(
                    "Warning: '{0}' needs about {1:.1f} GB of VRAM, "
                    "your GPU has {2:.1f} GB. It will likely fail to load."
                ).format(current, model_vram_gb(current), vram_gb)
            )
            _set_tone(self.model_recommendation_label, "bad")
        elif current and current != recommended:
            _set_tone(self.model_recommendation_label, "muted")
        else:
            _set_tone(self.model_recommendation_label, "good")
        self.model_recommendation_label.setText("<br/>".join(msgs))
