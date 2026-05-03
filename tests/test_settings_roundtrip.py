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

"""Tests for INI persistence (global + last-session snapshot + geometry)."""

from __future__ import annotations

import os

import pytest

from mufidiwiwhi.gui.settings import (
    GlobalSettings,
    ProjectSnapshot,
    SettingsManager,
)


def _ini_path(tmp_path) -> str:
    return str(tmp_path / "mufidiwiwhi-settings.ini")


def test_global_settings_round_trip(tmp_path):
    path = _ini_path(tmp_path)
    mgr = SettingsManager(path)
    custom = GlobalSettings(
        model_name="medium",
        device="cpu",
        compute_type="int8",
        default_language="fr",
        dictionary_path="/tmp/d.txt",
        phonetic_lang="fr",
        phonetic_lang_secondary="en",
        correct_low_conf=0.4,
        correct_high_conf=0.9,
        correct_edit_distance=3,
        hunspell_primary="/usr/share/hunspell/fr_FR",
        use_hunspell=True,
        conf_threshold_excellent=0.97,
    )
    mgr.save_global(custom)

    mgr2 = SettingsManager(_ini_path(tmp_path))
    loaded = mgr2.load_global()
    assert loaded.model_name == "medium"
    assert loaded.device == "cpu"
    assert loaded.correct_low_conf == pytest.approx(0.4)
    assert loaded.dictionary_path == "/tmp/d.txt"
    assert loaded.hunspell_primary == "/usr/share/hunspell/fr_FR"
    assert loaded.use_hunspell is True
    assert loaded.conf_threshold_excellent == pytest.approx(0.97)


def test_session_snapshot_round_trip(tmp_path):
    mgr = SettingsManager(_ini_path(tmp_path))
    snap = ProjectSnapshot(
        speakers=[
            {"speaker": "Alice", "file_path": "/tmp/a.wav"},
            {"speaker": "Bob", "file_path": "/tmp/b.wav"},
        ],
        output_dir="/tmp/out",
        output_filename="podcast",
        output_formats=["srt", "txt"],
    )
    mgr.save_last_session(snap)

    mgr2 = SettingsManager(_ini_path(tmp_path))
    loaded = mgr2.load_last_session()
    assert loaded is not None
    assert loaded.speakers[0]["speaker"] == "Alice"
    assert loaded.output_filename == "podcast"
    assert loaded.output_formats == ["srt", "txt"]


def test_clear_last_session(tmp_path):
    mgr = SettingsManager(_ini_path(tmp_path))
    mgr.save_last_session(ProjectSnapshot(speakers=[{"speaker": "X", "file_path": "x"}]))
    assert mgr.load_last_session() is not None
    mgr.clear_last_session()
    assert mgr.load_last_session() is None


def test_geometry_does_not_clobber_global_settings(tmp_path):
    """Regression for the actual bug: writing geometry must not
    overwrite global settings written between two save_geometry calls.
    Previously QSettings cached the file and rewrote everything on
    sync, undoing configparser updates.
    """
    path = _ini_path(tmp_path)
    mgr = SettingsManager(path)
    gs = GlobalSettings(model_name="medium")
    mgr.save_global(gs)

    # Write geometry through a second SettingsManager (mimicking a
    # later closeEvent in the same session).
    mgr2 = SettingsManager(path)
    mgr2.save_geometry("main", b"\x01\x02\x03\xff\xfe")

    # Re-read globals: must still be 'medium'.
    mgr3 = SettingsManager(path)
    assert mgr3.load_global().model_name == "medium"
    assert mgr3.load_geometry("main") == b"\x01\x02\x03\xff\xfe"
