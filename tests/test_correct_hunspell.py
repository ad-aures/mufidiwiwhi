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

"""Tests for the Hunspell second-opinion."""

from __future__ import annotations

import os
import pytest

from mufidiwiwhi.correct_hunspell import (
    HunspellChecker,
    auto_detect_hunspell,
    list_installed_dictionaries,
)


def test_auto_detect_returns_none_when_no_dictionary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mufidiwiwhi.correct_hunspell._HUNSPELL_DIRS", (str(tmp_path),)
    )
    assert auto_detect_hunspell("fr") is None
    assert auto_detect_hunspell("xx") is None


def test_auto_detect_finds_basename(tmp_path, monkeypatch):
    base = tmp_path / "fr_FR"
    base.with_suffix(".aff").write_text("SET UTF-8\n", encoding="utf-8")
    base.with_suffix(".dic").write_text("1\nbonjour\n", encoding="utf-8")
    monkeypatch.setattr(
        "mufidiwiwhi.correct_hunspell._HUNSPELL_DIRS", (str(tmp_path),)
    )
    out = auto_detect_hunspell("fr")
    assert out == str(base)
    assert auto_detect_hunspell("FR") == str(base)


def test_list_installed_dictionaries(tmp_path, monkeypatch):
    (tmp_path / "fr_FR.aff").write_text("SET UTF-8\n", encoding="utf-8")
    (tmp_path / "fr_FR.dic").write_text("1\nbonjour\n", encoding="utf-8")
    (tmp_path / "en_US.aff").write_text("SET UTF-8\n", encoding="utf-8")
    (tmp_path / "en_US.dic").write_text("1\nhello\n", encoding="utf-8")
    # an .aff without a matching .dic should be ignored
    (tmp_path / "lonely.aff").write_text("SET UTF-8\n", encoding="utf-8")
    monkeypatch.setattr(
        "mufidiwiwhi.correct_hunspell._HUNSPELL_DIRS", (str(tmp_path),)
    )
    found = [os.path.basename(p) for p in list_installed_dictionaries()]
    assert sorted(found) == ["en_US", "fr_FR"]


def test_checker_no_paths_is_no_op():
    checker = HunspellChecker(None, None)
    assert checker.loaded is False
    assert checker.is_misspelled("anything") is False


def test_checker_handles_missing_files_gracefully(tmp_path):
    # Path doesn't exist; load fails, checker reports failed_paths.
    bad = str(tmp_path / "nope")
    checker = HunspellChecker(bad, None)
    assert checker.loaded is False
    assert checker.failed_paths == [bad]
    # is_misspelled stays False so the corrector behaves as if Hunspell
    # weren't configured.
    assert checker.is_misspelled("foo") is False


def test_checker_with_real_dictionary(tmp_path):
    """Smoke-test against a tiny synthetic Hunspell dictionary."""
    spylls = pytest.importorskip("spylls.hunspell")
    base = tmp_path / "tiny"
    base.with_suffix(".aff").write_text(
        "SET UTF-8\n", encoding="utf-8"
    )
    # Hunspell .dic format: first line is a count, then one word per line.
    base.with_suffix(".dic").write_text(
        "3\nbonjour\nmonde\nopen\n", encoding="utf-8"
    )
    checker = HunspellChecker(str(base), None)
    assert checker.loaded is True
    assert checker.is_misspelled("bonjour") is False
    assert checker.is_misspelled("Bonjour") is False  # case fallback
    # Made-up word not in our tiny dictionary -> misspelled.
    assert checker.is_misspelled("openrag") is True
