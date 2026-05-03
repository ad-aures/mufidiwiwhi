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

"""Unit tests for the Audacity `.aup3` extractor.

End-to-end extraction is delegated to the third-party `aup3` PyPI
package; we don't try to rebuild a fake project file in-process.
The integration test runs against a real `.aup3` fixture when the
lib + a fixture path are available, otherwise it's skipped.
"""

from __future__ import annotations

import os
import wave

import pytest

from mufidiwiwhi import aup3


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_sanitize_basic():
    assert aup3._sanitize("Alice", 0) == "Alice"


def test_sanitize_punctuation_collapses():
    assert aup3._sanitize("Alice 1 / micro left", 0) == "Alice_1_micro_left"


def test_sanitize_empty_falls_back_to_idx():
    assert aup3._sanitize("", 0) == "track_0"
    assert aup3._sanitize("---", 3) == "track_3"


def test_is_aup3_extension_check():
    assert aup3.is_aup3("foo.aup3")
    assert aup3.is_aup3("/abs/path/Project.AUP3")
    assert not aup3.is_aup3("foo.wav")
    assert not aup3.is_aup3("foo.aup")  # Audacity 2 format, not supported


def test_extract_aup3_without_lib_raises_clear_error(tmp_path, monkeypatch):
    """When `aup3` (the PyPI package) isn't importable, surface a
    clear error message instead of a generic ImportError. We check
    the message wording so users know what to install.
    """
    fake = tmp_path / "fake.aup3"
    fake.write_text("not a real aup3")
    import builtins

    real_import = builtins.__import__

    def block_aup3(name, *args, **kwargs):
        if name == "aup3":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_aup3)
    with pytest.raises(RuntimeError, match=r"pip install aup3"):
        aup3.extract_aup3(str(fake))


# ---------------------------------------------------------------------------
# Optional integration test: requires the `aup3` lib AND a real fixture.
# Set MUFIDIWIWHI_TEST_AUP3 to a path of an Audacity 3 project to run it.
# ---------------------------------------------------------------------------


def test_extract_real_fixture(tmp_path):
    pytest.importorskip("aup3")
    pytest.importorskip("scipy.signal")
    fixture = os.environ.get("MUFIDIWIWHI_TEST_AUP3", "")
    if not fixture or not os.path.isfile(fixture):
        pytest.skip(
            "set MUFIDIWIWHI_TEST_AUP3 to a real .aup3 path to enable"
        )
    # Copy the fixture into tmp_path so the extractor's sibling
    # output dir lands there (and gets auto-cleaned by pytest).
    import shutil
    local = tmp_path / os.path.basename(fixture)
    shutil.copyfile(fixture, str(local))
    speakers = aup3.extract_aup3(str(local))
    assert speakers, "expected at least one track"
    for s in speakers:
        assert os.path.isfile(s.file_path)
        with wave.open(s.file_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() > 0
    again = aup3.extract_aup3(str(local))
    assert [s.file_path for s in again] == [s.file_path for s in speakers]
