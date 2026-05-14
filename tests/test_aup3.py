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
import types
import wave

import numpy as np
import pytest

from mufidiwiwhi import aup3
from mufidiwiwhi.core import SpeakerInput


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
# Regression: per-track sample rate is honoured.
# ---------------------------------------------------------------------------


def _make_elem(tag: str, attrs: dict, children=()):
    """Build a minimal stand-in for `aup3.xml.Element`. The extractor
    only touches `.tag`, `.attrs`, and `.children`; attribute values
    are wrapped as `(type_name, value)` tuples to match the real
    parser. We don't need full XML decoding to exercise `_extract`.
    """
    wrapped = {k: ("Double", v) for k, v in attrs.items()}
    return types.SimpleNamespace(
        tag=tag, attrs=wrapped, children=list(children)
    )


def test_extract_uses_per_track_rate(tmp_path, monkeypatch):
    """When a WaveTrack carries its own `rate` attribute, the
    extractor must use that rate for sample-domain math and for the
    final resample step — NOT the project default. Pre-fix, every
    track was processed at the project rate, which caused 48 kHz
    tracks in 44.1 kHz projects to play ~8.8% slow and made crop
    boundaries (trimLeft/trimRight) land on the wrong samples.
    """
    # Project default = 44100, track override = 48000. The fix is
    # observable iff these differ.
    track_48k = _make_elem("wavetrack", {"name": "A", "rate": 48000.0})
    track_44k = _make_elem("wavetrack", {"name": "B", "rate": 44100.0})
    root = _make_elem(
        "project", {"rate": 44100.0}, children=[track_48k, track_44k]
    )

    fake_db = types.SimpleNamespace(raw_project=root)

    recorded: list[tuple[str, int]] = []

    def fake_track_samples(db, track, rate, log):
        recorded.append(("track_samples", rate))
        # Return a short non-empty buffer so the resample call fires.
        return np.zeros(10, dtype=np.float32)

    def fake_resample(samples, src_rate):
        recorded.append(("resample", src_rate))
        return samples

    def fake_write_wav(path, samples):
        # Touch the file so any later os.path.isfile check is happy;
        # we don't actually need real PCM data for this test.
        open(path, "wb").close()

    monkeypatch.setattr(aup3, "_track_samples", fake_track_samples)
    monkeypatch.setattr(aup3, "_resample_to_16k", fake_resample)
    monkeypatch.setattr(aup3, "_write_wav", fake_write_wav)

    speakers = aup3._extract(fake_db, str(tmp_path), log=None)

    assert [s.speaker for s in speakers] == ["A", "B"]
    # Two tracks, each routed through _track_samples once and
    # _resample_to_16k once, all with their own rate.
    assert recorded == [
        ("track_samples", 48000),
        ("resample", 48000),
        ("track_samples", 44100),
        ("resample", 44100),
    ]


def test_extract_falls_back_to_project_rate_when_track_has_no_rate(
    tmp_path, monkeypatch
):
    """A WaveTrack without an explicit `rate` attribute (defensive
    case — shouldn't happen for valid Audacity projects) must fall
    back to the project's default rate rather than crash."""
    track = _make_elem("wavetrack", {"name": "A"})  # no rate
    root = _make_elem("project", {"rate": 44100.0}, children=[track])

    fake_db = types.SimpleNamespace(raw_project=root)
    rates: list[int] = []

    monkeypatch.setattr(
        aup3,
        "_track_samples",
        lambda db, t, rate, log: (
            rates.append(rate) or np.zeros(10, dtype=np.float32)
        ),
    )
    monkeypatch.setattr(aup3, "_resample_to_16k", lambda s, r: s)
    monkeypatch.setattr(aup3, "_write_wav", lambda p, s: open(p, "wb").close())

    aup3._extract(fake_db, str(tmp_path), log=None)

    assert rates == [44100]


def test_track_samples_uses_track_rate_for_trim_math():
    """Trim boundaries (trimLeft/trimRight in seconds) must be
    converted to samples using the TRACK rate, not the project
    rate. Pre-fix, a 1 s trim at the head of a 48 kHz clip was
    converted as `1 * 44100 = 44100 samples` and 3900 samples of
    cropped audio leaked through.
    """
    # Build a clip with a single block of 96000 float32 samples
    # representing 2 seconds of audio at 48 kHz, and ask for a
    # 1-second left trim. Expected visible length: 1 s * 48000.
    block = _make_elem("waveblock", {"blockid": 1, "start": 0})
    sequence = _make_elem("sequence", {}, children=[block])
    clip = _make_elem(
        "waveclip",
        {"offset": 0.0, "trimLeft": 1.0, "trimRight": 0.0},
        children=[sequence],
    )
    track = _make_elem("wavetrack", {}, children=[clip])

    samples = np.arange(96000, dtype=np.float32)  # 2 s @ 48 kHz
    fake_db = types.SimpleNamespace(get_block=lambda bid: samples)

    out = aup3._track_samples(fake_db, track, 48000, log=None)

    # 1 s trim at 48 kHz drops 48000 samples; 96000 - 48000 = 48000.
    # play_start = offset(0) + trimLeft samples(48000) -> visible
    # lands at 48000..96000 on the global timeline.
    assert len(out) == 96000
    # Pre-trim samples (indices < 48000) must be ZERO.
    assert np.all(out[:48000] == 0.0)
    # Post-trim samples must be the back half of the source.
    np.testing.assert_array_equal(out[48000:], samples[48000:])


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
