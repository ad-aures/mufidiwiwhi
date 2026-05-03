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

"""Tests for the explicit RMS-minima chunker."""

from __future__ import annotations

import pytest

pytest.importorskip("pydub")

from pydub import AudioSegment
from pydub.generators import Sine

from mufidiwiwhi.transcribe import (
    _MAX_CHUNK_MS,
    _MIN_CHUNK_MS,
    _SEEK_STEP_MS,
    _iter_chunks_from,
)


def _silence(ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=ms, frame_rate=16_000)


def _tone(ms: int, freq: int = 440) -> AudioSegment:
    # `Sine.to_audio_segment` returns mono int16 by default.
    return Sine(freq).to_audio_segment(duration=ms).set_frame_rate(16_000)


def test_short_audio_is_a_single_chunk():
    audio = _tone(1_500)
    chunks = list(_iter_chunks_from(audio))
    assert len(chunks) == 1
    s, e, _ = chunks[0]
    assert s == 0
    assert e <= len(audio)
    assert e - s <= _MAX_CHUNK_MS


def test_no_audio_lost_between_chunks():
    """Build a 90 s clip with intermittent silence + tones. After
    chunking, every consecutive pair must satisfy
    `next.start_ms == prev.end_ms` once leading silence has been
    skipped on the next chunk - i.e. the only "gap" between
    chunks is silence the chunker chose to drop. We assert by
    confirming the union of chunk spans covers all non-silent
    audio."""
    audio = (
        _tone(15_000)
        + _silence(1_000)
        + _tone(20_000)
        + _silence(800)
        + _tone(25_000)
        + _silence(1_200)
        + _tone(20_000)
    )
    chunks = list(_iter_chunks_from(audio))
    assert chunks, "expected at least one chunk"
    # Each chunk respects the 30 s ceiling.
    for s, e, _ in chunks:
        assert e - s <= _MAX_CHUNK_MS, (s, e)
    # Boundaries: next.start_ms is >= prev.end_ms (cursor never
    # goes backwards) and any gap MUST be silence (no tone audio
    # is dropped).
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt[0] >= prev[1]
        if nxt[0] > prev[1]:
            gap = audio[prev[1] : nxt[0]]
            # Tolerate a tiny bit of energy in the "silence" gap
            # (DC + dither) but it should be way below the tone
            # level.
            assert gap.rms < 100, gap.rms


def test_max_chunk_length_enforced():
    """Build a 70 s pure tone (no silence) and verify the chunker
    still cuts at <= 30 s steps."""
    audio = _tone(70_000)
    chunks = list(_iter_chunks_from(audio))
    assert len(chunks) >= 3
    for s, e, _ in chunks:
        assert e - s <= _MAX_CHUNK_MS


def test_chunks_dont_start_in_silence():
    """Pad the start with 4 seconds of silence; the first chunk
    must start AFTER the silence."""
    audio = _silence(4_000) + _tone(10_000) + _silence(500) + _tone(10_000)
    chunks = list(_iter_chunks_from(audio))
    assert chunks
    first_start, _first_end, _ = chunks[0]
    # Cursor advances in SEEK_STEP_MS (50 ms) ticks, so the chunk
    # should start within one tick of the tone's onset.
    assert first_start >= 4_000 - _SEEK_STEP_MS
    assert first_start <= 4_000 + _SEEK_STEP_MS


def test_chunk_boundary_is_quietest_point_in_lookahead():
    """20s tone, 1s silence, 20s tone. The boundary should land
    inside the silence window (not somewhere in the loud tone)."""
    audio = _tone(20_000) + _silence(1_000) + _tone(20_000)
    chunks = list(_iter_chunks_from(audio))
    # Expect at least 2 chunks; the first ending in the silence
    # interval.
    assert len(chunks) >= 2
    # First chunk reaches beyond MIN_CHUNK_MS so the matcher had
    # a real choice.
    assert chunks[0][1] >= _MIN_CHUNK_MS
    # The boundary should fall inside the silence window
    # (20_000..21_000).
    boundary = chunks[0][1]
    assert 19_900 <= boundary <= 21_100, boundary
