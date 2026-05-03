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

"""Tests that merge_segments preserves overlapping speakers' segments."""

from __future__ import annotations

from mufidiwiwhi.transcribe import merge_segments


def _seg(start, end, speaker, text):
    return {
        "id": -1,
        "seek": start,
        "start": start,
        "end": end,
        "speaker": speaker,
        "text": text,
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
        "words": None,
    }


def test_two_speakers_overlap_kept_in_order():
    a = [_seg(0.0, 5.0, "A", "Speaker A talking"), _seg(6.0, 9.0, "A", "...")]
    b = [_seg(2.0, 3.0, "B", "yeah"), _seg(8.0, 10.0, "B", "right")]
    merged = merge_segments([a, b])
    assert [s["speaker"] for s in merged] == ["A", "B", "A", "B"]
    assert merged[1]["speaker"] == "B"
    assert merged[1]["start"] == 2.0 and merged[1]["end"] == 3.0
    # the originally-overlapping B segment is NOT deleted
    assert any(s["text"] == "yeah" for s in merged)


def test_ids_assigned_sequentially():
    a = [_seg(0.0, 1.0, "A", "x")]
    b = [_seg(0.5, 1.5, "B", "y")]
    merged = merge_segments([a, b])
    assert [s["id"] for s in merged] == [0, 1]


def test_empty_input_yields_empty_list():
    assert merge_segments([]) == []
    assert merge_segments([[]]) == []
