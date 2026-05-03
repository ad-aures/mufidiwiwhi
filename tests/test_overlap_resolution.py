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

"""Tests for the global overlap resolver."""

from __future__ import annotations

from mufidiwiwhi.transcribe import resolve_segment_overlaps


def _seg(start, end, text="", speaker="A", words=None):
    return {
        "id": 0,
        "seek": start,
        "start": float(start),
        "end": float(end),
        "speaker": speaker,
        "text": text,
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
        "words": words,
        "language": "en",
    }


def _word(text, start, end, prob=1.0):
    return {"word": text, "start": float(start), "end": float(end), "probability": prob}


def test_no_overlap_returns_input_unchanged():
    a = _seg(0, 3, "hello")
    b = _seg(5, 8, "world")
    out = resolve_segment_overlaps([a, b])
    assert [s["start"] for s in out] == [0.0, 5.0]
    assert [s["end"] for s in out] == [3.0, 8.0]
    assert [s["text"] for s in out] == ["hello", "world"]


def test_rule_a_partial_overlap_splits_at_midpoint():
    # [0, 5] and [3, 8]: midpoint of (5, 3) is 4.0
    a = _seg(0, 5, "alpha", speaker="A")
    b = _seg(3, 8, "beta", speaker="B")
    out = resolve_segment_overlaps([a, b])
    assert len(out) == 2
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 4.0
    assert out[1]["start"] == 4.0
    assert out[1]["end"] == 8.0


def test_rule_b_full_containment_splits_outer_into_two_pieces():
    # Outer [0, 10] with 10 words evenly spaced; inner [3, 7].
    # mid = 5, ratio = 5/10 = 0.5 -> xa = 5
    words = [_word(f" w{i}", i, i + 1, prob=0.9) for i in range(10)]
    outer = _seg(0, 10, " ".join(w["word"] for w in words), speaker="A", words=words)
    inner = _seg(3, 7, "inner", speaker="B")
    out = resolve_segment_overlaps([outer, inner])
    assert len(out) == 3
    # Time order: outer-left, inner, outer-right.
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 3.0
    assert out[1]["start"] == 3.0
    assert out[1]["end"] == 7.0
    assert out[2]["start"] == 7.0
    assert out[2]["end"] == 10.0
    # Word distribution: 5 words on each outer half.
    assert len(out[0]["words"]) == 5
    assert len(out[2]["words"]) == 5
    # First piece has w0..w4, second has w5..w9.
    assert [w["word"].strip() for w in out[0]["words"]] == ["w0", "w1", "w2", "w3", "w4"]
    assert [w["word"].strip() for w in out[2]["words"]] == ["w5", "w6", "w7", "w8", "w9"]


def test_rule_b_lopsided_inner_skews_word_count():
    # Outer [0, 10] with 10 words; inner [1, 3]. mid = 2,
    # ratio = 2/10 = 0.2 -> xa = 2.
    words = [_word(f" w{i}", i, i + 1, prob=0.9) for i in range(10)]
    outer = _seg(0, 10, " ".join(w["word"] for w in words), speaker="A", words=words)
    inner = _seg(1, 3, "x", speaker="B")
    out = resolve_segment_overlaps([outer, inner])
    assert len(out) == 3
    assert len(out[0]["words"]) == 2
    assert len(out[2]["words"]) == 8


def test_cross_speaker_overlap_is_resolved():
    """Per user spec: overlap rules apply globally, including
    across speakers (cross-speaker simultaneous-speech is cut)."""
    a = _seg(0, 5, "alpha", speaker="A")
    b = _seg(3, 8, "beta", speaker="B")
    out = resolve_segment_overlaps([a, b])
    assert out[0]["speaker"] == "A"
    assert out[1]["speaker"] == "B"
    assert out[0]["end"] == out[1]["start"] == 4.0


def test_chained_overlaps_each_pair_resolved():
    # Three segments: [0, 4], [3, 8], [7, 12]. After rule (a) on
    # the first pair: [0, 3.5], [3.5, 8]. Then rule (a) on the
    # second pair: [3.5, 7.5], [7.5, 12].
    s1 = _seg(0, 4, "one")
    s2 = _seg(3, 8, "two")
    s3 = _seg(7, 12, "three")
    out = resolve_segment_overlaps([s1, s2, s3])
    assert len(out) == 3
    starts = [s["start"] for s in out]
    ends = [s["end"] for s in out]
    assert starts == [0.0, 3.5, 7.5]
    assert ends == [3.5, 7.5, 12.0]


def test_zero_length_segments_filtered_out():
    a = _seg(0, 0, "noop")
    b = _seg(1, 4, "real")
    out = resolve_segment_overlaps([a, b])
    assert len(out) == 1
    assert out[0]["text"] == "real"


def test_rule_a_preserves_all_words_across_midpoint_cut():
    # s1 = [0, 5] with 5 words, last word at [4, 5] sits AFTER the
    # midpoint cut at 4.0. s2 = [3, 8] with 5 words, first at [3, 4]
    # sits BEFORE the midpoint cut. Both must survive: timestamps
    # may be clamped, text never dropped.
    w1 = [_word(f" a{i}", i, i + 1, prob=0.9) for i in range(5)]
    w2 = [_word(f" b{i}", 3 + i, 4 + i, prob=0.9) for i in range(5)]
    s1 = _seg(0, 5, " ".join(w["word"] for w in w1), speaker="A", words=w1)
    s2 = _seg(3, 8, " ".join(w["word"] for w in w2), speaker="B", words=w2)
    out = resolve_segment_overlaps([s1, s2])
    assert len(out) == 2
    # Cut at midpoint of (5, 3) -> 4.0.
    assert out[0]["end"] == 4.0
    assert out[1]["start"] == 4.0
    # No text loss: every original word survives in its segment.
    assert [w["word"].strip() for w in out[0]["words"]] == [
        "a0", "a1", "a2", "a3", "a4"
    ]
    assert [w["word"].strip() for w in out[1]["words"]] == [
        "b0", "b1", "b2", "b3", "b4"
    ]
