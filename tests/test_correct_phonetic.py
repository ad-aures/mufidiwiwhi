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

"""End-to-end tests of the phonetic correction pass on synthetic segments."""

from __future__ import annotations

import pytest

from mufidiwiwhi.core import CorrectionConfig
from mufidiwiwhi.correct import apply_corrections


def _seg(words):
    text = "".join(w["word"] for w in words)
    return {
        "id": 0,
        "seek": 0.0,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "speaker": "A",
        "text": text,
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
        "words": words,
        "language": "en",
    }


def _w(text, prob, start=0.0, end=1.0):
    return {"word": text, "start": start, "end": end, "probability": prob}


def test_high_confidence_word_is_left_alone(tmp_path):
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("OpenRAG\n", encoding="utf-8")
    seg = _seg([_w(" open", 0.95, 0, 0.5), _w(" rag", 0.95, 0.5, 1.0)])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="en",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    assert "OpenRAG" not in out[0]["text"]


def test_low_confidence_word_is_replaced(tmp_path):
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("OpenRAG\n", encoding="utf-8")
    seg = _seg([_w(" openrag", 0.30, 0, 1.0)])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="en",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    assert "OpenRAG" in out[0]["text"]


def test_numbers_are_skipped(tmp_path):
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("forty two\n", encoding="utf-8")
    seg = _seg([_w(" 42", 0.30, 0, 1.0)])
    cfg = CorrectionConfig(dictionary_path=str(dict_path), phonetic_lang="en")
    out = apply_corrections([seg], cfg)
    # numeric token should not be replaced
    assert "42" in out[0]["text"]


def test_french_clitic_preserved_in_replacement(tmp_path):
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("OpenRAG\n", encoding="utf-8")
    seg = _seg([_w(" l'openrag", 0.30, 0, 1.0)])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="fr",
        phonetic_lang_secondary="en",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    text = out[0]["text"]
    assert "l'OpenRAG" in text


def test_secondary_language_catches_term(tmp_path):
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("Mufidiwiwhi\n", encoding="utf-8")
    seg = _seg([_w(" mufidiwiwhi", 0.30, 0, 1.0)])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="fr",
        phonetic_lang_secondary="en",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    assert "Mufidiwiwhi" in out[0]["text"]


def test_segments_without_words_are_unchanged(tmp_path):
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("Castopod\n", encoding="utf-8")
    seg = _seg([_w(" castopod", 0.3, 0, 1.0)])
    seg["words"] = None
    seg["text"] = " something"
    cfg = CorrectionConfig(dictionary_path=str(dict_path), phonetic_lang="en")
    out = apply_corrections([seg], cfg)
    assert out[0]["text"] == " something"


def test_single_token_entry_matches_two_word_window(tmp_path):
    """Regression: dict has 'OpenRAG' (one token), Whisper transcribes
    as 'open rag' (two words). The collapsed-code variant should make
    that match work even though per-word codes differ.
    """
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("OpenRAG\n", encoding="utf-8")
    seg = _seg([
        _w(" open", 0.40, 0.0, 0.5),
        _w(" rag", 0.40, 0.5, 1.0),
    ])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="en",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    assert "OpenRAG" in out[0]["text"]


def test_input_list_is_not_mutated(tmp_path):
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("OpenRAG\n", encoding="utf-8")
    seg = _seg([_w(" openrag", 0.30, 0, 1.0)])
    original_text = seg["text"]
    cfg = CorrectionConfig(dictionary_path=str(dict_path), phonetic_lang="en")
    apply_corrections([seg], cfg)
    assert seg["text"] == original_text
