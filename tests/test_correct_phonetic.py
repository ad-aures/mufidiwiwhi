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


def test_function_word_absorbed_when_joining_improves_match(tmp_path):
    """Regression: Whisper transcribes a French proper noun whose
    initial vowel got mishead as a separate preposition ("Aeris"
    spoken aloud lands as "à Eris" in the transcript). The matcher
    must absorb the "à" — the function-word guard is normally what
    blocks collapsed matches from eating sentence connectives, but
    here joining the window IS strictly the better surface match to
    "Aeris" than just the tail "Eris", so it must be allowed.
    """
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("Aeris\n", encoding="utf-8")
    seg = _seg([
        _w(" à", 0.40, 0.0, 0.3),
        _w(" Eris", 0.40, 0.3, 1.0),
    ])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="fr",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    assert "Aeris" in out[0]["text"]
    # And critically: the leading "à" must be gone (i.e. it was
    # absorbed, not left dangling).
    assert "à Aeris" not in out[0]["text"]
    assert "àAeris" not in out[0]["text"]


def test_function_word_kept_when_tail_already_matches(tmp_path):
    """Counter-example to the case above: when Whisper correctly
    transcribed the proper noun and the leading word is a true
    preposition ("à Aeris" meaning "to Aeris"), joining does NOT
    improve the surface match, so the function-word guard fires
    and only the tail is considered — yielding an identity
    replacement that leaves the "à" untouched.
    """
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("Aeris\n", encoding="utf-8")
    seg = _seg([
        _w(" à", 0.40, 0.0, 0.3),
        _w(" Aeris", 0.40, 0.3, 1.0),
    ])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="fr",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    assert "à Aeris" in out[0]["text"]


def test_clitic_entry_matches_uncontracted_preposition(tmp_path):
    """Regression: a dictionary entry with an apostrophe-clitic
    ("D'Échirolles") must match Whisper outputs that wrote the
    elided preposition as a separate un-contracted function word
    ("Des Chirolle"). The fix is at indexing time: clitic-prefixed
    entries are also registered under the phonetic of their full
    un-elided surface ("déchirolles"), which collides with the
    phonetic of "deschirolle".
    """
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("D'Échirolles\n", encoding="utf-8")
    seg = _seg([
        _w(" Des", 0.40, 0.0, 0.3),
        _w(" Chirolle", 0.40, 0.3, 1.0),
    ])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="fr",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    assert "D'Échirolles" in out[0]["text"]
    assert "Des Échirolles" not in out[0]["text"]


def test_french_determiner_synthesized_before_vowel_initial_entry(tmp_path):
    """Whisper transcribed "la NCT" (the agency is "ANCT"). The dict
    only contains the bare proper-noun form "ANCT". The matcher
    should consume the "la" — because joining yields a strictly
    closer phonetic surface to the entry — AND output the French
    contracted clitic "l'ANCT" instead of either "la ANCT" (stray
    article) or bare "ANCT" (silently eaten article).
    """
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("ANCT\n", encoding="utf-8")
    seg = _seg([
        _w(" la", 0.40, 0.0, 0.3),
        _w(" NCT", 0.40, 0.3, 1.0),
    ])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="fr",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    assert "l'ANCT" in out[0]["text"]
    assert "la ANCT" not in out[0]["text"]


def test_french_determiner_not_synthesized_before_consonant_entry(tmp_path):
    """No French elision before a consonant-initial proper noun:
    "la table" must NOT become "l'table" or eat "la". With dict
    "Table" the leading "la" is a real article and should survive.
    """
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("Table\n", encoding="utf-8")
    seg = _seg([
        _w(" la", 0.40, 0.0, 0.3),
        _w(" table", 0.40, 0.3, 1.0),
    ])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="fr",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    text = out[0]["text"]
    assert " la " in text or text.endswith(" la")
    assert "l'" not in text
    assert "l'table" not in text


def test_short_word_pair_does_not_fuzzy_match_dict_entry(tmp_path):
    """Regression for the over-eager bypass that briefly turned
    bigram windows of short tokens into surprise replacements:
    "ne t" must NOT become "ANCT" just because the fuzzy fallback
    admits "ANCT" as an edit-2 candidate. The entry surface is not
    the tail of the joined input, so the function-word guard fires
    and rejects the absorption.
    """
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("ANCT\n", encoding="utf-8")
    seg = _seg([
        _w(" ne", 0.40, 0.0, 0.3),
        _w(" t",  0.40, 0.3, 1.0),
    ])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="fr",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    text = out[0]["text"]
    assert "ANCT" not in text
    assert " ne t" in text or text.endswith(" ne t")


def test_function_word_guard_still_blocks_pure_connective(tmp_path):
    """Sanity check: the function-word guard must keep working in
    the genuine sentence-connective case. Dict "GAFAM", transcript
    "des GAFAM" — the "des" is a real preposition and the entry
    surface is nowhere near "desgafam"; joining doesn't improve
    the match, so the guard fires and we only replace the
    single-word window "GAFAM" (identity).
    """
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("GAFAM\n", encoding="utf-8")
    seg = _seg([
        _w(" des", 0.40, 0.0, 0.3),
        _w(" GAFAM", 0.40, 0.3, 1.0),
    ])
    cfg = CorrectionConfig(
        dictionary_path=str(dict_path),
        phonetic_lang="fr",
        low_conf=0.5,
        high_conf=0.85,
    )
    out = apply_corrections([seg], cfg)
    # "des" must survive.
    assert "des" in out[0]["text"]
    assert "GAFAM" in out[0]["text"]


def test_input_list_is_not_mutated(tmp_path):
    dict_path = tmp_path / "d.txt"
    dict_path.write_text("OpenRAG\n", encoding="utf-8")
    seg = _seg([_w(" openrag", 0.30, 0, 1.0)])
    original_text = seg["text"]
    cfg = CorrectionConfig(dictionary_path=str(dict_path), phonetic_lang="en")
    apply_corrections([seg], cfg)
    assert seg["text"] == original_text
