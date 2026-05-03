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

"""Tests for dictionary loading and phonetic indexing."""

from __future__ import annotations

import pytest

from mufidiwiwhi.core import CorrectionConfig
from mufidiwiwhi.correct import load_dictionary


def test_load_skips_comments_and_blank_lines(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text(
        "# header comment\n\n"
        "Castopod\n"
        "  # indented comment\n"
        "OpenRAG\n",
        encoding="utf-8",
    )
    cfg = CorrectionConfig(dictionary_path=str(p), phonetic_lang="en")
    idx = load_dictionary(str(p), cfg)
    canonicals = sorted({e.canonical for entries in idx.by_code.values() for e in entries})
    assert canonicals == ["Castopod", "OpenRAG"]


def test_multi_word_entries_have_n_greater_than_one(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("Free Software Foundation\nCastopod\n", encoding="utf-8")
    cfg = CorrectionConfig(dictionary_path=str(p), phonetic_lang="en")
    idx = load_dictionary(str(p), cfg)
    by_canon = {e.canonical: e for entries in idx.by_code.values() for e in entries}
    assert by_canon["Free Software Foundation"].n == 3
    assert by_canon["Castopod"].n == 1
    # The matcher needs window headroom beyond the longest entry so a
    # single dictionary token can match Whisper's 2-or-3-word output.
    # `max_n` is the max sliding-window size, which is the longest
    # entry's token count plus a small buffer.
    assert idx.max_n >= 3


def test_pipe_joined_code_for_multi_word(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("Free Software\n", encoding="utf-8")
    cfg = CorrectionConfig(dictionary_path=str(p), phonetic_lang="en")
    idx = load_dictionary(str(p), cfg)
    # the index keys must contain "|" because the entry is multi-word
    assert any("|" in code for code in idx.by_code.keys())


def test_secondary_language_codes_indexed(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("Mufidiwiwhi\n", encoding="utf-8")
    cfg = CorrectionConfig(
        dictionary_path=str(p),
        phonetic_lang="fr",
        phonetic_lang_secondary="en",
    )
    idx = load_dictionary(str(p), cfg)
    # The entry must be reachable via at least one phonetic code.
    # When the optional language-specific libraries are installed,
    # the primary and secondary languages produce different codes
    # and the entry shows up in two buckets; when only the soundex
    # fallback is available both languages collapse to a single
    # bucket. Either is fine; what matters is that the lookup works.
    canonicals_seen = [
        e.canonical
        for entries in idx.by_code.values()
        for e in entries
    ]
    assert "Mufidiwiwhi" in canonicals_seen
    assert idx.secondary_lang == "en"
