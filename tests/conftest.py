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

"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest


# Make the package importable when running `pytest` from the repo root
# without an installed copy.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def fake_word(text: str, start: float, end: float, prob: float) -> dict:
    return {"word": text, "start": start, "end": end, "probability": prob}


def fake_segment(
    seg_id: int,
    start: float,
    end: float,
    text: str | None = None,
    speaker: str = "A",
    words: list[dict] | None = None,
) -> dict:
    if text is None and words is not None:
        text = "".join(w["word"] for w in words)
    if text is None:
        text = ""
    return {
        "id": seg_id,
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
        "words": words,
        "language": "en",
    }


@pytest.fixture
def make_segment():
    return fake_segment


@pytest.fixture
def make_word():
    return fake_word


@pytest.fixture
def fr_dict_path(tmp_path) -> str:
    p = tmp_path / "fr_dict.txt"
    p.write_text(
        "# French podcast vocabulary\n"
        "Castopod\n"
        "OpenRAG\n"
        "Saint-Étienne\n"
        "Free Software Foundation\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture
def en_dict_path(tmp_path) -> str:
    p = tmp_path / "en_dict.txt"
    p.write_text(
        "Mufidiwiwhi\n"
        "Castopod\n",
        encoding="utf-8",
    )
    return str(p)
