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

"""Tests for the output filename suggestion heuristic."""

from __future__ import annotations

import pytest

from mufidiwiwhi.core import _suggest_output_filename


def test_two_episode_files_with_speaker_suffix():
    assert (
        _suggest_output_filename(["s02e15_b.wav", "s02e15_g.wav"]) == "s02e15"
    )


def test_interview_prefix():
    assert (
        _suggest_output_filename(
            ["interview_lucy.wav", "interview_samir.wav", "interview_rachel.wav"]
        )
        == "interview"
    )


def test_single_file_keeps_stem():
    assert _suggest_output_filename(["foo.wav"]) == "foo"


def test_short_common_prefix_falls_back_to_first_stem():
    assert _suggest_output_filename(["alpha.wav", "beta.wav"]) == "alpha"


def test_paths_in_subdirs_strip_directory():
    assert (
        _suggest_output_filename(
            ["/tmp/abc/podcast_a.wav", "/tmp/abc/podcast_b.wav"]
        )
        == "podcast"
    )


def test_empty_list_returns_default():
    assert _suggest_output_filename([]) == "transcript"


def test_unicode_basenames():
    assert (
        _suggest_output_filename(["épisode_1.wav", "épisode_2.wav"])
        == "épisode"
    )
