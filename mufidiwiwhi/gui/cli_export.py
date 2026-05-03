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

"""Translate a `RunConfig` back into the equivalent CLI command.

Used by the Project tab's "Copy CLI command" button so a user can
reproduce the exact GUI run from a shell or a script.
"""

from __future__ import annotations

import shlex
from typing import Optional

from ..core import RunConfig


def _quote(value: str) -> str:
    """Shell-quote `value` only when needed (POSIX-safe)."""
    return shlex.quote(value)


def runconfig_to_cli(cfg: RunConfig) -> str:
    """Return the `mufidiwiwhi ...` command equivalent to `cfg`.

    The command is built so that running it produces the same result
    as the GUI: same speakers, model, language, output directory,
    output formats, output filename, and correction settings.
    """
    parts: list[str] = ["mufidiwiwhi"]

    # Speakers (positional, alternating NAME PATH ...)
    for spk in cfg.speakers:
        parts.append(_quote(spk.speaker))
        parts.append(_quote(spk.file_path))

    parts.extend(["--model", _quote(cfg.model_name)])
    if cfg.model_dir:
        parts.extend(["--model_dir", _quote(cfg.model_dir)])
    parts.extend(["--device", _quote(cfg.device)])
    parts.extend(["--compute_type", _quote(cfg.compute_type)])
    if cfg.language:
        parts.extend(["--language", _quote(cfg.language)])
    if cfg.task and cfg.task != "transcribe":
        parts.extend(["--task", _quote(cfg.task)])
    if cfg.initial_prompt:
        parts.extend(["--initial_prompt", _quote(cfg.initial_prompt)])
    if not cfg.vad_filter:
        parts.extend(["--vad_filter", "False"])

    formats = _normalise_formats(cfg.output_formats)
    if formats and formats != ("all",):
        parts.extend(["--output_format", _quote(",".join(formats))])
    parts.extend(["--output_dir", _quote(cfg.output_dir)])
    if cfg.output_filename:
        parts.extend(["--output_filename", _quote(cfg.output_filename)])

    correction = cfg.correction
    if correction is not None:
        if correction.dictionary_path:
            parts.extend(["--dictionary", _quote(correction.dictionary_path)])
        if correction.phonetic_lang and correction.phonetic_lang != "auto":
            parts.extend(["--phonetic-lang", correction.phonetic_lang])
        if correction.phonetic_lang_secondary:
            parts.extend(
                ["--phonetic-lang-secondary", correction.phonetic_lang_secondary]
            )
        if correction.low_conf != 0.50:
            parts.extend(["--correct-low-conf", f"{correction.low_conf}"])
        if correction.high_conf != 0.95:
            parts.extend(["--correct-high-conf", f"{correction.high_conf}"])
        if correction.edit_distance_threshold != 2:
            parts.extend(
                ["--correct-edit-distance", f"{correction.edit_distance_threshold}"]
            )
        if correction.hunspell_primary:
            parts.extend(["--hunspell-primary", _quote(correction.hunspell_primary)])
        if correction.hunspell_secondary:
            parts.extend(
                ["--hunspell-secondary", _quote(correction.hunspell_secondary)]
            )
        if not correction.use_hunspell:
            parts.append("--no-hunspell")

    return " ".join(parts)


def _normalise_formats(formats) -> tuple[str, ...]:
    if isinstance(formats, str):
        return (formats,)
    out: list[str] = []
    for f in formats:
        if "," in f:
            out.extend(p.strip() for p in f.split(",") if p.strip())
        else:
            out.append(str(f))
    return tuple(out)
