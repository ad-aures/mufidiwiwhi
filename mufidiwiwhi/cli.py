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

"""Command-line entry point for Mufidiwiwhi.

Builds a `RunConfig` from argparse and delegates to
`mufidiwiwhi.core.run_pipeline`. The GUI worker calls the same
`run_pipeline` so business logic is shared.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from . import aup3
from .core import (
    Cancelled,
    CorrectionConfig,
    LogCb,
    RunConfig,
    SpeakerInput,
    run_pipeline,
    set_verbose,
)
from .writers import WRITER_FORMATS, optional_float, optional_int, str2bool


# Common faster-whisper model sizes. Any string is accepted; this list
# is just for --help discoverability.
_KNOWN_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mufidiwiwhi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Transcribe a podcast with one audio file per speaker, "
            "using faster-whisper."
        ),
    )
    parser.add_argument(
        "audio_args",
        nargs="+",
        type=str,
        help="alternating speaker names and audio files: NAME PATH NAME PATH ...",
    )
    parser.add_argument(
        "--model",
        default="small",
        help=(
            "faster-whisper model name (e.g. tiny, small, medium, large-v3, "
            "distil-large-v3). Known: " + ", ".join(_KNOWN_MODELS)
        ),
    )
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="inference device",
    )
    parser.add_argument(
        "--compute_type",
        default="default",
        help="faster-whisper compute_type, e.g. int8, float16, float32",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=".",
        help="directory to save the outputs",
    )
    parser.add_argument(
        "--output_format", "-f", type=str, default="all",
        help=(
            "output format(s): a single name, a comma-separated list, or "
            "'all'. Choices: " + ", ".join(WRITER_FORMATS) + ", all"
        ),
    )
    parser.add_argument(
        "--output_filename", type=str, default=None,
        help="output base filename (without extension). Auto if omitted.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="print progress and per-chunk debug messages on stderr",
    )
    parser.add_argument(
        "--task", default="transcribe", choices=["transcribe", "translate"]
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="ISO language code, e.g. en or fr. Auto-detect if omitted.",
    )
    parser.add_argument("--initial_prompt", type=str, default=None)
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="initial decoding temperature",
    )
    parser.add_argument(
        "--temperature_increment_on_fallback",
        type=optional_float,
        default=0.2,
    )
    parser.add_argument("--log_prob_threshold", type=optional_float, default=-0.57)
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6)
    parser.add_argument(
        "--compression_ratio_threshold", type=optional_float, default=2.4
    )
    parser.add_argument(
        "--condition_on_previous_text", type=str2bool, default=True
    )
    parser.add_argument("--word_timestamps", type=str2bool, default=False)
    parser.add_argument(
        "--vad_filter", type=str2bool, default=True,
        help="use Silero VAD to drop non-speech regions before transcription",
    )

    # Task 2: post-correction flags.
    parser.add_argument(
        "--dictionary", type=str, default=None,
        help="path to a plain-text dictionary file. Enables phonetic correction.",
    )
    parser.add_argument(
        "--phonetic-lang", dest="phonetic_lang",
        choices=["auto", "fr", "en"], default="auto",
    )
    parser.add_argument(
        "--phonetic-lang-secondary", dest="phonetic_lang_secondary",
        choices=["fr", "en"], default=None,
    )
    parser.add_argument(
        "--correct-low-conf", dest="correct_low_conf", type=float, default=0.50
    )
    parser.add_argument(
        "--correct-high-conf", dest="correct_high_conf", type=float, default=0.95
    )
    parser.add_argument(
        "--correct-edit-distance", dest="correct_edit_distance",
        type=int, default=2,
    )
    parser.add_argument(
        "--hunspell-primary", dest="hunspell_primary", type=str, default=None,
        help="basename of a Hunspell .aff/.dic pair (auto-detected if omitted)",
    )
    parser.add_argument(
        "--hunspell-secondary", dest="hunspell_secondary", type=str, default=None,
    )
    parser.add_argument(
        "--no-hunspell", dest="use_hunspell", action="store_false",
        help="disable the Hunspell second-opinion pass",
    )
    parser.set_defaults(use_hunspell=True)

    return parser


def _build_correction_config(args: argparse.Namespace) -> CorrectionConfig | None:
    if args.dictionary is None:
        return None
    return CorrectionConfig(
        dictionary_path=args.dictionary,
        phonetic_lang=args.phonetic_lang,
        phonetic_lang_secondary=args.phonetic_lang_secondary,
        low_conf=args.correct_low_conf,
        high_conf=args.correct_high_conf,
        edit_distance_threshold=args.correct_edit_distance,
        hunspell_primary=args.hunspell_primary,
        hunspell_secondary=args.hunspell_secondary,
        use_hunspell=args.use_hunspell,
    )


def _audio_dicts(
    audio_args: list[str], log: LogCb | None = None
) -> list[SpeakerInput]:
    """Expand the positional `SPEAKER FILE ...` args, with the
    extra rule that any standalone `.aup3` argument is unpacked
    into one `SpeakerInput` per Audacity track. The other args
    keep the original (speaker, file) pairing.
    """
    speakers: list[SpeakerInput] = []
    i = 0
    while i < len(audio_args):
        arg = audio_args[i]
        if aup3.is_aup3(arg):
            speakers.extend(aup3.extract_aup3(arg, log=log))
            i += 1
            continue
        if i + 1 >= len(audio_args):
            raise SystemExit(
                "Argument count must be even: SPEAKER1 FILE1 SPEAKER2 "
                "FILE2 ... (or a single .aup3 path)"
            )
        speakers.append(
            SpeakerInput(speaker=arg, file_path=audio_args[i + 1])
        )
        i += 2
    return speakers


def _temperature_tuple(start: float, increment) -> tuple[float, ...]:
    if increment is None:
        return (start,)
    out: list[float] = []
    t = start
    while t <= 1.0 + 1e-6:
        out.append(round(t, 4))
        t += increment
    return tuple(out)


def _log(msg: str) -> None:
    print(msg, flush=True)


def args_to_runconfig(args: argparse.Namespace) -> RunConfig:
    speakers = _audio_dicts(args.audio_args, log=_log)
    temperature = _temperature_tuple(
        args.temperature, args.temperature_increment_on_fallback
    )
    return RunConfig(
        speakers=speakers,
        model_name=args.model,
        model_dir=args.model_dir,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        task=args.task,
        initial_prompt=args.initial_prompt,
        temperature=temperature,
        log_prob_threshold=args.log_prob_threshold,
        no_speech_threshold=args.no_speech_threshold,
        compression_ratio_threshold=args.compression_ratio_threshold,
        condition_on_previous_text=args.condition_on_previous_text,
        word_timestamps=args.word_timestamps,
        vad_filter=args.vad_filter,
        output_dir=args.output_dir,
        output_formats=[args.output_format],
        output_filename=args.output_filename,
        correction=_build_correction_config(args),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # `--verbose` only gates the stderr breadcrumbs (chunker
    # timing, per-replacement decisions). High-level progress
    # (model load, transcribe-speaker, merge counts, writers)
    # always goes to stdout via `_log` so the CLI shows signs of
    # life by default.
    set_verbose(bool(getattr(args, "verbose", False)))
    cfg = args_to_runconfig(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    try:
        run_pipeline(cfg, log=_log)
    except Cancelled:
        print("Cancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
