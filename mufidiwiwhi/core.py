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

"""Top-level pipeline orchestrator shared by the CLI and the GUI.

`run_pipeline(cfg, ...)` loads a Whisper model, transcribes each
speaker file, merges the segments, optionally applies post-correction,
and writes the requested output formats. Both `mufidiwiwhi.cli:main`
and the GUI worker thread call this function.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


ProgressCb = Callable[[str, float], None]
LogCb = Callable[[str], None]


# Verbose console toggle. Default OFF; wired from `cli.main` and
# `gui.app.main` based on the user's `--verbose` flag. Every stderr
# diagnostic in transcribe / correct guards on `is_verbose()` so
# the GUI run log and bare CLI stay quiet by default.
_VERBOSE: bool = False


def set_verbose(value: bool) -> None:
    """Toggle the module-level verbose flag."""
    global _VERBOSE
    _VERBOSE = bool(value)


def is_verbose() -> bool:
    return _VERBOSE


def vstderr(msg: str) -> None:
    """Print a debug breadcrumb to stderr only when verbose is on."""
    if not _VERBOSE:
        return
    sys.stderr.write(msg if msg.endswith("\n") else msg + "\n")
    sys.stderr.flush()
CancelCb = Callable[[], bool]


class Cancelled(Exception):
    """Raised inside the pipeline when the caller requests cancellation."""


@dataclass
class SpeakerInput:
    speaker: str
    file_path: str


@dataclass
class CorrectionConfig:
    dictionary_path: Optional[str] = None
    phonetic_lang: str = "auto"
    phonetic_lang_secondary: Optional[str] = None
    low_conf: float = 0.50
    high_conf: float = 0.95
    edit_distance_threshold: int = 2
    # Hunspell second-opinion: paths to .aff/.dic basename
    # (e.g. "/usr/share/hunspell/fr_FR"). Auto-detected if None when
    # `use_hunspell` is True.
    hunspell_primary: Optional[str] = None
    hunspell_secondary: Optional[str] = None
    use_hunspell: bool = True
    # When a word's Whisper confidence is below this threshold, ask
    # Hunspell for a suggestion and substitute it (rendered as a
    # strikethrough-original + highlighted-correction in the live
    # log). Set to 0 to disable this pass entirely.
    hunspell_threshold: float = 0.5


@dataclass
class RunConfig:
    speakers: list[SpeakerInput]
    model_name: str = "small"
    model_dir: Optional[str] = None
    device: str = "auto"
    compute_type: str = "default"
    language: Optional[str] = None
    task: str = "transcribe"
    initial_prompt: Optional[str] = None
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    log_prob_threshold: float = -0.57
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    condition_on_previous_text: bool = True
    word_timestamps: bool = False
    vad_filter: bool = True
    output_dir: str = "."
    output_formats: list[str] = field(default_factory=lambda: ["all"])
    output_filename: Optional[str] = None
    correction: Optional[CorrectionConfig] = None
    # Confidence-colour thresholds for the live transcript
    # (excellent, high, mid, low). Tier the word into the matching
    # background colour. See `mufidiwiwhi.transcribe._confidence_bg`.
    conf_thresholds: tuple[float, float, float, float] = (0.99, 0.80, 0.70, 0.60)
    # Optional chapters: a list of {"t": float, "title": str} pairs
    # extracted from an Audacity labeltrack. When set, a sibling
    # `<output_filename>-chapters.json` is written at the start of
    # the run (timestamps rounded to 1 decimal).
    chapters: Optional[list[dict]] = None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    # Prefer pynvml: it's already bundled (we use it for the metrics
    # strip) and doesn't drag torch / the full CUDA toolchain into
    # the PyInstaller bundle. Fall back to torch when available, then
    # CPU as a last resort.
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() > 0:
                return "cuda"
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return "cpu"
    except Exception:
        pass
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(value: str, device: str) -> str:
    """Translate 'auto' (and 'default') to a concrete compute_type."""
    if not value or value in ("auto", "default"):
        return "float16" if device == "cuda" else "int8"
    return value


def _correction_active(cfg: RunConfig) -> bool:
    return (
        cfg.correction is not None
        and cfg.correction.dictionary_path is not None
    )


def run_pipeline(
    cfg: RunConfig,
    *,
    progress: Optional[ProgressCb] = None,
    log: Optional[LogCb] = None,
    log_html: Optional[LogCb] = None,
    cancel: Optional[CancelCb] = None,
) -> dict[str, Any]:
    """Run the full transcription pipeline.

    Returns the result dict (segments, language, text). Output files
    are written to `cfg.output_dir`. Raises `Cancelled` if `cancel()`
    returns True at one of the cooperative checkpoints.
    """
    from faster_whisper import WhisperModel  # type: ignore

    from . import transcribe as _t
    from .writers import get_writer

    if not cfg.speakers:
        raise ValueError("At least one speaker input is required")

    # Force word_timestamps when correction needs them.
    if _correction_active(cfg) and not cfg.word_timestamps:
        cfg.word_timestamps = True
    # The explicit chunker in `transcribe.py` handles silence
    # boundaries; faster-whisper's internal Silero VAD would
    # otherwise drop audio at chunk edges (the s02e32_b 160 ms
    # word-loss incident). Keep the field for API compatibility
    # but always pass False to the model.
    cfg.vad_filter = False

    def emit_log(msg: str) -> None:
        # When verbose is on, mirror every high-level progress line
        # to stderr too. The `log` callback routes to the GUI's
        # run-page log view (or the CLI's stdout); without this
        # mirror, `mufidiwiwhi-gui --verbose` would only show the
        # low-level chunker / correction breadcrumbs in the
        # terminal and the user couldn't see overall progress.
        vstderr(msg)
        if log is not None:
            log(msg)

    def check_cancel() -> bool:
        return bool(cancel and cancel())

    # Write the chapters JSON sidecar (if any) BEFORE transcription
    # starts: the user wants the chapter file produced even if the
    # run is later cancelled or the model fails to load.
    if cfg.chapters:
        try:
            chapter_path = _write_chapters_json(cfg)
            emit_log(f"Wrote chapter file: {chapter_path}")
        except OSError as exc:
            emit_log(f"WARNING: could not write chapter file: {exc}")

    device = _resolve_device(cfg.device)
    compute_type = _resolve_compute_type(cfg.compute_type, device)

    is_cached = _is_model_cached(cfg.model_name, cfg.model_dir)
    if is_cached:
        emit_log(
            f"Loading model '{cfg.model_name}' on {device} "
            f"(compute_type={compute_type}). Using cached weights."
        )
    else:
        emit_log(
            f"Loading model '{cfg.model_name}' on {device} "
            f"(compute_type={compute_type})."
        )
        emit_log(
            "Model not yet cached: downloading from Hugging Face. "
            "This can take several minutes for medium/large models. "
            "(Subsequent runs will be instant.)"
        )
    if progress is not None:
        progress("Loading model", 0.0)

    model = WhisperModel(
        cfg.model_name,
        device=device,
        compute_type=compute_type,
        download_root=cfg.model_dir,
    )
    emit_log("Model ready.")

    # Build the correction state once, BEFORE transcription starts,
    # so each segment can be corrected (phonetic + low-confidence
    # Hunspell) the moment Whisper yields it - and the corrected
    # segment is what shows up in the live log.
    correction_state = None
    if _correction_active(cfg):
        from . import correct as _c

        correction_state = _c.build_correction_state(cfg.correction, log=log)

    n = len(cfg.speakers)
    # Orchestrator dispatch. The default is the chunk-interleaved
    # orchestrator, which advances every speaker track in lockstep
    # by chunk start time. Export MUFIDIWIWHI_LEGACY_ORCHESTRATOR=1
    # to opt back into the per-speaker serial loop for A/B
    # comparison; that branch will be removed once the parallel
    # path has been validated against real podcast inputs.
    use_legacy = os.environ.get(
        "MUFIDIWIWHI_LEGACY_ORCHESTRATOR", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if not use_legacy:
        emit_log("")
        emit_log(
            f"=== Parallel orchestrator: {n} "
            f"speaker{'s' if n != 1 else ''} interleaved by chunk ==="
        )
        from . import orchestrator as _o

        tracks = _o.init_tracks(cfg.speakers, cfg, log=log)
        flat = _o.run_parallel(
            tracks,
            model,
            cfg,
            correction_state=correction_state,
            log=log,
            log_html=log_html,
            cancel=cancel,
            progress=progress,
        )
        per_speaker = [flat]
    else:
        per_speaker = []
        for idx, spk in enumerate(cfg.speakers):
            if check_cancel():
                raise Cancelled()
            emit_log("")  # blank separator line in the log
            emit_log(
                f"=== Speaker {idx + 1}/{n}: {spk.speaker} "
                f"({spk.file_path}) ==="
            )

            def on_segment_progress(frac: float, _spk=spk) -> None:
                if progress is not None:
                    progress(f"Transcribing {_spk.speaker}", frac)

            segments = _t.transcribe_speaker(
                model,
                spk.file_path,
                spk.speaker,
                cfg,
                log=log,
                log_html=log_html,
                cancel=cancel,
                progress=on_segment_progress,
                correction_state=correction_state,
            )
            per_speaker.append(segments)
            if progress is not None:
                progress("Transcribing", (idx + 1) / n)

    if check_cancel():
        raise Cancelled()
    emit_log("")
    total = sum(len(s) for s in per_speaker)
    emit_log(
        f"Merging segments from {n} speaker file{'s' if n != 1 else ''} "
        f"({total} total)…"
    )
    merged = _t.merge_segments(per_speaker)
    emit_log(f"Merged {len(merged)} segments.")
    # Global overlap resolution: applies the user's pairwise rules
    # so the writers don't emit overlapping cues.
    pre_overlap_count = len(merged)
    merged = _t.resolve_segment_overlaps(merged)
    if len(merged) != pre_overlap_count:
        emit_log(
            f"Overlap resolver: {pre_overlap_count} -> {len(merged)} "
            f"segments after splitting."
        )

    # Drop Whisper hallucinations (Amara-style stock fallbacks) before
    # they reach the writers. They've already been rendered with
    # strikethrough in the log so the user can see what got dropped.
    pre_hallu_count = len(merged)
    merged = [s for s in merged if not s.get("hallucination")]
    dropped = pre_hallu_count - len(merged)
    if dropped:
        emit_log(
            f"Dropped {dropped} hallucinated segment"
            f"{'s' if dropped != 1 else ''} from output."
        )

    detected_language = _detect_language(per_speaker, cfg.language)

    # Per-chunk correction has already mutated the segments in place
    # during transcription; here we just emit the recap.
    if correction_state is not None:
        from . import correct as _c

        emit_log("")
        for line in _c.format_corrections_recap(correction_state):
            emit_log(line)

    result = {
        "text": "".join(seg.get("text", "") for seg in merged),
        "segments": merged,
        "language": detected_language,
    }

    output_filename = cfg.output_filename or _suggest_output_filename(
        [s.file_path for s in cfg.speakers]
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    fmts = _normalise_format_list(cfg.output_formats)
    emit_log("")
    if len(fmts) == 1:
        emit_log(
            f"Writing output to {cfg.output_dir!r} as "
            f"'{output_filename}.{fmts[0]}'…"
        )
    else:
        joined = ", ".join(f"'{output_filename}.{f}'" for f in fmts)
        emit_log(
            f"Writing outputs to {cfg.output_dir!r} as {joined}…"
        )
    writer = get_writer(cfg.output_formats, cfg.output_dir)
    output_paths = writer(result, output_filename)
    for p in output_paths:
        emit_log(f"  wrote {p}")
    result["output_paths"] = output_paths
    return result


def _is_model_cached(model_name: str, model_dir: Optional[str]) -> bool:
    """Heuristic: True if the model directory contains an entry whose
    name encodes the requested faster-whisper model.
    """
    base = model_dir
    if not base:
        try:
            from .gui.helpers import default_model_dir

            base = default_model_dir()
        except Exception:
            return False
    if not base or not os.path.isdir(base):
        return False
    target = f"--faster-whisper-{model_name}"
    try:
        for entry in os.listdir(base):
            if entry.endswith(target):
                return True
    except OSError:
        return False
    return False


def _normalise_format_list(formats) -> list[str]:
    if isinstance(formats, str):
        return [formats] if formats != "all" else ["txt", "vtt", "srt", "tsv", "json"]
    out: list[str] = []
    for f in formats:
        if f == "all":
            return ["txt", "vtt", "srt", "tsv", "json"]
        out.extend(p.strip() for p in str(f).split(",") if p.strip())
    return out


def _detect_language(
    per_speaker: list[list[dict]], requested: Optional[str]
) -> Optional[str]:
    if requested:
        return requested
    for segs in per_speaker:
        for s in segs:
            lang = s.get("language")
            if lang:
                return lang
    return None


def _suggest_output_filename(paths: list[str]) -> str:
    """Longest common prefix of basenames, with sensible fallbacks.

    Examples:
        ["s02e15_b.wav", "s02e15_g.wav"] -> "s02e15"
        ["interview_lucy.wav", "interview_samir.wav"] -> "interview"
        ["foo.wav"] -> "foo"
        ["a.wav", "b.wav"] -> "a" (fallback to first stem)

    Special case: when every input file lives in a `<X>_tracks/`
    directory (the layout produced by `aup3.extract_aup3`), use
    `<X>` directly. The track WAVs are named per-speaker so a
    common prefix would collapse to junk (e.g. "" or "track_"),
    losing the project identity.
    """
    if not paths:
        return "transcript"
    aup3_stem = _aup3_tracks_stem(paths)
    if aup3_stem:
        return aup3_stem
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(stems) == 1:
        stem = stems[0]
    else:
        stem = os.path.commonprefix(stems)
    stem = stem.rstrip("_-. ")
    if len(stem) < 3:
        stem = stems[0].rstrip("_-. ")
    if len(stem) < 3:
        from datetime import datetime

        stem = f"transcript_{datetime.now():%Y%m%d_%H%M%S}"
    return stem or "transcript"


def _write_chapters_json(cfg: "RunConfig") -> str:
    """Serialise `cfg.chapters` to `<output_filename>-chapters.json`
    with timestamps rounded to one decimal. Returns the written
    path. Format mirrors the Podcasting 2.0 chapter spec, version
    1.0.0 (the same shape as Castopod's per-episode chapter files).
    """
    import json

    out_filename = cfg.output_filename or _suggest_output_filename(
        [s.file_path for s in cfg.speakers]
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, f"{out_filename}-chapters.json")
    payload = {
        "version": "1.0.0",
        "chapters": [
            {
                "startTime": round(float(c.get("t", 0.0)), 1),
                "title": str(c.get("title", "")),
            }
            for c in cfg.chapters or []
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return path


def _aup3_tracks_stem(paths: list[str]) -> Optional[str]:
    """If every path's parent directory ends in `_tracks` AND all
    paths share the same parent, return that parent's `<X>` part.
    Otherwise None.
    """
    if not paths:
        return None
    suffix = "_tracks"
    parent = None
    for p in paths:
        d = os.path.dirname(os.path.abspath(p))
        if not d.endswith(suffix):
            return None
        if parent is None:
            parent = d
        elif d != parent:
            return None
    assert parent is not None
    base = os.path.basename(parent)
    return base[: -len(suffix)] or None
