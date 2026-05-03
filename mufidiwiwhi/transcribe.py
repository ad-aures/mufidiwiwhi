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

"""Per-speaker transcription, explicit chunking, and merge.

Audio is split into chunks BEFORE faster-whisper sees it. The
chunker is a re-implementation of the original handwritten one
(commit `e64e524`) that the migration to faster-whisper had
silently dropped. Faster-whisper's own VAD is bypassed because
it was found to drop audio that carried real words at chunk
boundaries (the s02e32_b 160 ms gap incident).

Chunker invariants:
  * each chunk is at most 30 s long;
  * each chunk ends at the local-minimum-RMS point inside its
    look-ahead window (i.e. the quietest 50 ms slice in
    `[start + 2 s, start + 30 s]`);
  * chunks DON'T start with leading silence (the cursor advances
    past it before recording the chunk's start);
  * NO non-silent audio is ever skipped between chunks: the next
    chunk starts exactly where the previous one ended;
  * the silence threshold is adaptive: `audio.max_dBFS - 35 dB`.

Segments are streamed out as faster-whisper produces them, then
their timestamps are offset by the chunk's start position before
the rest of the pipeline sees them.

Cross-chunk overlap resolution lives in `resolve_segment_overlaps`
and runs once globally, after `merge_segments` collects every
speaker's stream.
"""

from __future__ import annotations

import html
import sys
import time
import unicodedata
from typing import Any, Callable, Iterator, Optional

from .core import Cancelled, LogCb, CancelCb, RunConfig, vstderr

ProgressCb = Callable[[float], None]


_DEFAULT_CONF_THRESHOLDS = (0.99, 0.80, 0.70, 0.60)


# ---------------------------------------------------------------------------
# Whisper-hallucination filter
# ---------------------------------------------------------------------------


# Whisper has a known habit of falling back to a small set of stock
# captions when fed silence or noise. The strings below are
# normalised (lowercased, accents stripped, punctuation collapsed)
# substrings that, when contained in a segment's text, mark the
# segment as a hallucination. Add to this list as you spot new
# offenders; per-language is OK but the matcher is language-agnostic.
_HALLUCINATION_NEEDLES: tuple[str, ...] = (
    # French - Amara.org subtitle credits (the canonical case).
    # We only match the distinctive trailing fragment so misspellings
    # of the leading wording (e.g. "para" instead of "par") still get
    # caught.
    "communaute d'amara",
    "merci d'avoir regarde",
    "abonnez-vous a la chaine",
    "abonnez vous a la chaine",
    "soustitreur.com",
    "sous-titreur.com",
    # English - same family of stock fillers.
    "amara.org community",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "please like and subscribe",
)


def _hallucination_normalise(text: str) -> str:
    """Lowercase + strip combining marks so hallucination patterns
    match whatever case / accents Whisper chose for the offending
    segment."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_marks.lower()


def _is_hallucination(text: str) -> bool:
    """True when `text` looks like one of Whisper's known stock
    fallbacks. Substring match on the normalised form."""
    if not text:
        return False
    norm = _hallucination_normalise(text)
    if not norm.strip():
        return False
    return any(needle in norm for needle in _HALLUCINATION_NEEDLES)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


# Chunker constants. Tweaked from the original (e64e524) values to keep
# parity with the user's expectations.
_MAX_CHUNK_MS = 30_000
_MIN_CHUNK_MS = 2_000
_SEEK_STEP_MS = 50
_SILENCE_DROP_DB = 35


def _silence_rms_threshold(audio) -> float:
    """RMS threshold in raw amplitude units corresponding to
    `audio.max_dBFS - SILENCE_DROP_DB`. Anything below this is
    treated as silence by the chunker.
    """
    from pydub.utils import db_to_float  # type: ignore

    threshold_dbfs = audio.max_dBFS - _SILENCE_DROP_DB
    return float(db_to_float(threshold_dbfs)) * float(audio.max_possible_amplitude)


def _iter_chunks(audio_path: str) -> Iterator[tuple[int, int, Any]]:
    """Public wrapper: load the file with pydub, then delegate to
    `_iter_chunks_from`. Useful when the caller only has a path.
    """
    from pydub import AudioSegment  # type: ignore

    audio = AudioSegment.from_file(audio_path)
    yield from _iter_chunks_from(audio)


def _audiosegment_to_float32(audio) -> "Any":
    """Convert a pydub `AudioSegment` to a mono float32 numpy
    array in [-1, 1], the format `WhisperModel.transcribe` expects
    when handed in-memory audio.
    """
    import numpy as np  # type: ignore

    audio = audio.set_frame_rate(16_000).set_channels(1)
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


# ---------------------------------------------------------------------------
# Word-confidence colouring (unchanged)
# ---------------------------------------------------------------------------


def _confidence_bg(
    probability: float,
    thresholds: Optional[tuple[float, float, float, float]] = None,
) -> Optional[str]:
    """Background colour for a word given its Whisper confidence.

    `thresholds` is a 4-tuple `(excellent, high, mid, low)`. When the
    probability is at or above `excellent`, the word gets the green
    tint; between `high` and `excellent`, no background; between
    `mid` and `high`, pale yellow; between `low` and `mid`, light
    orange; below `low`, pink.

    Defaults: (0.99, 0.80, 0.70, 0.60).
    """
    excellent, high, mid, low = thresholds or _DEFAULT_CONF_THRESHOLDS
    if probability >= excellent:
        return "#e5ffd5"
    if probability >= high:
        return None
    if probability >= mid:
        return "#fff6d5"
    if probability >= low:
        return "#ffe6d5"
    return "#ffd5d5"


def _format_ts(seconds: float) -> str:
    """Compact MM:SS.mmm timestamp for log lines."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms -= hours * 3_600_000
    minutes = total_ms // 60_000
    total_ms -= minutes * 60_000
    secs = total_ms // 1_000
    ms = total_ms - secs * 1_000
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def _build_segment_dict(
    s,
    speaker: str,
    cfg: RunConfig,
    seg_id: int,
    time_offset_s: float,
    language: Optional[str],
) -> dict:
    """Materialise a faster-whisper segment object into the dict
    shape the rest of the pipeline expects, applying `time_offset_s`
    to every timestamp (segment-level and word-level)."""
    words_payload: Optional[list[dict]] = None
    if cfg.word_timestamps:
        raw_words = getattr(s, "words", None) or []
        words_payload = [
            {
                "word": w.word,
                "start": float(w.start) + time_offset_s,
                "end": float(w.end) + time_offset_s,
                "probability": float(w.probability),
            }
            for w in raw_words
        ]
    text = s.text
    return {
        "id": seg_id,
        "seek": float(s.start) + time_offset_s,
        "start": float(s.start) + time_offset_s,
        "end": float(s.end) + time_offset_s,
        "speaker": speaker,
        "text": text,
        "tokens": list(s.tokens) if getattr(s, "tokens", None) else [],
        "temperature": float(getattr(s, "temperature", 0.0) or 0.0),
        "avg_logprob": float(getattr(s, "avg_logprob", 0.0) or 0.0),
        "compression_ratio": float(
            getattr(s, "compression_ratio", 0.0) or 0.0
        ),
        "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0) or 0.0),
        "words": words_payload,
        "language": language,
        # Tagged here, filtered out of the writer pipeline in
        # `core.run_pipeline`. The live log still renders the
        # segment so the user sees what got dropped.
        "hallucination": _is_hallucination(text),
    }


def _emit_segment_log(
    seg_dict: dict,
    speaker: str,
    cfg: RunConfig,
    log: Optional[LogCb],
    log_html: Optional[LogCb],
) -> None:
    """Stream a per-segment line to the log_html receiver (with
    confidence-colouring + correction diffs) and / or the plain
    log receiver."""
    words_payload = seg_dict.get("words")
    is_hallucination = bool(seg_dict.get("hallucination"))
    emitted_rich = False
    if log_html is not None and (words_payload or is_hallucination):
        thresholds = getattr(cfg, "conf_thresholds", None)
        ts_text = (
            f"{_format_ts(seg_dict['start'])} -&gt; "
            f"{_format_ts(seg_dict['end'])}"
        )
        prefix = (
            "<span style='background-color:#4d4d4d;color:#ffffff;'>"
            f"&nbsp;{ts_text}&nbsp;</span>"
            "<span style='background-color:transparent;color:inherit;'>"
            "&nbsp;</span>"
            "<span style='background-color:#2ea44f;color:#ffffff;'>"
            f"&nbsp;{html.escape(speaker)}&nbsp;</span>"
            "<span style='background-color:transparent;color:inherit;'>"
            "&nbsp;</span>"
        )
        parts = [
            "<pre style='margin:0;font-family:inherit;"
            "background-color:transparent;'>",
            prefix,
        ]
        if is_hallucination:
            # Whole segment is a known Whisper hallucination - render
            # it with strikethrough on a neutral grey so the user
            # sees what got dropped, then skip the per-word loop.
            text = html.escape(seg_dict.get("text", "").strip())
            parts.append(
                "<span style='text-decoration:line-through;"
                "background-color:#e6e6e6;color:#5e5e5e;'>"
                f"{text}</span>"
            )
            parts.append(
                "<span style='background-color:transparent;color:inherit;'>"
                "​&nbsp;</span>"
            )
            parts.append("</pre>")
            log_html("".join(parts))
            return
        for w in words_payload:
            correction = w.get("correction")
            if correction:
                orig = html.escape(correction.get("original", ""))
                repl = html.escape(correction.get("replacement", ""))
                orig_conf = float(
                    correction.get("original_confidence", 1.0)
                )
                orig_bg = _confidence_bg(orig_conf, thresholds)
                raw = w.get("word", "")
                leading_ws = ""
                k = 0
                while k < len(raw) and raw[k].isspace():
                    leading_ws += raw[k]
                    k += 1
                parts.append(html.escape(leading_ws))
                if orig:
                    bg_style = (
                        f"background-color:{orig_bg};"
                        if orig_bg
                        else ""
                    )
                    parts.append(
                        "<span style='text-decoration:line-through;"
                        f"{bg_style}'>{orig}</span> "
                    )
                parts.append(
                    "<span style='background-color:#c2ebff;'>"
                    f"{repl}</span>"
                )
            elif w.get("uncorrectable_misspelled"):
                # Hunspell flagged the word as misspelled at low
                # confidence and we had no usable suggestion: paint
                # it grey so the user spots "definitely wrong, but
                # we don't know what's right" at a glance.
                token = html.escape(w["word"])
                parts.append(
                    f"<span style='background-color:#e6e6e6;'>"
                    f"{token}</span>"
                )
            else:
                bg = _confidence_bg(w["probability"], thresholds)
                token = html.escape(w["word"])
                if bg is None:
                    parts.append(token)
                else:
                    parts.append(
                        f"<span style='background-color:{bg};'>"
                        f"{token}</span>"
                    )
        parts.append(
            "<span style='background-color:transparent;color:inherit;'>"
            "​&nbsp;</span>"
        )
        parts.append("</pre>")
        log_html("".join(parts))
        emitted_rich = True
    if log is not None and not emitted_rich:
        log(
            f"[{_format_ts(seg_dict['start'])} -> "
            f"{_format_ts(seg_dict['end'])}] "
            f"[{speaker}] {seg_dict['text'].strip()}"
        )


# ---------------------------------------------------------------------------
# Per-speaker transcription
# ---------------------------------------------------------------------------


def transcribe_speaker(
    model: Any,
    audio_path: str,
    speaker: str,
    cfg: RunConfig,
    *,
    log: Optional[LogCb] = None,
    log_html: Optional[LogCb] = None,
    cancel: Optional[CancelCb] = None,
    progress: Optional[ProgressCb] = None,
    correction_state: Any = None,
) -> list[dict]:
    """Transcribe one speaker file with faster-whisper.

    The audio is chunked locally with `_iter_chunks` (see top of
    module). Each chunk is decoded independently with
    `vad_filter=False`; segment timestamps are offset back into
    the full file's time base before emission. Returns the list of
    segment dicts in chunk-then-decode order.
    """
    from pydub import AudioSegment  # type: ignore

    if log is not None:
        log(f"Opening {audio_path!r} for speaker '{speaker}'")

    vstderr(f"[{speaker}] AudioSegment.from_file({audio_path!r}) starting...")
    t_load = time.monotonic()
    full_audio = AudioSegment.from_file(audio_path)
    vstderr(
        f"[{speaker}] AudioSegment.from_file done in "
        f"{time.monotonic() - t_load:.2f}s"
    )
    total_ms = len(full_audio)
    duration_s = total_ms / 1000.0
    detected_language: Optional[str] = cfg.language

    if log is not None:
        log(
            f"Audio duration {duration_s:.1f}s. "
            f"Chunking with explicit RMS-minima cuts (no VAD)."
        )

    out: list[dict] = []
    chunk_index = 0
    for chunk_start_ms, chunk_end_ms, chunk_audio in _iter_chunks_from(
        full_audio, log=log
    ):
        if cancel and cancel():
            raise Cancelled()
        chunk_index += 1
        offset_s = chunk_start_ms / 1000.0
        chunk_dur_s = (chunk_end_ms - chunk_start_ms) / 1000.0
        vstderr(
            f"[{speaker}] chunk {chunk_index}: "
            f"{_format_ts(chunk_start_ms / 1000.0)} -> "
            f"{_format_ts(chunk_end_ms / 1000.0)} "
            f"({chunk_dur_s:.1f}s) decoding..."
        )
        t_chunk = time.monotonic()
        samples = _audiosegment_to_float32(chunk_audio)
        t_whisper_start = time.monotonic()
        segments_iter, info = model.transcribe(
            samples,
            language=cfg.language,
            task=cfg.task,
            initial_prompt=cfg.initial_prompt,
            temperature=cfg.temperature,
            # Disable Whisper's segment-drop filters. The chunker
            # has already decided this chunk contains speech (via
            # the RMS test), so we trust that and force Whisper to
            # emit a transcription for every sample. Whisper's own
            # `log_prob_threshold` / `no_speech_threshold` are
            # opinionated quality gates that drop any segment the
            # decoder isn't sure about, including real but quiet
            # speech. Compression-ratio filtering stays on because
            # it catches genuine hallucination loops (e.g. the
            # "thanks for watching" repeats), not silence.
            log_prob_threshold=None,
            no_speech_threshold=1.0,
            compression_ratio_threshold=cfg.compression_ratio_threshold,
            condition_on_previous_text=cfg.condition_on_previous_text,
            word_timestamps=cfg.word_timestamps,
            vad_filter=False,
        )
        if detected_language is None:
            detected_language = getattr(info, "language", None)
        chunk_seg_count = 0
        for s in segments_iter:
            if cancel and cancel():
                raise Cancelled()
            chunk_seg_count += 1
            seg_dict = _build_segment_dict(
                s, speaker, cfg, len(out), offset_s, detected_language
            )
            if correction_state is not None:
                try:
                    from . import correct as _c

                    t_corr = time.monotonic()
                    _c.correct_segment_in_place(seg_dict, correction_state)
                    corr_dt = time.monotonic() - t_corr
                    if corr_dt > 1.0:
                        vstderr(
                            f"[{speaker}] chunk {chunk_index} seg "
                            f"{seg_dict['id']}: correction took "
                            f"{corr_dt:.2f}s"
                        )
                except Exception as exc:
                    if log is not None:
                        log(
                            f"Per-chunk correction failed on segment "
                            f"{seg_dict['id']}: {exc}"
                        )
            out.append(seg_dict)
            _emit_segment_log(seg_dict, speaker, cfg, log, log_html)
        chunk_dt = time.monotonic() - t_chunk
        whisper_dt = time.monotonic() - t_whisper_start
        vstderr(
            f"[{speaker}] chunk {chunk_index}: done in {chunk_dt:.2f}s "
            f"({chunk_seg_count} segments, whisper {whisper_dt:.2f}s)"
        )
        if progress is not None and total_ms > 0:
            progress(min(1.0, chunk_end_ms / total_ms))

    if log is not None:
        log(f"Finished '{speaker}': {len(out)} segments")
    return out


def _iter_chunks_from(
    audio, log: Optional[LogCb] = None
) -> Iterator[tuple[int, int, Any]]:
    """Numpy-vectorised chunker. Decodes the AudioSegment to a
    mono int16 numpy array once, computes per-50ms RMS in a
    single vectorised pass, then walks the RMS array to find
    chunk boundaries.

    The previous pure-pydub implementation called
    `audio[lo:hi].rms` per 50 ms slice; pydub creates a new
    AudioSegment object and iterates samples in pure Python for
    each call, which made the chunker the dominant stall on
    long files (multi-hour audio took minutes just to probe).
    Now the entire chunking decision is O(N) numpy and runs in
    well under a second for hours of audio.

    Silence handling: cursor advances by `_MIN_CHUNK_MS` at a
    time. If EVERY 50 ms slice in the lookahead window is below
    the silence threshold, the cursor advances past the window
    without yielding a chunk. As soon as ANY slice in the
    window has speech-level RMS, the cursor stops and the chunk
    starts THERE, including the leading silence inside that
    window - so a faint speech tail just past a chunk boundary
    survives into the next chunk's transcription window.

    Boundaries: chunk_end is the local-RMS minimum in
    `[start + MIN, start + MAX]`. Each chunk starts exactly
    where the previous one ended; audio inside a "has-speech"
    region is never skipped.
    """
    import numpy as np  # type: ignore

    total_ms = len(audio)
    if total_ms <= 0:
        return
    rate = audio.frame_rate
    step = max(1, rate * _SEEK_STEP_MS // 1000)  # samples per 50 ms slice
    vstderr(
        f"chunker: decoding {total_ms / 1000.0:.1f}s audio "
        f"({rate} Hz, {audio.channels}ch) for RMS scan..."
    )
    t0 = time.monotonic()
    if audio.channels > 1:
        mono = audio.set_channels(1)
    else:
        mono = audio
    samples = np.frombuffer(mono.raw_data, dtype=np.int16).astype(np.float32)
    n = samples.shape[0]
    if n < step:
        # Audio shorter than one slice: yield the whole thing.
        yield 0, total_ms, audio
        return
    n_slices = n // step
    sliced = samples[: n_slices * step].reshape(n_slices, step)
    # Per-slice RMS via vectorised mean of squares.
    rms = np.sqrt(np.mean(sliced * sliced, axis=1))
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.0:
        vstderr("chunker: audio is fully silent, no chunks yielded")
        return
    silence_rms = peak * (10.0 ** (-_SILENCE_DROP_DB / 20.0))
    vstderr(
        f"chunker: {n_slices} slices, peak={peak:.0f}, "
        f"silence threshold={silence_rms:.1f} "
        f"(scan in {time.monotonic() - t0:.2f}s)"
    )

    min_slices = max(1, _MIN_CHUNK_MS // _SEEK_STEP_MS)
    max_slices = max(min_slices + 1, _MAX_CHUNK_MS // _SEEK_STEP_MS)

    cursor = 0
    silent_skip_start: Optional[int] = None
    chunks_yielded = 0
    while cursor < n_slices:
        probe_end = min(cursor + min_slices, n_slices)
        if not bool(np.any(rms[cursor:probe_end] >= silence_rms)):
            if silent_skip_start is None:
                silent_skip_start = cursor
            cursor = probe_end
            continue
        # The probe found speech somewhere in [cursor, probe_end);
        # advance past any leading silence within that probe so the
        # chunk starts at the first non-silent slice. The skipped
        # silence is reported alongside any earlier accumulated
        # silent skips.
        speech_start = cursor
        while speech_start < probe_end and rms[speech_start] < silence_rms:
            speech_start += 1
        if silent_skip_start is None:
            silent_skip_start = cursor
        cursor = speech_start
        if silent_skip_start is not None and cursor > silent_skip_start:
            skipped_ms = (cursor - silent_skip_start) * _SEEK_STEP_MS
            vstderr(
                f"chunker: skipped {skipped_ms / 1000.0:.1f}s of silence "
                f"({_format_ts(silent_skip_start * _SEEK_STEP_MS / 1000.0)} -> "
                f"{_format_ts(cursor * _SEEK_STEP_MS / 1000.0)})"
            )
        silent_skip_start = None
        chunk_start = cursor
        look_lo = min(chunk_start + min_slices, n_slices)
        look_hi = min(chunk_start + max_slices, n_slices)
        if look_hi <= look_lo:
            chunk_end = n_slices
        else:
            chunk_end = look_lo + int(np.argmin(rms[look_lo:look_hi]))
        if chunk_end <= chunk_start:
            chunk_end = min(chunk_start + 1, n_slices)
        # Trim trailing silence: shrink the chunk back to the last
        # non-silent slice so silence between speech bursts stays
        # in the next iteration's silence-skip probe rather than
        # getting sent to Whisper. Bounded below by min_slices to
        # avoid degenerate zero-length chunks.
        floor = min(chunk_start + min_slices, chunk_end)
        trimmed_end = chunk_end
        while trimmed_end > floor and rms[trimmed_end - 1] < silence_rms:
            trimmed_end -= 1
        if trimmed_end > chunk_start:
            chunk_end = trimmed_end
        chunk_start_ms = chunk_start * _SEEK_STEP_MS
        chunk_end_ms = chunk_end * _SEEK_STEP_MS
        chunks_yielded += 1
        yield chunk_start_ms, chunk_end_ms, audio[chunk_start_ms:chunk_end_ms]
        cursor = chunk_end
    vstderr(
        f"chunker: done, {chunks_yielded} chunks yielded "
        f"(total {time.monotonic() - t0:.2f}s)"
    )


# ---------------------------------------------------------------------------
# Cross-speaker merge
# ---------------------------------------------------------------------------


def merge_segments(per_speaker: list[list[dict]]) -> list[dict]:
    """Sort all segments by start time and assign sequential ids.

    Overlap RESOLUTION (cutting and word redistribution) lives
    in `resolve_segment_overlaps` and runs once after this; the
    merge itself just orders the streams.
    """
    all_segs = sorted(
        (seg for spk in per_speaker for seg in spk),
        key=lambda d: (d["start"], d["end"]),
    )
    for i, seg in enumerate(all_segs):
        seg["id"] = i
    return all_segs


# ---------------------------------------------------------------------------
# Global overlap resolution
# ---------------------------------------------------------------------------


def _retime(seg: dict, new_start: float, new_end: float) -> dict:
    """Return a copy of `seg` with its timestamps clamped to
    `[new_start, new_end]`. ALL words are preserved (overlap
    resolution must never drop text); their timestamps are clamped
    into the new range. Words whose original times fall entirely
    outside the new window collapse to a zero-duration entry at the
    nearest boundary, but their surface text is kept.
    """
    new_start = float(new_start)
    new_end = float(new_end)
    out = dict(seg)
    out["start"] = new_start
    out["end"] = new_end
    out["seek"] = new_start
    words = seg.get("words")
    if words:
        new_words = []
        for w in words:
            w2 = dict(w)
            ws = max(new_start, min(new_end, float(w["start"])))
            we = max(new_start, min(new_end, float(w["end"])))
            if we < ws:
                we = ws
            w2["start"] = ws
            w2["end"] = we
            new_words.append(w2)
        out["words"] = new_words
        out["text"] = "".join(w.get("word", "") for w in new_words)
    return out


def _split_segment_3way(
    seg: dict,
    inner_start: float,
    inner_end: float,
) -> tuple[dict, dict]:
    """Split `seg` (timed `[a, b]` containing `[inner_start, inner_end]`)
    into a left half `[a, inner_start]` and a right half `[inner_end, b]`,
    distributing its words by COUNT (per the user's formula):
        xa = int(x * (mid - a) / (b - a))   with mid = (c + d) / 2
        xb = x - xa
    The first xa words go to the left half, the remaining xb to the
    right half. If word timestamps are absent, words are split by
    text length proportion using the same midpoint ratio.
    """
    a = float(seg["start"])
    b = float(seg["end"])
    c = float(inner_start)
    d = float(inner_end)
    mid = (c + d) / 2.0
    span = max(b - a, 1e-9)
    ratio = max(0.0, min(1.0, (mid - a) / span))

    words = seg.get("words")
    if words:
        x = len(words)
        xa = max(0, min(x, int(x * ratio)))
        left_words = [dict(w) for w in words[:xa]]
        right_words = [dict(w) for w in words[xa:]]
        # Clamp word timestamps to the resulting segment ranges so
        # writers don't surface impossible word times.
        for w in left_words:
            w["start"] = max(a, min(c, float(w["start"])))
            w["end"] = max(a, min(c, float(w["end"])))
        for w in right_words:
            w["start"] = max(d, min(b, float(w["start"])))
            w["end"] = max(d, min(b, float(w["end"])))
        left_text = "".join(w.get("word", "") for w in left_words)
        right_text = "".join(w.get("word", "") for w in right_words)
    else:
        text = seg.get("text", "") or ""
        cut = int(round(len(text) * ratio))
        left_text = text[:cut]
        right_text = text[cut:]
        left_words = None
        right_words = None

    left = dict(seg)
    left["start"] = a
    left["end"] = c
    left["seek"] = a
    left["text"] = left_text
    left["words"] = left_words

    right = dict(seg)
    right["start"] = d
    right["end"] = b
    right["seek"] = d
    right["text"] = right_text
    right["words"] = right_words
    return left, right


def resolve_segment_overlaps(segments: list[dict]) -> list[dict]:
    """Apply the user's pairwise overlap rules in time order.

    Notation: segment 1 = `[a, b]`, segment 2 = `[c, d]`, both
    sorted such that `a <= c`.

      * Rule (a): partial overlap (`a < c < b < d`) -> shrink each
        side to the midpoint `m = (b + c) / 2`:
            seg1 -> [a, m],  seg2 -> [m, d].

      * Rule (b): full containment (`a < c < d < b`) -> split
        seg1 in two pieces and keep seg2 in between:
            [a, c]  +  [c, d] (seg2)  +  [d, b]
        seg1's `x` words are distributed by count using
        `xa = int(x * ((c + d)/2 - a) / (b - a))`.

    Cross-speaker overlaps are resolved with the same rules
    (per user direction). Zero-length segments produced by the
    rules are filtered out.
    """
    if not segments:
        return []
    work = sorted(
        (dict(s) for s in segments),
        key=lambda s: (float(s["start"]), float(s["end"])),
    )
    out: list[dict] = []
    while work:
        s1 = work.pop(0)
        if not work:
            out.append(s1)
            break
        s2 = work[0]
        a = float(s1["start"])
        b = float(s1["end"])
        c = float(s2["start"])
        d = float(s2["end"])
        # No overlap: emit s1 as-is.
        if b <= c:
            out.append(s1)
            continue
        # Defensive: keep `a <= c` invariant.
        if c < a:
            # Shouldn't happen after sort, but if it did, swap.
            s1, s2 = s2, s1
            a, b, c, d = c, d, a, b
        if d > b:
            # Rule (a): partial overlap [a < c < b < d].
            m = (b + c) / 2.0
            out.append(_retime(s1, a, m))
            new_s2 = _retime(s2, m, d)
            work[0] = new_s2
            continue
        # Rule (b): full containment [a < c < d <= b].
        left, right = _split_segment_3way(s1, c, d)
        out.append(left)
        # s2 stays as-is at the head of `work`. The right half of
        # s1 must be re-inserted in time order so it gets compared
        # against everything that follows s2.
        _insert_sorted(work, right)
    # Drop zero / negative duration leftovers; renumber ids.
    out = [s for s in out if float(s["end"]) > float(s["start"])]
    for i, s in enumerate(out):
        s["id"] = i
    return out


def _insert_sorted(work: list[dict], seg: dict) -> None:
    """Insert `seg` into `work` keeping `work` sorted by start time
    (then end time)."""
    key = (float(seg["start"]), float(seg["end"]))
    lo, hi = 0, len(work)
    while lo < hi:
        mid = (lo + hi) // 2
        cur = work[mid]
        cur_key = (float(cur["start"]), float(cur["end"]))
        if cur_key < key:
            lo = mid + 1
        else:
            hi = mid
    work.insert(lo, seg)
