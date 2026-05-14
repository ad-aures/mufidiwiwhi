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

"""Cross-track parallel orchestration of chunked transcription.

Companion to `transcribe.transcribe_speaker`, which processes one
speaker file end-to-end before moving on to the next. The
orchestrator here interleaves all speaker tracks by chunk start
time, so the segment list grows chronologically as chunks come back
instead of being assembled per-speaker and merged at the end.

Algorithm:
  1. For each speaker, decode the audio once and prime the
     `_iter_chunks_from` generator. Peek one chunk ahead per track.
  2. While any track still has a peeked chunk, pick the track whose
     peeked chunk has the earliest start_ms. Tie-break by track
     index (stable, deterministic) so two tracks with identical
     start_ms are processed in the order the speakers were given.
  3. Hand that chunk to `transcribe.process_chunk`, which makes the
     single Whisper invocation and runs per-chunk correction
     in-place. Append the returned segment dicts to a flat list.
  4. Pull the next chunk from the chosen track's generator into
     `peeked` (None when exhausted) and loop.

Whisper calls stay serial: one `model.transcribe()` in flight at
any moment, matching the legacy path's GPU usage profile. The
"parallel" in the module name refers to the LOGICAL interleaving
of tracks, not concurrent decoding.

The returned flat list is in chunk-arrival (chronological by
chunk start) order, NOT fully sorted by segment start, because
chunks from different tracks can produce overlapping segment time
ranges. Callers feed it to `transcribe.merge_segments` /
`resolve_segment_overlaps` exactly as today, which sorts and
resolves the final transcript.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from .core import (
    Cancelled,
    CancelCb,
    LogCb,
    ProgressCb,
    RunConfig,
    SpeakerInput,
    vstderr,
)
from . import transcribe as _t


@dataclass
class TrackState:
    """Per-track mutable state held by the orchestrator loop.

    `chunks` is the live `_iter_chunks_from` generator. `peeked`
    holds the next chunk waiting to be processed, or `None` when
    the generator is exhausted (so `exhausted` is just `peeked is
    None` once `init_tracks` has primed each track).

    `segments_done` counts segments already emitted for this
    speaker and feeds `base_seg_id` so each `process_chunk` call
    hands out unique-within-track ids. Final ids are reassigned
    globally in `merge_segments`, but the local ids let log lines
    and any in-flight references stay coherent.

    `processed_ms` is the audio position consumed so far (the end
    of the most recently transcribed chunk). The orchestrator uses
    the sum across all tracks to drive whole-pipeline progress.
    """

    speaker: str
    audio_path: str
    audio: Any  # pydub.AudioSegment
    total_ms: int
    chunks: Iterator[tuple[int, int, Any]]
    peeked: Optional[tuple[int, int, Any]] = None
    detected_language: Optional[str] = None
    chunk_index: int = 0
    segments_done: int = 0
    processed_ms: int = 0

    @property
    def exhausted(self) -> bool:
        return self.peeked is None


def init_tracks(
    speakers: list[SpeakerInput],
    cfg: RunConfig,
    *,
    log: Optional[LogCb] = None,
) -> list[TrackState]:
    """Decode each speaker's audio and prime its chunk generator.

    Mirrors the per-speaker setup that lives at the top of
    `transcribe.transcribe_speaker` (file open, AudioSegment
    decode, total_ms, detected_language seeded from `cfg.language`,
    the same "Audio duration X.Ys" log line, the same vstderr
    breadcrumbs around AudioSegment.from_file). The chunk
    generator is `transcribe._iter_chunks_from`, called with the
    decoded AudioSegment exactly as the legacy path does.

    Each returned track has its `peeked` slot already populated
    with the first chunk, or `None` if the audio contains only
    silence and the generator yields nothing.
    """
    from pydub import AudioSegment  # type: ignore

    tracks: list[TrackState] = []
    for spk in speakers:
        if log is not None:
            log(f"Opening {spk.file_path!r} for speaker '{spk.speaker}'")
        vstderr(
            f"[{spk.speaker}] AudioSegment.from_file("
            f"{spk.file_path!r}) starting..."
        )
        t_load = time.monotonic()
        audio = AudioSegment.from_file(spk.file_path)
        vstderr(
            f"[{spk.speaker}] AudioSegment.from_file done in "
            f"{time.monotonic() - t_load:.2f}s"
        )
        total_ms = len(audio)
        duration_s = total_ms / 1000.0
        if log is not None:
            log(
                f"Audio duration {duration_s:.1f}s. "
                f"Chunking with explicit RMS-minima cuts (no VAD)."
            )
        track = TrackState(
            speaker=spk.speaker,
            audio_path=spk.file_path,
            audio=audio,
            total_ms=total_ms,
            chunks=_t._iter_chunks_from(audio, log=log),
            detected_language=cfg.language,
        )
        _advance(track)
        tracks.append(track)
    return tracks


def _advance(track: TrackState) -> None:
    """Pull the next chunk from `track.chunks` into `track.peeked`.

    Sets `peeked` to `None` when the generator raises
    StopIteration, which is how the main loop detects exhausted
    tracks.
    """
    try:
        track.peeked = next(track.chunks)
    except StopIteration:
        track.peeked = None


def run_parallel(
    tracks: list[TrackState],
    model: Any,
    cfg: RunConfig,
    *,
    correction_state: Any = None,
    log: Optional[LogCb] = None,
    log_html: Optional[LogCb] = None,
    cancel: Optional[CancelCb] = None,
    progress: Optional[ProgressCb] = None,
) -> list[dict]:
    """Drive all tracks in chronological-by-chunk-start order.

    On each iteration the un-exhausted track whose peeked chunk
    has the smallest `start_ms` is selected (ties broken by track
    index). Its chunk is run through `transcribe.process_chunk`,
    which performs the single `model.transcribe()` call and applies
    per-chunk correction in place; the returned segment dicts are
    appended to the running flat list and that track is advanced
    to its next chunk.

    Progress, when wired, reports `total_consumed_ms /
    total_all_tracks_ms` so the bar reflects whole-pipeline
    completion across speakers, not per-speaker as the legacy loop
    did. Cancellation goes through the same `Cancelled` exception
    `process_chunk` already raises; the loop also checks the cancel
    callback once per iteration so a click between chunks is
    honoured promptly.

    Returns the flat list of segment dicts, in chunk-arrival
    order. The downstream `merge_segments` + `resolve_segment_overlaps`
    pass runs unchanged in `core.run_pipeline` and produces a
    transcript byte-identical to the legacy path on the same
    inputs.
    """
    total_ms_all = sum(t.total_ms for t in tracks)
    out: list[dict] = []
    while any(not t.exhausted for t in tracks):
        if cancel and cancel():
            raise Cancelled()
        chosen_idx = -1
        chosen_start: Optional[int] = None
        for i, t in enumerate(tracks):
            if t.exhausted:
                continue
            start = t.peeked[0]
            if chosen_start is None or start < chosen_start:
                chosen_start = start
                chosen_idx = i
        if chosen_idx < 0:
            break
        track = tracks[chosen_idx]
        assert track.peeked is not None  # invariant: not exhausted
        chunk_start_ms, chunk_end_ms, chunk_audio = track.peeked
        track.chunk_index += 1
        chunk_segs, track.detected_language = _t.process_chunk(
            model,
            track.speaker,
            track.chunk_index,
            chunk_start_ms,
            chunk_end_ms,
            chunk_audio,
            cfg,
            track.segments_done,
            track.detected_language,
            correction_state=correction_state,
            log=log,
            log_html=log_html,
            cancel=cancel,
        )
        out.extend(chunk_segs)
        track.segments_done += len(chunk_segs)
        track.processed_ms = chunk_end_ms
        if progress is not None and total_ms_all > 0:
            done = sum(t.processed_ms for t in tracks)
            progress(
                f"Transcribing {track.speaker}",
                min(1.0, done / total_ms_all),
            )
        _advance(track)
    if log is not None:
        for t in tracks:
            log(f"Finished '{t.speaker}': {t.segments_done} segments")
    return out
