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

"""Audacity `.aup3` project import.

`.aup3` is a SQLite database with two project columns: a name table
in `project.dict` and a binary record stream in `project.doc` that
encodes the XML tree (Audacity's `ProjectSerializer` format, NOT
plain XML). Audio samples live in `sampleblocks.samples` as raw
PCM blobs.

Decoding the binary doc and reading sample blocks is the job of
the third-party `aup3` package (https://pypi.org/project/aup3/),
a pure-Python parser shipped under MIT. We use:

  * `AUP3.raw_project` to get the parsed XML tree as an `Element`
    NamedTuple (`tag`, `attrs`, `children`). Attributes come
    pre-typed: `attrs['rate']` is `('Double', 44100.0)`, etc.
    We use the raw tree (not the strict `AUP3.project` dataclass)
    so we don't break when newer Audacity versions add or rename
    attributes that the lib's dataclass schema doesn't know about.
  * `AUP3.get_block(blockid)` to fetch each sample block as a
    float32 numpy array normalised to [-1, +1].

The aup3 library opens a writable SQLite connection by default. We
bypass its constructor and slot in our own read-only connection
(`file:...?mode=ro`) so the source `.aup3` is never modified.

Output: WAVs are written next to the `.aup3` file in a sibling
directory `<basename>_tracks/`. They are NOT auto-deleted, so a
re-run uses the cached files (the `.done` marker is the cache key)
and the user can re-import or inspect them on disk.

Public API:
  * `is_aup3(path)`: extension check.
  * `extract_aup3(aup3_path, log)`: returns a list of
    `SpeakerInput` (one per WaveTrack, stereo pairs collapsed to
    mono), pointing at 16 kHz / 16-bit / mono WAVs.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import wave
from typing import Optional

import numpy as np

from .core import LogCb, SpeakerInput, vstderr


_TARGET_RATE = 16_000
_TARGET_SAMPWIDTH = 2  # 16-bit PCM
_DONE_MARKER = ".done"
_MANIFEST = "manifest.json"
_SUBDIR_SUFFIX = "_tracks"
# Bump when the extractor's output semantics change so stale caches
# from older versions are invalidated even though the source .aup3
# mtime hasn't moved. v2: per-track sample rate (was project rate).
_EXTRACTOR_VERSION = 2


def is_aup3(path: str) -> bool:
    """True when `path` looks like an Audacity 3 project file."""
    return os.path.splitext(path)[1].lower() == ".aup3"


def extract_label_tracks(aup3_path: str) -> list[dict]:
    """Return every `<labeltrack>` in an `.aup3` project as
    `[{"name": str, "labels": [{"t": float, "title": str}, ...]}]`.

    Cheap (XML-only, no audio); meant to be called when the GUI
    needs to show the available label tracks before kicking off
    transcription.
    """
    abs_path = os.path.abspath(aup3_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(abs_path)
    db = _open_readonly(abs_path)
    try:
        root = db.raw_project
    finally:
        try:
            db.close()
        except Exception:
            pass
    out: list[dict] = []
    for track in _children(root, "labeltrack"):
        labels = []
        for label in _children(track, "label"):
            t = float(_attr(label, "t", 0.0) or 0.0)
            title = str(_attr(label, "title", "") or "")
            labels.append({"t": t, "title": title})
        labels.sort(key=lambda d: d["t"])
        out.append(
            {
                "name": str(_attr(track, "name", "") or ""),
                "labels": labels,
            }
        )
    return out


def _sanitize(name: str, idx: int) -> str:
    """Reduce a track name to a filesystem-safe identifier while
    preserving accented characters (Unicode `\\w`). Falls back to
    `track_<idx>` when the input collapses to empty."""
    cleaned = re.sub(r"[^\w]+", "_", name or "", flags=re.UNICODE).strip("_")
    return cleaned or f"track_{idx}"


def _project_subdir(aup3_abs_path: str) -> str:
    """Return the sibling folder where extracted WAVs are written:
    `<aup3-dir>/<basename>_tracks/`. Persistent (not in /tmp) so
    re-runs are a cache hit and the user can inspect / reuse the
    extracted WAVs."""
    parent = os.path.dirname(aup3_abs_path) or "."
    base = os.path.splitext(os.path.basename(aup3_abs_path))[0]
    return os.path.join(parent, f"{base}{_SUBDIR_SUFFIX}")


def _emit(log: Optional[LogCb], msg: str) -> None:
    # Mirror to stderr when verbose so the terminal sees aup3
    # extraction progress alongside the GUI / CLI log routing.
    vstderr(msg)
    if log is not None:
        log(msg)


def _open_readonly(abs_path: str):
    """Build an `aup3.AUP3` instance backed by a read-only SQLite
    connection. The library's own constructor opens the file
    writable, which we don't want; we bypass it via __new__ and
    set the connection explicitly. `ignore_autosave` is implicitly
    True because we skip the constructor checks."""
    try:
        import aup3 as _aup3lib  # PyPI package, NOT this module
    except ImportError as exc:
        raise RuntimeError(
            "The 'aup3' package is required to import Audacity .aup3 "
            "projects. Install it with `pip install aup3`."
        ) from exc
    db = _aup3lib.AUP3.__new__(_aup3lib.AUP3)
    db.conn = sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)
    return db


def extract_aup3(
    aup3_path: str, log: Optional[LogCb] = None
) -> list[SpeakerInput]:
    """Extract every audio track from an `.aup3` Audacity project
    into 16 kHz mono 16-bit WAVs under the session temp dir.

    Returns a list of `SpeakerInput`, one per WaveTrack. Stereo
    tracks (a WaveTrack with `linked == 1` followed by its right
    channel) are downmixed to a single mono WAV.

    Re-extracting the same `.aup3` in the same session is a no-op:
    the cached WAVs and manifest are reused.
    """
    abs_path = os.path.abspath(aup3_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(abs_path)

    dest_dir = _project_subdir(abs_path)
    done_marker = os.path.join(dest_dir, _DONE_MARKER)
    manifest_path = os.path.join(dest_dir, _MANIFEST)
    if os.path.isfile(done_marker) and os.path.isfile(manifest_path):
        # Cache is valid only if the source .aup3 hasn't been
        # modified since extraction AND the manifest was written
        # by the current extractor version. The .done marker is
        # written last, so its mtime represents extraction
        # completion time.
        cache_fresh = (
            os.path.getmtime(abs_path) <= os.path.getmtime(done_marker)
        )
        if cache_fresh:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            # v1 wrote a bare list of entries; v2+ wraps them in
            # {"version": N, "entries": [...]}. Treat anything not
            # matching the current version as stale.
            if (
                isinstance(manifest, dict)
                and manifest.get("version") == _EXTRACTOR_VERSION
            ):
                _emit(
                    log,
                    f"Reusing cached aup3 extraction for {abs_path}",
                )
                return [
                    SpeakerInput(
                        speaker=e["speaker"], file_path=e["file_path"]
                    )
                    for e in manifest["entries"]
                ]
            _emit(
                log,
                f"Cached aup3 extraction is from an older extractor "
                f"version; re-extracting {abs_path}",
            )
        else:
            _emit(
                log,
                f"Source .aup3 modified since last extraction; "
                f"re-extracting {abs_path}",
            )
        try:
            os.remove(done_marker)
        except OSError:
            pass

    os.makedirs(dest_dir, exist_ok=True)
    db = _open_readonly(abs_path)
    try:
        speakers = _extract(db, dest_dir, log)
    finally:
        try:
            db.close()
        except Exception:
            pass

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "version": _EXTRACTOR_VERSION,
                "entries": [
                    {"speaker": s.speaker, "file_path": s.file_path}
                    for s in speakers
                ],
            },
            fh,
        )
    open(done_marker, "w").close()
    return speakers


def _attr(elem, name: str, default=None):
    """Read a typed attribute off an `aup3.xml.Element`. Each
    attribute is stored as `(type_name, value)`; we just want the
    value."""
    pair = elem.attrs.get(name)
    if pair is None:
        return default
    return pair[1]


def _children(elem, tag: str):
    """Iterate the immediate children of `elem` whose tag matches."""
    for child in elem.children:
        if getattr(child, "tag", None) == tag:
            yield child


def _extract(db, dest_dir: str, log: Optional[LogCb]) -> list[SpeakerInput]:
    root = db.raw_project
    project_rate = int(round(float(_attr(root, "rate", 44100.0))))
    wave_tracks = list(_children(root, "wavetrack"))

    # Pair stereo tracks: a wavetrack with `linked != 0` owns the
    # immediately following wavetrack as its right channel.
    # Audacity historically used 1 for stereo; recent versions use
    # 3 (a bitfield meaning "stereo + aligned"). Accept any non-
    # zero value as "this track is paired with the next".
    pairs: list[tuple[object, Optional[object]]] = []
    i = 0
    while i < len(wave_tracks):
        t = wave_tracks[i]
        linked = int(_attr(t, "linked", 0) or 0)
        if linked != 0 and i + 1 < len(wave_tracks):
            pairs.append((t, wave_tracks[i + 1]))
            i += 2
        else:
            pairs.append((t, None))
            i += 1

    used_names: dict[str, int] = {}
    speakers: list[SpeakerInput] = []
    for idx, (left, right) in enumerate(pairs):
        base_name = _sanitize(str(_attr(left, "name", "") or ""), idx)
        # Disambiguate when two tracks share a name so the second
        # WAV doesn't overwrite the first.
        used_names[base_name] = used_names.get(base_name, 0) + 1
        count = used_names[base_name]
        if count > 1:
            speaker = f"{base_name} ({count})"
            filename = f"{base_name}_{count}"
        else:
            speaker = base_name
            filename = base_name
        out_path = os.path.join(dest_dir, f"{filename}.wav")
        # Audacity stores a per-track sample rate on each WaveTrack.
        # Using the project's default rate for every track corrupts
        # tracks imported at a different rate (e.g. 48 kHz audio in
        # a 44.1 kHz project plays ~8.8% slow, and clip trim values
        # land on the wrong sample boundary).
        track_rate = int(
            round(float(_attr(left, "rate", project_rate)))
        )
        _emit(
            log,
            f"Extracting aup3 track {idx + 1}/{len(pairs)}: {speaker} "
            f"(rate {track_rate} Hz, "
            f"{'stereo' if right is not None else 'mono'})",
        )
        left_samples = _track_samples(db, left, track_rate, log)
        if right is not None:
            right_samples = _track_samples(db, right, track_rate, log)
            length = max(len(left_samples), len(right_samples))
            left_samples = _pad_to(left_samples, length)
            right_samples = _pad_to(right_samples, length)
            mono = (left_samples + right_samples) * 0.5
        else:
            mono = left_samples
        resampled = _resample_to_16k(mono, track_rate)
        _write_wav(out_path, resampled)
        speakers.append(SpeakerInput(speaker=speaker, file_path=out_path))

    return speakers


def _track_samples(
    db, track, track_rate: int, log: Optional[LogCb]
) -> np.ndarray:
    """Decode one wavetrack Element to a float32 timeline.

    For each waveclip:
      * Build a sequence-local buffer covering its blocks at
        their `start` sample positions.
      * Apply `trimLeft` / `trimRight` (seconds): Audacity stores
        the entire un-trimmed source in `sampleblocks` and uses
        these attributes to define what part is actually visible.
        Without this trim the WAV emits raw source content the
        user already cut.
      * Place the trimmed slice at `(offset + trimLeft) * rate`
        on the global timeline. Audacity's `offset` is
        `mSequenceOffset` (where the source's t=0 lands), and
        the visible play-start is `offset + trimLeft`. Placing
        at just `offset` is wrong on multi-clip tracks where
        different clips have different trim values, because the
        visible content slides relative to the timeline by the
        per-clip `trimLeft`.

    Gaps between clips remain silent (the global buffer is
    pre-zeroed). Negative play-start values pre-roll into t<0
    and get truncated to keep all speaker streams aligned to a
    common origin.
    """
    placed: list[tuple[int, np.ndarray]] = []
    total_end = 0
    for clip in _children(track, "waveclip"):
        offset_s = float(_attr(clip, "offset", 0.0) or 0.0)
        trim_left_s = float(_attr(clip, "trimLeft", 0.0) or 0.0)
        trim_right_s = float(_attr(clip, "trimRight", 0.0) or 0.0)
        offset_samples = int(round(offset_s * track_rate))
        trim_left_samples = max(0, int(round(trim_left_s * track_rate)))
        trim_right_samples = max(0, int(round(trim_right_s * track_rate)))

        clip_blocks: list[tuple[int, np.ndarray]] = []
        clip_local_end = 0
        for sequence in _children(clip, "sequence"):
            for block in _children(sequence, "waveblock"):
                blockid = _attr(block, "blockid")
                if blockid is None:
                    continue
                try:
                    samples = db.get_block(int(blockid))
                except Exception as exc:
                    _emit(
                        log,
                        f"  warning: could not read blockid "
                        f"{blockid}: {exc}",
                    )
                    continue
                samples = np.asarray(samples, dtype=np.float32)
                rel_start = int(_attr(block, "start", 0) or 0)
                clip_blocks.append((rel_start, samples))
                clip_local_end = max(
                    clip_local_end, rel_start + len(samples)
                )
        if not clip_blocks or clip_local_end <= 0:
            continue

        clip_buf = np.zeros(clip_local_end, dtype=np.float32)
        for rel_start, samples in clip_blocks:
            if rel_start < 0:
                # Block extends before clip-local t=0; truncate.
                if rel_start + len(samples) <= 0:
                    continue
                samples = samples[-rel_start:]
                rel_start = 0
            clip_buf[rel_start : rel_start + len(samples)] = samples

        # Apply trim: slice off hidden head + tail.
        visible_lo = min(trim_left_samples, len(clip_buf))
        visible_hi = max(visible_lo, len(clip_buf) - trim_right_samples)
        visible = clip_buf[visible_lo:visible_hi]
        if len(visible) == 0:
            continue

        # Visible content lands at the play-start, not at the
        # sequence-start. Audacity's `offset` attribute is
        # `mSequenceOffset`; the true play-start is
        # `offset + trimLeft`.
        play_start = offset_samples + trim_left_samples
        placed.append((play_start, visible))
        total_end = max(total_end, play_start + len(visible))

    if not placed or total_end <= 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(total_end, dtype=np.float32)
    for start, samples in placed:
        end = start + len(samples)
        if end <= 0:
            continue
        if start < 0:
            samples = samples[-start:]
            start = 0
        out[start : start + len(samples)] = samples
    return out


def _pad_to(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr
    out = np.zeros(n, dtype=np.float32)
    out[: len(arr)] = arr
    return out


def _resample_to_16k(samples: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == _TARGET_RATE or len(samples) == 0:
        return samples.astype(np.float32, copy=False)
    from scipy.signal import resample_poly  # type: ignore

    g = math.gcd(src_rate, _TARGET_RATE)
    up = _TARGET_RATE // g
    down = src_rate // g
    return resample_poly(samples, up, down).astype(np.float32, copy=False)


def _write_wav(path: str, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(_TARGET_SAMPWIDTH)
        wf.setframerate(_TARGET_RATE)
        wf.writeframes(pcm.tobytes())
