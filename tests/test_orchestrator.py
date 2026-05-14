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

"""Logical tests for the chunk-interleaved orchestrator.

These tests stub out `_iter_chunks_from` and `model.transcribe` so
they exercise only the orchestrator's pick-by-earliest-start logic
and per-track state machine. The real Whisper invocation, audio
decoding, and correction passes are covered elsewhere (and require
a model + audio fixture); here we just want to prove the
interleaving order matches what the legacy per-speaker loop would
produce after `merge_segments` sorts everything.
"""

from __future__ import annotations

import types
from typing import Iterator

from mufidiwiwhi import orchestrator as _o
from mufidiwiwhi import transcribe as _t
from mufidiwiwhi.core import RunConfig, SpeakerInput


def _make_cfg() -> RunConfig:
    """Minimal RunConfig good enough for the orchestrator path.
    The orchestrator only reads cfg.language; everything else is
    forwarded to process_chunk, which is stubbed out here.
    """
    return RunConfig(
        speakers=[],
        model_name="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        output_dir=".",
        output_filename="out",
        output_formats=("srt",),
    )


def _stub_chunks(*ranges: tuple[int, int]) -> Iterator[tuple[int, int, object]]:
    """Yield (start_ms, end_ms, sentinel) tuples mimicking
    `_iter_chunks_from`. The third element is opaque to the
    orchestrator because process_chunk is stubbed too; we just
    need something hashable / identifiable per chunk.
    """
    for start, end in ranges:
        yield (start, end, ("audio", start, end))


def _build_orchestrator(monkeypatch_state, speakers_chunks):
    """Wire init_tracks to use the synthetic chunk streams instead
    of decoding real audio. `speakers_chunks` maps speaker name to
    a list of (start_ms, end_ms) tuples. Each track gets its total
    duration set to the end of its last chunk, or 0 if empty.
    """
    chunk_streams: dict[str, list[tuple[int, int]]] = {}
    for spk, chunks in speakers_chunks.items():
        chunk_streams[spk] = list(chunks)

    # Stub AudioSegment.from_file to a fake with __len__ matching
    # the last chunk end. The orchestrator never reaches into the
    # audio object's contents because process_chunk is stubbed.
    class _FakeAudio:
        def __init__(self, total_ms: int) -> None:
            self.total_ms = total_ms

        def __len__(self) -> int:
            return self.total_ms

    fake_module = types.SimpleNamespace(
        AudioSegment=types.SimpleNamespace(
            from_file=lambda path: _FakeAudio(
                max(
                    (end for _, end in chunk_streams.get(_path_to_spk(path), [])),
                    default=0,
                )
            )
        )
    )

    def _path_to_spk(path: str) -> str:
        return path.replace(".wav", "")

    # Replace the orchestrator's deferred `from pydub import
    # AudioSegment` with our stub by monkeypatching the pydub
    # module before init_tracks does its import-then-call.
    monkeypatch_state["pydub"] = fake_module

    # Replace _iter_chunks_from with one that emits our synthetic
    # tuples per speaker. We dispatch by inspecting the audio
    # object's identity / total_ms back to the speaker name.
    def fake_iter_chunks(audio, log=None):
        total_ms = audio.total_ms
        for spk, chunks in chunk_streams.items():
            last_end = max((end for _, end in chunks), default=0)
            if last_end == total_ms:
                return _stub_chunks(*chunks)
        return iter(())

    monkeypatch_state["_iter_chunks_from"] = fake_iter_chunks


def _install_stubs(speakers_chunks, recorded_calls):
    """Install the monkeypatches needed for a synthetic run.
    Returns a callable that undoes them, for test cleanup."""
    import sys

    real_pydub = sys.modules.get("pydub")
    real_iter = _t._iter_chunks_from
    real_process = _t.process_chunk

    chunk_streams = {spk: list(chunks) for spk, chunks in speakers_chunks.items()}
    total_by_total = {
        max((end for _, end in chunks), default=0): spk
        for spk, chunks in chunk_streams.items()
    }

    class _FakeAudio:
        def __init__(self, total_ms: int) -> None:
            self.total_ms = total_ms

        def __len__(self) -> int:
            return self.total_ms

    def _from_file(path):
        spk = path.replace(".wav", "")
        last_end = max((end for _, end in chunk_streams.get(spk, [])), default=0)
        return _FakeAudio(last_end)

    fake_pydub = types.SimpleNamespace(
        AudioSegment=types.SimpleNamespace(from_file=_from_file)
    )
    sys.modules["pydub"] = fake_pydub

    def fake_iter_chunks(audio, log=None):
        spk = total_by_total.get(audio.total_ms)
        if spk is None:
            return iter(())
        return _stub_chunks(*chunk_streams[spk])

    _t._iter_chunks_from = fake_iter_chunks

    def fake_process_chunk(
        model,
        speaker,
        chunk_index,
        start_ms,
        end_ms,
        chunk_audio,
        cfg,
        base_seg_id,
        detected_language,
        **kw,
    ):
        # Record the dispatch order. Return one segment per chunk
        # with deterministic timing so the test can verify the
        # final flat list is what the legacy per-speaker loop +
        # merge_segments would produce.
        recorded_calls.append((speaker, start_ms, end_ms))
        seg = {
            "id": base_seg_id,
            "seek": start_ms / 1000.0,
            "start": start_ms / 1000.0,
            "end": end_ms / 1000.0,
            "speaker": speaker,
            "text": f"[{speaker}] {start_ms}-{end_ms}",
            "words": None,
            "language": detected_language or "fr",
        }
        return [seg], detected_language or "fr"

    _t.process_chunk = fake_process_chunk

    def undo():
        if real_pydub is None:
            sys.modules.pop("pydub", None)
        else:
            sys.modules["pydub"] = real_pydub
        _t._iter_chunks_from = real_iter
        _t.process_chunk = real_process

    return undo


def test_single_track_pulls_chunks_in_emitted_order():
    """One speaker, three chunks. The orchestrator must dispatch
    them in the order the generator yields them. This is the
    minimum sanity check; nothing about cross-track interleaving
    matters with one track.
    """
    recorded: list[tuple] = []
    undo = _install_stubs(
        {"Alice": [(0, 5000), (5000, 12000), (12000, 20000)]}, recorded
    )
    try:
        cfg = _make_cfg()
        tracks = _o.init_tracks(
            [SpeakerInput(speaker="Alice", file_path="Alice.wav")], cfg
        )
        out = _o.run_parallel(tracks, model=None, cfg=cfg)
    finally:
        undo()
    assert recorded == [
        ("Alice", 0, 5000),
        ("Alice", 5000, 12000),
        ("Alice", 12000, 20000),
    ]
    assert [s["text"] for s in out] == [
        "[Alice] 0-5000",
        "[Alice] 5000-12000",
        "[Alice] 12000-20000",
    ]


def test_two_tracks_interleave_by_chunk_start():
    """Alice: chunks at 0, 10s. Bob: chunks at 3s, 13s. The
    orchestrator must dispatch Alice@0, Bob@3000, Alice@10000,
    Bob@13000 in that order.
    """
    recorded: list[tuple] = []
    undo = _install_stubs(
        {
            "Alice": [(0, 9000), (10000, 19000)],
            "Bob": [(3000, 11000), (13000, 21000)],
        },
        recorded,
    )
    try:
        cfg = _make_cfg()
        tracks = _o.init_tracks(
            [
                SpeakerInput(speaker="Alice", file_path="Alice.wav"),
                SpeakerInput(speaker="Bob", file_path="Bob.wav"),
            ],
            cfg,
        )
        out = _o.run_parallel(tracks, model=None, cfg=cfg)
    finally:
        undo()
    assert recorded == [
        ("Alice", 0, 9000),
        ("Bob", 3000, 11000),
        ("Alice", 10000, 19000),
        ("Bob", 13000, 21000),
    ]


def test_ties_break_by_track_index():
    """Both tracks have a chunk starting at exactly 0 ms. Tie
    must go to the track with the lower index, which is the
    order the speakers were passed in. Determinism matters so
    runs are reproducible.
    """
    recorded: list[tuple] = []
    undo = _install_stubs(
        {
            "First": [(0, 5000)],
            "Second": [(0, 5000)],
        },
        recorded,
    )
    try:
        cfg = _make_cfg()
        tracks = _o.init_tracks(
            [
                SpeakerInput(speaker="First", file_path="First.wav"),
                SpeakerInput(speaker="Second", file_path="Second.wav"),
            ],
            cfg,
        )
        _o.run_parallel(tracks, model=None, cfg=cfg)
    finally:
        undo()
    assert [r[0] for r in recorded] == ["First", "Second"]


def test_shorter_track_exhausts_first_then_longer_continues():
    """Alice has one short chunk; Bob has three. After Alice's
    only chunk is dispatched, Bob's remaining chunks must be
    processed in order without the loop hanging or skipping any.
    """
    recorded: list[tuple] = []
    undo = _install_stubs(
        {
            "Alice": [(0, 2000)],
            "Bob": [(1000, 5000), (5000, 10000), (10000, 15000)],
        },
        recorded,
    )
    try:
        cfg = _make_cfg()
        tracks = _o.init_tracks(
            [
                SpeakerInput(speaker="Alice", file_path="Alice.wav"),
                SpeakerInput(speaker="Bob", file_path="Bob.wav"),
            ],
            cfg,
        )
        _o.run_parallel(tracks, model=None, cfg=cfg)
    finally:
        undo()
    assert recorded == [
        ("Alice", 0, 2000),
        ("Bob", 1000, 5000),
        ("Bob", 5000, 10000),
        ("Bob", 10000, 15000),
    ]


def test_progress_reports_whole_pipeline_completion():
    """Progress callback must weight chunk completion against the
    SUM of all tracks' durations, not per-speaker. With Alice
    finishing first (10s of 30s total) and Bob then finishing
    (20s of 30s total), the bar should pass 10/30 then 30/30.
    """
    recorded: list[tuple] = []
    progress_calls: list[tuple[str, float]] = []
    undo = _install_stubs(
        {
            "Alice": [(0, 10000)],
            "Bob": [(2000, 22000)],  # total 20s, dispatched second (start 2000)
        },
        recorded,
    )
    try:
        cfg = _make_cfg()
        tracks = _o.init_tracks(
            [
                SpeakerInput(speaker="Alice", file_path="Alice.wav"),
                SpeakerInput(speaker="Bob", file_path="Bob.wav"),
            ],
            cfg,
        )
        _o.run_parallel(
            tracks,
            model=None,
            cfg=cfg,
            progress=lambda label, frac: progress_calls.append((label, frac)),
        )
    finally:
        undo()
    # Two chunks, two progress callbacks. Total audio = 10s + 20s
    # = 30s. After Alice's chunk: 10s done -> 1/3. After Bob's:
    # 30s done -> 1.0. Order of dispatch was Alice (start 0) then
    # Bob (start 2000).
    assert len(progress_calls) == 2
    label_a, frac_a = progress_calls[0]
    label_b, frac_b = progress_calls[1]
    assert label_a == "Transcribing Alice"
    assert label_b == "Transcribing Bob"
    assert abs(frac_a - (10000 / 32000)) < 1e-6
    assert abs(frac_b - 1.0) < 1e-6
