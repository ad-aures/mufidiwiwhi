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

"""Tests for the GUI worker thread, with run_pipeline mocked out."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QCoreApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv[:1])
    return app


def _build_cfg():
    from mufidiwiwhi.core import RunConfig, SpeakerInput

    return RunConfig(
        speakers=[SpeakerInput("A", "/tmp/a.wav")],
        model_name="tiny",
        output_dir="/tmp",
    )


def test_worker_emits_finished_ok(monkeypatch, qapp):
    from mufidiwiwhi.gui import worker as worker_mod

    captured = {}

    def fake_run(cfg, *, progress=None, log=None, log_html=None, cancel=None):
        captured["cfg"] = cfg
        if progress is not None:
            progress("Transcribing A", 1.0)
        if log is not None:
            log("ok")
        return {"output_paths": ["/tmp/out.srt"], "segments": []}

    monkeypatch.setattr(worker_mod, "run_pipeline", fake_run)

    w = worker_mod.TranscriptionWorker(_build_cfg())
    received_paths: list = []
    log_lines: list = []
    file_progress: list = []
    w.finished_ok.connect(lambda paths: received_paths.extend(paths))
    w.log.connect(lambda msg: log_lines.append(msg))
    w.progress_file.connect(lambda spk, frac: file_progress.append((spk, frac)))

    w.run()  # run on the calling thread, no event loop needed
    assert received_paths == ["/tmp/out.srt"]
    assert "ok" in log_lines
    assert file_progress and file_progress[0][0] == "A"
    assert captured["cfg"].speakers[0].speaker == "A"


def test_worker_emits_cancelled(monkeypatch, qapp):
    from mufidiwiwhi.core import Cancelled
    from mufidiwiwhi.gui import worker as worker_mod

    def fake_run(cfg, *, progress=None, log=None, log_html=None, cancel=None):
        raise Cancelled()

    monkeypatch.setattr(worker_mod, "run_pipeline", fake_run)
    w = worker_mod.TranscriptionWorker(_build_cfg())
    cancelled = []
    w.finished_cancelled.connect(lambda: cancelled.append(True))
    w.run()
    assert cancelled == [True]


def test_worker_emits_error(monkeypatch, qapp):
    from mufidiwiwhi.gui import worker as worker_mod

    def fake_run(cfg, *, progress=None, log=None, log_html=None, cancel=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(worker_mod, "run_pipeline", fake_run)
    w = worker_mod.TranscriptionWorker(_build_cfg())
    errors: list = []
    w.error.connect(lambda msg: errors.append(msg))
    w.run()
    assert errors and "boom" in errors[0]
