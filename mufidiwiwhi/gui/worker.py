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

"""Background worker that runs the transcription pipeline.

A `QThread` so the Qt event loop stays responsive. The worker emits
signals the Run page connects to. Cancellation is cooperative: the
worker checks a flag between speakers and between segments inside
`transcribe_speaker`.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ..core import Cancelled, RunConfig, run_pipeline


class TranscriptionWorker(QThread):
    started_run = pyqtSignal()
    progress_overall = pyqtSignal(float)
    progress_file = pyqtSignal(str, float)
    log = pyqtSignal(str)
    log_html = pyqtSignal(str)
    file_done = pyqtSignal(str, int)
    correction_started = pyqtSignal()
    error = pyqtSignal(str)
    finished_ok = pyqtSignal(list)
    finished_cancelled = pyqtSignal()

    def __init__(self, cfg: RunConfig, parent=None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _on_progress(self, label: str, frac: float) -> None:
        if label.startswith("Transcribing "):
            speaker = label[len("Transcribing "):]
            self.progress_file.emit(speaker, float(frac))
            return
        if label == "Applying corrections":
            self.correction_started.emit()
            return
        self.progress_overall.emit(float(frac))

    def _on_log(self, msg: str) -> None:
        self.log.emit(str(msg))

    def _on_log_html(self, msg: str) -> None:
        self.log_html.emit(str(msg))

    def _on_cancel(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        self.started_run.emit()
        try:
            result = run_pipeline(
                self._cfg,
                progress=self._on_progress,
                log=self._on_log,
                log_html=self._on_log_html,
                cancel=self._on_cancel,
            )
        except Cancelled:
            self.finished_cancelled.emit()
            return
        except Exception as exc:  # surface any pipeline error to the UI
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished_ok.emit(list(result.get("output_paths", [])))
