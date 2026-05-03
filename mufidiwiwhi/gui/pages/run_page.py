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

"""Run page: progress, live log, cancel, summary."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...core import RunConfig
from ..settings import SettingsManager
from ..widgets.dual_progress import DualColorProgressBar
from ..widgets.icons import labeled_icon_button
from ..widgets.switch import Switch
from ..worker import TranscriptionWorker


# Maximum log entries drained in a single timer tick. Bigger means
# fewer paints (faster overall) but each tick blocks the event loop
# longer. 200 keeps a single tick under ~50 ms on typical segments
# while still keeping up with even very chatty runs.
_LOG_DRAIN_BATCH = 200


class RunPage(QWidget):
    """Live progress + log + cancel; turns into a summary at the end."""

    newProjectRequested = pyqtSignal()
    runStateChanged = pyqtSignal(bool)  # True = running, False = idle
    # text + tone ('', 'good', 'warn'). Drives the centered title at
    # the top of the right content panel; nothing else listens.
    statusChanged = pyqtSignal(str, str)

    def __init__(
        self,
        settings: SettingsManager,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._worker: Optional[TranscriptionWorker] = None
        self._cfg: Optional[RunConfig] = None
        self._start_time: float = 0.0
        self._status_text: str = ""
        self._status_tone: str = ""
        # Set in `_append` so the scrollbar valueChanged listener
        # ignores our OWN cursor-follow scrolls; only USER-initiated
        # scrolls disable autoscroll.
        self._suppress_scroll_signal: bool = False
        # Log batching: incoming log signals queue up here and a
        # single-shot timer flushes them in one append+paint pass.
        # Without this, dozens of per-segment appendHtml calls per
        # second saturate the GUI's event loop and the entire app
        # stops responding to clicks and scroll. 250 ms is fast
        # enough to feel live, slow enough to amortise the cost.
        self._log_queue: list[tuple[bool, str]] = []  # (is_html, text)
        self._build_ui()
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(250)
        self._log_flush_timer.setSingleShot(True)
        self._log_flush_timer.timeout.connect(self._flush_log_queue)
        self._set_status(self.tr("Idle"), "")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        self.overall_bar = DualColorProgressBar()
        self.overall_bar.setObjectName("overallBar")
        self.overall_bar.setRange(0, 100)
        outer.addWidget(self.overall_bar)

        self.file_bar = DualColorProgressBar()
        self.file_bar.setObjectName("fileBar")
        self.file_bar.setRange(0, 100)
        outer.addWidget(self.file_bar)

        from ..widgets.metrics_strip import MetricsStrip

        self.metrics_strip = MetricsStrip(self._settings, self)
        outer.addWidget(self.metrics_strip)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        # Cap the log document at a fixed number of blocks. Without
        # this, a long transcription accumulates tens of thousands
        # of HTML-formatted blocks and every append re-lays-out the
        # whole document, making the entire GUI unresponsive (the
        # signal pump on the main thread saturates). Qt drops the
        # oldest blocks automatically once the cap is reached.
        self.log_view.setMaximumBlockCount(5000)
        # The QSS uses a universal `*` selector which would otherwise
        # override the QFont we set programmatically. Tag the widget
        # with an objectName so the stylesheet can re-apply a
        # monospace family specifically for it.
        self.log_view.setObjectName("runLog")
        log_font = QFont("DejaVu Sans Mono")
        log_font.setStyleHint(QFont.StyleHint.Monospace)
        log_font.setFixedPitch(True)
        log_font.setPointSize(9)
        self.log_view.setFont(log_font)
        outer.addWidget(self.log_view, stretch=1)
        # Watch for user-initiated scrolls: if the user drags the
        # scrollbar away from the bottom while autoscroll is on,
        # disable autoscroll so we don't yank them back to the
        # tail every time a new segment streams in. Our own
        # cursor-follow scrolls are masked via
        # `_suppress_scroll_signal` inside `_append`.
        self.log_view.verticalScrollBar().valueChanged.connect(
            self._on_scroll_value_changed
        )

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.setContentsMargins(0, 8, 0, 0)
        # Pill-style switch + label, matching the Hunspell toggle on
        # the Settings tab. Clicking the label also toggles the
        # switch so the whole "Autoscroll log" pair behaves like one
        # control.
        self.autoscroll_check = Switch()
        self.autoscroll_check.setChecked(True)
        autoscroll_label = QLabel(self.tr("Autoscroll log"))
        autoscroll_label.setCursor(Qt.CursorShape.PointingHandCursor)
        autoscroll_label.mouseReleaseEvent = (  # type: ignore[assignment]
            lambda _e: self.autoscroll_check.toggle()
        )
        bottom.addWidget(self.autoscroll_check)
        bottom.addSpacing(4)
        bottom.addWidget(autoscroll_label)
        bottom.addStretch(1)
        # Elapsed-time readout: a small monospace label that ticks
        # at 1 Hz while a transcription is running. Sits next to the
        # action buttons so it's in the user's eye line when they're
        # watching progress.
        # Use a non-empty placeholder so the first layout pass
        # reserves the right width; the timer flips this to live
        # values once a run starts. Min width is set generously so
        # crossing the H:MM:SS boundary doesn't shift the buttons.
        self.elapsed_label = QLabel("Elapsed 00:00")
        self.elapsed_label.setObjectName("runElapsed")
        self.elapsed_label.setMinimumWidth(110)
        self.elapsed_label.setToolTip(self.tr("Elapsed time since the run started."))
        bottom.addWidget(self.elapsed_label)
        bottom.addSpacing(8)
        self.cancel_btn = labeled_icon_button(
            self.tr("Stop"),
            "stop-circle-svgrepo-com.svg",
            destructive=True,
            tooltip=self.tr("Stop the running transcription."),
        )
        self.cancel_btn.setEnabled(False)
        self.open_btn = labeled_icon_button(
            self.tr("Open output folder"),
            "folder-open-svgrepo-com.svg",
            tooltip=self.tr("Open the destination folder in your file manager."),
        )
        self.open_btn.setEnabled(False)
        self.new_btn = labeled_icon_button(
            self.tr("New project"),
            "add-folder-svgrepo-com.svg",
            tooltip=self.tr("Return to the Project tab to start a new run."),
        )
        self.new_btn.setEnabled(False)
        bottom.addWidget(self.cancel_btn)
        bottom.addWidget(self.open_btn)
        bottom.addWidget(self.new_btn)
        outer.addLayout(bottom)

        self.cancel_btn.clicked.connect(self._on_cancel)
        self.open_btn.clicked.connect(self._open_output_folder)
        self.new_btn.clicked.connect(self.newProjectRequested.emit)

        # Live re-highlight when new log content arrives. Throttled
        # to avoid lag during heavy log streaming.
        self._search_query: str = ""
        self._search_matches: list[tuple[int, int]] = []
        self._search_current: int = 0
        self._search_throttle = QTimer(self)
        self._search_throttle.setInterval(150)
        self._search_throttle.setSingleShot(True)
        self._search_throttle.timeout.connect(self._reapply_search_highlights)
        # 1 Hz ticker that refreshes the elapsed-time label while a
        # run is active. Kicked off in `start()` and stopped in the
        # finish / cancel / error handlers.
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        # The placeholder set during widget construction was just
        # for layout sizing; the user shouldn't see it at idle.
        self._tick_elapsed()
        self.log_view.textChanged.connect(self._on_log_text_changed)

    # ----- public search API --------------------------------------------
    def apply_search(self, text: str) -> tuple[int, int]:
        """Set / replace the search query. Highlights all matches and
        scrolls to the first one. Returns ``(current_1_based, total)``;
        ``(0, 0)`` when the query is empty or no matches."""
        self._search_query = text or ""
        self._search_current = 0
        self._compute_matches()
        self._update_extra_selections()
        if self._search_matches:
            self._scroll_to_current()
        return self._search_stats()

    def next_match(self) -> tuple[int, int]:
        if not self._search_matches:
            return (0, 0)
        self._search_current = (self._search_current + 1) % len(
            self._search_matches
        )
        self._update_extra_selections()
        self._scroll_to_current()
        return self._search_stats()

    def prev_match(self) -> tuple[int, int]:
        if not self._search_matches:
            return (0, 0)
        self._search_current = (self._search_current - 1) % len(
            self._search_matches
        )
        self._update_extra_selections()
        self._scroll_to_current()
        return self._search_stats()

    def clear_search(self) -> None:
        self._search_query = ""
        self._search_matches = []
        self._search_current = 0
        self.log_view.setExtraSelections([])

    # ----- internal search helpers --------------------------------------
    _MATCH_BG = QColor("#ffe066")
    _CURRENT_MATCH_BG = QColor("#ffa94d")

    def _on_log_text_changed(self) -> None:
        if self._search_query:
            self._search_throttle.start()

    def _reapply_search_highlights(self) -> None:
        self._compute_matches()
        self._update_extra_selections()

    def _compute_matches(self) -> None:
        self._search_matches = []
        if not self._search_query:
            return
        needle = self._search_query.lower()
        haystack = self.log_view.toPlainText().lower()
        i = 0
        n = len(needle)
        if n == 0:
            return
        while True:
            pos = haystack.find(needle, i)
            if pos < 0:
                break
            self._search_matches.append((pos, n))
            i = pos + n
        if self._search_current >= len(self._search_matches):
            self._search_current = 0

    def _update_extra_selections(self) -> None:
        if not self._search_matches:
            self.log_view.setExtraSelections([])
            return
        yellow = QTextCharFormat()
        yellow.setBackground(self._MATCH_BG)
        orange = QTextCharFormat()
        orange.setBackground(self._CURRENT_MATCH_BG)
        document = self.log_view.document()
        selections = []
        for i, (pos, length) in enumerate(self._search_matches):
            cursor = QTextCursor(document)
            cursor.setPosition(pos)
            cursor.setPosition(
                pos + length, QTextCursor.MoveMode.KeepAnchor
            )
            sel = QTextEdit.ExtraSelection()
            sel.cursor = cursor
            sel.format = orange if i == self._search_current else yellow
            selections.append(sel)
        self.log_view.setExtraSelections(selections)

    def _scroll_to_current(self) -> None:
        if not self._search_matches:
            return
        pos, length = self._search_matches[self._search_current]
        cursor = QTextCursor(self.log_view.document())
        cursor.setPosition(pos)
        cursor.setPosition(pos + length, QTextCursor.MoveMode.KeepAnchor)
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _search_stats(self) -> tuple[int, int]:
        total = len(self._search_matches)
        if total == 0:
            return (0, 0)
        return (self._search_current + 1, total)

    # ----- worker lifecycle ---------------------------------------------
    def start(self, cfg: RunConfig) -> None:
        self._cfg = cfg
        self.log_view.clear()
        self._reset_log_cursor_format()
        self.overall_bar.setValue(0)
        self.file_bar.setValue(0)
        self._set_status(self.tr("Starting..."), "")
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.new_btn.setEnabled(False)
        self.metrics_strip.start()
        self._start_time = time.time()
        self._tick_elapsed()
        self._elapsed_timer.start()
        self._worker = TranscriptionWorker(cfg, self)
        self._worker.started_run.connect(lambda: self.runStateChanged.emit(True))
        self._worker.progress_overall.connect(self._on_progress_overall)
        self._worker.progress_file.connect(self._on_progress_file)
        self._worker.log.connect(self._append_log)
        self._worker.log_html.connect(self._append_log_html)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.correction_started.connect(self._on_correction_started)
        self._worker.error.connect(self._on_error)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_cancelled.connect(self._on_finished_cancelled)
        # Emit running=True immediately so the UI locks even before the
        # thread's `started_run` slot fires.
        self.runStateChanged.emit(True)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.request_cancel()
        self._reset_log_cursor_format()
        self._set_status(self.tr("Stopping..."), "warn")
        self.cancel_btn.setEnabled(False)
        self._append_log("Stop requested. Aborting at next checkpoint...")
        # Cooperative cancel only fires between segments. The
        # watchdog forcibly terminates if the worker hasn't
        # noticed within 400 ms - users shouldn't wait for the
        # current segment to finish. terminate() leaves the
        # model in an undefined state (restart the app if
        # anything looks wrong) but immediate stop matters more.
        QTimer.singleShot(400, self._force_terminate_if_still_running)

    def _force_terminate_if_still_running(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._append_log(
            "Cooperative cancel did not return in time. "
            "Forcibly terminating the worker thread."
        )
        self._worker.terminate()
        self._worker.wait(200)
        # The worker will not emit finished_cancelled when terminated;
        # transition the UI manually.
        self._on_finished_cancelled()

    # ----- worker signal handlers --------------------------------------
    def _on_progress_overall(self, frac: float) -> None:
        self.overall_bar.setValue(int(round(max(0.0, min(1.0, frac)) * 100)))

    def _on_progress_file(self, speaker: str, frac: float) -> None:
        self._set_status(self.tr("Transcribing {0}").format(speaker), "")
        self.file_bar.setValue(int(round(max(0.0, min(1.0, frac)) * 100)))

    def _on_file_done(self, speaker: str, count: int) -> None:
        self._append_log(
            self.tr("File done: {0} ({1} segments)").format(speaker, count)
        )

    def _on_correction_started(self) -> None:
        self._set_status(self.tr("Applying corrections"), "")
        self.file_bar.setValue(0)

    def _on_error(self, msg: str) -> None:
        self._reset_log_cursor_format()
        self._set_status(self.tr("Error"), "warn")
        self._append_log(f"ERROR: {msg}")
        self._drain_log_queue()
        self.cancel_btn.setEnabled(False)
        self.new_btn.setEnabled(True)
        self._elapsed_timer.stop()
        self._tick_elapsed()
        self.runStateChanged.emit(False)

    def _on_finished_ok(self, output_paths: list) -> None:
        elapsed = time.time() - self._start_time
        self.overall_bar.setValue(100)
        self.file_bar.setValue(100)
        self._reset_log_cursor_format()
        self._set_status(self.tr("Done in {0:.1f}s").format(elapsed), "good")
        self._append_log(self.tr("Wrote: {0}").format(", ".join(map(str, output_paths))))
        self._drain_log_queue()
        self.cancel_btn.setEnabled(False)
        self.open_btn.setEnabled(True)
        self.new_btn.setEnabled(True)
        self._elapsed_timer.stop()
        self._tick_elapsed()
        self.runStateChanged.emit(False)

    def _on_finished_cancelled(self) -> None:
        self._reset_log_cursor_format()
        self._set_status(self.tr("Stopped"), "warn")
        self._drain_log_queue()
        self.cancel_btn.setEnabled(False)
        self.new_btn.setEnabled(True)
        self._elapsed_timer.stop()
        self._tick_elapsed()
        self.runStateChanged.emit(False)

    # ----- helpers -------------------------------------------------------
    def _tick_elapsed(self) -> None:
        """Refresh the elapsed-time label. Format is `H:MM:SS` once
        the run goes past one hour, otherwise `MM:SS`. Cleared
        entirely while idle (no run started yet)."""
        if self._start_time <= 0.0:
            self.elapsed_label.setText("")
            return
        elapsed_s = int(time.time() - self._start_time)
        if elapsed_s < 0:
            elapsed_s = 0
        h, rem = divmod(elapsed_s, 3600)
        m, s = divmod(rem, 60)
        if h:
            txt = f"{self.tr('Elapsed')} {h:d}:{m:02d}:{s:02d}"
        else:
            txt = f"{self.tr('Elapsed')} {m:02d}:{s:02d}"
        self.elapsed_label.setText(txt)

    def _set_status(self, text: str, tone: str = "") -> None:
        """Update the run-status string + emit so the MainWindow can
        reflect it in the centered content title."""
        self._status_text = text
        self._status_tone = tone
        self.statusChanged.emit(text, tone)

    def status_text(self) -> str:
        return self._status_text

    def status_tone(self) -> str:
        return self._status_tone

    def _append_log(self, msg: str) -> None:
        self._log_queue.append((False, msg))
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _append_log_html(self, html_msg: str) -> None:
        self._log_queue.append((True, html_msg))
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _flush_log_queue(self) -> None:
        """Drain the buffered log entries in one paint pass. Called
        at most 4x/second by `_log_flush_timer`, plus on demand at
        run end via `_drain_log_queue`.

        Two optimisations vs. naive per-item appends:
        * Reset cursor format ONCE at the start (not per item).
          The previous per-item reset called `setTextCursor()`
          which scrolls the viewport, dozens of scroll ops per
          drain saturated the event loop.
        * Coalesce consecutive plain-text items into a single
          `appendPlainText` call so 50 status lines become 1
          append + 1 paint.

        Drain is capped at `_LOG_DRAIN_BATCH` items per tick; if
        more are queued, the timer is re-armed so the event loop
        gets a chance to handle clicks/scroll between batches.
        """
        if not self._log_queue:
            return
        # Cap so a backlogged queue can't lock the GUI on one tick.
        batch = self._log_queue[:_LOG_DRAIN_BATCH]
        self._log_queue = self._log_queue[_LOG_DRAIN_BATCH:]
        sb = self.log_view.verticalScrollBar()
        autoscroll = self.autoscroll_check.isChecked()
        prev = sb.value() if not autoscroll else 0
        self._suppress_scroll_signal = True
        # Block widget signals (textChanged, scrollbar valueChanged)
        # for the duration of the drain. Each appendHtml fires
        # textChanged, which kicks the search throttle and
        # cascades through Qt's signal queue - 200 of those per
        # drain crush the event loop. Block once, re-emit search
        # update once at the end.
        self.log_view.blockSignals(True)
        self.log_view.setUpdatesEnabled(False)
        try:
            self._reset_log_cursor_format()
            # Walk the batch, coalescing consecutive plain-text
            # items into one appendPlainText. HTML items still
            # require one appendHtml each because Qt's parser
            # treats each call as one HTML fragment.
            i = 0
            n = len(batch)
            while i < n:
                if batch[i][0]:
                    self.log_view.appendHtml(batch[i][1])
                    i += 1
                    continue
                j = i
                while j < n and not batch[j][0]:
                    j += 1
                self.log_view.appendPlainText(
                    "\n".join(payload for _, payload in batch[i:j])
                )
                i = j
            if autoscroll:
                sb.setValue(sb.maximum())
            else:
                sb.setValue(prev)
        finally:
            self.log_view.setUpdatesEnabled(True)
            self.log_view.blockSignals(False)
            self._suppress_scroll_signal = False
        # Re-arm the search throttle ONCE for the whole batch
        # (it would otherwise have fired 200 times during the
        # drain if signals weren't blocked).
        if self._search_query:
            self._search_throttle.start()
        if self._log_queue:
            # Still more items pending: re-arm the timer so the
            # event loop processes input between batches instead
            # of looping in a single drain.
            self._log_flush_timer.start()

    def _drain_log_queue(self) -> None:
        """Force-flush any pending log entries. Called at the end
        of a run so the final lines are visible immediately rather
        than after the next 250 ms tick. Loops until the queue is
        empty (the regular flush is batch-capped)."""
        if self._log_flush_timer.isActive():
            self._log_flush_timer.stop()
        while self._log_queue:
            self._flush_log_queue()
            self._log_flush_timer.stop()

    def _append(self, fn, payload: str) -> None:
        """Append via `fn` (appendPlainText or appendHtml) with
        scroll-position preserved when autoscroll is off, and the
        cursor's character format reset to default beforehand so a
        previously-coloured span doesn't leak into the next line.

        Wrapped in `setUpdatesEnabled(False)` so Qt batches the
        cursor manipulation, append, and scrollbar restore into a
        single paint instead of repainting after each step. With
        thousands of segments per transcription this is the
        difference between a responsive UI and one whose event
        loop is permanently behind.
        """
        sb = self.log_view.verticalScrollBar()
        autoscroll = self.autoscroll_check.isChecked()
        # CRITICAL: capture `prev` BEFORE any cursor manipulation.
        # `_reset_log_cursor_format` calls `setTextCursor()` which
        # auto-scrolls the cursor into view, so reading `sb.value()`
        # after it would give us `sb.maximum()` and the "restore"
        # below would still scroll to the bottom even with autoscroll
        # off.
        prev = sb.value() if not autoscroll else 0
        self._suppress_scroll_signal = True
        self.log_view.setUpdatesEnabled(False)
        try:
            # Reset cursor character format BEFORE appending so a span
            # left coloured (e.g. red confidence-band) by the previous
            # line can't bleed onto plain-text status lines like "Stop
            # requested. Aborting at next checkpoint…".
            self._reset_log_cursor_format()
            fn(payload)
            if autoscroll:
                sb.setValue(sb.maximum())
            else:
                sb.setValue(prev)
        finally:
            self.log_view.setUpdatesEnabled(True)
            self._suppress_scroll_signal = False

    def _on_scroll_value_changed(self, value: int) -> None:
        """Disable autoscroll when the user scrolls the log away
        from the bottom. Suppressed during our own cursor-follow
        scrolls (set by `_append`)."""
        if self._suppress_scroll_signal:
            return
        sb = self.log_view.verticalScrollBar()
        if value < sb.maximum() - 4 and self.autoscroll_check.isChecked():
            self.autoscroll_check.setChecked(False)

    def _reset_log_cursor_format(self) -> None:
        """Drop any colour / weight / strikethrough state on the
        log view's cursor so the next append starts from defaults.
        Called before every append AND on every lifecycle
        transition (start / stop / finish / error)."""
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.setCharFormat(QTextCharFormat())
        self.log_view.setTextCursor(cursor)

    def _open_output_folder(self) -> None:
        if self._cfg is None:
            return
        path = self._cfg.output_dir
        if not path or not os.path.isdir(path):
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            self._append_log(self.tr("Could not open folder: {0}").format(exc))
