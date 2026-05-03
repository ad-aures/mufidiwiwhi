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

"""Top-level frameless window: vertical sidebar + page stack.

GNOME 49 / libadwaita inspired chrome: NO OS-rendered title bar.
The window uses `Qt.FramelessWindowHint`, draws its own three
round window-control buttons in the top-right corner, and hands
move / resize off to the compositor via
`QWindow.startSystemMove()` and `startSystemResize()` so it works
correctly on both X11 and Wayland.

Layout from left to right:

  * `Sidebar`: search bar + brand header + nav rows on `#ebebeb`.
  * `QStackedWidget` page area on `#fafafa`.
  * `WindowControls` overlay: 3 round buttons floating in the
    top-right corner of the page area.

Move / resize behaviour:

  * Click + drag on the top 40 px strip (above the cards) starts
    a system move via the windowing protocol.
  * Click within 6 px of a window edge starts a system resize on
    those edges.
  * Double-click on the top strip toggles maximize / restore.
  * Interactive widgets (line edits, combos, push buttons, list
    items, ...) intercept mouse events as usual, so no risk of
    accidentally moving the window from inside a control.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, QPoint, QRectF, QSize, Qt
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QKeySequence,
    QPainterPath,
    QPalette,
    QRegion,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core import RunConfig, SpeakerInput
from .pages.help_page import HelpPage
from .pages.project_page import ProjectPage
from .pages.run_page import RunPage
from .pages.setup_page import SetupPage
from .settings import SettingsManager
from .style import SURFACE
from .widgets.sidebar import Sidebar
from .widgets.window_controls import WindowControls


_RUN_INDEX = 2
_SETTINGS_INDEX = 0
_PROJECT_INDEX = 1
_HELP_INDEX = 3

def _to_edges(mask: int):
    """Build a `Qt.Edges` value from an int bitmask. PyQt6 accepts
    a piecewise OR of `Qt.Edge` members but rejects `Qt.Edge(0)`,
    so we OR members one at a time starting from a known member.
    """
    result = None
    for m in (
        Qt.Edge.LeftEdge,
        Qt.Edge.RightEdge,
        Qt.Edge.TopEdge,
        Qt.Edge.BottomEdge,
    ):
        if mask & m.value:
            result = m if result is None else (result | m)
    return result


# Top strip height treated as the "title bar" for drag / double-click.
_DRAG_STRIP_PX = 40
# Outer corner radius applied to the rounded mask + matching QSS.
_CORNER_RADIUS_PX = 12
# Edge-detection thickness used for resize hot zones.
_RESIZE_MARGIN_PX = 6
# Widget classes that must NOT trigger window drag when clicked.
_INTERACTIVE = (
    QLineEdit,
    QPushButton,
    QToolButton,
    QComboBox,
    QAbstractSpinBox,
    QPlainTextEdit,
    QTextEdit,
    QListWidget,
)


class MainWindow(QMainWindow):
    """Frameless sidebar nav + page stack."""

    def __init__(self, settings: Optional[SettingsManager] = None) -> None:
        super().__init__()
        self.setWindowTitle("Mufidiwiwhi")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        # Translucent background lets the corners we mask away
        # become actually transparent rather than showing default
        # window grey.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._settings = settings or SettingsManager()
        self._running: bool = False
        self.setMouseTracking(True)
        self._build_ui()
        self._restore_geometry()
        # First mask pass once the layout has settled.
        self._update_round_mask()

    def _build_ui(self) -> None:
        self.resize(1200, 1020)
        self.setMinimumSize(QSize(1024, 480))

        central = QWidget(self)
        central.setObjectName("rootContainer")
        central.setMouseTracking(True)
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self.sidebar = Sidebar(
            "Mufidiwiwhi",
            self.tr("Multi-file diarisation transcription with Whisper"),
        )
        self.sidebar.add_segment(
            self.tr("Settings"),
            self.tr("Persistent settings (model, dictionary, thresholds, ...)."),
            icon_svg="settings-svgrepo-com.svg",
        )
        self.sidebar.add_segment(
            self.tr("Project"),
            self.tr("Per-run inputs (speakers, output, formats)."),
            icon_svg="dialog-2-svgrepo-com.svg",
        )
        self.sidebar.add_segment(
            self.tr("Run"),
            self.tr("Live progress, log, and metrics during transcription."),
            icon_svg="play-circle-svgrepo-com.svg",
        )
        self.sidebar.add_segment(
            self.tr("Help"),
            self.tr("User manual: how Mufidiwiwhi works, recommended settings."),
            icon_svg="question-square-svgrepo-com.svg",
        )
        self.sidebar.set_segment_enabled(_RUN_INDEX, False)
        self.sidebar.set_segment_tooltip(
            _RUN_INDEX,
            self.tr("Available once a transcription run is started."),
        )
        self.sidebar.currentChanged.connect(self._on_sidebar_changed)
        central_layout.addWidget(self.sidebar)

        central_layout.addWidget(self._build_stack(), stretch=1)

        # Apply the initial title now that pages exist.
        self._update_content_title()

        # Wire the sidebar's bottom search to the Run-page log search.
        self.sidebar.search_input.textChanged.connect(self._on_search_text)
        self.sidebar.search_input.nextRequested.connect(self._on_search_next)
        self.sidebar.search_input.prevRequested.connect(self._on_search_prev)
        self.sidebar.search_input.escapePressed.connect(self._on_search_escape)
        # Window-wide shortcuts (only act when the Run tab is active).
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(
            self._focus_run_search
        )
        QShortcut(QKeySequence("F3"), self).activated.connect(self._on_search_next)
        QShortcut(QKeySequence("Shift+F3"), self).activated.connect(
            self._on_search_prev
        )

        self.setCentralWidget(central)
        # No QStatusBar: QMainWindow auto-creates one which renders
        # a rectangular strip below the rounded mask. The centered
        # content title + the run-page progress bars already convey
        # state.
        self.setStatusBar(None)

        # Window controls overlay: float in the top-right corner.
        self._controls = WindowControls(self)
        self._controls.minimizeRequested.connect(self.showMinimized)
        self._controls.maximizeToggleRequested.connect(self._toggle_maximize)
        self._controls.closeRequested.connect(self.close)
        self._reposition_controls()
        self._controls.raise_()

    def _build_stack(self) -> QWidget:
        wrapper = QWidget()
        wrapper.setObjectName("contentArea")
        wrapper.setMouseTracking(True)
        wrapper.setAutoFillBackground(True)
        pal = wrapper.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(SURFACE))
        wrapper.setPalette(pal)
        self._content_area = wrapper
        # Top padding leaves room for the window-control overlay
        # (24 px buttons + 12 px top margin + a little extra) so the
        # page content never collides with the controls.
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(24, 16, 24, 24)
        layout.setSpacing(0)

        # Centered title that swaps between Settings / Project /
        # Run-status. A right-side margin is reserved so a long
        # status string ("Transcribing ...") can't overlap the
        # window-control buttons in the top-right.
        self.content_title = QLabel("")
        self.content_title.setObjectName("contentTitle")
        self.content_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_title.setContentsMargins(120, 0, 120, 0)
        layout.addWidget(self.content_title)
        layout.addSpacing(16)

        self.stack = QStackedWidget()

        self.setup_page = SetupPage(self._settings, self)
        self.project_page = ProjectPage(self._settings, self)
        self.run_page = RunPage(self._settings, self)
        self.help_page = HelpPage(self)

        self.stack.addWidget(self.setup_page)
        self.stack.addWidget(self.project_page)
        self.stack.addWidget(self.run_page)
        self.stack.addWidget(self.help_page)

        layout.addWidget(self.stack)

        self.project_page.runRequested.connect(self._start_run)
        self.run_page.newProjectRequested.connect(self._new_project)
        self.run_page.runStateChanged.connect(self._on_run_state_changed)
        self.run_page.statusChanged.connect(self._on_run_status_changed)

        return wrapper

    def _on_sidebar_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        is_run = index == _RUN_INDEX
        self.sidebar.set_search_visible(is_run)
        if is_run:
            # Re-apply the existing query (preserved across tab
            # switches) to the current log content.
            text = self.sidebar.search_input.text()
            cur, total = self.run_page.apply_search(text)
            self.sidebar.set_match_count(cur, total)
        else:
            # Drop highlights so the log view is clean if the user
            # comes back via the Run tab being launched again later.
            self.run_page.clear_search()
        self._update_content_title()

    def _on_run_status_changed(self, text: str, tone: str) -> None:
        # Run-page status changed; if Run is the active tab, reflect
        # it in the centered content title.
        if self.stack.currentIndex() == _RUN_INDEX:
            self._update_content_title()

    def _update_content_title(self) -> None:
        idx = self.stack.currentIndex()
        if idx == _SETTINGS_INDEX:
            text, tone = self.tr("Settings"), ""
        elif idx == _PROJECT_INDEX:
            text, tone = self.tr("Project"), ""
        elif idx == _RUN_INDEX:
            text = self.run_page.status_text() or self.tr("Run")
            tone = self.run_page.status_tone()
        elif idx == _HELP_INDEX:
            text, tone = self.tr("Help"), ""
        else:
            text, tone = "", ""
        self.content_title.setText(text)
        self.content_title.setProperty("tone", tone)
        self.content_title.style().unpolish(self.content_title)
        self.content_title.style().polish(self.content_title)

    # ----- log search coordination --------------------------------------
    def _on_search_text(self, text: str) -> None:
        if self.stack.currentIndex() != _RUN_INDEX:
            return
        cur, total = self.run_page.apply_search(text)
        self.sidebar.set_match_count(cur, total)

    def _on_search_next(self) -> None:
        if self.stack.currentIndex() != _RUN_INDEX:
            return
        cur, total = self.run_page.next_match()
        self.sidebar.set_match_count(cur, total)

    def _on_search_prev(self) -> None:
        if self.stack.currentIndex() != _RUN_INDEX:
            return
        cur, total = self.run_page.prev_match()
        self.sidebar.set_match_count(cur, total)

    def _on_search_escape(self) -> None:
        self.sidebar.search_input.clear()
        self.run_page.clear_search()
        self.sidebar.set_match_count(0, 0)
        # Return focus to the log so keyboard scrolling works.
        self.run_page.log_view.setFocus()

    def _focus_run_search(self) -> None:
        if self.stack.currentIndex() != _RUN_INDEX:
            return
        self.sidebar.search_input.setFocus()
        self.sidebar.search_input.selectAll()

    def _start_run(self, cfg: RunConfig) -> None:
        self.sidebar.set_segment_enabled(_RUN_INDEX, True)
        self.sidebar.set_segment_tooltip(_RUN_INDEX, "")
        self.sidebar.set_current(_RUN_INDEX)
        self.run_page.start(cfg)

    def _on_run_state_changed(self, running: bool) -> None:
        self._running = running
        self.sidebar.set_segment_enabled(_SETTINGS_INDEX, not running)
        self.sidebar.set_segment_enabled(_PROJECT_INDEX, not running)
        if running:
            tip = self.tr("Locked while a transcription is running.")
            self.sidebar.set_segment_tooltip(_SETTINGS_INDEX, tip)
            self.sidebar.set_segment_tooltip(_PROJECT_INDEX, tip)
        else:
            self.sidebar.set_segment_tooltip(_SETTINGS_INDEX, "")
            self.sidebar.set_segment_tooltip(_PROJECT_INDEX, "")

    def _new_project(self) -> None:
        # Clear the speaker list so the new project starts fresh.
        try:
            self.project_page.speaker_view.remove_all()
        except Exception:
            pass
        self.sidebar.set_current(_PROJECT_INDEX)

    # ----- frameless drag / resize / maximize ---------------------------
    def _toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _reposition_controls(self) -> None:
        cw = self._controls.sizeHint().width()
        ch = self._controls.sizeHint().height()
        margin = 12
        self._controls.setGeometry(self.width() - cw - margin, margin, cw, ch)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._reposition_controls()
        self._update_round_mask()

    def changeEvent(self, event) -> None:  # type: ignore[override]
        if event.type() == QEvent.Type.WindowStateChange:
            self._controls.set_maximized(self.isMaximized())
            self._update_round_mask()
        super().changeEvent(event)

    def _update_round_mask(self) -> None:
        """Apply (or clear) the rounded mask + flip a 'maximized'
        property on the rounded surfaces so QSS can drop their
        border-radius on maximize / fullscreen.
        """
        maximized = self.isMaximized() or self.isFullScreen()
        if maximized:
            self.clearMask()
        else:
            path = QPainterPath()
            path.addRoundedRect(
                QRectF(self.rect()),
                float(_CORNER_RADIUS_PX),
                float(_CORNER_RADIUS_PX),
            )
            region = QRegion(path.toFillPolygon().toPolygon())
            self.setMask(region)
        for w in (
            self.centralWidget(),
            getattr(self, "sidebar", None),
            getattr(self, "_content_area", None),
        ):
            if w is None:
                continue
            w.setProperty("maximized", maximized)
            w.style().unpolish(w)
            w.style().polish(w)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            edges = self._edge_at(pos)
            if edges and self.windowHandle() is not None:
                self.windowHandle().startSystemResize(_to_edges(edges))
                event.accept()
                return
            if self._is_drag_zone(pos):
                if self.windowHandle() is not None:
                    self.windowHandle().startSystemMove()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.buttons() == Qt.MouseButton.NoButton:
            edges = self._edge_at(event.position().toPoint())
            cursor = self._cursor_for_edges(edges)
            if cursor is not None:
                self.setCursor(cursor)
            else:
                self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self._is_drag_zone(
            event.position().toPoint()
        ):
            self._toggle_maximize()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # ----- helpers ------------------------------------------------------
    def _edge_at(self, pos: QPoint):
        """Return the int bitmask of `Qt.Edge` values within
        `_RESIZE_MARGIN_PX` of `pos`, or None. We accumulate as a
        plain int because PyQt6's strict enum doesn't allow
        constructing `Qt.Edge(0)`.
        """
        if self.isMaximized():
            return 0
        m = _RESIZE_MARGIN_PX
        rect = self.rect()
        edges = 0
        if pos.x() <= m:
            edges |= Qt.Edge.LeftEdge.value
        if pos.x() >= rect.right() - m:
            edges |= Qt.Edge.RightEdge.value
        if pos.y() <= m:
            edges |= Qt.Edge.TopEdge.value
        if pos.y() >= rect.bottom() - m:
            edges |= Qt.Edge.BottomEdge.value
        return edges

    @staticmethod
    def _cursor_for_edges(edges: int):
        if not edges:
            return None
        L = Qt.Edge.LeftEdge.value
        R = Qt.Edge.RightEdge.value
        T = Qt.Edge.TopEdge.value
        B = Qt.Edge.BottomEdge.value
        if edges == (L | T) or edges == (R | B):
            return Qt.CursorShape.SizeFDiagCursor
        if edges == (R | T) or edges == (L | B):
            return Qt.CursorShape.SizeBDiagCursor
        if edges == L or edges == R:
            return Qt.CursorShape.SizeHorCursor
        if edges == T or edges == B:
            return Qt.CursorShape.SizeVerCursor
        return None

    def _is_drag_zone(self, pos: QPoint) -> bool:
        """True if a click at `pos` should start a system move."""
        if pos.y() >= _DRAG_STRIP_PX:
            return False
        # Don't drag if the user clicked on an interactive child
        # widget (line edit, combo, button, list item...).
        target = self.childAt(pos)
        cur = target
        while cur is not None and cur is not self:
            if isinstance(cur, _INTERACTIVE):
                return False
            cur = cur.parent() if isinstance(cur.parent(), QWidget) else None
        return True

    # ----- CLI passthrough ---------------------------------------------
    def apply_cli_overrides(
        self,
        speakers: list[SpeakerInput],
        dictionary: Optional[str],
        output_dir: Optional[str],
    ) -> None:
        self.project_page.apply_cli_overrides(speakers, dictionary, output_dir)
        if speakers:
            self.sidebar.set_current(_PROJECT_INDEX)

    # ----- geometry persistence ----------------------------------------
    def _restore_geometry(self) -> None:
        geom = self._settings.load_geometry("main")
        if geom is None:
            return
        try:
            from PyQt6.QtCore import QByteArray

            self.restoreGeometry(QByteArray(geom))
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._running:
            from PyQt6.QtWidgets import QMessageBox

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(self.tr("Stop transcription?"))
            box.setText(
                self.tr(
                    "A transcription is currently running. Closing "
                    "the window will abort it.\n\nQuit anyway?"
                )
            )
            quit_btn = box.addButton(
                self.tr("Quit"), QMessageBox.ButtonRole.AcceptRole
            )
            cancel_btn = box.addButton(
                self.tr("Keep running"), QMessageBox.ButtonRole.RejectRole
            )
            box.setDefaultButton(cancel_btn)
            box.exec()
            if box.clickedButton() is not quit_btn:
                event.ignore()
                return
            # User confirmed: ask the worker to stop cleanly. The app
            # is exiting anyway so we don't wait for the cancel to
            # complete.
            try:
                self.run_page._on_cancel()
            except Exception:
                pass
        try:
            self.project_page._snapshot()
        except Exception:
            pass
        try:
            self._settings.save_geometry("main", bytes(self.saveGeometry()))
        except Exception:
            pass
        super().closeEvent(event)
