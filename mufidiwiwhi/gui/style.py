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

"""Adwaita / GNOME-49 inspired styling.

Visual language

  Window background     #fafafa  (light neutral grey, NOT pure white)
  Card background       #ffffff  (white, 12 px corners)
  Hairline divider      #e0e0e0  (1 px, inset 16 px on either side)
  Floating label        #5e5e5e  (small, muted, sits above the value)
  Body text             #1e1e1e
  Accent (selection,
          focus ring,
          primary btn)  #006699  (we keep the project's strict accent)
  Running / good        #339966
  Error / destructive   #990000
  Tooltip background    #fff6d5
  Button bg (default)   #e8e8e8

Patterns

  * Cards (`QFrame[class~="card"]`) are white rounded panels with a
    1 px #e6e6e6 border. No drop shadows: the panel sits cleanly on
    the muted window background, like libadwaita's PreferencesGroup.
  * Rows (`QFrame[class~="row"]`) fill the card edge to edge. Visual
    separators are `QFrame[class~="rowDivider"]` lines, inset 16 px.
  * `EntryRow` is a row that contains a small floating label
    (`.floatingLabel`) and a borderless input (`.borderlessInput`).
    A `[focused="true"]` property on the row drives a 2 px accent
    outline, set by `EntryRow.eventFilter`.
  * Buttons are pill-shaped (16 px radius) with a soft grey
    background. Hover stays green (the user's preference). Primary
    buttons are accent-filled, slightly larger.
  * The HeaderBar is a thin band at the top of the window with the
    icon + name on the left and the ViewSwitcher centered.
"""

from __future__ import annotations

from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtWidgets import QApplication, QToolTip

from ..resources import check_white_path, chevron_down_path


ERROR = "#990000"
ACCENT = "#006699"
RUNNING = "#339966"
INK = "#1e1e1e"
MUTED = "#5e5e5e"
SURFACE = "#fafafa"
ACTIVE_BG = "#ffffff"
TOOLTIP_BG = "#fff6d5"

# Borders / dividers
CARD_BORDER = "#e6e6e6"
DIVIDER = "#e0e0e0"
INPUT_BORDER = "#cdcdd6"

# Button surfaces
BUTTON_BG = "#e8e8e8"
BUTTON_HOVER = RUNNING
PRESSED_BG = "#d8d8d8"
DISABLED_BG = "#efefef"
DISABLED_FG = "#a8a8a8"

# Sidebar surfaces (slightly darker than the content area so the
# sidebar reads as a continuous left strip even with no separator).
SIDEBAR_BG = "#ebebeb"
SIDEBAR_HOVER = "#dfdfdf"
SIDEBAR_SELECTED = "#d2d2d2"


_CHEVRON_DOWN = chevron_down_path().replace("\\", "/")
_CHECK_WHITE = check_white_path().replace("\\", "/")


_QSS = f"""
* {{
    font-family: "Inter", "Segoe UI", "Cantarell", "Noto Sans", sans-serif;
    font-size: 10pt;
    color: {INK};
}}

/* The QMainWindow itself stays transparent so the rounded mask
   we apply in code shows real transparency outside the rounded
   rect. The rootContainer (centralWidget) actually paints the
   surface color. */
QMainWindow {{
    background: transparent;
}}
QDialog {{
    background-color: {SURFACE};
}}
QWidget#rootContainer {{
    background-color: {SURFACE};
    border-radius: 12px;
}}
QWidget#rootContainer[maximized="true"] {{
    border-radius: 0px;
}}
QWidget#contentArea {{
    background-color: {SURFACE};
    border-top-right-radius: 12px;
    border-bottom-right-radius: 12px;
}}
QWidget#contentArea[maximized="true"] {{
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
}}

/* The centered tab title at the top of the right content panel.
   Settings/Project show static text; Run shows the live status
   ('Idle', 'Transcribing X', 'Done in 12.4s', 'Stopped', 'Error').
   The 'tone' property mirrors the previous inline status colour:
   green-good when finished, red-warn on stop/error. */
QLabel#contentTitle {{
    background: transparent;
    color: {INK};
    font-size: 12pt;
    font-weight: 600;
    padding: 0;
}}
QLabel#contentTitle[tone="good"] {{
    color: {RUNNING};
}}
QLabel#contentTitle[tone="warn"] {{
    color: {ERROR};
}}

QStatusBar {{
    background-color: {SURFACE};
    color: {MUTED};
    border-top: 1px solid {DIVIDER};
}}

/* -----------------------------------------------------------------
   Cards and rows (libadwaita PreferencesGroup / ActionRow / EntryRow)
   ----------------------------------------------------------------- */

QFrame[class~="card"] {{
    background-color: {ACTIVE_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
}}

QFrame[class~="row"] {{
    background-color: transparent;
    border: 0px solid transparent;
    border-radius: 0px;
}}
/* The focus ring on EntryRow uses an inner outline so it doesn't
   inflate the layout. */
QFrame[class~="row"][focused="true"] {{
    border: 2px solid {ACCENT};
    border-radius: 8px;
}}

QFrame[class~="rowDividerWrap"] {{
    background-color: transparent;
}}
QFrame[class~="rowDivider"] {{
    background-color: {DIVIDER};
    border: none;
    max-height: 1px;
    min-height: 1px;
}}

/* Section header sits ABOVE the card, slightly muted. */
QLabel[class~="sectionHeader"] {{
    color: {MUTED};
    font-weight: 600;
    font-size: 10pt;
    padding: 0 4px;
    margin-top: 4px;
}}

/* Row title: regular weight body text. */
QLabel[class~="rowTitle"] {{
    color: {INK};
    font-weight: 500;
    font-size: 10pt;
}}
/* Speaker rows specifically: regular weight title (matches GNOME
   Input Sources). Override via a more-specific class declared
   AFTER `rowTitle` so the cascade picks the lighter weight. */
QLabel[class~="speakerTitle"] {{
    font-weight: 400;
}}
QLabel[class~="rowSubtitle"] {{
    color: {MUTED};
    font-size: 9pt;
}}
/* Subtle hover background on the trailing "+ Add speaker..." row. */
QFrame[class~="row"][addRow="true"]:hover {{
    background-color: #f5f5f5;
}}
QLabel[class~="floatingLabel"] {{
    color: {MUTED};
    font-size: 9pt;
}}

/* Borderless input used inside an EntryRow: no frame, transparent
   bg, normal-sized text. */
QLineEdit[class~="borderlessInput"],
QPlainTextEdit[class~="borderlessInput"] {{
    border: none;
    background-color: transparent;
    padding: 0;
    color: {INK};
    selection-background-color: {ACCENT};
    selection-color: {ACTIVE_BG};
}}
QComboBox[class~="borderlessInput"],
QSpinBox[class~="borderlessInput"],
QDoubleSpinBox[class~="borderlessInput"] {{
    border: none;
    background-color: transparent;
    padding: 0;
    color: {INK};
    selection-background-color: {ACCENT};
    selection-color: {ACTIVE_BG};
}}
QComboBox[class~="borderlessInput"]::drop-down {{
    border: none;
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    padding-right: 4px;
}}
QComboBox[class~="borderlessInput"]::down-arrow {{
    image: url("{_CHEVRON_DOWN}");
    width: 12px;
    height: 12px;
}}

/* -----------------------------------------------------------------
   Standalone form inputs (kept for places that don't use EntryRow,
   e.g. dialogs and the speakers table editor).
   ----------------------------------------------------------------- */
QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
    border: 1px solid {INPUT_BORDER};
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 24px;
    background-color: {ACTIVE_BG};
    color: {INK};
    selection-background-color: {ACCENT};
    selection-color: {ACTIVE_BG};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 2px solid {ACCENT};
    padding: 5px 9px;
}}

/* Borderless inputs sit inside an EntryRow whose outer 2 px focus
   ring already signals focus. Suppress the standalone :focus
   border on these and instead show a very light grey background
   so the active field is still distinguishable from siblings. */
QLineEdit[class~="borderlessInput"]:focus,
QPlainTextEdit[class~="borderlessInput"]:focus,
QComboBox[class~="borderlessInput"]:focus,
QSpinBox[class~="borderlessInput"]:focus,
QDoubleSpinBox[class~="borderlessInput"]:focus {{
    border: none;
    background-color: #f0f0f0;
    padding: 0;
}}
QLineEdit:disabled, QComboBox:disabled,
QDoubleSpinBox:disabled, QSpinBox:disabled {{
    background-color: {DISABLED_BG};
    color: {DISABLED_FG};
}}
QComboBox {{
    padding-right: 32px;
    /* combobox-popup: 0 forces an item-view popup that honours
       QComboBox.maxVisibleCount, instead of Fusion's clamped
       native menu. */
    combobox-popup: 0;
}}
QComboBox::drop-down {{
    border: none;
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 28px;
    padding-right: 8px;
}}
QComboBox::down-arrow {{
    image: url("{_CHEVRON_DOWN}");
    width: 12px;
    height: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: {ACTIVE_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 4px;
    color: {INK};
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    padding: 8px 12px;
    margin: 1px 2px;
    border-radius: 6px;
    color: {INK};
    background-color: transparent;
}}
QComboBox QAbstractItemView::item:hover {{
    background-color: {SURFACE};
    color: {INK};
}}
QComboBox QAbstractItemView::item:selected {{
    background-color: {ACCENT};
    color: {ACTIVE_BG};
}}

/* -----------------------------------------------------------------
   Buttons. Both primary and secondary share the same metrics so
   buttons sitting next to each other line up cleanly.
   ----------------------------------------------------------------- */
QPushButton {{
    padding: 8px 16px;
    min-height: 36px;
    border: none;
    border-radius: 8px;
    background-color: {BUTTON_BG};
    color: {INK};
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {BUTTON_HOVER};
    color: {ACTIVE_BG};
}}
QPushButton:pressed {{
    background-color: {ACCENT};
    color: {ACTIVE_BG};
}}
QPushButton:disabled {{
    background-color: {DISABLED_BG};
    color: {DISABLED_FG};
}}
QPushButton[primary="true"] {{
    background-color: {ACCENT};
    color: {ACTIVE_BG};
    border: none;
    font-weight: 600;
    padding: 8px 16px;
    min-height: 36px;
    border-radius: 8px;
}}
QPushButton[primary="true"]:hover {{
    background-color: {RUNNING};
    color: {ACTIVE_BG};
}}
QPushButton[primary="true"]:pressed {{
    background-color: {INK};
    color: {ACTIVE_BG};
}}
QPushButton[primary="true"]:disabled {{
    background-color: {DISABLED_BG};
    color: {DISABLED_FG};
}}

QPushButton[destructive="true"] {{
    background-color: {ERROR};
    color: {ACTIVE_BG};
    border: none;
    font-weight: 600;
    padding: 8px 16px;
    min-height: 36px;
    border-radius: 8px;
}}
QPushButton[destructive="true"]:hover {{
    background-color: #b71212;
    color: {ACTIVE_BG};
}}
QPushButton[destructive="true"]:pressed {{
    background-color: {INK};
    color: {ACTIVE_BG};
}}
QPushButton[destructive="true"]:disabled {{
    background-color: {DISABLED_BG};
    color: {DISABLED_FG};
}}

/* Hide spinbox stepper arrows entirely. Keyboard arrows still work. */
QAbstractSpinBox::up-button,
QAbstractSpinBox::down-button {{
    width: 0;
    height: 0;
    border: none;
    background: transparent;
}}
QAbstractSpinBox::up-arrow,
QAbstractSpinBox::down-arrow {{ width: 0; height: 0; }}

/* Pill toggle buttons used for the Format multi-select. */
QPushButton[class~="pillToggle"] {{
    background-color: transparent;
    color: {INK};
    border: 1px solid {INPUT_BORDER};
    border-radius: 16px;
    padding: 6px 14px;
    min-height: 24px;
    font-weight: 500;
}}
QPushButton[class~="pillToggle"]:hover {{
    background-color: {SIDEBAR_HOVER};
    color: {INK};
}}
QPushButton[class~="pillToggle"]:checked {{
    background-color: {ACCENT};
    color: {ACTIVE_BG};
    border: 1px solid {ACCENT};
}}
QPushButton[class~="pillToggle"]:checked:hover {{
    background-color: {RUNNING};
    border: 1px solid {RUNNING};
}}

/* Stat bubbles in the Run-tab metrics strip. The default border
   is muted; the bubble's `tone` property turns it accent blue,
   running green or error red as the underlying metric crosses
   the 25 / 50 / 75 % usage thresholds. All variants use the same
   2 px border so the bubble's outer size stays stable when the
   tone flips, no layout jitter. */
QFrame[class~="statBubble"] {{
    background-color: {ACTIVE_BG};
    border: 2px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 0;
}}
QFrame[class~="statBubble"][tone="info"] {{
    border: 2px solid {ACCENT};
}}
QFrame[class~="statBubble"][tone="ok"] {{
    border: 2px solid {RUNNING};
}}
QFrame[class~="statBubble"][tone="hot"] {{
    border: 2px solid {ERROR};
}}
QLabel[class~="statBubbleLabel"] {{
    color: {MUTED};
    font-size: 9pt;
    font-weight: 500;
}}
/* Elapsed-time readout next to the Run-tab action buttons. */
QLabel#runElapsed {{
    color: {MUTED};
    font-size: 9pt;
    font-weight: 500;
    font-family: "DejaVu Sans Mono", "Consolas", "Liberation Mono",
                 "Courier New", monospace;
}}
/* Monospace for the value so the bubble's width stays stable as
   digits change (e.g. CPU 9.5% -> 12.3% -> 8.1%). */
QLabel[class~="statBubbleValue"] {{
    color: {INK};
    font-size: 9pt;
    font-weight: 600;
    font-family: "DejaVu Sans Mono", "Consolas", "Liberation Mono",
                 "Courier New", monospace;
}}

/* Round window-control buttons (minimize / maximize / close) that
   float in the top-right corner of the frameless window. GNOME 49
   look: three identical pale-grey circles at rest; only the close
   button turns red on hover. */
QFrame#windowControls {{
    background-color: transparent;
}}
QPushButton[class~="windowControlBtn"] {{
    background-color: #d6d6d6;
    border: none;
    border-radius: 12px;
    padding: 0;
    min-width: 24px;
    max-width: 24px;
    min-height: 24px;
    max-height: 24px;
}}
/* Per-role hover colors. Minimize -> accent blue, Maximize ->
   running green, Close -> red. Pressed states darken each. */
QPushButton[class~="windowControlBtn"][role="min"]:hover {{
    background-color: {ACCENT};
}}
QPushButton[class~="windowControlBtn"][role="min"]:pressed {{
    background-color: #004d77;
}}
QPushButton[class~="windowControlBtn"][role="max"]:hover {{
    background-color: {RUNNING};
}}
QPushButton[class~="windowControlBtn"][role="max"]:pressed {{
    background-color: #2a7a52;
}}
QPushButton[class~="windowControlBtn"][role="close"]:hover {{
    background-color: #e01b24;
}}
QPushButton[class~="windowControlBtn"][role="close"]:pressed {{
    background-color: #b71212;
}}

/* Flat icon-only buttons used inside EntryRow extras (folder /
   pencil for path fields). Border-less, transparent until hovered. */
QToolButton[class~="iconButton"] {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 0;
}}
QToolButton[class~="iconButton"]:hover {{
    background-color: {SIDEBAR_HOVER};
}}
QToolButton[class~="iconButton"]:pressed {{
    background-color: {SIDEBAR_SELECTED};
}}
QToolButton[class~="iconButton"]:disabled {{
    background-color: transparent;
}}

/* -----------------------------------------------------------------
   Adwaita-style checkboxes: rounded square indicator, accent-filled
   with a white tick when checked. The Hunspell on/off Switch and
   the format pill toggles are different widget classes (Switch /
   QPushButton[class~="pillToggle"]), so these `QCheckBox` rules
   only catch real checkboxes (currently just `Autoscroll log`).
   ----------------------------------------------------------------- */
QCheckBox {{
    spacing: 8px;
    padding: 3px 0;
    color: {INK};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 6px;
    border: 1px solid #c0c0c0;
    background-color: {ACTIVE_BG};
}}
QCheckBox::indicator:hover {{
    border-color: #a0a0a0;
    background-color: #f5f5f5;
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    image: url("{_CHECK_WHITE}");
}}
QCheckBox::indicator:checked:hover {{
    background-color: #005580;
    border: 1px solid #005580;
}}
QCheckBox::indicator:disabled {{
    background-color: {DISABLED_BG};
    border: 1px solid {INPUT_BORDER};
}}
QCheckBox:disabled {{
    color: {DISABLED_FG};
}}

/* -----------------------------------------------------------------
   Progress bar
   ----------------------------------------------------------------- */
QProgressBar {{
    border: 1px solid {INPUT_BORDER};
    border-radius: 12px;
    background-color: {ACTIVE_BG};
    text-align: center;
    height: 22px;
    color: {INK};
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 11px;
}}
/* The per-file progress bar uses the running-green accent so the
   two stacked bars (overall + file) read as distinct tracks. */
QProgressBar#fileBar::chunk {{
    background-color: {RUNNING};
}}

/* -----------------------------------------------------------------
   Tables / log
   ----------------------------------------------------------------- */
QHeaderView::section {{
    background-color: {SURFACE};
    padding: 7px 10px;
    border: none;
    border-right: 1px solid {DIVIDER};
    border-bottom: 1px solid {DIVIDER};
    font-weight: 600;
    color: {INK};
}}
QTableView {{
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    gridline-color: {DIVIDER};
    background-color: {ACTIVE_BG};
    selection-background-color: {ACCENT};
    selection-color: {ACTIVE_BG};
    color: {INK};
}}
QTableView::item {{ padding: 5px 7px; }}

QPlainTextEdit#runLog {{
    font-family: "DejaVu Sans Mono", "Consolas", "Liberation Mono",
                 "Courier New", monospace;
    font-size: 9pt;
}}

/* -----------------------------------------------------------------
   Tab bar (when used directly; we mostly drive the new ViewSwitcher).
   ----------------------------------------------------------------- */
QTabWidget {{ background-color: {SURFACE}; }}
QTabWidget::pane {{
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    top: -1px;
    background-color: {SURFACE};
}}
QTabBar {{ background-color: {SURFACE}; }}
QTabBar::tab {{
    padding: 9px 22px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    color: {MUTED};
    background-color: transparent;
}}
QTabBar::tab:hover {{ background-color: {BUTTON_BG}; color: {INK}; }}
QTabBar::tab:selected {{
    background-color: {ACTIVE_BG};
    color: {ACCENT};
    font-weight: 700;
    border-color: {CARD_BORDER};
}}
QTabBar::tab:disabled {{
    color: {DISABLED_FG};
    background-color: transparent;
}}

/* -----------------------------------------------------------------
   ViewSwitcher: a single rounded pill with N segments.
   ----------------------------------------------------------------- */
QFrame#viewSwitcher {{
    background-color: {BUTTON_BG};
    border-radius: 18px;
    padding: 0;
}}
QFrame#viewSwitcher QPushButton[segment="true"] {{
    background-color: transparent;
    color: {INK};
    border: none;
    border-radius: 14px;
    padding: 6px 22px;
    min-height: 22px;
    font-weight: 500;
}}
QFrame#viewSwitcher QPushButton[segment="true"]:hover {{
    background-color: {ACTIVE_BG};
    color: {INK};
}}
QFrame#viewSwitcher QPushButton[segment="true"]:checked {{
    background-color: {ACCENT};
    color: {ACTIVE_BG};
    font-weight: 600;
}}
QFrame#viewSwitcher QPushButton[segment="true"]:disabled {{
    color: {DISABLED_FG};
    background-color: transparent;
}}

/* Run-tab search bar */
QFrame#runSearchBar {{
    background-color: {ACTIVE_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
}}
QFrame#runSearchBar QPushButton {{
    padding: 6px 14px;
    min-height: 22px;
    border-radius: 14px;
}}

/* Sysmon strip: just spacing; the bubbles carry the visual style. */
QFrame#metricsStrip {{
    background-color: transparent;
    border: none;
}}

/* -----------------------------------------------------------------
   Sidebar (libadwaita / GNOME Settings vertical navigation)
   ----------------------------------------------------------------- */
QFrame#sidebar {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid #d8d8d8;
    border-top-left-radius: 12px;
    border-bottom-left-radius: 12px;
}}
QFrame#sidebar[maximized="true"] {{
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
}}
QFrame#sidebarHeader {{
    background-color: transparent;
}}
QFrame#sidebarBottomSearch {{
    background-color: transparent;
}}
QLineEdit#sidebarSearchInput {{
    background-color: {ACTIVE_BG};
    border: 1px solid #d0d0d0;
    border-radius: 8px;
    padding: 7px 10px;
    color: {INK};
    selection-background-color: {ACCENT};
    selection-color: {ACTIVE_BG};
}}
QLineEdit#sidebarSearchInput:focus {{
    border: 2px solid {ACCENT};
    padding: 6px 9px;
}}
QLabel#sidebarSearchCount {{
    color: {MUTED};
    font-size: 9pt;
}}
QLabel#sidebarSearchCount[tone="warn"] {{
    color: {ERROR};
    font-weight: 600;
}}
QFrame#sidebarDivider {{
    background-color: {DIVIDER};
    border: none;
    max-height: 1px;
    min-height: 1px;
}}
QLabel#sidebarAppName {{
    font-size: 13pt;
    font-weight: 600;
    color: {INK};
}}
QLabel#sidebarSubtitle {{
    font-size: 9pt;
    color: {MUTED};
}}
QLabel#sidebarVersion {{
    font-size: 8pt;
    color: {MUTED};
}}
QListWidget#sidebarList {{
    background-color: transparent;
    border: none;
    padding: 8px 8px 8px 8px;
    outline: none;
}}
QListWidget#sidebarList::item {{
    padding: 10px 12px;
    margin: 1px 0;
    border-radius: 8px;
    color: {INK};
}}
QListWidget#sidebarList::item:hover {{
    background-color: {SIDEBAR_HOVER};
}}
QListWidget#sidebarList::item:selected {{
    background-color: {SIDEBAR_SELECTED};
    color: {INK};
}}
QListWidget#sidebarList::item:disabled {{
    color: {DISABLED_FG};
    background-color: transparent;
}}

/* -----------------------------------------------------------------
   Tooltips
   ----------------------------------------------------------------- */
QToolTip {{
    background-color: {TOOLTIP_BG};
    color: {INK};
    border: 1px solid {CARD_BORDER};
    padding: 6px 9px;
    border-radius: 8px;
    opacity: 240;
}}

/* -----------------------------------------------------------------
   Scroll areas (we don't want a frame around them).
   ----------------------------------------------------------------- */
QScrollArea {{
    background-color: {SURFACE};
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background-color: {SURFACE};
}}

/* Slim, modern scrollbar: 8 px when not hovered, 12 px on hover. */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px 4px 0;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: #c0c0c0;
    min-height: 28px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{
    background: #a8a8a8;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
    background: transparent;
}}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0 4px 2px 4px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: #c0c0c0;
    min-width: 28px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{
    background: #a8a8a8;
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
    background: transparent;
}}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* Tone classes for hint labels. */
QLabel[tone="muted"] {{ color: {MUTED}; }}
QLabel[tone="good"]  {{ color: {RUNNING}; font-weight: 600; }}
QLabel[tone="warn"]  {{ color: {ERROR};   font-weight: 600; }}
QLabel[tone="bad"]   {{ color: {ERROR};   font-weight: 700; }}
"""


def apply_app_style(app: QApplication) -> None:
    """Apply the Adwaita-flavoured palette and stylesheet."""
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(INK))
    palette.setColor(QPalette.ColorRole.Base, QColor(ACTIVE_BG))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(INK))
    palette.setColor(QPalette.ColorRole.Button, QColor(BUTTON_BG))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(INK))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(ACTIVE_BG))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(TOOLTIP_BG))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(INK))
    app.setPalette(palette)

    tooltip_palette = QPalette()
    tooltip_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(TOOLTIP_BG))
    tooltip_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(INK))
    tooltip_palette.setColor(QPalette.ColorRole.Window, QColor(TOOLTIP_BG))
    tooltip_palette.setColor(QPalette.ColorRole.WindowText, QColor(INK))
    QToolTip.setPalette(tooltip_palette)

    base_font = QFont(app.font())
    if base_font.pointSize() < 10:
        base_font.setPointSize(10)
    app.setFont(base_font)

    app.setStyleSheet(_QSS)
