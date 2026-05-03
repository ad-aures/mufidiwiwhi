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

"""Help page: renders the bundled markdown user manual.

The markdown source is `mufidiwiwhi/resources/help.md` (copy of the
top-level `DOCS.md`). It's loaded once at construction and rendered
into a read-only QTextBrowser via Qt's built-in markdown support.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import (
    QColor,
    QFont,
    QPixmap,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from ...resources import icon_path, resource_path
from ..widgets.icons import render_svg_to_pixmap


_HEADING_COLORS = {
    1: "#c92a2a",  # red
    2: "#2f9e44",  # green
    3: "#1971c2",  # blue
}
_HEADING_POINT_SIZES = {1: 32, 2: 22, 3: 16}
_PARAGRAPH_BOTTOM_MARGIN = 24.0
_CODE_BG = "#e8e8e8"
_LOGO_PX = 96
_LINK_COLOR = "#000000"
_HR_COLOR = "#cccccc"
_LOGO_RESOURCE = "logo://help"


class _HelpBrowser(QTextBrowser):
    """QTextBrowser that swaps a hovered anchor's underline style
    (dotted at rest, none on hover). Qt's QTextDocument CSS doesn't
    support `:hover`, so we track the hovered fragment via mouse
    moves and rewrite its char format directly.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._hover_pos: int = -1
        self._hover_len: int = 0

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        super().mouseMoveEvent(event)
        cursor = self.cursorForPosition(event.pos())
        anchor_pos, anchor_len = -1, 0
        block = cursor.block()
        it = block.begin()
        while not it.atEnd():
            frag = it.fragment()
            if frag.isValid() and frag.charFormat().isAnchor():
                start = frag.position()
                end = start + frag.length()
                if start <= cursor.position() < end:
                    anchor_pos, anchor_len = start, frag.length()
                    break
            it += 1
        if anchor_pos == self._hover_pos:
            return
        if self._hover_pos >= 0:
            self._set_link_underline(
                self._hover_pos, self._hover_len, dotted=True
            )
        if anchor_pos >= 0:
            self._set_link_underline(anchor_pos, anchor_len, dotted=False)
        self._hover_pos, self._hover_len = anchor_pos, anchor_len

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        if self._hover_pos >= 0:
            self._set_link_underline(
                self._hover_pos, self._hover_len, dotted=True
            )
            self._hover_pos, self._hover_len = -1, 0

    def _set_link_underline(self, pos: int, length: int, dotted: bool) -> None:
        cursor = QTextCursor(self.document())
        cursor.setPosition(pos)
        cursor.setPosition(
            pos + length, QTextCursor.MoveMode.KeepAnchor
        )
        cf = QTextCharFormat()
        cf.setFontUnderline(dotted)
        cf.setUnderlineStyle(
            QTextCharFormat.UnderlineStyle.DotLine
            if dotted
            else QTextCharFormat.UnderlineStyle.NoUnderline
        )
        cursor.mergeCharFormat(cf)


class HelpPage(QWidget):
    """Read-only markdown viewer wired to the bundled help file."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 32, 48, 32)
        layout.setSpacing(0)

        self._view = _HelpBrowser(self)
        self._view.setObjectName("helpView")
        self._view.setOpenExternalLinks(True)
        self._view.setFrameShape(QTextBrowser.Shape.NoFrame)
        self._view.document().setDefaultStyleSheet(
            "p, li { line-height: 1.45; }"
            "table { border-collapse: collapse; }"
            "th, td { padding: 4px 10px; border: 1px solid #d0d0d0; }"
        )
        layout.addWidget(self._view)

        self._load_markdown()

    def _load_markdown(self) -> None:
        path = resource_path("help.md")
        if not path:
            self._view.setPlainText(self.tr("Help file not bundled."))
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                md = fh.read()
        except OSError as exc:
            self._view.setPlainText(
                self.tr("Could not read help file: {0}").format(exc)
            )
            return
        # Qt's setMarkdown supports CommonMark + GFM tables; perfect
        # for our docs.
        doc = self._view.document()
        doc.setMarkdown(
            md, QTextDocument.MarkdownFeature.MarkdownDialectGitHub
        )
        # setMarkdown writes inline styles on every span, so a CSS
        # default-stylesheet rule for `h1` / `code` is silently
        # overridden. Walk the document and force heading colors +
        # font sizes on each block, and a grey background on every
        # monospace fragment (which is how Qt renders inline `code`
        # and ``` fenced blocks).
        self._restyle_headings(doc)
        self._restyle_code(doc)
        self._restyle_paragraphs(doc)
        self._restyle_links(doc)
        self._restyle_hrs(doc)
        self._insert_logo(doc)
        # All the cursor manipulation above leaves the visible
        # cursor at the END of the document, so the browser would
        # open scrolled to the bottom. Reset both the cursor and
        # the scrollbar to the top.
        top = QTextCursor(doc)
        top.movePosition(QTextCursor.MoveOperation.Start)
        self._view.setTextCursor(top)
        self._view.verticalScrollBar().setValue(0)

    @staticmethod
    def _restyle_headings(doc: QTextDocument) -> None:
        # Qt's setMarkdown stores heading sizes as FontSizeAdjustment
        # (a relative-size enum: xx-large = 3, x-large = 2, large = 1)
        # rather than FontPointSize. The renderer prefers the
        # adjustment and ignores any FontPointSize we add via
        # setFont, so we have to clear FontSizeAdjustment and set
        # FontPointSize explicitly for the override to land.
        fpt_prop = QTextFormat.Property.FontPointSize
        fsa_prop = QTextFormat.Property.FontSizeAdjustment
        for level, color in _HEADING_COLORS.items():
            pt = _HEADING_POINT_SIZES.get(level)
            block = doc.firstBlock()
            while block.isValid():
                if block.blockFormat().headingLevel() == level:
                    it = block.begin()
                    while not it.atEnd():
                        frag = it.fragment()
                        if frag.isValid():
                            cf = frag.charFormat()
                            if pt is not None:
                                cf.setProperty(fpt_prop, float(pt))
                                if cf.hasProperty(fsa_prop):
                                    cf.clearProperty(fsa_prop)
                            cf.setForeground(QColor(color))
                            cursor = QTextCursor(doc)
                            cursor.setPosition(frag.position())
                            cursor.setPosition(
                                frag.position() + frag.length(),
                                QTextCursor.MoveMode.KeepAnchor,
                            )
                            cursor.setCharFormat(cf)
                        it += 1
                block = block.next()

    @staticmethod
    def _restyle_paragraphs(doc: QTextDocument) -> None:
        """Increase bottom margin on plain paragraph blocks so prose
        doesn't feel cramped. Skip headings (already styled) and
        list items (their own spacing is fine).
        """
        block = doc.firstBlock()
        while block.isValid():
            bf = block.blockFormat()
            is_heading = bf.headingLevel() > 0
            is_list = block.textList() is not None
            if not is_heading and not is_list and block.text().strip():
                new_bf = QTextBlockFormat(bf)
                new_bf.setBottomMargin(_PARAGRAPH_BOTTOM_MARGIN)
                cursor = QTextCursor(block)
                cursor.setBlockFormat(new_bf)
            block = block.next()

    @staticmethod
    def _restyle_hrs(doc: QTextDocument) -> None:
        """Color the markdown horizontal rule blocks gray instead
        of the palette default (which lands on near-black). Qt
        renders `---` as a block carrying the
        `BlockTrailingHorizontalRulerWidth` property and draws it
        using the block's foreground brush.
        """
        hr_prop = QTextFormat.Property.BlockTrailingHorizontalRulerWidth
        gray = QColor(_HR_COLOR)
        block = doc.firstBlock()
        while block.isValid():
            bf = block.blockFormat()
            if bf.hasProperty(hr_prop):
                new_bf = QTextBlockFormat(bf)
                new_bf.setForeground(gray)
                cursor = QTextCursor(block)
                cursor.setBlockFormat(new_bf)
            block = block.next()

    def _insert_logo(self, doc: QTextDocument) -> None:
        """Insert the application logo as the first block in the
        document so it scrolls with the rest of the help content
        instead of staying pinned above it.
        """
        ipath = icon_path()
        if not ipath:
            return
        renderer = QSvgRenderer(ipath)
        if not renderer.isValid():
            return
        pix = render_svg_to_pixmap(renderer, _LOGO_PX)
        doc.addResource(
            QTextDocument.ResourceType.ImageResource,
            QUrl(_LOGO_RESOURCE),
            pix,
        )
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        # Save the original first-block format (typically the H1)
        # so we can restore it on the split-off block after we
        # re-style the current block as our centered image holder.
        original_bf = cursor.blockFormat()
        original_cf = cursor.charFormat()
        image_bf = QTextBlockFormat()
        image_bf.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        image_bf.setBottomMargin(_PARAGRAPH_BOTTOM_MARGIN * 2)
        cursor.setBlockFormat(image_bf)
        cursor.insertImage(_LOGO_RESOURCE)
        cursor.insertBlock(original_bf, original_cf)

    @staticmethod
    def _restyle_links(doc: QTextDocument) -> None:
        fg = QColor(_LINK_COLOR)
        block = doc.firstBlock()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid() and frag.charFormat().isAnchor():
                    cursor = QTextCursor(doc)
                    cursor.setPosition(frag.position())
                    cursor.setPosition(
                        frag.position() + frag.length(),
                        QTextCursor.MoveMode.KeepAnchor,
                    )
                    cf = QTextCharFormat()
                    cf.setForeground(fg)
                    cf.setFontUnderline(True)
                    cf.setUnderlineStyle(
                        QTextCharFormat.UnderlineStyle.DotLine
                    )
                    cursor.mergeCharFormat(cf)
                it += 1
            block = block.next()

    @staticmethod
    def _restyle_code(doc: QTextDocument) -> None:
        bg = QColor(_CODE_BG)
        block = doc.firstBlock()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid():
                    cf = frag.charFormat()
                    if cf.fontFixedPitch() or cf.font().fixedPitch():
                        cursor = QTextCursor(doc)
                        cursor.setPosition(frag.position())
                        cursor.setPosition(
                            frag.position() + frag.length(),
                            QTextCursor.MoveMode.KeepAnchor,
                        )
                        new_cf = QTextCharFormat()
                        new_cf.setBackground(bg)
                        cursor.mergeCharFormat(new_cf)
                it += 1
            block = block.next()
