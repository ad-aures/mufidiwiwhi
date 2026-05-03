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

"""Speakers and audio-files table model and view."""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QInputDialog,
    QTableView,
)

from .audio_drop import mime_audio_paths


class SpeakerTableModel(QAbstractTableModel):
    HEADERS = ("Speaker", "Audio file")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[dict] = []  # each: {"speaker": str, "file_path": str}

    # ----- public API ----------------------------------------------------
    def rows(self) -> list[dict]:
        return list(self._rows)

    def set_rows(self, rows: list[dict]) -> None:
        self.beginResetModel()
        self._rows = [
            {"speaker": str(r.get("speaker", "")), "file_path": str(r.get("file_path", ""))}
            for r in rows
        ]
        self.endResetModel()

    def add_row(self, speaker: str, file_path: str) -> None:
        self.beginInsertRows(QModelIndex(), len(self._rows), len(self._rows))
        self._rows.append({"speaker": speaker, "file_path": file_path})
        self.endInsertRows()

    def remove_row(self, row: int) -> None:
        if 0 <= row < len(self._rows):
            self.beginRemoveRows(QModelIndex(), row, row)
            del self._rows[row]
            self.endRemoveRows()

    def move_row(self, src: int, dst: int) -> None:
        if src == dst or not (0 <= src < len(self._rows)) or not (0 <= dst < len(self._rows)):
            return
        item = self._rows.pop(src)
        self._rows.insert(dst, item)
        self.layoutChanged.emit()

    def file_paths(self) -> list[str]:
        return [r["file_path"] for r in self._rows]

    # ----- QAbstractTableModel ------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._rows):
            return None
        row = self._rows[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if index.column() == 0:
                return row["speaker"]
            if index.column() == 1:
                return row["file_path"]
        if role == Qt.ItemDataRole.ToolTipRole and index.column() == 1:
            return row["file_path"]
        return None

    def setData(
        self,
        index: QModelIndex,
        value,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        if index.column() == 0:
            self._rows[index.row()]["speaker"] = str(value)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
            return True
        if index.column() == 1:
            self._rows[index.row()]["file_path"] = str(value)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
            return True
        return False

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return section + 1

    def flags(self, index: QModelIndex):
        base = super().flags(index)
        if not index.isValid():
            return base | Qt.ItemFlag.ItemIsDropEnabled
        return (
            base
            | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
        )


class SpeakerTableView(QTableView):
    """QTableView with audio-file drag-drop support."""

    def __init__(self, model: SpeakerTableModel, parent=None) -> None:
        super().__init__(parent)
        self.setModel(model)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        paths = mime_audio_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        paths = mime_audio_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        paths = mime_audio_paths(event.mimeData())
        if not paths:
            super().dropEvent(event)
            return
        model = self.model()
        assert isinstance(model, SpeakerTableModel)
        for path in paths:
            default_name = os.path.splitext(os.path.basename(path))[0]
            speaker, ok = QInputDialog.getText(
                self,
                self.tr("Speaker name"),
                self.tr("Speaker name for {0}:").format(os.path.basename(path)),
                text=default_name,
            )
            if ok and speaker.strip():
                model.add_row(speaker.strip(), path)
        event.acceptProposedAction()
