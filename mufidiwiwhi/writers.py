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

"""Output writers for transcription results.

These writers were originally provided by `whisper.utils`. Since
Mufidiwiwhi now depends on faster-whisper instead of openai-whisper,
the small set of writer helpers we need is vendored here.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional, TextIO


def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    if seconds < 0:
        seconds = 0.0
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


def str2bool(string: str) -> bool:
    table = {"True": True, "False": False, "true": True, "false": False}
    if string in table:
        return table[string]
    raise ValueError(f"Expected True or False, got {string!r}")


def optional_int(string: str) -> Optional[int]:
    return None if string == "None" else int(string)


def optional_float(string: str) -> Optional[float]:
    return None if string == "None" else float(string)


class ResultWriter:
    extension: str = ""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(
        self,
        result: dict,
        audio_path: str,
        options: Optional[dict] = None,
    ) -> str:
        basename = os.path.basename(audio_path)
        basename = os.path.splitext(basename)[0]
        output_path = os.path.join(
            self.output_dir, basename + "." + self.extension
        )
        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options)
        return output_path

    def write_result(
        self,
        result: dict,
        file: TextIO,
        options: Optional[dict] = None,
    ) -> None:
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result, file, options=None):
        for segment in result["segments"]:
            text = segment["text"].strip()
            if text:
                print(text, file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result, file, options=None):
        json.dump(result, file, ensure_ascii=False)


class WriteTSV(ResultWriter):
    extension: str = "tsv"

    def write_result(self, result, file, options=None):
        print("start", "end", "speaker", "text", sep="\t", file=file)
        for segment in result["segments"]:
            text = segment["text"].strip().replace("\t", " ")
            if not text:
                continue
            print(
                round(1000 * segment["start"]),
                round(1000 * segment["end"]),
                segment.get("speaker", ""),
                text,
                sep="\t",
                file=file,
                flush=True,
            )


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(self, result: dict):
        for segment in result["segments"]:
            segment_start = self.format_timestamp(segment["start"])
            segment_end = self.format_timestamp(segment["end"])
            segment_speaker = segment.get("speaker", "")
            segment_text = segment["text"].strip().replace("-->", "->")
            if not segment_text:
                continue
            yield segment_start, segment_end, segment_text, segment_speaker

    def format_timestamp(self, seconds: float) -> str:
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteSpeakerSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(self, result, file, options=None):
        for i, (start, end, text, speaker) in enumerate(
            self.iterate_result(result), start=1
        ):
            speaker_tag = f"[{speaker}] " if speaker else ""
            print(
                f"{i}\n{start} --> {end}\n{speaker_tag}{text}\n",
                file=file,
                flush=True,
            )


class WriteSpeakerVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = True
    decimal_marker: str = "."

    def write_result(self, result, file, options=None):
        print("WEBVTT\n", file=file)
        for i, (start, end, text, speaker) in enumerate(
            self.iterate_result(result), start=1
        ):
            speaker_tag = f"<v {speaker}>" if speaker else ""
            print(
                f"{i}\n{start} --> {end}\n{speaker_tag}{text}\n",
                file=file,
                flush=True,
            )


_WRITER_CLASSES = {
    "txt": WriteTXT,
    "vtt": WriteSpeakerVTT,
    "srt": WriteSpeakerSRT,
    "tsv": WriteTSV,
    "json": WriteJSON,
}

WRITER_FORMATS = tuple(_WRITER_CLASSES.keys())


def _normalise_formats(formats) -> list[str]:
    """Accept 'all', a single name, a comma-separated string, or a list."""
    if isinstance(formats, str):
        if formats == "all":
            return list(_WRITER_CLASSES.keys())
        if "," in formats:
            return [f.strip() for f in formats.split(",") if f.strip()]
        return [formats]
    out: list[str] = []
    for item in formats:
        if item == "all":
            return list(_WRITER_CLASSES.keys())
        if "," in item:
            out.extend(f.strip() for f in item.split(",") if f.strip())
        else:
            out.append(item)
    return out


def get_writer(
    output_formats,
    output_dir: str,
) -> Callable[[dict, str], list[str]]:
    """Return a callable `(result, audio_path) -> list[output_path]`.

    `output_formats` may be 'all', a single format name, a comma-separated
    string, or a list of format names.
    """
    formats = _normalise_formats(output_formats)
    unknown = [f for f in formats if f not in _WRITER_CLASSES]
    if unknown:
        raise ValueError(
            f"Unknown output format(s): {unknown}. "
            f"Known: {sorted(_WRITER_CLASSES)}"
        )

    writers = [_WRITER_CLASSES[f](output_dir) for f in formats]

    def write_all(result: dict, audio_path: str) -> list[str]:
        return [w(result, audio_path) for w in writers]

    return write_all
