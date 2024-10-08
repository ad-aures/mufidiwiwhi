from typing import Callable, TextIO
from whisper.utils import ResultWriter, WriteTXT, WriteVTT, WriteTSV, WriteJSON, format_timestamp


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(self, result: dict):
        for segment in result["segments"]:
            segment_start = self.format_timestamp(segment["start"])
            segment_end = self.format_timestamp(segment["end"])
            segment_speaker = segment["speaker"]
            segment_text = segment["text"].strip().replace("-->", "->")

            if word_timings := segment.get("words", None):
                all_words = [timing["word"] for timing in word_timings]
                all_words[0] = all_words[0].strip()  # remove the leading space, if any
                last = segment_start
                for i, this_word in enumerate(word_timings):
                    start = self.format_timestamp(this_word["start"])
                    end = self.format_timestamp(this_word["end"])
                    if last != start:
                        yield last, start, segment_text, segment_speaker

                    yield start, end, "".join(
                        [
                            f"<u>{word}</u>" if j == i else word
                            for j, word in enumerate(all_words)
                        ]
                    )
                    last = end

                if last != segment_end:
                    yield last, segment_end, segment_text, segment_speaker
            else:
                yield segment_start, segment_end, segment_text, segment_speaker

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )

class WriteSpeakerSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(self, result: dict, file: TextIO, options: dict = None):
        for i, (start, end, text, speaker) in enumerate(self.iterate_result(result), start=1):
            print(f"{i}\n{start} --> {end}\n[{speaker}] {text}\n", file=file, flush=True) 

class WriteSpeakerVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(self, result: dict, file: TextIO, options: dict = None):
        for i, (start, end, text, speaker) in enumerate(self.iterate_result(result), start=1):
            print(f"{i}\n{start} --> {end}\n<{speaker}>{text}\n", file=file, flush=True)

def get_writer(output_format: str, output_dir: str) -> Callable[[dict, TextIO], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteSpeakerVTT,
        "srt": WriteSpeakerSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, file: TextIO):
            for writer in all_writers:
                writer(result, file)

        return write_all

    return writers[output_format](output_dir)
