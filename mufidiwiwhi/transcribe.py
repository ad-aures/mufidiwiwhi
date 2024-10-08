#!env/bin/python3

import os
import sys
import argparse
import whisper
import pydub
import numpy as np
import torch
import tqdm
import warnings
from pydub import AudioSegment
from pydub.silence import db_to_float
from typing import Optional, Union, Tuple
from whisper.audio import (
    CHUNK_LENGTH,
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from whisper.tokenizer import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
    get_tokenizer,
)
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.utils import (
    exact_div,
    format_timestamp,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)
from utils import (
    get_writer,
)
from whisper import available_models

MIN_SILENCE_LEN = 500             # silence longer than MIN_SILENCE_LEN ms will be trimed
SEEK_STEP = 50                    # search for silence every SEEK_STEP ms
DBFS_THRESHOLD = 40               # cuts every time dbFS falls DBFS_THRESHOLD db under
MIN_CHUNK_LEN = 700               # each chunk must be longer than MIN_CHUNK_LEN ms
MAX_CHUNK_LEN = CHUNK_LENGTH*1000 # each chunk must be shorter than MAX_CHUNK_LEN ms

# Convert Pydub audio to numpy array:
def pydub_to_np(pydub_audio: pydub.AudioSegment) -> (np.ndarray):
    # Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    # where each value is in range [-1.0, 1.0].
    return np.array(pydub_audio.get_array_of_samples(), np.int16).flatten().astype(np.float32) / pydub_audio.max

# Let's transcribe some audio!
def transcribe(
    model: "Whisper",
    audio_dicts: list,
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -0.57,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    **decode_options,
):
    """
    Transcribe an audio file using Whisper
    Parameters
    ----------
    model: Whisper
        The Whisper model instance
    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform
    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything
    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.
    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed
    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed
    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent
    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.
    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word
    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word
    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.
    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances
    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    language: str

    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    for audio_dict in audio_dicts:
        speaker=audio_dict['speaker']
        if verbose is not None:
           print(
               f"Speaker: {speaker}"
           )
        
        pydub_audio = AudioSegment.from_file(audio_dict['file_path'])
        # resample audio to 16KHz, monophonic, 16bits/sample, remove offset
        pydub_audio = pydub_audio.set_frame_rate(SAMPLE_RATE)
        pydub_audio = pydub_audio.set_channels(1)
        pydub_audio = pydub_audio.set_sample_width(2)
        pydub_audio = pydub_audio.remove_dc_offset()

        # Pad 30-seconds of silence to the input audio, for slicing
        mel = log_mel_spectrogram(pydub_to_np(pydub_audio))
        content_frames = mel.shape[-1] - N_FRAMES

        if decode_options.get("language", None) is None:
            if not model.is_multilingual:
                decode_options["language"] = "en"
            else:
                if verbose:
                    print(
                        "Detecting language using up to the first 30 seconds from the first file. You should probably use `--language` to specify the language"
                    )
                mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
                _, probs = model.detect_language(mel_segment)
                decode_options["language"] = max(probs, key=probs.get)
                if verbose is not None:
                    print(
                        f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                    )

        language = decode_options["language"]
        task: str = decode_options.get("task", "transcribe")
        tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

        if word_timestamps and task == "translate":
            warnings.warn("Word-level timestamps on translations may not be reliable.")

        def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
            temperatures = (
                [temperature] if isinstance(temperature, (int, float)) else temperature
            )
            decode_result = None

            for t in temperatures:
                kwargs = {**decode_options}
                if t > 0:
                    # disable beam_size and patience when t > 0
                    kwargs.pop("beam_size", None)
                    kwargs.pop("patience", None)
                else:
                    # disable best_of when t == 0
                    kwargs.pop("best_of", None)

                options = DecodingOptions(**kwargs, temperature=t)
                decode_result = model.decode(segment, options)

                needs_fallback = False
                if (
                    compression_ratio_threshold is not None
                    and decode_result.compression_ratio > compression_ratio_threshold
                ):
                    needs_fallback = True  # too repetitive
                if (
                    logprob_threshold is not None
                    and decode_result.avg_logprob < logprob_threshold
                ):
                    needs_fallback = True  # average log probability is too low

                if not needs_fallback:
                    break

            return decode_result

        # Total length of the audio file:
        total_len = len(pydub_audio)
        # Cursor within audio file:
        file_cursor = 0
        # Silence Threshold, everything lower this will be trimed:
        silence_threshold = db_to_float(pydub_audio.max_dBFS-DBFS_THRESHOLD) * pydub_audio.max_possible_amplitude

        if initial_prompt is not None:
            initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
            all_tokens.extend(initial_prompt_tokens)
        else:
            initial_prompt_tokens = []

        def new_segment(
                *, start: float, end: float, speaker:str, tokens: torch.Tensor, result: DecodingResult
        ):
            text_tokens = [token for token in tokens.tolist() if token < tokenizer.eot]
            return {
                "id": len(all_segments),
                "seek": file_cursor/1000,
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": tokenizer.decode(text_tokens),
                "tokens": text_tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }

        # show the progress bar when verbose is False (if True, transcribed text will be printed)
        with tqdm.tqdm(
            total=total_len//1000, unit=" seconds", disable=verbose is not False
        ) as pbar:
            while file_cursor < (total_len - MIN_CHUNK_LEN):
                previous_cursor = file_cursor
                # Find the first non-silence piece:
                while file_cursor<(total_len - MIN_CHUNK_LEN) and (pydub_audio[file_cursor : file_cursor + SEEK_STEP].rms < silence_threshold):
                    file_cursor += SEEK_STEP
                chunk_cursor = file_cursor + MIN_CHUNK_LEN
                chunk_min_cursor = chunk_cursor            
                min_rms = pydub_audio[chunk_cursor : chunk_cursor + MIN_SILENCE_LEN].rms
                # Finding the lowest cursor:
                while chunk_cursor < (total_len - MIN_SILENCE_LEN) and chunk_cursor < (file_cursor + MAX_CHUNK_LEN):
                    chunk_rms = pydub_audio[chunk_cursor : chunk_cursor + MIN_SILENCE_LEN].rms
                    # If the current level is lower than the threshold, cut here, no need to go further:
                    if chunk_rms <= silence_threshold:
                        chunk_tmp_min_cursor = chunk_cursor
                        min_rms = chunk_rms
                        # We look for the best timestamp to cut in the next MIN_SILENCE_LEN interval:
                        while chunk_cursor < (total_len - MIN_SILENCE_LEN) and chunk_cursor < (chunk_tmp_min_cursor + MIN_SILENCE_LEN):
                            chunk_rms = pydub_audio[chunk_cursor : chunk_cursor + MIN_SILENCE_LEN].rms
                            if chunk_rms <= min_rms:
                                chunk_min_cursor = chunk_cursor
                                min_rms = chunk_rms
                            chunk_cursor += SEEK_STEP                        
                        chunk_cursor = file_cursor + MAX_CHUNK_LEN
                    # Else find the minimum:
                    else:
                        if chunk_rms <= min_rms:
                            chunk_min_cursor = chunk_cursor
                            min_rms = chunk_rms
                        chunk_cursor += SEEK_STEP

                audio_chunk = pydub_audio[file_cursor : chunk_min_cursor + MIN_SILENCE_LEN]

                mel_segment = pad_or_trim(log_mel_spectrogram(pydub_to_np(audio_chunk)), N_FRAMES).to(model.device).to(dtype)
                current_segments = []

                decode_options["prompt"] = all_tokens[prompt_reset_since:]
                result: DecodingResult = decode_with_fallback(mel_segment)
                tokens = torch.tensor(result.tokens)

                if(result.avg_logprob > logprob_threshold) and (result.no_speech_prob < no_speech_threshold):
                    current_segments.append(
                        new_segment(
                            start=file_cursor/1000,
                            end=chunk_min_cursor/1000,
                            speaker=speaker,
                            tokens=tokens,
                            result=result,
                        )
                    )
                    #current_tokens.append(tokens.tolist())
                file_cursor = chunk_min_cursor

                if not condition_on_previous_text or result.temperature > 0.5:
                    # do not feed the prompt tokens if a high temperature was used
                    prompt_reset_since = len(all_tokens)

                if word_timestamps:
                    add_word_timestamps(
                        segments=current_segments,
                        model=model,
                        tokenizer=tokenizer,
                        mel=mel_segment,
                        num_frames=segment_size,
                        prepend_punctuations=prepend_punctuations,
                        append_punctuations=append_punctuations,
                    )
                    word_end_timestamps = [
                        w["end"] for s in current_segments for w in s["words"]
                    ]
                    if not single_timestamp_ending and len(word_end_timestamps) > 0:
                        seek_shift = round(
                            (word_end_timestamps[-1] - time_offset) * FRAMES_PER_SECOND
                        )
                        if seek_shift > 0:
                            seek = previous_seek + seek_shift

                if verbose:
                    for segment in current_segments:
                        start, end, text = segment["start"], segment["end"], segment["text"]
                        line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                        print(make_safe(line))

                # if a segment is instantaneous or does not contain text, clear it
                for i, segment in enumerate(current_segments):
                    if segment["start"] == segment["end"] or segment["text"].strip() == "":
                        segment["text"] = ""
                        segment["tokens"] = []
                        segment["words"] = []
                        current_tokens[i] = []  

                all_segments.extend(current_segments)

                # update progress bar
                pbar.update((file_cursor-previous_cursor)//1000)

    # sort all transcriptions for all speakers:
    all_segments = sorted(all_segments, key=lambda d: d['start'])

    # We look for overlaps (speakers interrupting each other…):
    i = 0
    # we go from the first segment to the penultimate:
    while i < len(all_segments)-1:
        # we set the segment ID to the current cursor value (because we sorted all speakers together):
        all_segments[i]['id']=i
        # Is there a segment after that starts before the current one ends:
        while(i < len(all_segments)-1) and (all_segments[i]['end']>all_segments[i+1]['start']):
            # if the next segment ends before the current one, we just ignore it and delete it:          
            if(all_segments[i]['end']>=all_segments[i+1]['end']):
                del all_segments[i+1]
            # otherwise we split in the middle and we make the current one end when the next one starts
            else:
                avg=(all_segments[i]['end']+all_segments[i+1]['start'])/2
                all_segments[i]['end']=avg
                all_segments[i+1]['start']=avg
        i += 1

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=language,
    )

def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_args", nargs="+", type=str, help="speaker names and audio files to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["txt", "srt", "json", "all"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-0.57, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead."
            )
        args["language"] = "en"


    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    audio_args = args.pop("audio_args")

    if(len(audio_args)%2 != 0):
        raise Warning("Argument number should be even: speaker_name_1 audio_file_1 speaker_name_2 audio_file_2 …")

    audio_dicts = []
    for i in range(len(audio_args)//2):
        audio_dict={}
        audio_dict['speaker']=audio_args[i*2]
        audio_dict['file_path']=audio_args[i*2+1]
        audio_dicts.append(audio_dict)

    output_filename = audio_dicts[0]['file_path'][0:audio_dicts[0]['file_path'].rindex('.')]
    for separator in ['_', '-', ' ', '.']:
        try:
            output_filename=output_filename[0:output_filename.rindex(separator)]
            break
        except:
            pass

    from whisper import load_model

    model = load_model(model_name, device=device, download_root=model_dir)

    writer = get_writer(output_format, output_dir)
    result = transcribe(model, audio_dicts, temperature=temperature, **args)
    writer(result, output_filename)



if __name__ == "__main__":
    cli()

