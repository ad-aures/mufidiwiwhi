# Mufidiwiwhi

Mufidiwiwhi (Multi-file diarisation with Whisper) is a tiny, **quick-and-dirty** program using [Whisper](https://github.com/openai/whisper).

It will transcript audio files with reliable [speaker diarisation](https://en.wikipedia.org/wiki/Speaker_diarisation), by using **one file per speaker**: Mufidiwiwhi requires that you record each speaker in a separate file.  
In order to do that, you can [use Mumble to record a podcast with guests](https://blog.castopod.org/use-mumble-to-record-a-podcast-with-guests/) or use [Ardour DAW](https://ardour.org/) to [record a Podcast with several remote guests](https://blog.castopod.org/how-to-record-a-podcast-with-several-remote-guests/) (you can also use [Zrythm](https://blog.castopod.org/how-to-record-a-podcast-with-zrythm/)).  
This will create 100% accurate diarisation.

Of course, you should run Mufidiwiwhi before merging all audio files together.

More information: [Transcribe your Podcast with accurate speaker diarisation, for free, with Whisper](https://blog.castopod.org/transcribe-your-podcast-with-accurate-speaker-diarisation-for-free-with-whisper/)

Make sure that you choose a [podcast hosting platform that supports transcripts](https://podcastindex.org/apps?appTypes=hosting&elements=Transcript) (such as [Castopod](https://castopod.org/)!).

## Setup

You need [Whisper](https://github.com/openai/whisper) and [Pydub](http://pydub.com/) installed.

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
    pip install pydub
    pip install git+https://github.com/ad-aures/mufidiwiwhi.git

## Command-line usage

To get help, type

    mufidiwiwhi --help

Example:

    mufidiwiwhi Lucy interview_lucy.wav Samir interview_samir.wav Rachel interview_rachel.wav --model large --language French

See [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) for the list of all available languages.

## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.

Mufidiwiwhi's code is released under the MIT License. See [LICENSE](https://github.com/adaures/mufidiwiwhi/blob/main/LICENSE) for further details.

