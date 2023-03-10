# Mufidiwiwhi

Mufidiwhiwi (Multi-file diarization with Whisper) is a tiny program using [Whisper](https://github.com/openai/whisper).

It will transcript audio files with reliable diarization, using one file per speaker: Mufidiwhiwi requires that you record all speaker in a separate file.  
In order to do that, you can [use Mumble to record a podcast with guests](https://blog.castopod.org/use-mumble-to-record-a-podcast-with-guests/) or use [Ardour DAW](https://ardour.org/) to [record a Podcast with several remote guests](https://blog.castopod.org/how-to-record-a-podcast-with-several-remote-guests/) (you can also use [Zrythm](https://blog.castopod.org/how-to-record-a-podcast-with-zrythm/)).

Of course, you should run Mufidiwhiwi before merging all audio files together.

## Setup

You need Whisper and [Pydub](http://pydub.com/) installed.

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
    pip install pydub
    pip install git+https://github.com/adaures/mufidiwhiwi.git

## Command-line usage

To get help, type

    mufidiwhiwi --help

Example:

    mufidiwhiwi Lucy interview_lucy.wav Samir interview_samir.wav Rachel interview_rachel.wav --model large --language French

See [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) for the list of all available languages.

## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.

Mufidiwhiwi's code is released under the MIT License. See [LICENSE](https://github.com/adaures/mufidiwhiwi/blob/main/LICENSE) for further details.
