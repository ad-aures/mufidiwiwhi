<p align="center">
  <img src="mufidiwiwhi.svg" alt="Mufidiwiwhi" width="128" height="128">
</p>

# Mufidiwiwhi

**v2.0.0 &mdash; now with a GUI.** See [DOCS.md](DOCS.md) for the user manual (what it does, how the corrector works, recommended settings, hardware, privacy).

**Install on Ubuntu / Debian:**

    curl -fsSL https://codeberg.org/adaures/mufidiwiwhi/raw/branch/main/install_ubuntu.sh | bash

Or grab the binary from the [releases page](https://codeberg.org/adaures/mufidiwiwhi/releases).

Mufidiwiwhi (Multi-file diarisation with Whisper) is a tiny, **quick-and-dirty** program built on top of [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

It transcribes audio with reliable [speaker diarisation](https://en.wikipedia.org/wiki/Speaker_diarisation), by using **one file per speaker**: Mufidiwiwhi requires that you record each speaker in a separate file.
You can [use Mumble to record a podcast with guests](https://blog.castopod.org/use-mumble-to-record-a-podcast-with-guests/) or use [Ardour DAW](https://ardour.org/) to [record a Podcast with several remote guests](https://blog.castopod.org/how-to-record-a-podcast-with-several-remote-guests/) (you can also use [Zrythm](https://blog.castopod.org/how-to-record-a-podcast-with-zrythm/)).
This will create 100% accurate diarisation.

Of course, you should run Mufidiwiwhi before merging all audio files together.

More information: [Transcribe your Podcast with accurate speaker diarisation, for free, with Whisper](https://blog.castopod.org/transcribe-your-podcast-with-accurate-speaker-diarisation-for-free-with-whisper/)

Make sure that you choose a [podcast hosting platform that supports transcripts](https://podcastindex.org/apps?appTypes=hosting&elements=Transcript) (such as [Castopod](https://castopod.org/)!).

## Setup

Mufidiwiwhi requires Python 3.10 or newer. The transcription engine is `faster-whisper`, which uses CTranslate2 for inference (4 to 8 times faster than `openai-whisper` on CPU). `ffmpeg` must be available on the PATH.

    pip install git+https://codeberg.org/adaures/mufidiwiwhi.git

This installs both the `mufidiwiwhi` CLI and the `mufidiwiwhi-gui` GUI. All runtime dependencies (faster-whisper, PyQt6, the phonetic libraries, httpx) are pulled in automatically.

### Ubuntu / Debian one-liner

Pulls the latest release from Codeberg, drops the binary in `~/.local/bin`, registers the icon and an apps-menu entry. No sudo, no system packages.

    curl -fsSL https://codeberg.org/adaures/mufidiwiwhi/raw/branch/main/install_ubuntu.sh | bash

To uninstall, remove these three files:

    ~/.local/bin/mufidiwiwhi-gui
    ~/.local/share/icons/hicolor/scalable/apps/mufidiwiwhi.svg
    ~/.local/share/applications/mufidiwiwhi-gui.desktop

## Command-line usage

To get help, type

    mufidiwiwhi --help

Example:

    mufidiwiwhi Lucy interview_lucy.wav Samir interview_samir.wav Rachel interview_rachel.wav --model large-v3 --language fr

You can also point Mufidiwiwhi at an Audacity `.aup3` project; each track is extracted to a 16 kHz mono WAV in a temp folder, the speaker name comes from the Audacity track name, and the temp folder is cleaned up when the app exits:

    mufidiwiwhi project.aup3 --model medium --language fr

Add `-v` / `--verbose` for per-chunk timing, replacement decisions, and any pathological-word warnings on stderr; the high-level pipeline progress prints on stdout by default.

Overlapping speech (when one speaker interrupts another) is preserved as overlapping subtitle cues; SRT and VTT both support this. Most players render only one cue at a time, but the data is in the file.

### Optional post-correction

A plain-text dictionary of proper nouns and domain terms can be used to correct the transcript. The phonetic pass uses Whisper's per-word confidence: low-confidence words are replaced when phonetically close to a dictionary entry, mid-confidence words are replaced only when there is a single very close match, and high-confidence words are left alone.

    # one entry per line, # for comments, multi-word entries allowed
    cat > vocab.txt <<EOF
    # Podcast vocabulary
    Castopod
    OpenRAG
    Podcasting 2.0
    Free Software Foundation
    EOF

    mufidiwiwhi Alice a.wav Bob b.wav --model small --language fr \
        --dictionary vocab.txt \
        --phonetic-lang fr --phonetic-lang-secondary en

## GUI

A PyQt6 GUI is available via the `mufidiwiwhi-gui` console script. It exposes the same functionality as the CLI in a tabbed window (Setup, Project, Run). Settings persist via `QSettings`.

    mufidiwiwhi-gui

The GUI also accepts the same arguments as the CLI to prefill the project tab:

    mufidiwiwhi-gui Lucy interview_lucy.wav Samir interview_samir.wav --dictionary vocab.txt

A PyInstaller spec is provided in `packaging/` for single-file Linux distribution.

## Dependencies

Runtime:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) &mdash; CTranslate2-backed Whisper inference.
- [NumPy](https://numpy.org/) &mdash; numeric arrays for audio buffers.
- [phonetic-fr](https://pypi.org/project/phonetic-fr/) &mdash; French phonetic algorithm.
- [Metaphone](https://pypi.org/project/Metaphone/) &mdash; English phonetic algorithm.
- [jellyfish](https://github.com/jamesturk/jellyfish) &mdash; Levenshtein and other string-distance metrics.
- [spylls](https://github.com/zverok/spylls) &mdash; pure-Python Hunspell reader.
- [psutil](https://github.com/giampaolo/psutil) &mdash; CPU / memory introspection for the GUI metrics strip.
- [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) &mdash; NVIDIA GPU / VRAM introspection.
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) &mdash; the GUI toolkit.
- [pydub](https://github.com/jiaaro/pydub) &mdash; audio chunking via `ffmpeg`.

Build / dev:

- [pytest](https://pytest.org/) &mdash; test runner.
- [PyInstaller](https://pyinstaller.org/) &mdash; standalone Linux binary builds.

External tools:

- [ffmpeg](https://ffmpeg.org/) &mdash; audio decoding (must be on `PATH`).

## Credits

Sidebar and toolbar icons are from the [Solar Linear Icons Collection](https://www.svgrepo.com/collection/solar-linear-icons/) on SVG Repo.

## Author

Benjamin Bellamy &lt;benjamin@podlibre.org&gt;.

## License

Mufidiwiwhi is released under the **GNU General Public License v3**. Copyright &copy; 2026 Ad Aures. See [LICENSE](https://codeberg.org/adaures/mufidiwiwhi/src/branch/main/LICENSE) for further details.
