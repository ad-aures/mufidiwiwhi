# Mufidiwiwhi

**Version 2.0.0** · © 2026 Ad Aures · GPLv3

Mufidiwiwhi (**Mu**lti-**Fi**le **Di**arisation **Wi**th **Whi**sper) is a desktop transcription tool for podcasts and interviews. It takes one audio file per speaker and produces a single transcript with perfect speaker labels.

It runs entirely on your machine. **No audio, no text, no metadata is ever sent to any cloud service.** All processing happens locally.

> All Mufidiwiwhi features will eventually ship inside [Podlibre](https://podlibre.org/).

---

## What it is for

Mufidiwiwhi is built for people who record podcasts, interviews, or panel discussions and want a clean transcript with reliable speaker attribution.

It is not a general transcription tool: it requires that **each speaker is recorded on a separate file with full audio isolation** (no bleed). That single requirement is what gives it 100 % accurate diarisation without any speaker-identification model.

Hence the name: **MUlti-FIles** rather than diarisation models that try to guess "who is speaking" from a mixed track.

If you only have one mixed file with several speakers on the same track, this tool will not help you &mdash; you need a regular diariser.

### How to record clean per-speaker tracks

Any recording setup that keeps each microphone on its own track will do:

- [Mumble + per-user recording](https://blog.castopod.org/use-mumble-to-record-a-podcast-with-guests/) for remote podcasts.
- [Audacity](https://www.audacityteam.org/), [Ardour](https://ardour.org/) or [Zrythm](https://blog.castopod.org/how-to-record-a-podcast-with-zrythm/) when you control the local DAW.
- A multi-input audio interface routing each mic to its own track.

Run Mufidiwiwhi **before** mixing the tracks down to a single file.

**Audacity `.aup3` projects** can be opened directly, no manual export needed. Mufidiwiwhi reads the project read-only, extracts each track to a 16 kHz mono WAV in a session-scoped temp folder, and feeds those into the same speaker list as if you had picked the WAVs by hand. Stereo tracks are downmixed to mono. The temp folder is cached for the session (re-opening the same project reuses the extracted WAVs) and removed when the app exits.

**Recommended audio format:** WAV, 16-bit PCM, 16 kHz, mono. That is exactly what `faster-whisper` consumes internally, so anything else gets resampled / down-mixed on the fly. Feeding it the native format saves a conversion pass and keeps the audio bit-exact. `ffmpeg -i input.flac -ac 1 -ar 16000 -sample_fmt s16 output.wav` does the conversion in one shot.

---

## How it works

Pipeline overview, in order:

1. **Per-speaker chunking.** Each speaker's audio is split into 30-second-max chunks at local silence minima. No audio is ever dropped.
2. **Per-chunk transcription.** Each chunk is fed to [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (a CTranslate2-backed Whisper runtime). Word-level timestamps and per-word confidence are kept.
3. **Per-segment correction.** Phonetic + Hunspell corrections (see below) replace likely-misrecognised words, guided by Whisper's own per-word confidence.
4. **Global merge + overlap resolution.** Segments from all speakers are merged on the timeline. Overlapping cues are split so writers emit clean, non-overlapping output.
5. **Output writers.** Final transcript saved as SRT, VTT, TSV, JSON, or plain text.

Because each speaker file is processed independently, overlapping speech (interruptions, "yeah!" / "right" while another speaker is going) is preserved in the output: SRT and VTT both support overlapping cues.

---

## Correction system

There are two complementary passes, both opt-in: the **user dictionary** (phonetic) and the **Hunspell** spell-checker. They can be combined.

### User dictionary (phonetic correction)

A plain-text file, one entry per line. Use it for things Whisper has never seen: brand names, guest names, acronyms, project names, jargon.

```
# one entry per line, # for comments
Castopod
GAFAM
RdGP
```

For each transcribed word window, Mufidiwiwhi computes a phonetic code (Metaphone for English, [`phonetic-fr`](https://pypi.org/project/phonetic-fr/) for French) and looks for matching entries.

A match is then accepted or rejected based on:

- **Whisper confidence on the window** (taken as the *minimum* per-word probability &mdash; one shaky word in a phrase pulls the whole window down).
- **Edit-distance** between the input surface and the dictionary entry.
- **Surface guards**: anagrams of short acronyms are rejected (RGPD vs RdGP), single-token mismatches with a different first character are rejected (Linux vs DINUM), windows of all-short tokens are protected against random collisions.
- **Hunspell verdict** (when enabled): if the input is a real word, the bar is raised &mdash; unless the dictionary entry is an obvious near-spelling (Nerd &rarr; NIRD), which signals explicit user intent.

### Hunspell second opinion

Optional. Point Mufidiwiwhi at a Hunspell `.aff` / `.dic` pair (Ubuntu: `/usr/share/hunspell/fr_FR.*`) and the corrector will:

- Modulate the phonetic pass: misspelled words become more permissive, real words more conservative.
- Run a second pass on remaining low-confidence words: if Hunspell offers a single very close suggestion, apply it.

Hunspell on its own is too noisy (it would replace "Twake" with "Take"); the combination phonetic + Hunspell + Whisper-confidence is the sweet spot.

### Confidence bands

The corrector tiers each window using two thresholds (defaults shown):

- **High confidence** (`probability >= 0.95`): word is left alone.
- **Mid confidence** (`0.50 <= probability < 0.95`): a single very close candidate may replace the word.
- **Low confidence** (`probability < 0.50`): the closest phonetic candidate replaces the word.

Adjust the bands from the Settings tab.

### Verbose diagnostics

Pass `-v` / `--verbose` on the CLI (or launch the GUI as `mufidiwiwhi-gui --verbose`) to mirror per-chunk timing, individual replacement decisions, slow-Hunspell warnings, and safety-net vetoes to stderr. Useful when an unexpected substitution sneaks into a transcript: every replacement prints the input, the chosen entry, and the candidate list it was picked from.

---

## Recommended parameters

Sensible starting points for a French podcast on a modern laptop:

| Setting             | Recommended                                 | Why                                    |
| ------------------- | ------------------------------------------- | -------------------------------------- |
| Model               | `medium` (CPU) or `large-v3` (GPU)          | Best accuracy / speed balance.         |
| Language            | Set explicitly (e.g. `fr`)                  | Avoids language-detection slip-ups.    |
| Phonetic language   | Match the recording                         | Drives the phonetic-code algorithm.    |
| Phonetic 2nd lang   | `en` for FR podcasts (or vice versa)        | Picks up English brand / project names mixed into French speech. |
| Edit-distance       | `2`                                         | Default. Higher = more aggressive.     |
| Low / High conf     | `0.50` / `0.95`                             | Defaults. Lower the high band to be more conservative. |
| Output formats      | `srt` + `vtt`                               | Both support overlapping cues, both render in most players. |
| Hunspell            | `fr_FR` for FR, `en_US` for EN              | Apt-installable on Ubuntu / Debian.    |

For very short clips, drop down to `small` or `base` &mdash; the smaller models are fast and accurate enough on clean studio audio.

---

## Recommended hardware

Mufidiwiwhi runs on CPU only, but a CUDA-capable NVIDIA GPU makes the `large-v3` model practical on long files.

| Use case                     | Hardware                                         |
| ---------------------------- | ------------------------------------------------ |
| Quick checks, `tiny` / `base`| Any laptop with 8 GB RAM.                        |
| `medium` on CPU              | 6+ core CPU, 16 GB RAM. Real-time-ish on clean audio. |
| `large-v3` on GPU            | NVIDIA GPU with at least 6 GB VRAM (8 GB comfortable). |
| `large-v3` on CPU            | Possible but slow. 32 GB RAM, 8+ cores recommended. |

The Settings tab auto-detects available compute types (`int8`, `int8_float16`, `float16`, `float32`) and recommends a model that fits in your VRAM.

---

## Privacy

Mufidiwiwhi is fully local:

- The Whisper model is downloaded from Hugging Face the first time, then cached on disk.
- Hunspell dictionaries are read from your filesystem.
- No telemetry, no analytics, no remote API calls during transcription.
- The audio never leaves your machine.

Disconnect from the internet after the first model download; everything still works.

---

## Author and license

Author: Benjamin Bellamy &lt;benjamin@podlibre.org&gt;.

Mufidiwiwhi is **free software**, released under the **GNU General Public License v3**. Copyright &copy; 2026 Ad Aures.

You can use it, study it, modify it, and redistribute it under the terms of the GPLv3. See the `LICENSE` file shipped with the source.

---

## Roadmap

The features developed here are being upstreamed into [Podlibre](https://podlibre.org/), the broader open-source podcast publishing toolchain. Mufidiwiwhi is the standalone playground; Podlibre is the home for the user-facing product.
