# Manual GUI test checklist

Run after every meaningful change to `mufidiwiwhi/gui/`.

## Smoke

1. `mufidiwiwhi-gui` launches; the main window shows three tabs: Setup, Project, Run.
2. The Setup tab loads with sensible defaults (model `small`, device `auto`, no dictionary).
3. Edit any Setup field; close and relaunch the app; field is restored.

## Project tab

4. Drag two audio files (.wav or .flac) onto the speakers table; you are prompted for a speaker name per drop; rows appear.
5. Click "Add speaker..."; pick a file via the dialog; provide a name; row appears.
6. Edit a speaker name in place by double clicking the cell.
7. Use Move up / Move down to reorder rows; order persists when leaving the tab.
8. Set an output directory; the Start button enables.
9. Pick a dictionary file via Browse; the "Enable phonetic correction" checkbox auto checks.
10. Output filename placeholder shows the longest common prefix of the input basenames.

## Run tab

11. Click Start; tab switches to Run; progress bars and log start updating.
12. Cancel mid-run; current speaker file aborts at the next segment boundary; finishes "Cancelled".
13. Run to completion; the log ends with the output paths; "Open output folder" works.
14. Click "New project"; tab switches back to Project; previous selections are still there.

## Persistence

15. Close the window and relaunch; Project tab restores the last session (speakers, dictionary, formats).
16. Choose Setup -> Reset to defaults; the global settings revert to the canned values.

## CLI passthrough

17. `mufidiwiwhi-gui Lucy /tmp/spkA.wav Samir /tmp/spkB.wav --dictionary /tmp/dict.txt` opens the app with the speakers prefilled and the dictionary path filled in.

## Failure paths

18. With Ollama not running, click "Test connection" on Setup; the button shows "Failed: ..." rather than crashing.
19. Set `--llm-correct` to a non-existent model; the LLM pass logs a warning and the phonetic output is preserved.
20. Provide a non-existent audio file; the run completes with an error in the log; the GUI does not crash.
