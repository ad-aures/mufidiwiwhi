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

"""Persistent settings for the GUI.

Everything is stored in `~/.config/mufidiwiwhi/settings.ini` via
stdlib `configparser`. We deliberately do NOT use Qt's `QSettings`:
when QSettings shares an INI file with our own writes, its in-memory
cache silently overwrites configparser updates on the next sync()
(window-close path), which manifested as "settings won't persist".
Window geometry is stored as a base64-encoded blob in the INI so
configparser is the only writer.
"""

from __future__ import annotations

import base64
import configparser
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


CONFIG_DIR = Path.home() / ".config" / "mufidiwiwhi"
CONFIG_FILE = CONFIG_DIR / "settings.ini"


def _default_config_path() -> str:
    """Return the absolute path to the settings INI file.

    Creates the parent directory if it doesn't exist.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return str(CONFIG_FILE)


def _initial_model_dir() -> str:
    try:
        from .helpers import default_model_dir

        return default_model_dir()
    except Exception:
        return ""


def _initial_device() -> str:
    try:
        from .helpers import default_device

        return default_device()
    except Exception:
        return "cpu"


def _initial_dictionary_path() -> str:
    candidate = CONFIG_DIR / "dictionary.txt"
    return str(candidate) if candidate.exists() else ""


@dataclass
class GlobalSettings:
    model_name: str = "medium"
    model_dir: str = field(default_factory=_initial_model_dir)
    device: str = field(default_factory=_initial_device)
    compute_type: str = "auto"
    default_language: str = ""
    dictionary_path: str = field(default_factory=_initial_dictionary_path)
    phonetic_lang: str = "auto"
    phonetic_lang_secondary: str = ""
    correct_low_conf: float = 0.50
    correct_high_conf: float = 0.95
    correct_edit_distance: int = 2
    # Hunspell second-opinion paths (basename without extension); auto-
    # detected at run time when empty.
    hunspell_primary: str = ""
    hunspell_secondary: str = ""
    use_hunspell: bool = True
    hunspell_threshold: float = 0.5
    # Confidence-colour thresholds (greatest-to-lowest). Used by the
    # Run-tab live transcript and by writers when colour-coded output
    # is requested.
    conf_threshold_excellent: float = 0.99
    conf_threshold_high: float = 0.80
    conf_threshold_mid: float = 0.70
    conf_threshold_low: float = 0.60
    # Observed CPU / GPU temperature range, persisted across runs.
    # The Run-tab metrics strip uses these to colour the temp
    # bubbles by relative load. Sentinel -1.0 means "no observation
    # yet" so the bubble renders with the default border.
    cpu_temp_min: float = -1.0
    cpu_temp_max: float = -1.0
    gpu_temp_min: float = -1.0
    gpu_temp_max: float = -1.0


@dataclass
class ProjectSnapshot:
    speakers: list[dict] = field(default_factory=list)
    output_dir: str = ""
    output_filename: str = ""
    output_formats: list[str] = field(default_factory=lambda: ["srt"])


def _parse_session_json(raw: str) -> Any:
    """Parse a session JSON blob, tolerating legacy QSettings INI
    escaping (the value used to be wrapped in literal double-quotes
    with backslash-escaped inner quotes, which makes `json.loads`
    return a *string* instead of a dict).
    """
    if raw is None:
        return None
    candidate = raw.strip()
    # Strip QSettings-style outer quotes and unescape inner ones.
    if (
        len(candidate) >= 2
        and candidate.startswith('"')
        and candidate.endswith('"')
    ):
        inner = candidate[1:-1]
        candidate = inner.replace('\\"', '"').replace("\\\\", "\\")
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    # Some legacy writes JSON-encoded the JSON string a second time.
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    return data


def _coerce(raw: str, default_val: Any) -> Any:
    """Coerce a string from the INI file back to the dataclass type."""
    if isinstance(default_val, bool):
        return raw.strip().lower() in ("true", "yes", "1", "on")
    if isinstance(default_val, int) and not isinstance(default_val, bool):
        try:
            return int(raw)
        except ValueError:
            return default_val
    if isinstance(default_val, float):
        try:
            return float(raw)
        except ValueError:
            return default_val
    return raw


class SettingsManager:
    """Direct configparser-backed settings store with QSettings used
    only for binary window geometry. Both share the same INI file.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or _default_config_path()

    def config_path(self) -> str:
        return self._path

    # ----- low-level INI helpers ----------------------------------------
    def _read_parser(self) -> configparser.ConfigParser:
        cp = configparser.ConfigParser()
        # Preserve the case of keys.
        cp.optionxform = lambda s: s
        if os.path.isfile(self._path):
            try:
                cp.read(self._path, encoding="utf-8")
            except configparser.Error:
                # Leave cp empty if the file is malformed; we'll
                # rewrite it on the next save.
                pass
        return cp

    def _write_parser(self, cp: configparser.ConfigParser) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        # Atomic write via a tmp file to avoid partial writes.
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            cp.write(fh)
        os.replace(tmp, self._path)

    # ----- global settings ----------------------------------------------
    def load_global(self) -> GlobalSettings:
        defaults = GlobalSettings()
        cp = self._read_parser()
        kwargs: dict[str, Any] = {}
        for f in defaults.__dataclass_fields__.keys():
            default_val = getattr(defaults, f)
            if cp.has_option("global", f):
                raw = cp.get("global", f)
                kwargs[f] = _coerce(raw, default_val)
            else:
                kwargs[f] = default_val
        return GlobalSettings(**kwargs)

    # Keys that older versions of `GlobalSettings` wrote into the
    # `[global]` section of the INI file but that no longer exist.
    # Pruned on every save to keep the file tidy.
    _OBSOLETE_GLOBAL_KEYS = (
        "ollama_url",
        "ollama_model",
        "default_output_dir",
        "default_output_formats",
        "default_dictionary",
        "enable_llm",
        "llm_only",
    )

    def save_global(self, gs: GlobalSettings) -> None:
        cp = self._read_parser()
        if not cp.has_section("global"):
            cp.add_section("global")
        for f, v in asdict(gs).items():
            cp.set("global", f, "" if v is None else str(v))
        for stale in self._OBSOLETE_GLOBAL_KEYS:
            if cp.has_option("global", stale):
                cp.remove_option("global", stale)
        self._write_parser(cp)
        # Verify by reading back. If the round-trip doesn't match the
        # written values, surface the problem to stderr so we can spot
        # filesystem / permission issues quickly.
        verify = self.load_global()
        for f in gs.__dataclass_fields__.keys():
            if getattr(verify, f) != getattr(gs, f):
                print(
                    f"[mufidiwiwhi] WARNING: settings round-trip "
                    f"mismatch for {f!r}: wrote {getattr(gs, f)!r}, "
                    f"read back {getattr(verify, f)!r}",
                    file=sys.stderr,
                    flush=True,
                )

    # ----- last-session snapshot (project page state) -------------------
    def save_last_session(self, snap: ProjectSnapshot) -> None:
        cp = self._read_parser()
        if not cp.has_section("session"):
            cp.add_section("session")
        cp.set(
            "session",
            "last",
            json.dumps(asdict(snap), ensure_ascii=False),
        )
        self._write_parser(cp)

    def load_last_session(self) -> Optional[ProjectSnapshot]:
        cp = self._read_parser()
        if not cp.has_option("session", "last"):
            return None
        raw = cp.get("session", "last")
        if not raw:
            return None
        data = _parse_session_json(raw)
        if not isinstance(data, dict):
            return None
        return ProjectSnapshot(
            speakers=list(data.get("speakers", [])),
            output_dir=str(data.get("output_dir", "")),
            output_filename=str(data.get("output_filename", "")),
            output_formats=list(data.get("output_formats", ["srt"])),
        )

    def clear_last_session(self) -> None:
        cp = self._read_parser()
        if cp.has_section("session"):
            cp.remove_section("session")
        self._write_parser(cp)

    # ----- window geometry (base64-encoded in the INI) -------------------
    def save_geometry(self, key: str, geom: bytes) -> None:
        cp = self._read_parser()
        if not cp.has_section("geometry"):
            cp.add_section("geometry")
        cp.set("geometry", key, base64.b64encode(bytes(geom)).decode("ascii"))
        self._write_parser(cp)

    def load_geometry(self, key: str) -> Optional[bytes]:
        cp = self._read_parser()
        if not cp.has_option("geometry", key):
            return None
        raw = cp.get("geometry", key)
        if not raw:
            return None
        try:
            return base64.b64decode(raw)
        except (ValueError, base64.binascii.Error):
            return None
