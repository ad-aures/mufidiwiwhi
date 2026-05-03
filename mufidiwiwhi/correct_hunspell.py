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

"""Hunspell second-opinion for the phonetic corrector.

Uses `spylls` (pure-Python Hunspell) so we don't need any native build
step. The corrector calls `_hunspell_says_misspelled(token, lang)`
once per matched window and feeds the verdict into the phonetic
decision matrix:

  Whisper conf >= high   spelled       leave alone
  Whisper conf >= high   misspelled    downgrade to mid-conf rule
  low <= conf < high     spelled       leave alone
  low <= conf < high     misspelled    replace if any phonetic candidate
  conf < low                            replace if any candidate (unchanged)
"""

from __future__ import annotations

import os
from typing import Optional


# --------------------------------------------------------------------------
# Auto-detection of system Hunspell dictionaries.
# --------------------------------------------------------------------------


_HUNSPELL_VARIANTS: dict[str, tuple[str, ...]] = {
    "fr": ("fr_FR", "fr"),
    "en": ("en_US", "en_GB", "en"),
    "de": ("de_DE", "de_AT", "de_CH", "de"),
    "es": ("es_ES", "es"),
    "it": ("it_IT", "it"),
    "pt": ("pt_PT", "pt_BR", "pt"),
    "nl": ("nl_NL", "nl"),
    "ru": ("ru_RU", "ru"),
}


_HUNSPELL_DIRS = (
    "/usr/share/hunspell",
    "/usr/share/myspell",
    "/usr/share/myspell/dicts",
    "/usr/local/share/hunspell",
)


def auto_detect_hunspell(lang_code: str) -> Optional[str]:
    """Return the basename (no extension) of the .aff/.dic pair for
    `lang_code` if it can be found in a standard location. None
    otherwise.
    """
    if not lang_code:
        return None
    code = lang_code.lower().split("_")[0]
    variants = _HUNSPELL_VARIANTS.get(code, (lang_code,))
    for d in _HUNSPELL_DIRS:
        if not os.path.isdir(d):
            continue
        for variant in variants:
            base = os.path.join(d, variant)
            if os.path.isfile(base + ".aff") and os.path.isfile(base + ".dic"):
                return base
    return None


def list_installed_dictionaries() -> list[str]:
    """Return basenames of every Hunspell dictionary discoverable in
    the standard locations. Each entry is a basename suitable for
    `Dictionary.from_files`.
    """
    out: list[str] = []
    seen: set[str] = set()
    for d in _HUNSPELL_DIRS:
        if not os.path.isdir(d):
            continue
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for name in sorted(entries):
            if not name.endswith(".dic"):
                continue
            base = os.path.join(d, name[:-4])
            aff = base + ".aff"
            if not os.path.isfile(aff):
                continue
            if base in seen:
                continue
            seen.add(base)
            out.append(base)
    return out


# --------------------------------------------------------------------------
# Spell-check wrapper. spylls is loaded lazily so the rest of the
# module imports without it installed.
# --------------------------------------------------------------------------


class HunspellChecker:
    """Thin wrapper around `spylls.hunspell.Dictionary` that supports
    a primary + secondary language. A word is "spelled correctly" if
    either dictionary recognises it (matches the multilingual case).

    Failed loads degrade gracefully: a checker built from a missing
    or invalid path acts as a no-op (`is_misspelled` returns False
    for every word, i.e. the corrector behaves as if Hunspell were
    not consulted).
    """

    def __init__(
        self,
        primary: Optional[str],
        secondary: Optional[str] = None,
    ) -> None:
        self._dicts = []
        self._loaded_paths: list[str] = []
        self._failed_paths: list[str] = []
        for path in (primary, secondary):
            if not path:
                continue
            d = self._load(path)
            if d is not None:
                self._dicts.append(d)
                self._loaded_paths.append(path)
            else:
                self._failed_paths.append(path)

    @staticmethod
    def _load(path: str):
        try:
            from spylls.hunspell import Dictionary  # type: ignore
        except ImportError:
            return None
        try:
            return Dictionary.from_files(path)
        except Exception:
            return None

    @property
    def loaded(self) -> bool:
        return bool(self._dicts)

    @property
    def loaded_paths(self) -> list[str]:
        return list(self._loaded_paths)

    @property
    def failed_paths(self) -> list[str]:
        return list(self._failed_paths)

    def is_misspelled(self, token: str) -> bool:
        """True only when no loaded dictionary recognises the token.

        When no dictionary is loaded the answer is always False so
        the rest of the pipeline behaves as if Hunspell were not
        configured.
        """
        if not token or not self._dicts:
            return False
        # spylls' lookup is case-sensitive in some forms; check the
        # token verbatim plus a lowercase fallback so common-noun
        # capitalisation at sentence start doesn't trip the check.
        candidates = {token, token.lower()}
        for d in self._dicts:
            for cand in candidates:
                try:
                    if d.lookup(cand):
                        return False
                except Exception:
                    # Defensive: a malformed token shouldn't break the
                    # whole pipeline. Treat any lookup failure as
                    # "not in this dictionary" and try the next.
                    continue
        return True

    def suggest(self, token: str, limit: int = 1) -> list[str]:
        """Return up to `limit` Hunspell spelling suggestions for
        `token`. Empty list when no dictionary is loaded, the token
        is empty, or the dictionary returns nothing.

        We pull suggestions from the first dictionary that yields any.
        Lazy-evaluated so we don't generate the full alternative
        space when we only want the top suggestion.
        """
        if not token or not self._dicts:
            return []
        for d in self._dicts:
            try:
                gen = d.suggest(token)
            except Exception:
                continue
            out: list[str] = []
            try:
                for s in gen:
                    if not s:
                        continue
                    out.append(str(s))
                    if len(out) >= limit:
                        break
            except Exception:
                pass
            if out:
                return out
        return []
