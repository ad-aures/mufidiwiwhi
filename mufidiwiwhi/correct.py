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

"""Post-correction module: phonetic + word-confidence dictionary correction.

Independent of Whisper. Takes a list of segment dicts (Whisper-shape with
optional `words` lists carrying `probability`) and returns a new list of
segments with `text` and `words[i].word` mutated. Timestamps are not touched.
"""

from __future__ import annotations

import os
import re
import threading
import time as _time
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Sequence

from .core import CorrectionConfig, LogCb, vstderr


# ---------------------------------------------------------------------------
# Phonetic backends. We try the best-known function names for each library
# and degrade gracefully if a library is missing.
# ---------------------------------------------------------------------------


def _phonetic_fr(token: str) -> str:
    try:
        import phonetic_fr  # type: ignore

        for attr in ("main_phonetic", "phonetic"):
            fn = getattr(phonetic_fr, attr, None)
            if callable(fn):
                return str(fn(token))
        # older API
        from phonetic_fr.phonetic import Phonetic  # type: ignore

        return str(Phonetic.main_phonetic(token))
    except Exception:
        return _phonetic_soundex(token)


def _phonetic_en(token: str) -> str:
    try:
        from metaphone import doublemetaphone  # type: ignore

        primary, _secondary = doublemetaphone(token)
        return str(primary or _phonetic_soundex(token))
    except Exception:
        return _phonetic_soundex(token)


def _phonetic_soundex(token: str) -> str:
    try:
        import jellyfish  # type: ignore

        return str(jellyfish.soundex(token))
    except Exception:
        # last-resort: a degenerate fingerprint that still groups identical
        # tokens together
        return token.upper()


def _phonetic_token(token: str, lang: str) -> str:
    if not token:
        return ""
    if lang == "fr":
        return _phonetic_fr(token)
    if lang == "en":
        return _phonetic_en(token)
    return _phonetic_soundex(token)


def _levenshtein(a: str, b: str) -> int:
    try:
        import jellyfish  # type: ignore

        return int(jellyfish.levenshtein_distance(a, b))
    except Exception:
        # tiny pure-Python fallback
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[-1]


# ---------------------------------------------------------------------------
# Tokenization. Both dictionary entries and Whisper word-windows go through
# the same path so the phonetic codes are comparable.
# ---------------------------------------------------------------------------


_FR_CLITICS = ("qu", "l", "d", "j", "n", "m", "t", "s", "c")
_CLITIC_RE = re.compile(
    r"^(?:" + "|".join(_FR_CLITICS) + r")['’]",
    flags=re.IGNORECASE,
)


def _strip_outer_punct(text: str) -> str:
    text = re.sub(r"^[^\wÀ-ſ]+", "", text)
    text = re.sub(r"[^\wÀ-ſ]+$", "", text)
    return text


def _normalise_text_to_tokens(text: str) -> tuple[list[str], str]:
    """Return (tokens, leading_clitic).

    `tokens` is a list of normalised sub-tokens (lowercased, hyphens split).
    `leading_clitic` is the original-cased clitic prefix if any (so it can be
    re-prepended on replacement to preserve "l'OpenRAG"); empty otherwise.
    """
    if not text:
        return [], ""
    raw = text.strip()
    raw_norm = unicodedata.normalize("NFKC", raw)
    leading_clitic = ""
    m = _CLITIC_RE.match(raw_norm)
    if m:
        leading_clitic = m.group(0)
        raw_norm = raw_norm[m.end():]
    cleaned = _strip_outer_punct(raw_norm).lower()
    if not cleaned:
        return [], leading_clitic
    cleaned = cleaned.replace("’", "'")
    cleaned = cleaned.replace("'", " ")
    cleaned = cleaned.replace("-", " ")
    tokens: list[str] = []
    for chunk in cleaned.split():
        chunk = _strip_outer_punct(chunk)
        if chunk and not re.fullmatch(r"\d+", chunk):
            tokens.append(chunk)
    return tokens, leading_clitic


def _tokens_for_entry(canonical: str) -> list[str]:
    tokens, _ = _normalise_text_to_tokens(canonical)
    return tokens


def _phonetic_code(tokens: Sequence[str], lang: str) -> str:
    if not tokens:
        return ""
    return "|".join(_phonetic_token(t, lang) for t in tokens)


def _resolve_lang(cfg: CorrectionConfig, segment_language: Optional[str]) -> str:
    if cfg.phonetic_lang and cfg.phonetic_lang != "auto":
        return cfg.phonetic_lang
    if segment_language in ("fr", "en"):
        return segment_language
    return "en"


# ---------------------------------------------------------------------------
# Dictionary loading and indexing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DictionaryEntry:
    canonical: str
    tokens: tuple[str, ...]
    n: int
    code_primary: str
    code_secondary: Optional[str]


@dataclass
class PhoneticIndex:
    by_code: dict[str, list[DictionaryEntry]]
    max_n: int
    primary_lang: str
    secondary_lang: Optional[str]
    # Flat, deduplicated list of every entry. Used by the fuzzy
    # surface-level fallback so we don't iterate `by_code.values()`
    # (which contains duplicates) on every lookup.
    all_entries: tuple[DictionaryEntry, ...] = ()


@dataclass
class CorrectionState:
    """Bundles everything needed for per-segment correction so the
    pipeline can build it once and reuse it as Whisper streams
    segments. `corrections` accumulates a record per replacement so
    we can print a recap at the end of the run.
    """

    cfg: CorrectionConfig
    index: Optional[PhoneticIndex]
    hunspell: Optional[object]  # HunspellChecker
    corrections: list[dict] = None  # type: ignore[assignment]
    # Normalised surfaces of every user-dict entry, built once so
    # the per-segment Hunspell pass can skip recognised words in
    # O(1) instead of re-iterating the index.
    user_dict_surfaces: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.corrections is None:
            self.corrections = []
        if self.user_dict_surfaces is None:
            surfaces: set[str] = set()
            if self.index is not None:
                for entry in self.index.all_entries:
                    surfaces.add(_surface_normalize("".join(entry.tokens)))
                    surfaces.add(_surface_normalize(entry.canonical))
                surfaces.discard("")
            self.user_dict_surfaces = surfaces


def _entry_codes(
    tokens: Sequence[str], lang: str, clitic: str = ""
) -> set[str]:
    """Return the set of phonetic codes that should index a dict entry.

    We index each entry under three variants so it matches whether the
    speech recogniser produced the same word boundaries as the
    dictionary or not:

      * per-token joined: code_a|code_b|code_c   (matches when the
        Whisper window has the same number of words as the entry)
      * collapsed single-token: code(concat(a,b,c))   (matches when
        Whisper merged the words, e.g. dict has "OpenRAG" and the
        transcript has "openrag" as a single token, OR dict has
        "open rag" written as 2 tokens but Whisper produced 1)
      * clitic-merged: for entries with an apostrophe-clitic prefix
        (l'ANTS, d'Aures, D'Échirolles, ...) we also register the
        phonetic of the un-elided surface, i.e. the clitic letter
        glued to the body ("lants", "dechirolles"). Whisper often
        writes the un-contracted preposition as a separate word
        ("la NTS" -> l'ANTS, "Des Chirolle" -> D'Échirolles), and
        this code variant catches that.

    All forms are useful because Whisper word boundaries are not
    stable for proper nouns and made-up words.
    """
    out: set[str] = set()
    joined = _phonetic_code(tokens, lang)
    if joined:
        out.add(joined)
    if len(tokens) > 1:
        collapsed = "".join(tokens)
        if collapsed:
            collapsed_code = _phonetic_token(collapsed, lang)
            if collapsed_code:
                out.add(collapsed_code)
    if clitic:
        clitic_letter = clitic[0] if clitic[0].isalpha() else ""
        if clitic_letter:
            merged = clitic_letter + "".join(tokens)
            merged_code = _phonetic_token(merged, lang)
            if merged_code:
                out.add(merged_code)
    return out


def _sort_dictionary_file(path: str) -> None:
    """Re-sort the dictionary file alphabetically (case-insensitive)
    leaving comments and blank lines in place. Only writes back when
    the entry order would actually change so timestamps don't churn."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return
    lines = raw.splitlines()
    entries: list[str] = []
    keep_other: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            keep_other.append((i, line))
        else:
            entries.append(line)
    sorted_entries = sorted(
        entries, key=lambda s: (s.strip().casefold(), s.strip())
    )
    if entries == sorted_entries:
        return  # already sorted; don't rewrite
    # Reassemble: comments / blanks keep their slot; sorted entries
    # fill the remaining slots in order.
    out = list(lines)
    entry_iter = iter(sorted_entries)
    for i in range(len(out)):
        s = out[i].strip()
        if s and not s.startswith("#"):
            out[i] = next(entry_iter)
    new_text = "\n".join(out)
    if raw.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def load_dictionary(path: str, cfg: CorrectionConfig) -> PhoneticIndex:
    """Read a plain-text dictionary file and build a phonetic index.

    Lines starting with `#` are comments. Empty lines are ignored.
    Multi-word entries are allowed. The file is also re-sorted on
    disk (alphabetical, case-insensitive) so manually-edited
    dictionaries stay tidy without the user having to open the
    editor.
    """
    _sort_dictionary_file(path)
    primary_lang = cfg.phonetic_lang if cfg.phonetic_lang != "auto" else "en"
    secondary_lang = cfg.phonetic_lang_secondary

    by_code: dict[str, list[DictionaryEntry]] = {}
    max_n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            canonical = line.strip()
            tokens, clitic = _normalise_text_to_tokens(canonical)
            if not tokens:
                continue
            n = len(tokens)
            if n > 4:
                # entries longer than 4 tokens are dropped to keep the
                # sliding window small
                continue
            code_primary = _phonetic_code(tokens, primary_lang)
            code_secondary = (
                _phonetic_code(tokens, secondary_lang) if secondary_lang else None
            )
            entry = DictionaryEntry(
                canonical=canonical,
                tokens=tuple(tokens),
                n=n,
                code_primary=code_primary,
                code_secondary=code_secondary,
            )
            codes = _entry_codes(tokens, primary_lang, clitic=clitic)
            if secondary_lang:
                codes |= _entry_codes(tokens, secondary_lang, clitic=clitic)
            for code in codes:
                by_code.setdefault(code, []).append(entry)
            if n > max_n:
                max_n = n
    # The matcher will also try windows of `max_n + 1` Whisper words
    # against the collapsed code, in case one dictionary token comes
    # out as two Whisper words. Bump max_n once for that headroom.
    if max_n > 0:
        max_n += 1
    # Build the deduplicated flat list once.
    seen: set[str] = set()
    flat: list[DictionaryEntry] = []
    for entries in by_code.values():
        for e in entries:
            if e.canonical in seen:
                continue
            seen.add(e.canonical)
            flat.append(e)
    return PhoneticIndex(
        by_code=by_code,
        max_n=max_n,
        primary_lang=primary_lang,
        secondary_lang=secondary_lang,
        all_entries=tuple(flat),
    )


# ---------------------------------------------------------------------------
# Quality guards: filter out replacements where the surface is too
# dissimilar from the input, or where the input is purely common
# function words. The phonetic codes lose a lot of information for
# very short / common words and they tend to collide with random
# short dictionary entries (acronyms in particular), so an extra
# surface-level sanity check is necessary on top of the band logic.
# ---------------------------------------------------------------------------


# Generic short-word guard: regardless of language, a window of
# 1-3-character lowercase tokens is too short to phoneticise
# reliably and tends to collide with random short dict entries (the
# main offender is acronyms like "ANTS" matching "on", "en", "un",
# "une", "ya un", ...). When such a window has no token of length
# >= 4, refuse to replace it.
def _is_low_information_window(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    return all(len(t) <= 3 for t in tokens)


# Per-language function-word lists. ONLY used when the user has set
# `cfg.phonetic_lang` to that language. For "auto" / unsupported
# languages this is a no-op and the length-based fallback below
# still applies. Adding more languages is a matter of dropping
# another set in here.
_FUNCTION_WORDS_BY_LANG: dict[str, frozenset[str]] = {
    "fr": frozenset({
        # 1-letter elided forms ("l'arbre" -> token "l")
        "c", "d", "j", "l", "m", "n", "qu", "s", "t",
        # 2-3 letter common articles, prepositions, conjunctions
        "à", "ai", "au", "aux", "as", "ça", "ce", "ces", "cet", "cette",
        "de", "des", "du", "elle", "elles", "en", "es", "est", "et",
        "été", "été", "il", "ils", "je", "la", "le", "les", "leur",
        "leurs", "lui", "ma", "mais", "mes", "moi", "mon", "ne", "ni",
        "nos", "notre", "nous", "on", "ont", "ou", "où", "par", "pas",
        "plus", "pour", "qu", "que", "qui", "quoi", "sa", "sans",
        "se", "ses", "si", "son", "sur", "ta", "te", "tes", "toi",
        "ton", "tout", "tu", "très", "un", "une", "vos", "votre",
        "vous", "y",
    }),
    "en": frozenset({
        # English equivalents
        "a", "am", "an", "and", "are", "as", "at", "be", "but", "by",
        "for", "from", "he", "her", "his", "i", "if", "in", "is", "it",
        "its", "me", "my", "no", "nor", "not", "of", "on", "or", "our",
        "she", "so", "than", "that", "the", "their", "them", "these",
        "they", "this", "those", "to", "us", "was", "we", "were",
        "when", "where", "which", "with", "you", "your",
    }),
}


def _is_function_word(token: str, lang: Optional[str]) -> bool:
    """True if `token` is a known per-language function word. The
    check is keyed by the resolved phonetic language (FR / EN
    today). Words in unsupported languages are never flagged here.
    """
    if not token or not lang:
        return False
    table = _FUNCTION_WORDS_BY_LANG.get(lang)
    if not table:
        return False
    return token.lower() in table


# French determiners that contract before a vowel-initial word:
# "le ANCT" -> "l'ANCT", "la NCT" -> "l'ANCT", "de Aures" -> "d'Aures",
# "ce ANCT" -> "c'ANCT". Each maps to its contracted form including
# the apostrophe. Only French; English determiners don't elide this
# way. Kept tight on purpose — broader lists (que, ne, me, te, se,
# je) precede verbs in practice and would over-fire on proper-noun
# replacement.
_FR_CONTRACTABLE_DETERMINERS: dict[str, str] = {
    "le": "l'",
    "la": "l'",
    "de": "d'",
    "ce": "c'",
}

_VOWELS = frozenset("aeiouyàâéèêëïîôûùüœæAEIOUYÀÂÉÈÊËÏÎÔÛÙÜŒÆ")


def _starts_with_vowel(text: str) -> bool:
    """True when `text` begins with a French/English vowel after
    stripping any leading apostrophe-clitic prefix.
    """
    s = text.strip()
    if len(s) >= 2 and s[1] in ("'", "’"):
        s = s[2:].lstrip()
    return bool(s) and s[0] in _VOWELS


def _sub_window_already_matches(
    entry: "DictionaryEntry", input_tokens: Sequence[str]
) -> bool:
    """True when a smaller sub-window of `input_tokens` (size
    `entry.n`) already matches the entry's surface essentially
    exactly (edit distance ≤ 1, case/diacritic-insensitive). The
    bigger match would only be eating extra real words.

    Catches `part framasoft -> Framasoft` (the smaller window
    `framasoft` matches exactly, so the bigger window would just
    consume `part`) without breaking legitimate compounds like
    `open rag -> OpenRAG` (no sub-window matches `openrag`).
    """
    if entry.n >= len(input_tokens):
        return False
    entry_surface = _surface_normalize("".join(entry.tokens))
    if not entry_surface:
        return False
    for start in range(len(input_tokens) - entry.n + 1):
        sub = input_tokens[start : start + entry.n]
        sub_surface = _surface_normalize("".join(sub))
        if not sub_surface:
            continue
        if _levenshtein(sub_surface, entry_surface) <= 1:
            return True
    return False


def _is_acronym_canonical(entry: "DictionaryEntry") -> bool:
    """True when the entry's canonical is an all-uppercase
    acronym: every alphabetic character is uppercase. The
    apostrophe-clitic prefix (l'ANTS, d'Aures) is allowed and
    so are digits / punctuation. Used as the signal that the
    user explicitly wants this replacement to override Hunspell
    for single-token Hunspell-spelled inputs (Nerd -> NIRD).
    """
    s = (entry.canonical or "").strip()
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def _entry_has_clitic_prefix(entry: "DictionaryEntry") -> bool:
    """True when the entry's canonical form starts with a French
    apostrophe-clitic (`l'ANTS`, `d'Aures`, `n'Garra`, ...). For
    such entries, an input that contains the corresponding
    function word ("la", "des", "ne") in front of the matching
    body is EXPECTED to be consumed - the article and the clitic
    are the same thing in different grammatical positions.
    """
    s = (entry.canonical or "").strip()
    if len(s) < 2:
        return False
    if s[0].lower() not in {"l", "d", "j", "n", "m", "t", "s", "c"}:
        return False
    return s[1] in ("'", "’")


def _has_short_or_function_word(
    tokens: Sequence[str], lang: Optional[str]
) -> bool:
    """True when a multi-token window contains at least one token
    that's either ≤ 2 characters or a known function word for the
    active language. Such tokens (et / à / le / des / un / and / the /
    of …) are usually sentence connectives and should not be merged
    into a smaller dict entry; e.g. 'des GAFAM' must NOT become
    'GAFAM' by eating the 'des'.
    """
    if len(tokens) <= 1:
        return False
    for t in tokens:
        if len(t) <= 2 or _is_function_word(t, lang):
            return True
    return False


def _surface_normalize(s: str) -> str:
    """Lowercase + strip combining marks (accents). Whisper often
    swallows or mangles diacritics, so a fair surface comparison
    has to ignore them too: 'etielgonu' should be treated as 4
    edits from 'etiennegonnu', not 5 (which would happen if 'é'
    counted as a substitution against 'e')."""
    nfkd = unicodedata.normalize("NFKD", s)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_marks.lower()


def _is_best_surface_match(
    chosen: "DictionaryEntry",
    input_tokens: Sequence[str],
    index: "PhoneticIndex",
    log: Optional["LogCb"] = None,
) -> bool:
    """Return True only when no other dict entry has a smaller-or-
    equal surface edit distance to the input than `chosen`. Acts
    as a final sanity net after `_decide` so a candidate that
    survived the picker can still be vetoed when the full
    dictionary contains an equally good or better match.

    Tie-breaking is critical: if `chosen` is at edit distance 3
    from the input but another entry is also at edit distance 3,
    the picker has no way to choose between them - reject the
    replacement so we don't randomly pick the wrong sibling.
    Strictly less wouldn't be enough: the bug case
    "Cryptoast" vs ["Cryptoast", "Cryptpad"] hits when both are
    edit 0 / both are edit 3 / one wins by tie order.
    """
    input_surface = _surface_normalize("".join(input_tokens))
    if not input_surface:
        return True
    chosen_surface = _surface_normalize("".join(chosen.tokens))
    chosen_edit = _levenshtein(input_surface, chosen_surface)
    for idx, other in enumerate(index.all_entries):
        if idx and idx % 128 == 0:
            _time.sleep(0)
        if other.canonical == chosen.canonical and other.tokens == chosen.tokens:
            continue
        other_surface = _surface_normalize("".join(other.tokens))
        if not other_surface:
            continue
        if abs(len(other_surface) - len(input_surface)) > chosen_edit:
            continue
        other_edit = _levenshtein(input_surface, other_surface)
        # Reject on STRICT less OR equal-with-different-tokens.
        # The equality case catches "Cryptoast" picking
        # "Cryptpad" when both are edit 0 against input
        # "cryptoast" via lookup quirks.
        if other_edit < chosen_edit or (
            other_edit == chosen_edit
            and other.tokens != chosen.tokens
        ):
            if log is not None:
                log(
                    f"  rejecting replacement {input_tokens!r} -> "
                    f"{chosen.canonical!r}: dict entry "
                    f"{other.canonical!r} is an equally-or-closer "
                    f"surface match (edit {other_edit} <= {chosen_edit})"
                )
            vstderr(
                f"  safety-net veto: {input_tokens!r} -> "
                f"{chosen.canonical!r} blocked by {other.canonical!r} "
                f"(other_edit={other_edit}, chosen_edit={chosen_edit})"
            )
            return False
    return True


def _passes_surface_check(
    entry: "DictionaryEntry", input_tokens: Sequence[str]
) -> bool:
    """Reject replacements where the rebuilt surface differs by more
    than 50 % of its length from the input (short words: edit
    distance must be <= 2). Comparison ignores whitespace and
    accents because Whisper's word boundaries and diacritics are
    both unreliable.

    Also rejects pure anagrams of short surfaces (RGPD vs RdGP):
    same multiset of characters in a different order is almost
    always a different acronym, not a typo of the same one.
    """
    a = _surface_normalize("".join(input_tokens))
    b = _surface_normalize("".join(entry.tokens))
    if not a or not b:
        return True
    longer = max(len(a), len(b))
    # Short single-token mismatch with a different first character:
    # protects e.g. "Linux" (5) vs "DINUM" (5) which sit at edit
    # distance 2 but are obviously different words. Same first
    # letter is a much stronger signal than edit distance for
    # 4-6 letter words. EXCEPTION: when the shorter side is the
    # longer side stripped of one leading character (or vice
    # versa), this is just an alignment slip, not a different
    # word - e.g. "nts" vs "ants" or "lants" vs "ants". Allow it.
    if (
        len(input_tokens) == 1
        and entry.n == 1
        and longer <= 6
        and a[0] != b[0]
        and (len(a) <= len(b) or a[1:] != b)
        and (len(b) <= len(a) or b[1:] != a)
    ):
        return False
    # Anagram guard for short surfaces: protects e.g. RGPD <-> RdGP
    # which are two distinct French acronyms with the same letters.
    if (
        longer <= 6
        and len(a) == len(b)
        and a != b
        and sorted(a) == sorted(b)
    ):
        return False
    edit = _levenshtein(a, b)
    if longer < 6:
        return edit <= 2
    # Strict `<`: an edit distance equal to half the longer length is
    # already too suspicious (e.g. "dynumeet" vs "dinum" has edit
    # distance 4 against length 8 — exactly at the cutoff but in
    # practice always wrong).
    return edit < longer * 0.5


# ---------------------------------------------------------------------------
# Replacement decision.
# ---------------------------------------------------------------------------


def _decide(
    candidates: list[DictionaryEntry],
    input_tokens: Sequence[str],
    window_prob: float,
    cfg: CorrectionConfig,
    hunspell_misspelled: Optional[bool] = None,
    lang: Optional[str] = None,
) -> Optional[DictionaryEntry]:
    """Pick a replacement candidate (or None).

    The Hunspell verdict, when available, modulates the bands:
      * `hunspell_misspelled is True`  => one band lower (more permissive).
      * `hunspell_misspelled is False` => one band higher (more conservative).
      * `hunspell_misspelled is None`  => no modulation (Hunspell not consulted).
    """
    if not candidates:
        return None

    # Compare against `c.canonical` (not `"".join(c.tokens)`) so the
    # user's intentional punctuation counts toward the distance:
    # "chat" vs "chut !" is 3 edits (protection holds), while "nerd"
    # vs "NIRD" stays 1 (legitimate override fires).
    _input_surface = _surface_normalize("".join(input_tokens))
    _has_near_match = any(
        _levenshtein(_input_surface, _surface_normalize(c.canonical))
        <= 1
        for c in candidates
    )

    # A window of only very short tokens is normally too low-
    # information to phoneticise reliably (the codes collide with
    # random short dict entries). But at LOW confidence, that's
    # exactly the case where the user might be trying to fix a
    # genuine misrecognition like "la NTS" -> "l'ANTS", so skip
    # this guard when the window is below `low_conf` OR when one
    # of the candidates is a near-match (explicit user intent).
    if (
        _is_low_information_window(input_tokens)
        and window_prob >= cfg.low_conf
        and not _has_near_match
    ):
        return None

    input_joined = " ".join(input_tokens)

    def edit_distance(entry: DictionaryEntry) -> int:
        return _levenshtein(input_joined, " ".join(entry.tokens))

    # Tier the word into low / mid / high based on window confidence.
    if window_prob >= cfg.high_conf:
        tier = "high"
    elif window_prob >= cfg.low_conf:
        tier = "mid"
    else:
        tier = "low"

    # Hunspell modulation:
    #   misspelled => bump tier DOWN one band (more permissive)
    #   spelled    => protect the word: a single-token spelled word
    #                 is never phonetically replaced; multi-token
    #                 windows just get bumped UP one band.
    # When Hunspell wasn't consulted (no dict loaded), don't modulate.
    if hunspell_misspelled is True:
        if tier == "high":
            tier = "mid"
        elif tier == "mid":
            tier = "low"
    elif hunspell_misspelled is False:
        # Hunspell says the input is a real word. Single-token
        # spelled words are NEVER replaced (so "avril" stays
        # "avril" even when "April" is in the dict, and "Tchat"
        # stays "Tchat" even when "Chut !" is near). EXCEPTION:
        # all-uppercase dict canonicals (acronyms like NIRD,
        # GAFAM, ANTS) signal the user explicitly wants the
        # replacement regardless of Hunspell - the spelling
        # difference between Hunspell-recognised input ("Nerd")
        # and acronym output ("NIRD") is intentional.
        if len(input_tokens) <= 1:
            if not _has_near_match:
                return None
            if not any(_is_acronym_canonical(c) for c in candidates):
                return None
            # Fall through with normal tier handling.
        elif not _has_near_match:
            if tier == "low":
                tier = "mid"
            elif tier == "mid":
                tier = "high"

    if tier == "high":
        return None

    def _word_eaten(entry: DictionaryEntry) -> bool:
        # Reject collapsed-code matches that would silently consume
        # extra tokens. Two complementary checks:
        #   (a) the input contains a function word ("et", "des",
        #       "pour", ...) and the dict entry is shorter — it's
        #       almost certainly a sentence connective being absorbed.
        #   (b) a smaller sub-window of the input ALREADY matches
        #       the entry essentially exactly — the bigger match is
        #       just eating unrelated content words ("part" in
        #       "part framasoft").
        # EXCEPTION 1: when the dict entry's canonical starts with an
        # apostrophe-clitic ("l'ANTS", "d'Aures", ...), the leading
        # function word IS expected to be consumed, because the
        # clitic IS the contracted form of the article. Allow it.
        # EXCEPTION 2: when the entry's surface is the TAIL of the
        # joined input surface (so the leading tokens just prepend
        # extra letters in front of the entry, not noise mixed
        # throughout), the joined-surface edit distance is at most
        # 1, and joining doesn't make the match worse than any
        # sub-window alone, the leading word is part of the proper
        # noun's pronunciation (e.g. "à Eris" -> "Aeris", or
        # "la NCT" with "ANCT" in the dictionary). Without this
        # bypass the matcher leaves the stray leading word in front
        # of the replaced tail.
        # In every other case the smaller window gets a chance on
        # the next iteration of the outer matcher loop.
        if entry.n >= len(input_tokens):
            return False
        if _entry_has_clitic_prefix(entry):
            return False
        entry_surface = _surface_normalize("".join(entry.tokens))
        joined_surface = _surface_normalize("".join(input_tokens))
        if (
            entry_surface
            and joined_surface
            and len(joined_surface) >= len(entry_surface)
            and joined_surface.endswith(entry_surface)
        ):
            joined_edit = _levenshtein(joined_surface, entry_surface)
            if joined_edit <= 1:
                best_sub_edit: Optional[int] = None
                for start in range(len(input_tokens) - entry.n + 1):
                    sub = input_tokens[start : start + entry.n]
                    sub_surface = _surface_normalize("".join(sub))
                    if not sub_surface:
                        continue
                    d = _levenshtein(sub_surface, entry_surface)
                    if best_sub_edit is None or d < best_sub_edit:
                        best_sub_edit = d
                if best_sub_edit is None or joined_edit <= best_sub_edit:
                    return False
        if _has_short_or_function_word(input_tokens, lang):
            return True
        if _sub_window_already_matches(entry, input_tokens):
            return True
        return False

    if tier == "mid":
        # Require exactly one candidate within the edit-distance
        # threshold. Multi-token windows get one extra edit of
        # leniency (longer phrases pick up more transcription noise).
        if len(candidates) != 1:
            return None
        only = candidates[0]
        if _word_eaten(only):
            return None
        is_multiword = len(input_tokens) > 1 or only.n > 1
        threshold = cfg.edit_distance_threshold + (1 if is_multiword else 0)
        if edit_distance(only) <= threshold and _passes_surface_check(
            only, input_tokens
        ):
            return only
        return None

    # tier == "low": filter out candidates that would be word-eaten
    # or fail the surface check FIRST, then pick the closest by edit
    # distance among the survivors. Picking-then-rejecting loses
    # viable alternatives, e.g. "la NTS" with both "ANTS" and
    # "l'ANTS" in the user dict: both are edit-1 fuzzy matches, but
    # "ANTS" gets picked first (tie) and rejected by the function-
    # word eat-guard, while "l'ANTS" (clitic-prefix exempt) would
    # have been the right answer.
    viable = [
        c
        for c in candidates
        if not _word_eaten(c) and _passes_surface_check(c, input_tokens)
    ]
    if not viable:
        return None
    return min(viable, key=lambda e: (edit_distance(e), len(e.tokens)))


# ---------------------------------------------------------------------------
# Window replacement and segment rebuild.
# ---------------------------------------------------------------------------


def _merge_window_into_word(
    window: list[dict],
    canonical: str,
    leading_clitic: str,
    correction_kind: str = "phonetic",
) -> dict:
    """Collapse a multi-word window into a single Whisper-shape word dict
    carrying the canonical replacement, with leading whitespace, leading
    clitic, and trailing punctuation preserved.

    The original surface (joined from the input window) is stored on
    the result under `correction = {"original": ..., "type": ...}`
    so the live log can render the diff inline.
    """
    first = window[0]
    last = window[-1]

    raw_first = first["word"]
    leading_ws = ""
    i = 0
    while i < len(raw_first) and raw_first[i].isspace():
        leading_ws += raw_first[i]
        i += 1
    # Preserve any leading punctuation that prefixed the original
    # surface (a stray apostrophe, opening quote, etc.) so we don't
    # silently strip it. We don't capture leading punct when there's
    # already a clitic; the clitic itself is the leading punct in
    # that case.
    leading_punct = ""
    if not leading_clitic:
        while (
            i < len(raw_first)
            and not raw_first[i].isspace()
            and not raw_first[i].isalnum()
        ):
            leading_punct += raw_first[i]
            i += 1

    raw_last = last["word"]
    trailing_punct = ""
    j = len(raw_last)
    while j > 0 and re.match(
        r"[^\wÀ-ſ]", raw_last[j - 1]
    ):
        trailing_punct = raw_last[j - 1] + trailing_punct
        j -= 1

    new_text = (
        leading_ws + leading_punct + leading_clitic + canonical + trailing_punct
    )
    original_inner = "".join(w["word"] for w in window)
    # Strip the same outer whitespace / punctuation we just preserved
    # so the recorded original is just the substantive surface.
    original_inner_stripped = original_inner.strip()
    if trailing_punct and original_inner_stripped.endswith(trailing_punct):
        original_inner_stripped = original_inner_stripped[
            : -len(trailing_punct)
        ]

    probs = [w.get("probability", 0.0) for w in window if "probability" in w]
    # Keep the lowest pre-correction probability around so the live
    # log can still tint the struck-through original with the same
    # confidence-band background it would have had unchanged.
    original_confidence = min(probs) if probs else 1.0
    return {
        "word": new_text,
        "start": first["start"],
        "end": last["end"],
        "probability": max(probs) if probs else 1.0,
        "correction": {
            "original": original_inner_stripped,
            "replacement": leading_clitic + canonical,
            "type": correction_kind,
            "original_confidence": original_confidence,
        },
    }


def _correct_segment(
    segment: dict,
    index: Optional[PhoneticIndex],
    cfg: CorrectionConfig,
    log: Optional[LogCb] = None,
    hunspell=None,
    state: Optional[CorrectionState] = None,
) -> int:
    """Apply phonetic corrections to a single segment.

    Returns the number of replacements performed. Each replacement is
    logged (when `log` is given) with timestamp, before/after surface
    forms, and the minimum Whisper confidence across the replaced span.
    """
    words = segment.get("words")
    if not words:
        return 0
    if index is None:
        return 0

    seg_start = float(segment.get("start", 0.0))
    out: list[dict] = []
    i = 0
    fixes = 0
    while i < len(words):
        # Drop the GIL every word so the GUI's event loop on the
        # main thread keeps pumping clicks / scroll while the
        # correction pass runs (it's pure Python: Hunspell,
        # Levenshtein, dict scans).
        if i and i % 16 == 0:
            _time.sleep(0)
        replaced = False
        max_window = min(index.max_n, len(words) - i)
        for n in range(max_window, 0, -1):
            window = words[i : i + n]
            tokens_collected: list[str] = []
            leading_clitic = ""
            any_empty_raw = False
            for k, w in enumerate(window):
                tks, clitic = _normalise_text_to_tokens(w["word"])
                if k == 0:
                    leading_clitic = clitic
                if not tks:
                    any_empty_raw = True
                tokens_collected.extend(tks)
            if not tokens_collected:
                continue
            # Skip multi-raw-word windows where at least one raw word
            # contributed zero tokens (digits like "24", pure punct,
            # ...). Letting them through means the merge silently eats
            # those raw words, e.g. "24 avril" -> "April" swallowing
            # the "24". Smaller windows pick up the substantive parts
            # on the next iteration.
            if n > 1 and any_empty_raw:
                continue
            # Use MIN (not mean) across the window: a single
            # uncertain word in a multi-word phrase pulls the whole
            # window into the lower confidence band, so phrases like
            # "Fritjof Michelsen" still get replaced even if Whisper
            # is confidently wrong on the second word.
            window_prob = min(
                (w.get("probability", 1.0) for w in window),
                default=1.0,
            )
            misspelled: Optional[bool] = None
            if hunspell is not None and getattr(hunspell, "loaded", False):
                # Treat the window as misspelled when ANY of its
                # constituent tokens is unknown to Hunspell. The
                # window is what Whisper produced; we're judging
                # whether it looks like real natural-language text.
                misspelled = False
                for tok in tokens_collected:
                    if hunspell.is_misspelled(tok):
                        misspelled = True
                        break
            cands = _lookup_window(tokens_collected, index)
            if not cands:
                # Fuzzy surface-level fallback: catches near-misses
                # like "Julie Cahillard" -> "Julie Caillard" where
                # the phonetic algorithm produces close-but-not-equal
                # codes. The distance budget scales with confidence
                # and Hunspell verdict so misspelled low-confidence
                # words tolerate larger edits.
                cands = _fuzzy_lookup(
                    tokens_collected, index, window_prob, cfg, misspelled
                )
            if not cands:
                continue
            # Resolve the active language for function-word lookup:
            # the explicit cfg.phonetic_lang wins; otherwise fall
            # back to whatever Whisper detected on the segment.
            active_lang = cfg.phonetic_lang
            if not active_lang or active_lang == "auto":
                active_lang = segment.get("language") or ""
            chosen = _decide(
                cands,
                tokens_collected,
                window_prob,
                cfg,
                misspelled,
                lang=active_lang or None,
            )
            if chosen is None:
                continue
            # Safety net: before applying the replacement, verify
            # that no OTHER dict entry is a strictly better surface
            # match for the input. The phonetic+fuzzy lookup can
            # admit candidates whose surface is far from the input
            # (e.g. "La CNIL" entering the candidate list for input
            # "april" via fuzzy with a wide budget), and even though
            # the picker prefers the closer one, a misordered tie or
            # a bug elsewhere can let the wrong one through.
            # Scanning `index.all_entries` here is O(N) per window
            # but N is small (typical user dicts have <500 entries)
            # and this only fires on the rare cases that get past
            # _decide.
            if not _is_best_surface_match(
                chosen, tokens_collected, index, log
            ):
                continue
            original_surface = "".join(w["word"] for w in window).strip()
            if chosen.tokens == tuple(tokens_collected):
                if chosen.canonical == original_surface:
                    continue
            # French clitic synthesis: when the matcher absorbed a
            # leading contractable determiner ("le", "la", "de",
            # "ce") and the canonical begins with a vowel, emit the
            # contracted clitic. "la NCT" with "ANCT" in the dict
            # becomes "l'ANCT", not "ANCT" with the "la" silently
            # eaten.
            if (
                active_lang == "fr"
                and not leading_clitic
                and chosen.n < len(tokens_collected)
                and len(tokens_collected) - chosen.n == 1
                and not _entry_has_clitic_prefix(chosen)
            ):
                contraction = _FR_CONTRACTABLE_DETERMINERS.get(
                    tokens_collected[0]
                )
                if contraction and _starts_with_vowel(chosen.canonical):
                    leading_clitic = contraction
            new_word = _merge_window_into_word(
                window, chosen.canonical, leading_clitic
            )
            new_surface = new_word["word"].strip()
            # Belt-and-braces: even when the dictionary entry's
            # tokenisation differs from the input window, the merged
            # surface can still come out byte-for-byte identical to
            # what the user already had (e.g. clitic re-prepended
            # leaving the same string). Treat that as no-op and try
            # a smaller window or fall through.
            if new_surface == original_surface:
                continue
            # Stderr breadcrumb so the run terminal shows the
            # full decision chain when a replacement fires.
            vstderr(
                f"  phonetic-fix: {original_surface!r} -> "
                f"{new_surface!r} (cands="
                f"{[c.canonical for c in cands]}, "
                f"conf={window_prob:.2f}, hunspell="
                f"{misspelled})"
            )
            out.append(new_word)
            fixes += 1
            if state is not None:
                state.corrections.append(
                    {
                        "speaker": segment.get("speaker", ""),
                        "start": float(window[0].get("start", seg_start)),
                        "original": original_surface,
                        "replacement": new_surface,
                        "type": "phonetic",
                        "confidence": window_prob,
                    }
                )
            if log is not None:
                abs_start = float(window[0].get("start", seg_start))
                hunspell_tag = ""
                if hunspell is not None and getattr(hunspell, "loaded", False):
                    hunspell_tag = (
                        " hunspell=misspelled"
                        if misspelled
                        else " hunspell=spelled"
                    )
                log(
                    f"  fix [{_format_log_ts(abs_start):>8s}] "
                    f"{original_surface!r} -> {new_surface!r} "
                    f"(min conf {window_prob:.2f}{hunspell_tag})"
                )
            i += n
            replaced = True
            break
        if not replaced:
            out.append(words[i])
            i += 1

    segment["words"] = out
    segment["text"] = "".join(w["word"] for w in out)
    return fixes


def _lookup_window(tokens: list[str], index: PhoneticIndex) -> list[DictionaryEntry]:
    """Look up `tokens` in the index, trying both code variants in
    both languages. Deduplicates results while preserving order so
    the first hit wins.
    """
    seen_canonical: set[str] = set()
    out: list[DictionaryEntry] = []

    def _try(lang: Optional[str]) -> None:
        if not lang:
            return
        # Variant 1: per-token joined code
        code_a = _phonetic_code(tokens, lang)
        for code in (code_a,):
            if not code:
                continue
            for e in index.by_code.get(code, []):
                if e.canonical not in seen_canonical:
                    seen_canonical.add(e.canonical)
                    out.append(e)
        # Variant 2: collapsed-token code
        if len(tokens) > 1:
            collapsed = "".join(tokens)
            if collapsed:
                code_b = _phonetic_token(collapsed, lang)
                if code_b:
                    for e in index.by_code.get(code_b, []):
                        if e.canonical not in seen_canonical:
                            seen_canonical.add(e.canonical)
                            out.append(e)

    _try(index.primary_lang)
    _try(index.secondary_lang)
    return out


def _fuzzy_distance_budget(
    longer: int,
    window_prob: float,
    cfg: CorrectionConfig,
    hunspell_misspelled: Optional[bool],
) -> int:
    """Edit-distance ceiling for the surface-level fuzzy fallback.

    Scales with three signals:
      * input length (longer entries tolerate more typos);
      * Whisper confidence (lower => more permissive);
      * Hunspell verdict (misspelled => more permissive; spelled
        => no extra leniency).
    """
    if longer < 8:
        budget = 1
    elif longer < 12:
        budget = 2
    else:
        budget = 3
    if window_prob < cfg.low_conf:
        budget += 1
    if hunspell_misspelled is True:
        budget += 1
    return budget


def _fuzzy_lookup(
    tokens: Sequence[str],
    index: PhoneticIndex,
    window_prob: float,
    cfg: CorrectionConfig,
    hunspell_misspelled: Optional[bool],
) -> list[DictionaryEntry]:
    """Surface-level edit-distance fallback for when the phonetic
    code lookup returned nothing. Walks the flat entries list and
    keeps every entry whose surface (whitespace-stripped, lower-
    cased) is within the dynamic edit-distance budget of the input.
    """
    input_surface = _surface_normalize("".join(tokens))
    if not input_surface:
        return []
    out: list[DictionaryEntry] = []
    seen: set[str] = set()
    for entry in index.all_entries:
        entry_surface = _surface_normalize("".join(entry.tokens))
        if not entry_surface:
            continue
        # Quick length filter so we don't compute Levenshtein for
        # entries whose length differs more than any possible budget.
        len_diff = abs(len(entry_surface) - len(input_surface))
        if len_diff > 6:
            continue
        longer = max(len(input_surface), len(entry_surface))
        budget = _fuzzy_distance_budget(
            longer, window_prob, cfg, hunspell_misspelled
        )
        if len_diff > budget:
            continue
        edit = _levenshtein(input_surface, entry_surface)
        if edit <= budget:
            if entry.canonical not in seen:
                seen.add(entry.canonical)
                out.append(entry)
    return out


_hunspell_busy = threading.Event()
_hunspell_busy.set()
_hunspell_cache: dict[str, list] = {}


def _suggest_with_timeout(
    hunspell, core: str, timeout_s: float
) -> Optional[list]:
    """Wall-clock-bounded wrapper around `hunspell.suggest`.

    Critical detail: only ONE Hunspell suggest is allowed in
    flight at a time. spylls is pure Python and holds the GIL
    for the FULL duration of the original call (e.g. 35 s),
    even when our timeout abandons waiting - so the orphan
    thread starves the main thread for the original duration.
    The `_hunspell_busy` flag makes subsequent calls return
    `None` immediately while the orphan thread finishes, so
    we cap the worst-case main-thread starvation at the first
    pathological word's natural duration instead of summing
    them.

    Result is also memoised so the same word never goes
    through spylls twice.
    """
    if core in _hunspell_cache:
        return _hunspell_cache[core]
    if not _hunspell_busy.is_set():
        return None
    _hunspell_busy.clear()
    holder: list = []

    def _runner() -> None:
        try:
            holder.append(hunspell.suggest(core, limit=5))
        except Exception:
            holder.append([])
        finally:
            _hunspell_busy.set()

    t = threading.Thread(target=_runner, name="hunspell-suggest", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None
    suggestions = holder[0] if holder else []
    _hunspell_cache[core] = suggestions
    return suggestions


def _format_log_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    minutes = total_ms // 60_000
    total_ms -= minutes * 60_000
    secs = total_ms // 1_000
    ms = total_ms - secs * 1_000
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def _apply_low_conf_hunspell(
    segment: dict,
    cfg: CorrectionConfig,
    hunspell,
    state: Optional[CorrectionState],
    log: Optional[LogCb] = None,
) -> int:
    """Hunspell second pass on words below `cfg.hunspell_threshold`.

    For each candidate word, ask Hunspell for the top-3 suggestions
    and apply the one with the smallest case- and accent-
    insensitive edit distance, capped at 2. Words that already
    carry a `correction` (from the phonetic pass) are left alone.
    The conf gate is critical for performance: `hunspell.suggest()`
    is slow on unusual words (proper nouns, jargon) and would
    otherwise run on every word in the segment.
    """
    if hunspell is None or not getattr(hunspell, "loaded", False):
        return 0
    if cfg.hunspell_threshold <= 0:
        return 0
    words = segment.get("words")
    if not words:
        return 0

    # User-dict surfaces are cached on `state` so the project's
    # recognised vocabulary ("Aeris", "GAFAM", "Castopod") is
    # never replaced by a Hunspell suggestion - even when those
    # words are flagged misspelled by the system dictionary.
    user_dict_surfaces = (
        state.user_dict_surfaces
        if state is not None and state.user_dict_surfaces is not None
        else set()
    )

    seg_start = float(segment.get("start", 0.0))
    fixes = 0
    for w_idx, w in enumerate(words):
        # Drop the GIL periodically so the GUI stays responsive
        # during slow Hunspell passes.
        if w_idx and w_idx % 8 == 0:
            _time.sleep(0)
        if w.get("correction"):
            continue
        prob = float(w.get("probability", 1.0))
        if prob >= cfg.hunspell_threshold:
            continue
        raw = str(w.get("word", ""))
        leading_ws_n = len(raw) - len(raw.lstrip())
        leading_ws = raw[:leading_ws_n]
        body = raw[leading_ws_n:]
        # Strip outer punctuation for the Hunspell lookup; preserve
        # it in the rebuilt word.
        m = re.match(r"^([^\w\sÀ-ſ]*)(.*?)([^\w\sÀ-ſ]*)$", body, flags=re.DOTALL)
        if not m:
            continue
        lead_punct, core, trail_punct = m.group(1), m.group(2), m.group(3)
        if not core or core.isdigit():
            continue
        # Skip very short tokens: Hunspell's first suggestion for a
        # 1-3 letter "word" is almost always a wild guess (e.g.
        # "boé" -> "blé" when the actual utterance was something
        # entirely different). Need at least 4 characters before we
        # trust the suggestion.
        if len(core) < 4:
            continue
        # Skip very long tokens: spylls' edit-distance neighbourhood
        # search can spike to 10s of seconds on a single 20+ char
        # gibberish word (Whisper sometimes concatenates words at low
        # conf). 18 chars covers all real French words while keeping
        # the worst-case suggest() call bounded.
        if len(core) > 18:
            continue
        # User-dict words override Hunspell. The project's
        # vocabulary defines what's correct; "Aeris", "Castopod",
        # "GAFAM" etc. are all "misspelled" per Hunspell's French
        # / English dicts but explicitly recognised by the user.
        # Replacing them would corrupt the transcript.
        if _surface_normalize(core) in user_dict_surfaces:
            continue
        if not hunspell.is_misspelled(core):
            continue
        t_sug = _time.monotonic()
        suggestions = _suggest_with_timeout(hunspell, core, 0.5)
        sug_dt = _time.monotonic() - t_sug
        if suggestions is None:
            # Timed out: spylls is grinding the edit-distance
            # neighbourhood. Abandon this word; the orphan thread
            # is daemonised so it dies with the process.
            vstderr(
                f"  hunspell.suggest({core!r}) timed out at "
                f"{sug_dt:.1f}s, skipping"
            )
            w["uncorrectable_misspelled"] = True
            continue
        if sug_dt > 0.5:
            vstderr(
                f"  slow hunspell.suggest({core!r}): {sug_dt:.2f}s "
                f"({len(core)} chars, {len(suggestions)} suggestions)"
            )
        if not suggestions:
            w["uncorrectable_misspelled"] = True
            continue
        # Pick the closest suggestion by edit distance.
        core_norm = _surface_normalize(core)
        suggestion, best_edit = min(
            (
                (s, _levenshtein(core_norm, _surface_normalize(s)))
                for s in suggestions
            ),
            key=lambda pair: pair[1],
        )
        if best_edit > 2:
            w["uncorrectable_misspelled"] = True
            continue
        # First-character preservation. Genuine typos almost never
        # change the first letter ("compération" -> "coopération",
        # "Hmmmm" -> "Hmm"), but Hunspell's edit-1 neighbourhood
        # also contains semantic siblings that DO change it
        # ("oeufs" -> "keufs", "yeux" -> "deux"). The first-char
        # gate keeps the typo fixes while blocking those.
        suggestion_norm = _surface_normalize(suggestion)
        if (
            suggestion_norm
            and core_norm
            and suggestion_norm[0] != core_norm[0]
        ):
            w["uncorrectable_misspelled"] = True
            continue
        # Skip no-op suggestions (case-insensitive equality).
        if suggestion.strip().lower() == core.strip().lower():
            w["uncorrectable_misspelled"] = True
            continue
        new_text = leading_ws + lead_punct + suggestion + trail_punct
        original_surface = body.strip()
        w["word"] = new_text
        w["correction"] = {
            "original": original_surface,
            "replacement": suggestion,
            "type": "hunspell",
            "original_confidence": prob,
        }
        fixes += 1
        if state is not None:
            state.corrections.append(
                {
                    "speaker": segment.get("speaker", ""),
                    "start": float(w.get("start", seg_start)),
                    "original": original_surface,
                    "replacement": suggestion,
                    "type": "hunspell",
                    "confidence": prob,
                }
            )
        if log is not None:
            log(
                f"  hunspell-fix [{_format_log_ts(float(w.get('start', seg_start))):>8s}] "
                f"{original_surface!r} -> {suggestion!r} "
                f"(conf {prob:.2f})"
            )
    if fixes:
        # Rebuild segment text from the (possibly mutated) words.
        segment["text"] = "".join(w["word"] for w in words)
    return fixes


def build_correction_state(
    cfg: Optional[CorrectionConfig],
    log: Optional[LogCb] = None,
) -> Optional[CorrectionState]:
    """Construct a CorrectionState for per-segment correction. Returns
    None when correction is disabled or the dictionary can't be
    loaded. Both phonetic and Hunspell components are independent;
    the state is also built when only one of them would fire.
    """
    if cfg is None:
        return None
    if not cfg.dictionary_path:
        if log is not None:
            log("No dictionary configured; skipping phonetic correction.")
        return None
    try:
        index = load_dictionary(cfg.dictionary_path, cfg)
    except OSError as exc:
        if log is not None:
            log(f"Failed to load dictionary {cfg.dictionary_path!r}: {exc}")
        return None
    hunspell = _build_hunspell(cfg, log)
    if log is not None:
        seen: set[str] = set()
        for entries in index.by_code.values():
            for e in entries:
                seen.add(e.canonical)
        log(
            f"Per-chunk correction enabled: {len(seen)} dictionary "
            f"entr{'y' if len(seen) == 1 else 'ies'}, "
            f"low_conf={cfg.low_conf}, high_conf={cfg.high_conf}, "
            f"hunspell_threshold={cfg.hunspell_threshold}"
        )
    return CorrectionState(cfg=cfg, index=index, hunspell=hunspell)


def correct_segment_in_place(
    segment: dict,
    state: CorrectionState,
    log: Optional[LogCb] = None,
) -> int:
    """Apply phonetic + low-confidence Hunspell correction to a single
    segment, mutating it in place. Returns the total number of fixes.
    """
    t0 = _time.monotonic()
    fixes = _correct_segment(
        segment,
        state.index,
        state.cfg,
        log=log,
        hunspell=state.hunspell,
        state=state,
    )
    t_phon = _time.monotonic() - t0
    t1 = _time.monotonic()
    fixes += _apply_low_conf_hunspell(
        segment, state.cfg, state.hunspell, state, log=log
    )
    t_hun = _time.monotonic() - t1
    if t_phon + t_hun > 1.0:
        nwords = len(segment.get("words") or [])
        vstderr(
            f"  slow correction: {nwords} words, "
            f"phonetic {t_phon:.2f}s, hunspell {t_hun:.2f}s"
        )
    # Release the GIL briefly so the GUI's event loop can pump
    # clicks while the correction pass runs (pure Python).
    _time.sleep(0)
    return fixes


def format_corrections_recap(state: CorrectionState) -> list[str]:
    """Build a human-readable recap of every correction made during
    a run. Returns a list of log lines (no trailing newlines)."""
    if not state.corrections:
        return ["No corrections were applied during this run."]
    by_pair: dict[tuple, dict] = {}
    for c in state.corrections:
        key = (c["original"], c["replacement"], c["type"])
        if key not in by_pair:
            by_pair[key] = {
                "original": c["original"],
                "replacement": c["replacement"],
                "type": c["type"],
                "count": 0,
                "first": c["start"],
            }
        by_pair[key]["count"] += 1
    phonetic = [v for v in by_pair.values() if v["type"] == "phonetic"]
    hunspell = [v for v in by_pair.values() if v["type"] == "hunspell"]
    lines: list[str] = []
    lines.append(
        f"Corrections recap: {len(state.corrections)} total "
        f"({len(phonetic)} phonetic / {len(hunspell)} hunspell)."
    )
    if phonetic:
        lines.append("  Phonetic (user dictionary):")
        for v in sorted(phonetic, key=lambda x: -x["count"]):
            tag = "" if v["count"] == 1 else f" (x{v['count']})"
            lines.append(
                f"    {v['original']!r} -> {v['replacement']!r}{tag}"
            )
    if hunspell:
        lines.append("  Hunspell (low confidence):")
        for v in sorted(hunspell, key=lambda x: -x["count"]):
            tag = "" if v["count"] == 1 else f" (x{v['count']})"
            lines.append(
                f"    {v['original']!r} -> {v['replacement']!r}{tag}"
            )
    return lines


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def apply_corrections(
    segments: list[dict],
    cfg: CorrectionConfig,
    *,
    log: Optional[LogCb] = None,
) -> list[dict]:
    """Apply phonetic correction (with optional Hunspell second-opinion).

    Returns a new list of segment dicts. The input list is not mutated.
    Timestamps are preserved.
    """
    out = deepcopy(segments)
    if not cfg.dictionary_path:
        if log is not None:
            log("No dictionary configured; skipping phonetic correction.")
        return out

    index = load_dictionary(cfg.dictionary_path, cfg)
    seen: set[str] = set()
    dictionary_terms: list[str] = []
    for entries in index.by_code.values():
        for e in entries:
            if e.canonical not in seen:
                seen.add(e.canonical)
                dictionary_terms.append(e.canonical)

    hunspell = _build_hunspell(cfg, log)

    if log is not None:
        log(
            f"Phonetic pass: {len(out)} segment(s), "
            f"{len(dictionary_terms)} dictionary entr"
            f"{'y' if len(dictionary_terms) == 1 else 'ies'}, "
            f"low_conf={cfg.low_conf}, high_conf={cfg.high_conf}"
        )
    total_fixes = 0
    for seg in out:
        total_fixes += _correct_segment(
            seg, index, cfg, log=log, hunspell=hunspell
        )
    if log is not None:
        if total_fixes:
            log(f"Phonetic pass applied {total_fixes} replacement(s).")
        else:
            log("Phonetic pass: no replacements needed.")

    return out


def _build_hunspell(cfg: CorrectionConfig, log: Optional[LogCb]):
    """Construct a HunspellChecker for the configured paths, with
    auto-detection when the user didn't pick one. Returns None when
    Hunspell is disabled or no dictionary loaded successfully.
    """
    if not getattr(cfg, "use_hunspell", True):
        return None
    from .correct_hunspell import HunspellChecker, auto_detect_hunspell

    primary = cfg.hunspell_primary
    if not primary:
        # Auto-detect from the phonetic-language hint when set.
        primary_lang = cfg.phonetic_lang
        if not primary_lang or primary_lang == "auto":
            primary_lang = ""
        if primary_lang:
            primary = auto_detect_hunspell(primary_lang)

    secondary = cfg.hunspell_secondary
    if not secondary and cfg.phonetic_lang_secondary:
        secondary = auto_detect_hunspell(cfg.phonetic_lang_secondary)

    if not primary and not secondary:
        if log is not None:
            log(
                "Hunspell second-opinion: no dictionary auto-detected; "
                "phonetic pass runs without it."
            )
        return None

    checker = HunspellChecker(primary, secondary)
    if log is not None:
        if checker.loaded:
            log(
                f"Hunspell second-opinion: loaded "
                f"{', '.join(checker.loaded_paths)}"
            )
        else:
            log(
                "Hunspell second-opinion: dictionaries failed to load "
                f"({', '.join(checker.failed_paths)}); falling back to "
                "phonetic-only."
            )
    return checker if checker.loaded else None
