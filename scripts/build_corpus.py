#!/usr/bin/env python3
"""Build training corpora from the Rundata SQLite database.

Input:  data/runes.sqlite3  (from the rundata-net project, which ships the
        Scandinavian Runic-text Database as SQLite)
Output: data/corpus.jsonl   one record per inscription: signature, raw and
                            cleaned transliteration, best-effort Unicode runes,
                            Old Norse normalisation, English translation, dating
        data/runes.txt      concatenated Unicode rune text (char-level LM corpus)
        data/translit.txt   concatenated cleaned transliteration (alt corpus)

The rune rendering is a best-effort inverse of the Rundata transliteration,
Younger-Futhark-centric (the corpus is overwhelmingly Viking Age). Dotted /
medieval runes are used where the transliteration is unambiguous about them.
Characters with no defensible mapping are dropped and counted; the script
reports coverage at the end.

Usage: uv run python scripts/build_corpus.py   (or plain python3, stdlib only)
"""

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "runes.sqlite3"

# Transliteration letter -> Unicode rune. Younger futhark base row, with
# dotted/medieval runes for letters the 16-rune row can't express.
RUNE_MAP = {
    "f": "ᚠ",
    "u": "ᚢ",
    "þ": "ᚦ",
    "o": "ᚬ",  # ąss/óss
    "ô": "ᚬ",  # nasal o
    "r": "ᚱ",
    "k": "ᚴ",
    "h": "ᚼ",
    "n": "ᚾ",
    "i": "ᛁ",
    "a": "ᛅ",
    "s": "ᛋ",
    "t": "ᛏ",
    "b": "ᛒ",
    "m": "ᛘ",
    "l": "ᛚ",
    "R": "ᛦ",  # palatal r
    # dotted / medieval additions
    "e": "ᛂ",
    "g": "ᚵ",
    "y": "ᚤ",
    "d": "ᛑ",
    "p": "ᛔ",
    "v": "ᚡ",
    "w": "ᚹ",  # elder futhark wunjo (rare, early inscriptions)
    "z": "ᛉ",  # elder futhark algiz (rare, early inscriptions)
    "æ": "ᛅ",  # medieval futhork; collides with Viking-age 'a' by design
    "ø": "ᚯ",  # medieval futhork
    "ð": "ᚧ",
    "ï": "ᛇ",  # elder futhark eihwaz
    "ŋ": "ᛜ",  # elder futhark ingwaz
    "j": "ᛃ",  # elder futhark jera
    "c": "ᚴ",  # medieval c, phonetically k
    "q": "ᚴ",
    "x": "ᚴᛋ",
}

# Word separators as carved: normalise the zoo of editorial separator marks
# onto the three runic punctuation codepoints Unicode actually has.
SEPARATOR_MAP = {
    "·": "᛫",
    "'": "᛫",
    "¤": "᛫",
    "°": "᛫",
    ":": "᛬",
    "÷": "᛬",
    "×": "᛭",
    "+": "᛭",
    "*": "᛭",
}


def clean_transliteration(raw: str) -> str:
    """Strip Rundata editorial notation, keep the readable rune sequence."""
    text = raw
    text = re.sub(r"§[A-ZÖ]+", " ", text)  # side markers §A §B ...
    text = text.replace("¶", " ")  # line break on the stone
    text = re.sub(r"\{[^}]*\}", " ", text)  # {LATIN LETTER} segments
    text = re.sub(r"<[^>]*>", " ", text)  # <unintelligible> segments
    text = re.sub(r"\([^)]*\?\)", " ", text)  # (...?) fully doubtful readings
    # alternative readings "a/b": keep the first
    text = re.sub(r"([^\s/]+)/[^\s]+", r"\1", text)
    text = text.replace("(", "").replace(")", "")  # uncertain but read: keep
    text = text.replace("[", "").replace("]", "")  # restored: keep
    text = text.replace("^", "")  # bind rune marker: both runes remain
    text = re.sub(r"\.{2,}", " ", text)  # lost sequence
    text = re.sub(r"-+", " ", text)  # illegible runes
    text = re.sub(r"[?!=~|´`\"\\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_runes(clean: str) -> tuple[str, Counter]:
    """Best-effort Unicode rune rendering of a cleaned transliteration."""
    out = []
    dropped: Counter = Counter()
    for ch in clean:
        if ch in RUNE_MAP:
            out.append(RUNE_MAP[ch])
        elif ch in SEPARATOR_MAP:
            out.append(SEPARATOR_MAP[ch])
        elif ch == " ":
            out.append(" ")
        elif ch.lower() in RUNE_MAP and ch != "R":
            # Other capitals mark unusual rune forms; map by sound value.
            out.append(RUNE_MAP[ch.lower()])
        else:
            dropped[ch] += 1
    text = re.sub(r"\s+", " ", "".join(out)).strip()
    return text, dropped


def main() -> None:
    if not DB.exists():
        sys.exit(
            f"missing {DB}\nfetch it with:\n  curl -sL "
            "https://github.com/Snorresk/rundata-net/raw/master/rundatanet/static/runes/runes.sqlite3 "
            f"-o {DB}"
        )

    con = sqlite3.connect(DB)
    rows = con.execute(
        """
        SELECT s.signature_text, t.value, n.value, e.value,
               m.dating, m.rune_type, m.year_from, m.year_to,
               m.latitude, m.longitude, m.parish, m.municipality, m.district,
               m.found_location, m.material, mt.name
        FROM signatures s
        JOIN transliterated_text t ON t.signature_id = s.id
        LEFT JOIN normalisation_norse n ON n.signature_id = s.id
        LEFT JOIN translation_english e ON e.signature_id = s.id
        LEFT JOIN meta_information m ON m.signature_id = s.id
        LEFT JOIN material_types mt ON m.materialType_id = mt.id
        ORDER BY s.signature_text
        """
    ).fetchall()

    def clean_str(v):
        v = (v or "").strip()
        return v or None

    def coord(v):
        # Rundata uses 0 for "unknown"; treat as missing.
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f != 0.0 else None

    records = []
    dropped_total: Counter = Counter()
    rune_chars = translit_chars = 0
    for (sig, translit, norse, english, dating, rune_type, y0, y1,
         lat, lon, parish, municipality, district, found, material, mat_type) in rows:
        clean = clean_transliteration(translit or "")
        runes, dropped = to_runes(clean)
        dropped_total.update(dropped)
        rune_chars += len(runes)
        translit_chars += len(clean)
        records.append(
            {
                "signature": sig,
                "transliteration": translit,
                "transliteration_clean": clean,
                "runes": runes,
                "norse": norse,
                "english": english,
                "dating": dating,
                "rune_type": rune_type,
                "year_from": y0,
                "year_to": y1,
                "lat": coord(lat),
                "lon": coord(lon),
                "parish": clean_str(parish),
                "municipality": clean_str(municipality),
                "district": clean_str(district),
                "found_location": clean_str(found),
                "material": clean_str(material),
                "material_type": clean_str(mat_type),
            }
        )

    out_dir = ROOT / "data"
    with open(out_dir / "corpus.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "runes.txt", "w") as f:
        f.write("\n".join(r["runes"] for r in records if r["runes"]) + "\n")
    with open(out_dir / "translit.txt", "w") as f:
        f.write(
            "\n".join(
                r["transliteration_clean"]
                for r in records
                if r["transliteration_clean"]
            )
            + "\n"
        )

    mapped = rune_chars / translit_chars if translit_chars else 0
    n = len(records)
    print(f"{n} inscriptions -> data/corpus.jsonl")
    print(f"rune corpus: {rune_chars} chars -> data/runes.txt")
    print(f"translit corpus: {translit_chars} chars -> data/translit.txt")
    print(f"char coverage: {mapped:.1%} of cleaned transliteration rendered as runes")
    for field in ("lat", "material", "material_type", "parish"):
        have = sum(1 for r in records if r[field] is not None)
        print(f"  {field}: {have}/{n} ({have/n:.0%})")
    if dropped_total:
        top = ", ".join(f"{ch!r}×{n}" for ch, n in dropped_total.most_common(10))
        print(f"dropped (no mapping): {top}")


if __name__ == "__main__":
    main()
