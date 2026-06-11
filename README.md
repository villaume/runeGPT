# runeGPT
Weekend Project to see if one could perhaps train a small transformer on runes instead of tokens

## First problem: data — SOLVED
The Scandinavian Runic-text Database (Rundata, Uppsala) ships as a 45MB SQLite file inside the
[rundata-net](https://github.com/Snorresk/rundata-net) web clone — no Windows .exe needed:

```bash
curl -sL https://github.com/Snorresk/rundata-net/raw/master/rundatanet/static/runes/runes.sqlite3 \
  -o data/runes.sqlite3
python3 scripts/build_corpus.py
```

That yields **6,815 inscriptions**, each with transliteration + Old Norse normalisation +
English translation + dating/metadata, and produces:

- `data/corpus.jsonl` — full records: runes, transliteration, Old Norse, English, dating, plus
  **coordinates (97%), material/geology (80%), parish (95%)** lifted from the SQLite metadata
- `data/runes.txt` — ~316k chars of Unicode Younger Futhark, the char-level LM corpus
- `data/translit.txt` — the same text in scholarly transliteration

The build script strips Rundata's editorial notation (`§A` side markers, `[]`/`()` uncertainty
brackets, `^` bind runes) and inverts the transliteration back to Unicode runes
(younger futhark + dotted medieval runes), at 100% character coverage.

Other sources considered: www.runesdb.eu, Skaldic Poetry of the Scandinavian Middle Ages.

## Training: runes ARE the tokens

nanoGPT's `shakespeare_char` recipe ported to MLX (runs on the Mac GPU, no CUDA).
The corpus has a **33-symbol vocabulary** — 28 runes, 3 runic punctuation marks,
space, newline — so no tokenizer at all: every token is literally a rune.
(nanochat was considered and rejected: it's the full ChatGPT pipeline for 8×H100
with a 65k BPE vocab — four orders of magnitude too big, and BPE defeats the premise.)

```bash
uv sync
uv run python scripts/train.py     # ~3M param char-GPT, minutes on Apple Silicon
uv run python scripts/sample.py    # generate fake runestone inscriptions
uv run python scripts/sample.py --prompt "ᚴᚢᚦᚱᚢᚾ ᛬ ᛚᛁᛏ"   # continue a prompt
```

Best checkpoint by validation loss lands in `checkpoints/`. The val split is by
inscription (shuffled lines), not a raw slice, so it isn't just one region of Sweden.

## Ithaca for the younger futhark

[DeepMind's Ithaca](https://github.com/google-deepmind/ithaca) restores damaged ancient
Greek inscriptions and attributes them in space and time. The same three tasks map onto
Rundata — so `scripts/ithaca.py` is a small **bidirectional** multitask transformer (a
masked encoder, not the causal LM above — restoration needs to see runes on *both* sides
of a gap):

```bash
uv run python scripts/ithaca.py    # ~3.2M params, ~6 min on Apple Silicon
```

| Head | Task | Label source | Val accuracy | Majority baseline |
|------|------|--------------|--------------|-------------------|
| restoration | fill masked runes (BERT-style) | self-supervised | ~29% top-1 | ~3% (1/34) |
| period | dating U / V / M | Rundata dating prefix (100%) | ~83% | ~56% (Viking) |
| region | geographic attribution | signature prefix (U, Ög, DR…), 11 classes + other (94%) | ~47% | ~24% (Norway) |
| material | geology: stone / wood / metal / bone / … | Rundata material_type (100%) | ~71% | ~62% (stone) |

Restoration is the hard task (short inscriptions give each gap little context) but is
self-supervised, so its training signal is unbounded. It also doubles as a scoring engine
for ranking competing readings of contested strings — the original runeGPT goal, and the
exact problem behind 150 years of argument over Rök's `raiþ þiaurikR` (Theodoric) vs Bo
Ralph's re-segmentation `raið iau rinkR`.

## Runestone Atlas (static site)

A map of all **6,412 geolocated inscriptions**, in the spirit of
[ithaca.deepmind.com](https://ithaca.deepmind.com) — except our model runs on MLX, so
instead of serving live inference we **bake the model's predictions into static JSON**.
Colour the map by dating period, by material/geology, or by *model agreement* (where the
Ithaca model's period guess matches the ground truth — 87% of mapped stones); click any
stone for its runes, transliteration, translation, dating, material, and the model's
period/region/material predictions.

```bash
uv run python scripts/ithaca.py        # train (writes checkpoints-ithaca/)
uv run python scripts/export_atlas.py  # corpus + predictions -> web/atlas.json
python3 -m http.server -d web 8777     # open http://localhost:8777
```

`web/` is a self-contained Leaflet site (no build step) — deployable to GitHub Pages as-is.

## Later: a runic Gemma via LoRA

Build instruction pairs from `corpus.jsonl` (runes → transliteration → Old Norse →
English) and fine-tune e.g. `mlx-community/gemma-3-4b-it-4bit` with `mlx_lm.lora`.

## Fun fact
The Elder Futhark is supported in unicode
ᚠᚢᚦ
