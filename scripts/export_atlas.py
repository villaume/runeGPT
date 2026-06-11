#!/usr/bin/env python3
"""Export the geolocated corpus + model predictions to a static JSON for the Atlas.

Reads data/corpus.jsonl, keeps the inscriptions that have coordinates, and runs
the trained Ithaca model (checkpoints-ithaca/) over each one so its period /
region / material predictions ship as static data — the web map needs no live
inference. Output: web/atlas.json.

Usage: uv run python scripts/export_atlas.py
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ithaca import IthacaRunes, PERIODS, period_of, region_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints-ithaca"


def predict(rows):
    """Attach model predictions to each row in place; no-op if no checkpoint."""
    if not (CKPT / "config.json").exists():
        print("no checkpoint — exporting ground truth only", file=sys.stderr)
        return
    meta = json.loads((CKPT / "config.json").read_text())
    cfg, itos = meta["cfg"], meta["itos"]
    regions, materials = meta["regions"], meta["materials"]
    stoi = {c: i for i, c in enumerate(itos)}
    bs = cfg["block_size"]
    model = IthacaRunes(cfg)
    model.load_weights(str(CKPT / "model.safetensors"))
    model.eval()

    B = 256
    for start in range(0, len(rows), B):
        chunk = rows[start : start + B]
        x = np.zeros((len(chunk), bs), dtype=np.int32)
        pm = np.zeros((len(chunk), bs), dtype=np.float32)
        for i, r in enumerate(chunk):
            ids = [stoi[c] for c in r["runes"] if c in stoi][:bs]
            x[i, : len(ids)] = ids
            pm[i, : len(ids)] = 1.0
        _, pl, gl, ml = model(mx.array(x), mx.array(pm))
        pp = np.array(pl.argmax(-1))
        gg = np.array(gl.argmax(-1))
        mm = np.array(ml.argmax(-1))
        for i, r in enumerate(chunk):
            r["pred_period"] = PERIODS[pp[i]]
            r["pred_region"] = regions[gg[i]]
            r["pred_material"] = materials[mm[i]]


def main():
    rows = [json.loads(l) for l in open(ROOT / "data" / "corpus.jsonl")]
    geo = [r for r in rows if r.get("lat") and r.get("lon") and r.get("runes")]

    predict(geo)

    out = []
    for r in geo:
        out.append(
            {
                "sig": r["signature"],
                "lat": round(r["lat"], 5),
                "lon": round(r["lon"], 5),
                "runes": r["runes"],
                "translit": r["transliteration_clean"],
                "norse": r["norse"],
                "english": r["english"],
                "dating": r["dating"],
                "period": period_of(r["dating"]),       # ground-truth coarse period
                "region": region_of(r["signature"]),    # ground-truth region prefix
                "material": r.get("material"),
                "material_type": r.get("material_type"),
                "parish": r.get("parish"),
                "p_period": r.get("pred_period"),
                "p_region": r.get("pred_region"),
                "p_material": r.get("pred_material"),
            }
        )

    web = ROOT / "web"
    web.mkdir(exist_ok=True)
    (web / "atlas.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    size_mb = (web / "atlas.json").stat().st_size / 1e6
    print(f"{len(out)} geolocated inscriptions -> web/atlas.json ({size_mb:.1f} MB)")
    if out and "p_period" in out[0]:
        pa = sum(1 for r in out if r["period"] and r["period"] == r["p_period"]) / len(out)
        print(f"model period prediction matches ground truth on {pa:.0%} of mapped stones")


if __name__ == "__main__":
    main()
