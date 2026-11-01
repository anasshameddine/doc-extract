"""Etape 0 - Inspecte les donnees exportees : volumes, champs manquants,
formats de dates/montants, tailles d'images, et planche d'apercu de 5 recus.

Usage :
    python scripts/01_inspect_data.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from docextract import config
from docextract.datasets_io import read_jsonl

FIELDS = ("company", "date", "total", "tax", "items")


def shape(value: str) -> str:
    """'25/12/2018' -> '99/99/9999' ; 'RM9.00' -> 'AA9.99' (lettres -> A)."""
    s = re.sub(r"\d", "9", value)
    return re.sub(r"[A-Za-z]", "A", s)


def describe(name: str, recs: list[dict]) -> dict:
    print(f"\n=== {name} : {len(recs)} docs ===")
    missing_images = [r["doc_id"] for r in recs if not (config.ROOT / r["image"]).exists()]
    print(f"  images manquantes sur disque : {len(missing_images)}")

    print("  champs renseignes :")
    filled = {}
    for f in FIELDS:
        n = sum(1 for r in recs if r["gt"].get(f))
        filled[f] = n
        print(f"    {f:<8} {n:>4}/{len(recs)}  ({100 * n / max(len(recs), 1):.0f} %)")

    for f in ("date", "total"):
        shapes = Counter(shape(r["gt"][f]) for r in recs if r["gt"].get(f))
        if shapes:
            top = ", ".join(f"'{s}' x{c}" for s, c in shapes.most_common(6))
            print(f"  formats {f} ({len(shapes)} distincts) : {top}")

    widths = [r["width"] for r in recs]
    heights = [r["height"] for r in recs]
    nwords = [len(r["words"]) for r in recs]
    if recs:
        print(f"  largeur image  min/med/max : {min(widths)}/{int(statistics.median(widths))}/{max(widths)} px")
        print(f"  hauteur image  min/med/max : {min(heights)}/{int(statistics.median(heights))}/{max(heights)} px")
        print(f"  mots OCR annotes  med/max  : {int(statistics.median(nwords))}/{max(nwords)}")
        ex = recs[0]
        gt_show = {k: v for k, v in ex["gt"].items() if v and k != "items"}
        print(f"  exemple {ex['doc_id']} : {json.dumps(gt_show, ensure_ascii=False)}")
    return {"n": len(recs), "filled": filled, "missing_images": len(missing_images)}


def preview(recs: list[dict], out: Path, n: int = 5, thumb_h: int = 600) -> None:
    """Planche : n recus cote a cote + verite terrain sous chacun."""
    recs = recs[:n]
    if not recs:
        return
    thumbs = []
    for r in recs:
        img = Image.open(config.ROOT / r["image"]).convert("RGB")
        w = max(1, int(img.width * thumb_h / img.height))
        thumbs.append(img.resize((w, thumb_h)))
    text_h, pad = 110, 10
    total_w = sum(t.width for t in thumbs) + pad * (len(thumbs) + 1)
    sheet = Image.new("RGB", (total_w, thumb_h + text_h + 2 * pad), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    x = pad
    for t, r in zip(thumbs, recs):
        sheet.paste(t, (x, pad))
        lines = [r["doc_id"][-14:]] + [f"{k}: {(r['gt'].get(k) or '-')[:22]}"
                                        for k in ("company", "date", "total")]
        for j, line in enumerate(lines):
            draw.text((x, thumb_h + pad + 5 + j * 18), line, fill="black", font=font)
        x += t.width + pad
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=90)
    print(f"\n  apercu -> {out.relative_to(config.ROOT)}")


def main() -> int:
    report = {}
    for path in sorted(config.PROCESSED_DIR.glob("*_*.jsonl")):
        recs = list(read_jsonl(path))
        report[path.stem] = describe(path.stem, recs)

    if not report:
        print("Aucune donnee : lance d'abord  python scripts/00_prepare_data.py")
        return 1

    if config.EVAL_SAMPLE_FILE.exists():
        sample = json.loads(config.EVAL_SAMPLE_FILE.read_text(encoding="utf-8"))
        ids = set(sample.get("sroie_test", []))
        recs = [r for r in read_jsonl(config.jsonl_path("sroie", "test")) if r["doc_id"] in ids]
        print(f"\nEchantillon d'eval : {len(sample.get('sroie_test', []))} SROIE, "
              f"{len(sample.get('cord_test', []))} CORD (graine {sample.get('seed')})")
        preview(recs, config.OUTPUTS_DIR / "inspect" / "apercu_sroie.jpg")

    out = config.OUTPUTS_DIR / "inspect" / "rapport_donnees.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  rapport -> {out.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
