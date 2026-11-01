"""Conversion des jeux SROIE et CORD vers un format commun (JSONL + images JPEG).

Format d'un enregistrement (une ligne JSONL) :
{
  "doc_id": "sroie_test_X51005230605",
  "dataset": "sroie" | "cord",
  "split": "train" | "test" | ...,
  "image": "data/raw/sroie/test/X51005230605.jpg",   # chemin relatif a la racine
  "width": 463, "height": 1013,
  "gt": {"company", "date", "address", "total", "subtotal", "tax", "items"},
  "words": [...], "bboxes": [...]                     # OCR annote (verite terrain)
}
Les valeurs de gt sont BRUTES (non normalisees) : la normalisation arrive a l'etape 1.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image

GT_FIELDS = ("company", "date", "address", "total", "subtotal", "tax", "items")


def _empty_gt() -> dict[str, Any]:
    return {k: None for k in GT_FIELDS}


def _clean(value: Any) -> str | None:
    """Chaine nettoyee, ou None si vide."""
    if value is None:
        return None
    if isinstance(value, list):  # CORD met parfois une liste de valeurs
        value = " ".join(str(v) for v in value if v is not None)
    text = str(value).strip()
    return text or None


def _as_dict(value: Any) -> dict:
    """Accepte un dict ou une chaine JSON."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# ---------------------------------------------------------------- SROIE
def parse_sroie_row(row: dict, split: str, index: int) -> dict:
    """Ligne HF SROIE -> enregistrement commun (sans l'image)."""
    key = _clean(row.get("key")) or f"{index:04d}"
    ent = _as_dict(row.get("entities"))
    gt = _empty_gt()
    for field in ("company", "date", "address", "total"):
        gt[field] = _clean(ent.get(field))
    return {
        "doc_id": f"sroie_{split}_{key}",
        "dataset": "sroie",
        "split": split,
        "file_stem": key,
        "gt": gt,
        "words": [str(w) for w in (row.get("words") or [])],
        "bboxes": _to_list(row.get("bboxes")),
    }


# ---------------------------------------------------------------- CORD
def _quad_to_box(quad: dict) -> list[int]:
    xs = [quad.get(f"x{i}", 0) for i in range(1, 5)]
    ys = [quad.get(f"y{i}", 0) for i in range(1, 5)]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _cord_items(menu: Any) -> list[dict] | None:
    if not menu:
        return None
    entries = menu if isinstance(menu, list) else [menu]
    items = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        items.append({
            "name": _clean(e.get("nm")),
            "qty": _clean(e.get("cnt")),
            "price": _clean(e.get("price")),
        })
    return items or None


def parse_cord_row(row: dict, split: str, index: int) -> dict:
    """Ligne HF CORD-v2 -> enregistrement commun (sans l'image).

    CORD public n'a ni fournisseur ni date (retires pour la vie privee) :
    company/date/address restent None.
    """
    gt_full = _as_dict(row.get("ground_truth"))
    parse = gt_full.get("gt_parse", {}) or {}
    sub = parse.get("sub_total", {}) or {}
    tot = parse.get("total", {}) or {}
    if isinstance(sub, list):
        sub = sub[0] if sub else {}
    if isinstance(tot, list):
        tot = tot[0] if tot else {}

    gt = _empty_gt()
    gt["total"] = _clean(tot.get("total_price"))
    gt["subtotal"] = _clean(sub.get("subtotal_price"))
    gt["tax"] = _clean(sub.get("tax_price"))
    gt["items"] = _cord_items(parse.get("menu"))

    words, bboxes = [], []
    for line in gt_full.get("valid_line", []) or []:
        for w in line.get("words", []) or []:
            text = _clean(w.get("text"))
            if text:
                words.append(text)
                bboxes.append(_quad_to_box(w.get("quad", {}) or {}))

    stem = f"cord_{split}_{index:04d}"
    return {
        "doc_id": stem,
        "dataset": "cord",
        "split": split,
        "file_stem": stem,
        "gt": gt,
        "words": words,
        "bboxes": bboxes,
    }


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    out = []
    for box in value:
        if hasattr(box, "tolist"):
            box = box.tolist()
        out.append(list(box) if isinstance(box, (list, tuple)) else box)
    return out


# ---------------------------------------------------------------- export
PARSERS = {"sroie": parse_sroie_row, "cord": parse_cord_row}


def export_split(rows: Iterable[dict], dataset: str, split: str,
                 root: Path, raw_dir: Path, jsonl_file: Path,
                 jpeg_quality: int = 92) -> int:
    """Ecrit les images en JPEG et les annotations en JSONL. Retourne le nb de docs."""
    parser = PARSERS[dataset]
    img_dir = raw_dir / dataset / split
    img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with jsonl_file.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            rec = parser(row, split, i)
            img = row.get("image")
            if not isinstance(img, Image.Image):
                raise TypeError(f"{rec['doc_id']}: champ 'image' absent ou invalide")
            img = img.convert("RGB")
            img_path = img_dir / f"{rec.pop('file_stem')}.jpg"
            img.save(img_path, "JPEG", quality=jpeg_quality)
            rec["image"] = img_path.relative_to(root).as_posix()
            rec["width"], rec["height"] = img.size
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def make_eval_sample(doc_ids: list[str], size: int, seed: int) -> list[str]:
    """Tirage deterministe (meme graine -> meme echantillon), trie pour la lisibilite."""
    ids = sorted(doc_ids)
    if size >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, size))
