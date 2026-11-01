"""Etape 4a : prepare les donnees du fine-tuning Donut (approche 3, Document AI).

Donut lit l'image et ecrit directement le JSON : pas d'OCR. Pour l'entrainer on lui donne des couples
(image, JSON attendu). Format Hugging Face "imagefolder" : un dossier d'images + un metadata.jsonl.

    data/donut/
        sroie_train/   526 recus : le split train de SROIE SANS les 100 recus dev   (entrainement)
        sroie_dev/     100 recus dev (avec verite)                                  (suivi, reglages)
        sroie_eval/    100 recus eval (SANS verite : images seules)                 (score final, etape 5)
        cord_dev/      100 recus (images seules)  \\ pour les predictions du checkpoint public
        cord_eval/     100 recus (images seules)  /  de Donut sur CORD (approche A)
    data/donut_data.zip    tout ca, a deposer sur Google Drive pour Colab

Garde-fous : aucun recu dev ou eval n'est dans l'entrainement ; l'eval n'embarque aucune verite.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import config
from .datasets_io import read_jsonl
from .samples import get_docs

SROIE_FIELDS = ("company", "date", "total")      # ce que Donut apprend a ecrire sur SROIE
TASK_PROMPT = "<s_receipt>"


def sroie_target(gt: dict) -> dict:
    """Verite SROIE -> cible Donut {company, date, total} (texte brut, "" si absent)."""
    return {k: (gt.get(k) or "") for k in SROIE_FIELDS}


def metadata_line(rec: dict, file_name: str, with_truth: bool) -> dict:
    line = {"file_name": file_name, "doc_id": rec["doc_id"]}
    if with_truth:
        line["ground_truth"] = json.dumps({"gt_parse": sroie_target(rec["gt"])}, ensure_ascii=False)
    return line


def _export_set(out_dir: Path, recs: list[dict], with_truth: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for rec in recs:
        src = config.ROOT / rec["image"]
        name = f"{rec['doc_id']}{src.suffix.lower() or '.jpg'}"
        shutil.copyfile(src, out_dir / name)
        lines.append(metadata_line(rec, name, with_truth))
    (out_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    return len(lines)


def split_sroie_train() -> tuple[list[dict], list[dict]]:
    """(train sans dev, dev) pour SROIE. Verifie qu'aucun recu n'est dans les deux."""
    dev = get_docs("sroie", "dev")
    dev_ids = {r["doc_id"] for r in dev}
    train = [r for r in read_jsonl(config.jsonl_path("sroie", "train")) if r["doc_id"] not in dev_ids]
    assert not dev_ids & {r["doc_id"] for r in train}
    return train, dev


def export_all(out_root: Path | None = None, make_zip: bool = True) -> dict:
    """Ecrit tous les jeux Donut (et le zip). Renvoie {nom du jeu: nombre de recus}."""
    out_root = Path(out_root) if out_root else config.DATA_DIR / "donut"
    if out_root.exists():
        shutil.rmtree(out_root)
    train, dev = split_sroie_train()
    sroie_eval = get_docs("sroie", "eval")
    clash = {r["doc_id"] for r in sroie_eval} & {r["doc_id"] for r in train + dev}
    if clash:
        raise RuntimeError(f"recus eval presents dans l'entrainement ou le dev : {sorted(clash)[:3]}")
    counts = {
        "sroie_train": _export_set(out_root / "sroie_train", train, with_truth=True),
        "sroie_dev": _export_set(out_root / "sroie_dev", dev, with_truth=True),
        "sroie_eval": _export_set(out_root / "sroie_eval", sroie_eval, with_truth=False),
    }
    for role in ("dev", "eval"):
        counts[f"cord_{role}"] = _export_set(out_root / f"cord_{role}", get_docs("cord", role), with_truth=False)
    if make_zip:
        base = config.DATA_DIR / "donut_data"
        shutil.make_archive(str(base), "zip", root_dir=out_root)
    return counts
