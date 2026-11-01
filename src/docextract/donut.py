"""Approche 3 : Document AI (Donut). Lit les predictions produites sur Colab et les met au format commun.

Donut ecrit un JSON, sans OCR :
    SROIE (notre fine-tuning) : {"company": ..., "date": ..., "total": ...}
    CORD  (checkpoint public) : {"menu": [...], "sub_total": {"tax_price": ...}, "total": {"total_price": ...}}
Les valeurs restent du texte brut : la normalisation et les metriques sont celles des approches 1 et 2.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from . import config

FIELDS = ("company", "date", "total", "tax")


def pred_path(dataset: str, role: str, pred_dir: Path | None = None) -> Path:
    return Path(pred_dir or config.DONUT_PRED_DIR) / f"predictions_{dataset}_{role}.json"


def load_predictions(path: Path) -> tuple[dict, dict]:
    """(meta, {doc_id: {parsed, sequence, ms}}). FileNotFoundError si le fichier n'existe pas."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("meta", {}), data["predictions"]


def _text(value: Any) -> str | None:
    """Une valeur Donut -> texte, ou None. Une liste -> son premier element non vide ; un dict n'est pas un champ."""
    if isinstance(value, list):
        value = next((v for v in value if _text(v)), None)
    if value is None or isinstance(value, dict):
        return None
    text = " ".join(str(value).split())
    return text or None


def _first_dict(value: Any) -> dict:
    if isinstance(value, list):
        value = next((v for v in value if isinstance(v, dict)), {})
    return value if isinstance(value, dict) else {}


def fields_from_prediction(dataset: str, parsed: dict) -> dict[str, str | None]:
    """JSON de Donut -> {company, date, total, tax} (None = non trouve). Sequence illisible -> tout a None."""
    out: dict[str, str | None] = {k: None for k in FIELDS}
    if not isinstance(parsed, dict):
        return out
    if dataset == "sroie":
        for key in ("company", "date", "total"):
            out[key] = _text(parsed.get(key))
    else:
        out["total"] = _text(_first_dict(parsed.get("total")).get("total_price"))
        out["tax"] = _text(_first_dict(parsed.get("sub_total")).get("tax_price"))
    return out


def save_predictions(results: dict, path, meta: dict | None = None) -> None:
    """Meme format que les fichiers ecrits par le notebook Colab : {'meta': ..., 'predictions': {doc_id: ...}}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "predictions": results}, indent=1, ensure_ascii=False), encoding="utf-8")


def timing(preds: dict) -> dict:
    ms = [v["ms"] for v in preds.values() if "ms" in v]
    if not ms:
        return {}
    out = {"ms_moyen": round(statistics.fmean(ms), 0), "ms_median": round(statistics.median(ms), 0),
           "ms_max": round(max(ms), 0)}
    total = [v["ms_total"] for v in preds.values() if "ms_total" in v]          # present si mesure en local
    if total:
        out["ms_total_moyen"] = round(statistics.fmean(total), 0)             # image -> JSON, preparation comprise
    return out


def unreadable(preds: dict) -> int:
    """Nombre de recus dont la sequence n'a pas pu etre lue comme du JSON de champs."""
    return sum(1 for v in preds.values() if "_error" in v["parsed"] or list(v["parsed"]) == ["text_sequence"])
