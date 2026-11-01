"""Quels recus pour quoi ? Separation stricte entre DEV et EVAL.

    dev  : SROIE = 100 recus tires dans train (graine DEV_SEED), CORD = split validation
           -> sert a regler les regles, le prompt du LLM, a choisir le moteur OCR.
    eval : SROIE = 100 recus tires dans test (graine EVAL_SEED), CORD = split test
           -> sert UNIQUEMENT au score final (etape 5). On ne regle rien dessus.

Regler sur le jeu d'evaluation donnerait un score gonfle : il ne mesurerait plus la
capacite a generaliser a de nouveaux recus, mais la memorisation de ces 100 recus.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .datasets_io import make_eval_sample, read_jsonl

ROLES = ("dev", "eval")
DATASETS = ("sroie", "cord")


def split_for(dataset: str, role: str) -> str:
    return {("sroie", "dev"): "train", ("sroie", "eval"): "test",
            ("cord", "dev"): "validation", ("cord", "eval"): "test"}[(dataset, role)]


def _sample_ids(dataset: str, role: str) -> list[str] | None:
    """Ids imposes par un fichier d'echantillon, ou None = tout le split."""
    if dataset == "cord":
        return None
    split = split_for(dataset, role)
    path = Path(config.DEV_SAMPLE_FILE if role == "dev" else config.EVAL_SAMPLE_FILE)
    key = f"{dataset}_{split}"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if key in data:
            return data[key]
    if role == "eval":
        raise FileNotFoundError("eval_sample.json absent : lance scripts/00_prepare_data.py")
    # dev : tirage deterministe, ecrit une fois pour garder le meme echantillon ensuite
    ids = [r["doc_id"] for r in read_jsonl(config.jsonl_path(dataset, split))]
    chosen = make_eval_sample(ids, config.DEV_SAMPLE_SIZE, config.DEV_SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update({"seed": config.DEV_SEED, key: chosen})
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return chosen


def get_docs(dataset: str, role: str) -> list[dict]:
    """Enregistrements JSONL du jeu `dataset` pour le role `dev` ou `eval`."""
    if role not in ROLES:
        raise ValueError(f"role inconnu : {role!r}")
    path = config.jsonl_path(dataset, split_for(dataset, role))
    if not path.exists():
        hint = "--skip-sroie" if dataset == "cord" else ""
        raise FileNotFoundError(f"{path.name} absent : lance  python scripts/00_prepare_data.py {hint}".strip())
    recs = list(read_jsonl(path))
    ids = _sample_ids(dataset, role)
    if ids is None:
        return recs
    wanted = set(ids)
    return [r for r in recs if r["doc_id"] in wanted]


def available(role: str) -> list[str]:
    """Jeux dont les donnees existent pour ce role."""
    return [d for d in DATASETS if config.jsonl_path(d, split_for(d, role)).exists()]
