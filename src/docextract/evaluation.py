"""Evaluation commune aux 3 approches.

Pour chaque document et chaque champ, on compare la prediction et la verite terrain
APRES normalisation (normalize.py), puis on classe le cas :

    correct  : les deux existent et sont egaux
    wrong    : les deux existent mais different     -> compte en FP ET en FN
    missed   : verite presente, prediction absente   -> FN
    extra    : prediction presente, verite absente   -> FP
    empty    : les deux absents                      -> ignore

Metriques par champ :
    exact match (accuracy) = correct / nb de docs ayant une verite
    precision = correct / (correct + wrong + extra)
    rappel    = correct / (correct + wrong + missed)
    F1        = moyenne harmonique precision / rappel
    token F1  = recouvrement de mots (champs texte), pour les erreurs "presque justes"
Global :
    taux de champs corrects = somme des correct / somme des verites presentes
    docs 100 % corrects     = docs dont tous les champs evalues sont corrects
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import config
from .datasets_io import read_jsonl
from .normalize import normalize_field

FIELDS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "sroie": ("company", "date", "total"),   # CORD n'a ni fournisseur ni date
    "cord": ("total", "tax"),
}
TEXT_FIELDS = {"company", "address"}


# ------------------------------------------------------------------ comparaison
def norm(field: str, value: Any):
    """Normalise une valeur (str, date, Decimal...) pour la comparaison."""
    if value is None:
        return None
    if isinstance(value, dt.date):
        value = value.isoformat()
    return normalize_field(field, value)


def token_f1(pred: str | None, gt: str | None) -> float:
    """F1 sur les mots (deja normalises). 1.0 = identique, 0.0 = rien en commun."""
    if not pred or not gt:
        return float(pred == gt)
    p, g = Counter(pred.split()), Counter(gt.split())
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    precision, recall = common / sum(p.values()), common / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def classify(pred_norm, gt_norm) -> str:
    if gt_norm is None and pred_norm is None:
        return "empty"
    if gt_norm is None:
        return "extra"
    if pred_norm is None:
        return "missed"
    return "correct" if pred_norm == gt_norm else "wrong"


# ------------------------------------------------------------------ rapport
@dataclass
class FieldScore:
    field: str
    correct: int = 0
    wrong: int = 0
    missed: int = 0
    extra: int = 0
    token_f1_sum: float = 0.0

    @property
    def n_gt(self) -> int:
        return self.correct + self.wrong + self.missed

    @property
    def exact_match(self) -> float:
        return self.correct / self.n_gt if self.n_gt else 0.0

    @property
    def precision(self) -> float:
        d = self.correct + self.wrong + self.extra
        return self.correct / d if d else 0.0

    @property
    def recall(self) -> float:
        return self.exact_match

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mean_token_f1(self) -> float | None:
        return self.token_f1_sum / self.n_gt if self.field in TEXT_FIELDS and self.n_gt else None

    def as_dict(self) -> dict:
        d = {"field": self.field, "n_gt": self.n_gt, "correct": self.correct,
             "wrong": self.wrong, "missed": self.missed, "extra": self.extra,
             "exact_match": round(self.exact_match, 4), "precision": round(self.precision, 4),
             "recall": round(self.recall, 4), "f1": round(self.f1, 4)}
        if self.mean_token_f1 is not None:
            d["token_f1"] = round(self.mean_token_f1, 4)
        return d


@dataclass
class EvalReport:
    fields: dict[str, FieldScore]
    n_docs: int
    docs_all_correct: int
    rows: list[dict] = field(default_factory=list)   # detail par doc et champ (analyse d'erreurs)

    @property
    def field_rate(self) -> float:
        total = sum(s.n_gt for s in self.fields.values())
        return sum(s.correct for s in self.fields.values()) / total if total else 0.0

    @property
    def doc_rate(self) -> float:
        return self.docs_all_correct / self.n_docs if self.n_docs else 0.0

    def summary(self) -> dict:
        return {"n_docs": self.n_docs,
                "field_rate": round(self.field_rate, 4),
                "doc_rate": round(self.doc_rate, 4),
                "fields": {k: v.as_dict() for k, v in self.fields.items()}}

    def to_markdown(self) -> str:
        lines = ["| Champ | Exact match | Précision | Rappel | F1 | Manqués | Faux |",
                 "|---|---|---|---|---|---|---|"]
        for s in self.fields.values():
            lines.append(f"| {s.field} | {s.exact_match:.1%} | {s.precision:.1%} | "
                         f"{s.recall:.1%} | {s.f1:.1%} | {s.missed} | {s.wrong + s.extra} |")
        lines.append(f"\nChamps corrects : {self.field_rate:.1%} · "
                     f"Docs 100 % corrects : {self.doc_rate:.1%} ({self.n_docs} docs)")
        return "\n".join(lines)


def evaluate(predictions: dict[str, dict], ground_truth: dict[str, dict],
             fields: Iterable[str]) -> EvalReport:
    """predictions / ground_truth : {doc_id: {champ: valeur brute}}.
    Un doc absent des predictions = rien extrait (tous les champs manques)."""
    fields = tuple(fields)
    scores = {f: FieldScore(f) for f in fields}
    rows, all_ok = [], 0
    for doc_id, gt in ground_truth.items():
        pred = predictions.get(doc_id) or {}
        doc_ok = True
        for f in fields:
            g, p = norm(f, gt.get(f)), norm(f, pred.get(f))
            status = classify(p, g)
            s = scores[f]
            if status != "empty":
                setattr(s, status, getattr(s, status) + 1)
            if f in TEXT_FIELDS and g is not None:
                s.token_f1_sum += token_f1(p, g)
            if status in ("wrong", "missed", "extra"):
                doc_ok = False
            rows.append({"doc_id": doc_id, "field": f, "status": status,
                         "gt": gt.get(f), "pred": pred.get(f),
                         "gt_norm": None if g is None else str(g),
                         "pred_norm": None if p is None else str(p)})
        all_ok += doc_ok
    return EvalReport(scores, len(ground_truth), all_ok, rows)


# ------------------------------------------------------------------ chargement
def load_ground_truth(dataset: str, split: str = "test", sample_only: bool = True) -> dict[str, dict]:
    """{doc_id: gt} pour un jeu ; par defaut limite a l'echantillon d'evaluation fixe."""
    recs = {r["doc_id"]: r for r in read_jsonl(config.jsonl_path(dataset, split))}
    if sample_only:
        sample = json.loads(Path(config.EVAL_SAMPLE_FILE).read_text(encoding="utf-8"))
        ids = sample.get(f"{dataset}_{split}", [])
        recs = {i: recs[i] for i in ids if i in recs}
    return {doc_id: r["gt"] for doc_id, r in recs.items()}
