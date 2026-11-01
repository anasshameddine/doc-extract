"""Etape 1 - Verifie la normalisation sur TOUTE la verite terrain, et teste l'evaluation.

1. Pour chaque jeu et chaque champ : combien de valeurs brutes se normalisent bien,
   et la liste de celles qui echouent (a corriger avant l'etape 2).
2. Controle de l'evaluation : verite terrain comparee a elle-meme -> doit donner 100 %.

Usage :
    python scripts/02_check_ground_truth.py
"""
from __future__ import annotations

import csv
import json
import sys

from docextract import config
from docextract.datasets_io import read_jsonl
from docextract.evaluation import FIELDS_BY_DATASET, evaluate, load_ground_truth, norm

MAX_EXAMPLES = 10
PLACEHOLDERS = {"-", "--", "\u2014", "n/a", "na"}   # CORD ecrit "-" quand il n'y a pas de valeur


def check_file(name: str, dataset: str) -> tuple[dict, list[dict]]:
    recs = list(read_jsonl(config.PROCESSED_DIR / f"{name}.jsonl"))
    print(f"\n=== {name} : {len(recs)} docs ===")
    stats, failures = {}, []
    for f in FIELDS_BY_DATASET[dataset]:
        raw = [(r["doc_id"], r["gt"].get(f)) for r in recs
               if r["gt"].get(f) and str(r["gt"][f]).strip().lower() not in PLACEHOLDERS]
        normed = [(d, v, norm(f, v)) for d, v in raw]
        bad = [(d, v) for d, v, n in normed if n is None]
        ok = len(raw) - len(bad)
        stats[f] = {"present": len(raw), "normalized": ok, "failed": len(bad)}
        rate = ok / len(raw) if raw else 0.0
        print(f"  {f:<8} normalises {ok:>4}/{len(raw):<4} ({rate:.1%})")
        values = [n for _, _, n in normed if n is not None]
        if values and f in ("date", "total", "tax"):
            print(f"           min {min(values)}   max {max(values)}")
        for d, v in bad[:MAX_EXAMPLES]:
            print(f"           ECHEC  {d}  {v!r}")
        failures += [{"file": name, "doc_id": d, "field": f, "value": v} for d, v in bad]
    return stats, failures


def main() -> int:
    files = [("sroie_train", "sroie"), ("sroie_test", "sroie"), ("cord_test", "cord")]
    files = [(n, d) for n, d in files if (config.PROCESSED_DIR / f"{n}.jsonl").exists()]
    if not files:
        print("Aucune donnee : lance d'abord  python scripts/00_prepare_data.py")
        return 1

    summary, all_failures = {}, []
    for name, dataset in files:
        summary[name], fails = check_file(name, dataset)
        all_failures += fails

    print("\n=== Controle de l'evaluation (verite vs verite, echantillon fixe) ===")
    sanity_ok = True
    for dataset in ("sroie", "cord"):
        if not config.jsonl_path(dataset, "test").exists():
            continue
        gt = load_ground_truth(dataset)
        rep = evaluate(gt, gt, FIELDS_BY_DATASET[dataset])
        print(f"  {dataset}: {rep.n_docs} docs, champs corrects {rep.field_rate:.1%}")
        sanity_ok &= rep.field_rate == 1.0 or rep.n_docs == 0

    out_dir = config.OUTPUTS_DIR / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "normalisation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "echecs_normalisation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "doc_id", "field", "value"])
        w.writeheader()
        w.writerows(all_failures)
    print(f"\n  {len(all_failures)} echec(s) -> {(out_dir / 'echecs_normalisation.csv').relative_to(config.ROOT)}")
    print("  controle evaluation :", "OK" if sanity_ok else "PROBLEME")
    return 0 if sanity_ok else 2


if __name__ == "__main__":
    sys.exit(main())
