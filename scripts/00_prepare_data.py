"""Etape 0 - Telecharge SROIE (train+test) et CORD-v2 (test + validation), exporte images + JSONL,
puis fixe l'echantillon d'evaluation (100 recus SROIE test + 100 CORD test).

Usage :
    python scripts/00_prepare_data.py              # tout
    python scripts/00_prepare_data.py --skip-cord  # SROIE seulement
    python scripts/00_prepare_data.py --skip-sroie # CORD seulement (ajoute le split validation)
    python scripts/00_prepare_data.py --limit 5    # essai rapide
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # avertissement Windows

from docextract import config
from docextract.datasets_io import export_split, make_eval_sample, read_jsonl

# (jeu, depot HF, splits)
SOURCES = [
    ("sroie", config.SROIE_REPO, ["train", "test"]),
    ("cord", config.CORD_REPO, ["test", "validation"]),  # test = evaluation, validation = dev (regles, prompt)
]


def load_split(repo: str, split: str, limit: int | None):
    from datasets import load_dataset  # import tardif : les tests le remplacent

    ds = load_dataset(repo, split=split, cache_dir=str(config.HF_CACHE_DIR))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-cord", action="store_true", help="ne pas telecharger CORD")
    ap.add_argument("--skip-sroie", action="store_true", help="ne pas retelecharger SROIE")
    ap.add_argument("--limit", type=int, default=None, help="nb max de docs par split (essai)")
    args = ap.parse_args(argv)

    counts: dict[str, int] = {}
    for dataset, repo, splits in SOURCES:
        if (dataset == "cord" and args.skip_cord) or (dataset == "sroie" and args.skip_sroie):
            continue
        for split in splits:
            t0 = time.time()
            print(f"[{dataset}/{split}] telechargement depuis {repo} ...", flush=True)
            ds = load_split(repo, split, args.limit)
            n = export_split(ds, dataset, split, root=config.ROOT, raw_dir=config.RAW_DIR,
                             jsonl_file=config.jsonl_path(dataset, split))
            counts[f"{dataset}_{split}"] = n
            print(f"[{dataset}/{split}] {n} docs exportes en {time.time() - t0:.0f} s")

    # Echantillon d'evaluation fixe
    sample: dict[str, list[str]] = {}
    for key in ("sroie_test", "cord_test"):
        path = config.PROCESSED_DIR / f"{key}.jsonl"
        if path.exists():
            ids = [r["doc_id"] for r in read_jsonl(path)]
            sample[key] = make_eval_sample(ids, config.EVAL_SAMPLE_SIZE, config.EVAL_SEED)
    config.EVAL_SAMPLE_FILE.write_text(
        json.dumps({"seed": config.EVAL_SEED, **sample}, indent=2), encoding="utf-8")

    print("\n=== RESUME ===")
    for k, v in counts.items():
        print(f"  {k:<12} {v:>4} docs")
    for k, v in sample.items():
        print(f"  echantillon eval {k}: {len(v)} docs")
    print(f"  -> {config.EVAL_SAMPLE_FILE.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
