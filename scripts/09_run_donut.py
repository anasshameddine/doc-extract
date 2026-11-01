"""Etape 4d - Fait tourner Donut sur CE PC : predictions + temps + memoire GPU mesures en local.

Usage :
    python scripts/09_run_donut.py --set dev
    python scripts/09_run_donut.py --set dev --dataset sroie --limit 10      # test rapide
    python scripts/09_run_donut.py --set eval --final                        # score final (etape 5)
Ecrit data/donut_preds_local/predictions_<jeu>_<role>.json : a noter avec
    python scripts/08_eval_donut.py --set dev --preds-dir data/donut_preds_local
Prerequis : requirements-donut.txt installe ; notre modele dans data/donut_models/sroie_best (copie de Drive).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from docextract import config
from docextract.donut import save_predictions
from docextract.samples import DATASETS, get_docs


def run_dataset(dataset: str, role: str, source: str, device: str | None, limit: int) -> tuple[Path, dict]:
    from docextract.donut_infer import DonutExtractor
    extractor = DonutExtractor(source, dataset, device=device)
    extractor.warm_up()
    if extractor.device == "cuda":
        extractor._torch.cuda.reset_peak_memory_stats()
    docs = get_docs(dataset, role)
    if limit:
        docs = docs[:limit]
    results = {}
    for rec in tqdm(docs, desc=dataset, unit="recu"):
        results[rec["doc_id"]] = extractor.run(Image.open(config.ROOT / rec["image"]))
    ms = [v["ms"] for v in results.values()]
    meta = {"model": source, "set": role, "gpu": extractor.gpu, "device": extractor.device, "amp_fp16": extractor.amp,
            "load_seconds": extractor.load_seconds, "peak_vram_gb": extractor.peak_vram_gb(), "local": True,
            "ms_median": round(statistics.median(ms), 1), "ms_total_median": round(
                statistics.median(v["ms_total"] for v in results.values()), 1)}
    out = config.DONUT_LOCAL_PRED_DIR / f"predictions_{dataset}_{role}.json"
    save_predictions(results, out, meta)
    return out, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", dest="role", choices=["dev", "eval"], default="dev")
    ap.add_argument("--final", action="store_true", help="obligatoire pour le jeu eval")
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--sroie-model", default=str(config.DONUT_MODEL_DIR / "sroie_best"))
    ap.add_argument("--cord-model", default=config.DONUT_CORD_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--limit", type=int, default=0, help="n premiers recus par jeu (test rapide)")
    args = ap.parse_args(argv)
    if args.role == "eval" and not args.final:
        print("Le jeu eval est reserve au score final (etape 5). Ajoute --final si c'est bien le moment.")
        return 3
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("torch / transformers absents. Installe-les :\n"
              "  pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
              "  pip install -r requirements-donut.txt")
        return 2
    sources = {"sroie": args.sroie_model, "cord": args.cord_model}
    wanted = DATASETS if args.dataset == "all" else (args.dataset,)
    if "sroie" in wanted and not Path(args.sroie_model).exists():
        print(f"Modele introuvable : {args.sroie_model}\n"
              "Copie le dossier donut_sroie/best de Google Drive vers data/donut_models/sroie_best")
        return 1
    print("Rappel : ferme les applications inutiles avant (Donut utilise le GPU).")
    t0 = time.time()
    for dataset in wanted:
        try:
            out, meta = run_dataset(dataset, args.role, sources[dataset], args.device, args.limit)
        except FileNotFoundError as e:
            print(e)
            return 1
        print(f"[{dataset}] modele charge en {meta['load_seconds']} s | {meta['gpu']} | mediane {meta['ms_median']:.0f} ms "
              f"(generation) / {meta['ms_total_median']:.0f} ms (image comprise) | pic memoire GPU {meta['peak_vram_gb']} Go")
        print(f"         -> {out.relative_to(config.ROOT)}")
    print(f"\nTermine en {(time.time() - t0) / 60:.1f} min. Note-les :  python scripts/08_eval_donut.py "
          f"--set {args.role} --preds-dir data/donut_preds_local" + (" --final" if args.role == "eval" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
