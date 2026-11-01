"""Etape 2 - Lance l'OCR sur les recus du jeu dev (ou eval) et met les sorties en cache.

A faire UNE fois par moteur : ensuite les regles (et le LLM, etape 3) relisent le cache.

Usage :
    python scripts/03_run_ocr.py --engine tesseract --set dev --limit 3   # essai
    python scripts/03_run_ocr.py --engine tesseract --set dev             # 200 recus
    python scripts/03_run_ocr.py --engine rapidocr  --set dev
    python scripts/03_run_ocr.py --engine tesseract --set eval            # seulement pour l'etape 5
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from docextract import config
from docextract.ocr import ENGINES, cached_ocr
from docextract.samples import available, get_docs

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", choices=list(ENGINES), default="tesseract")
    ap.add_argument("--set", dest="role", choices=["dev", "eval"], default="dev")
    ap.add_argument("--limit", type=int, default=None, help="nb max de recus par jeu (essai)")
    ap.add_argument("--force", action="store_true", help="refaire l'OCR meme si le cache existe")
    ap.add_argument("--psm", type=int, default=6, help="Tesseract : mode de segmentation de page")
    args = ap.parse_args(argv)

    datasets = available(args.role)
    if not datasets:
        print("Aucune donnee : lance d'abord  python scripts/00_prepare_data.py")
        return 1
    params = {"psm": args.psm} if args.engine == "tesseract" else {}

    print(f"Moteur {args.engine} | jeu {args.role} | cache : {config.OCR_CACHE_DIR / args.engine}")
    print("Rappel : ferme les applications inutiles avant (OCR en serie, ~2 s par recu).\n")
    all_ms: list[float] = []
    for dataset in datasets:
        docs = get_docs(dataset, args.role)
        if args.limit:
            docs = docs[:args.limit]
        iterator = tqdm(docs, desc=dataset, unit="recu") if tqdm else docs
        fresh = cached = 0
        ms: list[float] = []
        t0 = time.time()
        for rec in iterator:
            try:
                res, from_cache = cached_ocr(rec["doc_id"], config.ROOT / rec["image"],
                                             args.engine, force=args.force, **params)
            except RuntimeError as e:        # moteur absent : message d'installation
                print(f"\nERREUR : {e}")
                return 2
            cached += from_cache
            fresh += not from_cache
            ms.append(res.ms)
        all_ms += ms
        print(f"  {dataset}: {len(docs)} recus ({fresh} calcules, {cached} depuis le cache) "
              f"en {time.time() - t0:.0f} s | OCR par recu : mediane {statistics.median(ms):.0f} ms, "
              f"max {max(ms):.0f} ms")
    if all_ms:
        print(f"\nTemps OCR moyen : {statistics.fmean(all_ms):.0f} ms par recu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
