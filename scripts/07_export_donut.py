"""Etape 4a - Prepare les donnees du fine-tuning Donut et les zippe pour Google Colab.

Usage :
    python scripts/07_export_donut.py
Ecrit data/donut/ (images + metadata.jsonl) et data/donut_data.zip : a deposer sur Google Drive.
Rien n'est entraine ici : l'entrainement se fait sur Colab (etape 4b).
"""
from __future__ import annotations

import argparse
import sys

from docextract import config
from docextract.donut_data import export_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-zip", action="store_true", help="ne cree pas data/donut_data.zip")
    args = ap.parse_args(argv)
    try:
        counts = export_all(make_zip=not args.no_zip)
    except FileNotFoundError as e:
        print(f"{e}\nLance d'abord :  python scripts/00_prepare_data.py")
        return 1
    print("Jeux exportes :")
    for name, n in counts.items():
        print(f"  {name:<12} {n:>4} recus")
    print(f"\nEntrainement : sroie_train ({counts['sroie_train']}) | suivi : sroie_dev | "
          "eval : images seules (aucune verite dans le zip)")
    if not args.no_zip:
        zip_path = config.DATA_DIR / "donut_data.zip"
        print(f"Zip pour Colab : {zip_path.relative_to(config.ROOT)}  ({zip_path.stat().st_size / 1e6:.0f} Mo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
