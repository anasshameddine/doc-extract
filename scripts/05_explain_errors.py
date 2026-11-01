"""Etape 2 - Explique les erreurs de l'approche 1 : pour chaque erreur, les lignes OCR utiles.

Lecture seule : ne modifie ni les regles ni les donnees. Reserve au jeu dev (jamais eval).

Usage :
    python scripts/05_explain_errors.py --engine gt
    python scripts/05_explain_errors.py --engine tesseract --fields total date --max 15
Ecrit outputs/rules/explication_dev_<moteur>.txt : a ouvrir, ou a m'envoyer en piece jointe.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

from docextract import config
from docextract.ocr import ENGINES, Line
from docextract.samples import available, get_docs

# Reutilise le calcul du script 04 (son nom commence par un chiffre : import par chemin)
_spec = importlib.util.spec_from_file_location("eval_rules", Path(__file__).with_name("04_eval_rules.py"))
eval_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_rules)

HINTS = {
    "date": re.compile(r"\d\s*[/\-.]\s*\d|DATE|TARIKH|\b\d{1,2}\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)",
                       re.I),
    "total": re.compile(r"TOTAL|AMOUNT|DUE|CASH|CHANGE|ROUND|PAY|JUMLAH|TUNAI|BAYAR|NET|BALANCE", re.I),
    "tax": re.compile(r"GST|SST|TAX|VAT|PPN|PB\s*1|PAJAK|SERVICE|SERVIS|%", re.I),
}
ALL_FIELDS = ("company", "date", "total", "tax")


def relevant_lines(field: str, lines: list[Line]) -> list[tuple[int, str]]:
    """Lignes (numero, texte) a regarder pour comprendre une erreur sur `field`."""
    if field == "company":
        picked = list(enumerate(lines[:6]))
    else:
        picked = [(i, l) for i, l in enumerate(lines) if HINTS[field].search(l.text)][:10]
        if not picked:
            picked = list(enumerate(lines[:6]))          # rien de typique : on montre le debut
    return [(i + 1, l.text.strip()) for i, l in picked]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", choices=[*ENGINES, "gt"], default="gt")
    ap.add_argument("--fields", nargs="+", choices=ALL_FIELDS, default=list(ALL_FIELDS))
    ap.add_argument("--max", type=int, default=25, help="erreurs montrees par jeu et par champ")
    args = ap.parse_args(argv)

    datasets = available("dev")
    if not datasets:
        print("Aucune donnee : lance d'abord  python scripts/00_prepare_data.py")
        return 1

    blocks: list[str] = []
    for dataset in datasets:
        res = eval_rules.run_dataset(dataset, "dev", args.engine)
        if res is None:
            print(f"[{dataset}] aucun OCR en cache pour {args.engine}")
            continue
        docs = {r["doc_id"]: r for r in get_docs(dataset, "dev")}
        shown: dict[str, int] = {}
        counts: dict[str, int] = {}
        for row in res["report"].rows:
            field, status = row["field"], row["status"]
            if field not in args.fields or status in ("correct", "empty"):
                continue
            counts[field] = counts.get(field, 0) + 1
            if shown.get(field, 0) >= args.max:
                continue
            shown[field] = shown.get(field, 0) + 1
            lines, _ = eval_rules.lines_for(docs[row["doc_id"]], args.engine)
            info = res["rule_of"][row["doc_id"]]
            block = [f"[{dataset}] {row['doc_id']} | champ={field} | statut={status}",
                     f"  verite     : {row['gt']}",
                     f"  prediction : {row['pred']}   (regle : {info[field]})",
                     "  lignes OCR :"]
            block += [f"    {n:>3}| {text[:110]}" for n, text in relevant_lines(field, lines)]
            blocks.append("\n".join(block))
        summary = ", ".join(f"{f} {n}" for f, n in counts.items()) or "aucune"
        print(f"[{dataset}] erreurs : {summary}")

    out = config.RULES_OUTPUT_DIR / f"explication_dev_{args.engine}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"\n{len(blocks)} erreur(s) detaillee(s) -> {out.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
