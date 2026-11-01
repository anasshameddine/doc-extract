"""Etape 2 - Approche 1 (OCR + regles) : extrait les champs et calcule les scores.

Usage :
    python scripts/04_eval_rules.py --engine gt        --set dev   # OCR parfait = borne haute des regles
    python scripts/04_eval_rules.py --engine tesseract --set dev
    python scripts/04_eval_rules.py --engine rapidocr  --set dev
    python scripts/04_eval_rules.py --engine tesseract --set eval --final   # score final, 1 seule fois

Le jeu `eval` est reserve au score final : on regle les regles sur `dev`, jamais sur `eval`.
Ecrit dans outputs/rules/ : un CSV des erreurs (avec la regle fautive) et un resume JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict

from docextract import config
from docextract.evaluation import FIELDS_BY_DATASET, evaluate
from docextract.ocr import ENGINES, Line, build_lines, load_cached, segments_from_gt
from docextract.rules import extract_rules
from docextract.samples import available, get_docs, split_for


def lines_for(rec: dict, engine: str) -> tuple[list[Line], float] | None:
    """(lignes, temps OCR en ms) pour un recu, ou None si pas d'OCR en cache."""
    if engine == "gt":
        return build_lines(segments_from_gt(rec["words"], rec["bboxes"])), 0.0
    cached = load_cached(rec["doc_id"], engine)
    return (cached.lines(), cached.ms) if cached else None


def run_dataset(dataset: str, role: str, engine: str) -> dict | None:
    docs = get_docs(dataset, role)
    preds, gts, rule_of, heads, ocr_ms, rules_ms, missing = {}, {}, {}, {}, [], [], 0
    for rec in docs:
        got = lines_for(rec, engine)
        if got is None:
            missing += 1
            continue
        lines, ms = got
        out = extract_rules(lines)
        f = out.fields
        preds[rec["doc_id"]] = {"company": f.company, "date": f.date, "total": f.total, "tax": f.tax}
        gts[rec["doc_id"]] = rec["gt"]
        rule_of[rec["doc_id"]] = {k: (v.rule if v else "none") for k, v in out.found.items()}
        heads[rec["doc_id"]] = " | ".join(line.text for line in lines[:3])[:160]
        evidence = {k: out.evidence(k) for k in out.found}
        rule_of[rec["doc_id"]]["_evidence"] = evidence
        ocr_ms.append(ms)
        rules_ms.append(out.ms)
    if not gts:
        return None

    fields = FIELDS_BY_DATASET[dataset]
    report = evaluate(preds, gts, fields)

    # precision de chaque regle : quelle regle se trompe ?
    by_rule: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in report.rows:
        if row["status"] == "empty":
            continue
        rule = rule_of[row["doc_id"]][row["field"]]
        stat = by_rule[row["field"]][rule]
        stat[0] += row["status"] == "correct"
        stat[1] += 1

    return {"dataset": dataset, "split": split_for(dataset, role), "report": report, "missing": missing,
            "preds": preds, "rule_of": rule_of, "heads": heads, "by_rule": by_rule,
            "ocr_ms": ocr_ms, "rules_ms": rules_ms}


def write_outputs(res: dict, role: str, engine: str) -> tuple[str, str]:
    out_dir = config.RULES_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{role}_{engine}_{res['dataset']}"
    report = res["report"]

    csv_path = out_dir / f"{stem}_erreurs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "champ", "statut", "verite", "prediction", "regle", "ligne_source", "debut_ocr"])
        for row in report.rows:
            if row["status"] in ("correct", "empty"):
                continue
            info = res["rule_of"][row["doc_id"]]
            w.writerow([row["doc_id"], row["field"], row["status"], row["gt"], row["pred"],
                        info[row["field"]], info["_evidence"][row["field"]], res["heads"][row["doc_id"]]])

    summary = report.summary()
    summary.update({
        "engine": engine, "set": role, "docs_sans_ocr": res["missing"],
        "ocr_ms_moyen": round(statistics.fmean(res["ocr_ms"]), 1),
        "regles_ms_moyen": round(statistics.fmean(res["rules_ms"]), 3),
        "par_regle": {k: {r: {"corrects": v[0], "total": v[1]} for r, v in d.items()}
                      for k, d in res["by_rule"].items()}})
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(csv_path.relative_to(config.ROOT)), str(json_path.relative_to(config.ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", choices=[*ENGINES, "gt"], default="tesseract")
    ap.add_argument("--set", dest="role", choices=["dev", "eval"], default="dev")
    ap.add_argument("--final", action="store_true", help="obligatoire pour le jeu eval")
    args = ap.parse_args(argv)

    if args.role == "eval" and not args.final:
        print("Le jeu eval est reserve au score final (etape 5) : on regle les regles sur dev.\n"
              "Si c'est bien le score final, relance avec --final.")
        return 3
    datasets = available(args.role)
    if not datasets:
        print("Aucune donnee : lance d'abord  python scripts/00_prepare_data.py")
        return 1

    status = 0
    for dataset in datasets:
        try:
            res = run_dataset(dataset, args.role, args.engine)
        except FileNotFoundError as e:
            print(f"\n[{dataset}] {e}")
            status = 1
            continue
        if res is None:
            print(f"\n[{dataset}] aucun OCR en cache pour {args.engine} : lance "
                  f"python scripts/03_run_ocr.py --engine {args.engine} --set {args.role}")
            status = 1
            continue
        report = res["report"]
        print(f"\n=== {dataset} ({res['split']}) | moteur {args.engine} | jeu {args.role} | {report.n_docs} recus ===")
        if res["missing"]:
            print(f"  ATTENTION : {res['missing']} recu(s) sans OCR en cache (ignores)")
            if args.role == "eval":
                status = 1
        print(report.to_markdown())
        if args.engine != "gt":
            print(f"\n  Temps : OCR {statistics.fmean(res['ocr_ms']):.0f} ms + regles "
                  f"{statistics.fmean(res['rules_ms']):.2f} ms par recu")
        print("\n  Par regle (corrects / total) :")
        for field, rules in res["by_rule"].items():
            parts = ", ".join(f"{r} {v[0]}/{v[1]}" for r, v in sorted(rules.items()))
            print(f"    {field:<8} {parts}")
        csv_path, json_path = write_outputs(res, args.role, args.engine)
        print(f"\n  erreurs -> {csv_path}\n  resume  -> {json_path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
