"""Etape 4c - Approche 3 (Donut) : note les predictions faites sur Colab, avec les memes metriques que les approches 1 et 2.

Usage :
    python scripts/08_eval_donut.py --set dev
    python scripts/08_eval_donut.py --set eval --final     # score final : 1 seule fois (etape 5)
Lit data/donut_preds/predictions_<sroie|cord>_<dev|eval>.json (telecharges depuis Google Drive).
Ecrit dans outputs/donut/ : CSV des erreurs, resume JSON, fichier d'explication.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from docextract import config
from docextract.donut import fields_from_prediction, load_predictions, pred_path, timing, unreadable
from docextract.evaluation import FIELDS_BY_DATASET, evaluate
from docextract.samples import DATASETS, get_docs, split_for

MODELS = {"sroie": "donut-base fine-tune sur SROIE (notre entrainement)",
          "cord": "donut-base-finetuned-cord-v2 (checkpoint public)"}


def run_dataset(dataset: str, role: str, pred_dir=None) -> dict | None:
    path = pred_path(dataset, role, pred_dir)
    if not path.exists():
        return None
    meta, raw = load_predictions(path)
    docs = get_docs(dataset, role)
    preds = {d["doc_id"]: fields_from_prediction(dataset, raw[d["doc_id"]]["parsed"]) for d in docs if d["doc_id"] in raw}
    gts = {d["doc_id"]: d["gt"] for d in docs}
    missing = [d["doc_id"] for d in docs if d["doc_id"] not in raw]
    report = evaluate(preds, gts, FIELDS_BY_DATASET[dataset])
    return {"dataset": dataset, "split": split_for(dataset, role), "report": report, "meta": meta, "raw": raw,
            "preds": preds, "missing": missing, "n": len(docs), "timing": timing(raw), "unreadable": unreadable(raw)}


def write_outputs(res: dict, role: str, max_shown: int) -> tuple[str, str, str]:
    out_dir = config.DONUT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{role}_{res['dataset']}{'_local' if res['meta'].get('local') else ''}"
    report, raw = res["report"], res["raw"]

    csv_path = out_dir / f"{stem}_erreurs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "champ", "statut", "verite", "prediction", "sequence_donut"])
        for row in report.rows:
            if row["status"] in ("correct", "empty"):
                continue
            w.writerow([row["doc_id"], row["field"], row["status"], row["gt"], row["pred"],
                        raw.get(row["doc_id"], {}).get("sequence", "")[:300]])

    blocks, shown = [], {}
    for row in report.rows:
        field, status = row["field"], row["status"]
        if status in ("correct", "empty") or shown.get(field, 0) >= max_shown:
            continue
        shown[field] = shown.get(field, 0) + 1
        seq = raw.get(row["doc_id"], {}).get("sequence", "")
        blocks.append("\n".join([f"[{res['dataset']}] {row['doc_id']} | champ={field} | statut={status}",
                                 f"  verite     : {row['gt']}", f"  prediction : {row['pred']}",
                                 f"  sequence Donut : {seq[:260]}"]))
    txt_path = out_dir / f"explication_{stem}.txt"
    txt_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    summary = report.summary()
    summary.update({"modele": MODELS[res["dataset"]], "set": role, "predictions_meta": res["meta"],
                    "temps": res["timing"], "sequences_illisibles": res["unreadable"], "docs_sans_prediction": res["missing"]})
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    rel = lambda p: str(p.relative_to(config.ROOT))
    return rel(csv_path), rel(json_path), rel(txt_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", dest="role", choices=["dev", "eval"], default="dev")
    ap.add_argument("--final", action="store_true", help="obligatoire pour le jeu eval")
    ap.add_argument("--preds-dir", default=None,
                    help="dossier des predictions (defaut : data/donut_preds = Colab ; data/donut_preds_local = ce PC)")
    ap.add_argument("--max", type=int, default=25, help="erreurs detaillees par jeu et par champ")
    args = ap.parse_args(argv)
    if args.role == "eval" and not args.final:
        print("Le jeu eval est reserve au score final (etape 5). Ajoute --final si c'est bien le moment.")
        return 3

    status, found = 0, False
    for dataset in DATASETS:
        try:
            res = run_dataset(dataset, args.role, args.preds_dir)
        except FileNotFoundError as e:
            print(e)
            return 1
        if res is None:
            print(f"[{dataset}] pas de fichier {pred_path(dataset, args.role, args.preds_dir).name} dans "
                  f"{args.preds_dir or config.DONUT_PRED_DIR.relative_to(config.ROOT)} : copie-le depuis Google Drive "
                  "(donut_preds) ou lance scripts/09_run_donut.py.")
            status = 1
            continue
        found = True
        report, t = res["report"], res["timing"]
        print(f"\n=== {dataset} ({res['split']}) | Donut : {MODELS[dataset]} | jeu {args.role} | {report.n_docs} recus ===")
        if res["missing"]:
            print(f"  ATTENTION : {len(res['missing'])} recu(s) sans prediction (comptes comme manques)")
            if args.role == "eval":
                status = 1
        print(report.to_markdown())
        if t:
            gpu = res["meta"].get("gpu", "GPU inconnu")
            where = "sur ton PC" if res["meta"].get("local") else "pas sur ton PC"
            print(f"\n  Temps Donut par recu : moyenne {t['ms_moyen']:.0f} ms, mediane {t['ms_median']:.0f} ms, "
                  f"max {t['ms_max']:.0f} ms  (mesure sur {gpu}, {where})")
            if res["meta"].get("peak_vram_gb") is not None:
                print(f"  Pic de memoire GPU : {res['meta']['peak_vram_gb']} Go")
        print(f"  Sequences illisibles : {res['unreadable']}")
        csv_path, json_path, txt_path = write_outputs(res, args.role, args.max)
        print(f"\n  erreurs  -> {csv_path}\n  resume   -> {json_path}\n  analyse  -> {txt_path}")
    return status if found else 1


if __name__ == "__main__":
    sys.exit(main())
