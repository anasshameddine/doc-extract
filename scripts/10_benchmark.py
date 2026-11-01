"""Etape 5 - Tableau comparatif des 3 approches : precision par champ, temps par recu, memoire.

Lit les resumes JSON ecrits par les scripts 04 (regles), 06 (LLM) et 08 (Donut). Ne calcule aucun score lui-meme.

Usage :
    python scripts/10_benchmark.py --set dev          # repetition a blanc sur le jeu dev (aucun risque)
    python scripts/10_benchmark.py --set eval --final # tableau FINAL, a faire une fois les 4 evaluations eval terminees
Ecrit outputs/benchmark/benchmark_<jeu>.md et .csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from docextract import config
from docextract.evaluation import FIELDS_BY_DATASET
from docextract.llm import safe_name

DATASETS = ("sroie", "cord")


def method_files(role: str, engine: str, llm_models: list[str]) -> list[dict]:
    """Les approches comparees et le resume JSON de chacune, par jeu."""
    methods = [{"key": "rules", "label": "1 · OCR + règles", "hardware": f"CPU ({engine})",
                "path": lambda ds: config.RULES_OUTPUT_DIR / f"{role}_{engine}_{ds}.json"}]
    for model in llm_models:
        methods.append({"key": f"llm:{model}", "label": f"2 · OCR + LLM {model} + vérif", "hardware": "GPU (Ollama)",
                        "path": lambda ds, m=model: config.LLM_OUTPUT_DIR / f"{role}_{safe_name(m)}_{engine}_verif_{ds}.json"})
    methods.append({"key": "donut", "label": "3 · Donut (Document AI)", "hardware": "GPU (local)",
                    "path": lambda ds: config.DONUT_OUTPUT_DIR / f"{role}_{ds}_local.json"})
    return methods


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def time_ms(method_key: str, data: dict, rules_data: dict | None) -> float | None:
    """Temps moyen d'un recu, de l'image au JSON (OCR compris pour les approches 1 et 2)."""
    if method_key == "rules":
        return data["ocr_ms_moyen"] + data["regles_ms_moyen"]
    if method_key.startswith("llm:"):
        return (rules_data["ocr_ms_moyen"] if rules_data else 0.0) + data["temps"]["ms_moyen"]
    t = data.get("temps", {})
    return t.get("ms_total_moyen", t.get("ms_moyen"))


def gpu_memory(method_key: str, data: dict) -> str:
    if method_key == "rules":
        return "aucune"
    if method_key.startswith("llm:"):
        mem = data.get("memoire_ollama")
        return f"{mem['vram_gb']} Go" if mem else "?"
    peak = data.get("predictions_meta", {}).get("peak_vram_gb")
    return f"{peak} Go" if peak is not None else "?"


def collect(role: str, engine: str, llm_models: list[str]) -> tuple[list[dict], list[Path]]:
    """Lignes du tableau + chemins manquants."""
    methods, missing, rows = method_files(role, engine, llm_models), [], []
    for ds in DATASETS:
        rules_data = load(methods[0]["path"](ds))
        for m in methods:
            path = m["path"](ds)
            data = load(path)
            if data is None:
                missing.append(path)
                continue
            rows.append({"dataset": ds, "method": m, "data": data, "path": path, "ms": time_ms(m["key"], data, rules_data),
                         "vram": gpu_memory(m["key"], data)})
    return rows, missing


def loose_company(row: dict) -> float | None:
    """DIAGNOSTIC : part des entreprises justes si on ignore espaces et ponctuation (OCR sans espaces).

    Calcule sans relancer aucun modele : entreprises deja justes + erreurs du CSV qui ne different que par la
    ponctuation/les espaces. Ce n'est pas la mesure officielle (exact match).
    """
    stats = row["data"]["fields"].get("company")
    csv_path = row["path"].with_name(row["path"].stem + "_erreurs.csv")
    if not stats or not csv_path.exists():
        return None
    squash = lambda t: re.sub(r"[^A-Z0-9]", "", (t or "").upper())
    with csv_path.open(encoding="utf-8") as f:
        near = sum(1 for e in csv.DictReader(f)
                   if e["champ"] == "company" and e["statut"] == "wrong" and squash(e["verite"]) == squash(e["prediction"]))
    return (stats["correct"] + near) / stats["n_gt"] if stats["n_gt"] else None


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def markdown(role: str, rows: list[dict], llm_models: list[str], engine: str) -> str:
    out = [f"# Comparaison des 3 approches · jeu {role}\n"]
    for ds in DATASETS:
        sub = [r for r in rows if r["dataset"] == ds]
        if not sub:
            continue
        n = sub[0]["data"]["n_docs"]
        out.append(f"## {ds.upper()} ({n} reçus) · exact match par champ, en %\n")
        out.append("| Champ | " + " | ".join(f"{r['method']['label']}" + (f" (n={r['data']['n_docs']})" if r["data"]["n_docs"] != n else "")
                                          for r in sub) + " |")
        out.append("|---|" + "---|" * len(sub))
        for field in FIELDS_BY_DATASET[ds]:
            vals = [r["data"]["fields"][field]["exact_match"] for r in sub]
            best = max(vals)
            cells = [f"**{pct(v)}**" if v == best else pct(v) for v in vals]
            out.append(f"| {field} | " + " | ".join(cells) + " |")
        if ds == "sroie":
            loose = [loose_company(r) for r in sub]
            if all(v is not None for v in loose):
                best = max(loose)
                out.append("| *company, sans espaces (diagnostic)* | "
                           + " | ".join(f"**{pct(v)}**" if v == best else pct(v) for v in loose) + " |")
        rates = [r["data"]["field_rate"] for r in sub]
        docs = [r["data"]["doc_rate"] for r in sub]
        out.append("| *champs corrects* | " + " | ".join(f"**{pct(v)}**" if v == max(rates) else pct(v) for v in rates) + " |")
        out.append("| *reçus 100 % corrects* | " + " | ".join(f"**{pct(v)}**" if v == max(docs) else pct(v) for v in docs) + " |\n")

    out.append("## Ressources · temps moyen par reçu, de l'image au JSON\n")
    out.append("| Approche | Jeu | Matériel | Temps moyen / reçu | Mémoire GPU |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        ms = r["ms"]
        t = "?" if ms is None else (f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s")
        out.append(f"| {r['method']['label']} | {r['dataset'].upper()} | {r['method']['hardware']} | {t} | {r['vram']} |")
    out.append(f"""
## À lire avec ce tableau

- Les temps des approches 1 et 2 comptent l'OCR ({engine}, sur CPU) ; Donut n'a pas d'OCR. Tout est mesuré sur le même PC (RTX 2050, 4 Go).
- Approche 3, jeu CORD : checkpoint public entraîné par les auteurs de Donut sur le train CORD (dans le domaine). Jeu SROIE : notre propre fine-tuning (526 reçus, 75 min sur un GPU gratuit).
- « company, sans espaces » est un diagnostic ajouté après l'évaluation : il ignore espaces et ponctuation, car l'OCR colle les mots (« KEDAIPAPANYEWCHUAN »). La mesure officielle reste l'exact match ; aucun modèle n'a été relancé ni réglé pour ce diagnostic.
- Les 3 approches sont notées avec le même code (normalisation des dates et montants, mêmes métriques) sur les mêmes reçus.
- Réglages faits sur le jeu dev uniquement ; le jeu eval n'est évalué qu'une fois, avec des réglages figés.
""")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jeu", "approche", "champ", "n_docs", "exact_match", "precision", "rappel", "f1", "ms_par_recu", "gpu"])
        for r in rows:
            for field, s in r["data"]["fields"].items():
                w.writerow([r["dataset"], r["method"]["label"], field, r["data"]["n_docs"], s["exact_match"],
                            s["precision"], s["recall"], s["f1"], round(r["ms"] or 0), r["vram"]])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", dest="role", choices=["dev", "eval"], default="dev")
    ap.add_argument("--final", action="store_true", help="obligatoire pour le jeu eval")
    ap.add_argument("--engine", default="rapidocr", help="OCR des approches 1 et 2")
    ap.add_argument("--llm-models", nargs="+", default=["qwen2.5:3b", "qwen2.5:7b"])
    args = ap.parse_args(argv)
    if args.role == "eval" and not args.final:
        print("Le tableau eval est reserve au score final (etape 5). Ajoute --final quand les 4 evaluations eval sont faites.")
        return 3

    rows, missing = collect(args.role, args.engine, args.llm_models)
    if not rows:
        print("Aucun resultat trouve. Lance d'abord les evaluations (04, 06, 08).")
        return 1
    text = markdown(args.role, rows, args.llm_models, args.engine)
    out_dir = config.OUTPUTS_DIR / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"benchmark_{args.role}.md").write_text(text, encoding="utf-8")
    write_csv(out_dir / f"benchmark_{args.role}.csv", rows)
    print(text)
    if missing:
        print("RESULTATS MANQUANTS (ces approches n'apparaissent pas dans le tableau) :")
        for p in missing:
            print(f"  - {p.relative_to(config.ROOT)}")
    print(f"ecrit -> {(out_dir / f'benchmark_{args.role}.md').relative_to(config.ROOT)} (+ .csv)")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
