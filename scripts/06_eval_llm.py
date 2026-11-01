"""Etape 3 - Approche 2 (OCR + LLM local via Ollama) : extrait les champs et calcule les scores.

Usage :
    python scripts/06_eval_llm.py --engine rapidocr --set dev --limit 3    # test rapide (3 recus par jeu)
    python scripts/06_eval_llm.py --engine rapidocr --set dev
    python scripts/06_eval_llm.py --engine gt --set dev                    # OCR parfait = borne haute du LLM
    python scripts/06_eval_llm.py --engine rapidocr --set eval --final     # score final, 1 seule fois

Le LLM lit le texte deja produit par l'OCR (cache de l'etape 2). Chaque reponse est mise en cache :
relancer ne rappelle pas le LLM (sauf --force), et une coupure en cours de route se reprend ou elle s'est arretee.
Le jeu `eval` est reserve au score final : le prompt se regle sur `dev`, jamais sur `eval`.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

from tqdm import tqdm

from docextract import config, llm
from docextract.evaluation import FIELDS_BY_DATASET, evaluate
from docextract.ocr import ENGINES
from docextract.samples import available, get_docs, split_for


def _load(script: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_rules = _load("04_eval_rules.py", "eval_rules")          # lines_for : lignes OCR d'un recu
explain = _load("05_explain_errors.py", "explain_errors")      # relevant_lines : lignes utiles a une erreur


def run_dataset(dataset: str, role: str, args) -> dict | None:
    docs = get_docs(dataset, role)
    if args.limit:
        docs = docs[:args.limit]
    preds, gts, runs, lines_of, missing, from_cache = {}, {}, {}, {}, 0, 0
    checks: dict[str, dict[str, int]] = {"kept": {}, "fixed": {}, "rejected": {}}
    for rec in tqdm(docs, desc=dataset, unit="recu"):
        got = eval_rules.lines_for(rec, args.engine)
        if got is None:
            missing += 1
            continue
        lines, _ = got
        run, cached = llm.cached_llm(rec["doc_id"], lines, model=args.model, ocr_engine=args.engine,
                                     force=args.force, few_shot=not args.no_few_shot,
                                     use_schema=args.format == "schema", base_url=args.url)
        from_cache += cached
        out = run.output
        preds[rec["doc_id"]] = {k: (getattr(out, k) if out else None) for k in llm.FIELDS}
        if args.verify:
            preds[rec["doc_id"]], verdict = llm.verified_fields(run, lines)
            for kind in ("kept", "fixed", "rejected"):
                for name in getattr(verdict, kind):
                    checks[kind][name] = checks[kind].get(name, 0) + 1
        gts[rec["doc_id"]] = rec["gt"]
        runs[rec["doc_id"]] = run
        lines_of[rec["doc_id"]] = lines
    if not gts:
        return None
    report = evaluate(preds, gts, FIELDS_BY_DATASET[dataset])
    return {"dataset": dataset, "split": split_for(dataset, role), "report": report, "missing": missing,
            "runs": runs, "lines": lines_of, "from_cache": from_cache, "n": len(gts), "checks": checks}


def timing(res: dict) -> dict:
    runs = list(res["runs"].values())
    ms = [r.ms for r in runs]
    tps = [r.meta["output_tokens"] / (r.meta["eval_ms"] / 1000) for r in runs
           if r.meta.get("output_tokens") and r.meta.get("eval_ms")]
    return {"ms_moyen": round(statistics.fmean(ms), 0), "ms_median": round(statistics.median(ms), 0),
            "ms_max": round(max(ms), 0), "jetons_par_s": round(statistics.fmean(tps), 1) if tps else None,
            "reponses_invalides": sum(r.output is None for r in runs),
            "reessais": sum(r.attempts > 1 for r in runs)}


def write_outputs(res: dict, role: str, args) -> tuple[str, str, str]:
    out_dir = config.LLM_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{role}_{llm.safe_name(args.model)}_{args.engine}{'_verif' if args.verify else ''}_{res['dataset']}"
    report, runs = res["report"], res["runs"]

    csv_path = out_dir / f"{stem}_erreurs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "champ", "statut", "verite", "prediction", "reponse_llm", "debut_ocr"])
        for row in report.rows:
            if row["status"] in ("correct", "empty"):
                continue
            head = " | ".join(line.text for line in res["lines"][row["doc_id"]][:3])[:160]
            w.writerow([row["doc_id"], row["field"], row["status"], row["gt"], row["pred"],
                        runs[row["doc_id"]].raw.replace("\n", " "), head])

    blocks, shown = [], {}
    for row in report.rows:
        field, status = row["field"], row["status"]
        if status in ("correct", "empty") or shown.get(field, 0) >= args.max:
            continue
        shown[field] = shown.get(field, 0) + 1
        run = runs[row["doc_id"]]
        block = [f"[{res['dataset']}] {row['doc_id']} | champ={field} | statut={status}",
                 f"  verite     : {row['gt']}", f"  prediction : {row['pred']}",
                 f"  reponse LLM: {run.raw.strip()[:200]}", "  lignes OCR :"]
        block += [f"    {n:>3}| {text[:110]}" for n, text in explain.relevant_lines(field, res["lines"][row["doc_id"]])]
        blocks.append("\n".join(block))
    txt_path = out_dir / f"explication_{stem}.txt"
    txt_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    summary = report.summary()
    summary.update({"model": args.model, "ocr": args.engine, "set": role, "prompt": llm.PROMPT_VERSION,
                    "few_shot": not args.no_few_shot, "format": args.format, "docs_sans_ocr": res["missing"],
                    "temps": timing(res), "verification": args.verify,
                    "controles": res["checks"] if args.verify else None,
                    "memoire_ollama": getattr(args, "mem", None)})
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    rel = lambda p: str(p.relative_to(config.ROOT))
    return rel(csv_path), rel(json_path), rel(txt_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", choices=[*ENGINES, "gt"], default="rapidocr", help="OCR dont on lit le cache")
    ap.add_argument("--model", default=config.LLM_MODEL, help="modele Ollama (defaut : %(default)s)")
    ap.add_argument("--set", dest="role", choices=["dev", "eval"], default="dev")
    ap.add_argument("--final", action="store_true", help="obligatoire pour le jeu eval")
    ap.add_argument("--limit", type=int, default=0, help="n premiers recus par jeu (test rapide)")
    ap.add_argument("--force", action="store_true", help="ignore le cache des reponses LLM")
    ap.add_argument("--format", choices=["schema", "json"], default="schema",
                    help="schema = JSON impose par un schema (defaut) ; json = JSON libre")
    ap.add_argument("--no-few-shot", action="store_true", help="sans les 2 exemples du prompt")
    ap.add_argument("--verify", action="store_true",
                    help="verifie chaque champ contre le texte OCR (rejette les valeurs absentes, rend le format imprime)")
    ap.add_argument("--url", default=config.OLLAMA_URL)
    ap.add_argument("--max", type=int, default=25, help="erreurs detaillees par jeu et par champ")
    args = ap.parse_args(argv)

    if args.role == "eval" and not args.final:
        print("Le jeu eval est reserve au score final (etape 5) : on regle le prompt sur dev.\n"
              "Si c'est bien le score final, relance avec --final.")
        return 3
    datasets = available(args.role)
    if not datasets:
        print("Aucune donnee : lance d'abord  python scripts/00_prepare_data.py")
        return 1
    try:
        installed = llm.installed_models(args.url)
    except llm.OllamaError as e:
        print(f"{e}\nInstallation : winget install --id Ollama.Ollama -e   (puis rouvre PowerShell)")
        return 2
    if not llm.has_model(args.model, installed):
        print(f"Modele {args.model} absent. Telecharge-le :  ollama pull {args.model}\n"
              f"Modeles installes : {', '.join(installed) or 'aucun'}")
        return 2
    print(f"Modele {args.model} | OCR {args.engine} | prompt {llm.PROMPT_VERSION} | jeu {args.role}")
    print("Rappel : ferme les applications inutiles avant (le LLM chauffe le PC).")
    try:
        load_ms = llm.warm_up(args.model, args.url)
    except llm.OllamaError as e:
        print(e)
        return 2
    print(f"Modele charge en memoire ({load_ms / 1000:.0f} s, hors des temps mesures).")
    args.mem = llm.loaded_model_info(args.model, args.url)
    if args.mem:
        print(f"Memoire Ollama : {args.mem['size_gb']} Go dont {args.mem['vram_gb']} Go en VRAM "
              f"({args.mem['gpu_share']:.0%} sur GPU).")

    status = 0
    for dataset in datasets:
        try:
            res = run_dataset(dataset, args.role, args)
        except FileNotFoundError as e:
            print(f"\n[{dataset}] {e}")
            status = 1
            continue
        except llm.OllamaError as e:
            print(f"\n[{dataset}] {e}\nRelance la meme commande : les reponses deja obtenues sont en cache.")
            return 2
        if res is None:
            print(f"\n[{dataset}] aucun OCR en cache pour {args.engine} : lance "
                  f"python scripts/03_run_ocr.py --engine {args.engine} --set {args.role}")
            status = 1
            continue
        report, t = res["report"], timing(res)
        print(f"\n=== {dataset} ({res['split']}) | LLM {args.model} | OCR {args.engine} | jeu {args.role} "
              f"| {report.n_docs} recus ===")
        if res["missing"]:
            print(f"  ATTENTION : {res['missing']} recu(s) sans OCR en cache (ignores)")
            if args.role == "eval":
                status = 1
        print(report.to_markdown())
        speed = f", {t['jetons_par_s']} jetons/s" if t["jetons_par_s"] else ""
        print(f"\n  Temps LLM par recu : moyenne {t['ms_moyen']:.0f} ms, mediane {t['ms_median']:.0f} ms, "
              f"max {t['ms_max']:.0f} ms{speed}")
        print(f"  Reponses invalides : {t['reponses_invalides']} | reessais : {t['reessais']} "
              f"| depuis le cache : {res['from_cache']}/{res['n']}")
        if args.verify:
            c = res["checks"]
            fmt = lambda d: ", ".join(f"{k} {v}" for k, v in sorted(d.items())) or "aucun"
            print(f"  Verification : gardes [{fmt(c['kept'])}] | format corrige [{fmt(c['fixed'])}] "
                  f"| rejetes [{fmt(c['rejected'])}]")
        csv_path, json_path, txt_path = write_outputs(res, args.role, args)
        print(f"\n  erreurs  -> {csv_path}\n  resume   -> {json_path}\n  analyse  -> {txt_path}")
    print("\nCharge du modele (GPU ou CPU) : ollama ps      | Decharger : ollama stop " + args.model)
    return status


if __name__ == "__main__":
    sys.exit(main())
