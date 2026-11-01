import csv
import json

from conftest import load_script
from docextract import config
from docextract.evaluation import FIELDS_BY_DATASET, evaluate


def summary(dataset, correct_fields, n=10, **extra):
    """Resume JSON realiste (celui de evaluate().summary()) ou chaque champ de `correct_fields` est juste."""
    gts = {f"d{i}": {"company": "A", "date": "2018-01-01", "total": "5.00", "tax": "1.00"} for i in range(n)}
    preds = {k: {f: (v if f in correct_fields else "x") for f, v in gt.items()} for k, gt in gts.items()}
    out = evaluate(preds, gts, FIELDS_BY_DATASET[dataset]).summary()
    out.update(extra)
    return out


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def populate(role="dev", donut=True, llm7=True):
    for ds in ("sroie", "cord"):
        write(config.RULES_OUTPUT_DIR / f"{role}_rapidocr_{ds}.json",
              summary(ds, {"date", "total"}, ocr_ms_moyen=1200.0, regles_ms_moyen=2.0))
        for model, fields in (("qwen2.5_3b", {"company"}), ("qwen2.5_7b", {"company", "total"})):
            if model.endswith("7b") and not llm7:
                continue
            write(config.LLM_OUTPUT_DIR / f"{role}_{model}_rapidocr_verif_{ds}.json",
                  summary(ds, fields, temps={"ms_moyen": 3500.0}, memoire_ollama={"vram_gb": 2.2}))
        if donut:
            write(config.DONUT_OUTPUT_DIR / f"{role}_{ds}_local.json",
                  summary(ds, {"company", "date", "total", "tax"}, temps={"ms_moyen": 400.0, "ms_total_moyen": 500.0},
                          predictions_meta={"peak_vram_gb": 1.47}))


def test_benchmark_builds_markdown_and_csv(tmp_project, capsys):
    populate()
    assert load_script("10_benchmark").main(["--set", "dev"]) == 0
    text = (tmp_project / "outputs" / "benchmark" / "benchmark_dev.md").read_text(encoding="utf-8")
    assert "## SROIE (10 reçus)" in text and "## CORD (10 reçus)" in text
    assert "| company | 0.0 | **100.0** | **100.0** | **100.0** |" in text            # regles / LLM 3B / LLM 7b / Donut
    assert "| total | **100.0** | 0.0 | **100.0** | **100.0** |" in text       # meilleure valeur en gras, ex aequo compris
    assert "1.2 s" in text and "3.5 s" not in text and "4.7 s" in text         # OCR 1,2 s + LLM 3,5 s = 4,7 s
    assert "| 3 · Donut (Document AI) | SROIE | GPU (local) | 500 ms | 1.47 Go |" in text
    rows = list(csv.DictReader((tmp_project / "outputs" / "benchmark" / "benchmark_dev.csv").open(encoding="utf-8")))
    assert {r["jeu"] for r in rows} == {"sroie", "cord"} and any(r["champ"] == "tax" for r in rows)


def test_benchmark_lists_missing_results(tmp_project, capsys):
    populate(donut=False)
    assert load_script("10_benchmark").main(["--set", "dev"]) == 1
    out = capsys.readouterr().out
    assert "RESULTATS MANQUANTS" in out and "dev_sroie_local.json" in out
    assert "| 3 · Donut (Document AI) |" not in out


def test_benchmark_flags_a_smaller_sample(tmp_project, capsys):
    populate()
    write(config.LLM_OUTPUT_DIR / "dev_qwen2.5_7b_rapidocr_verif_sroie.json",
          summary("sroie", {"company"}, n=4, temps={"ms_moyen": 7500.0}, memoire_ollama={"vram_gb": 3.9}))
    load_script("10_benchmark").main(["--set", "dev"])
    assert "(n=4)" in (tmp_project / "outputs" / "benchmark" / "benchmark_dev.md").read_text(encoding="utf-8")


def test_benchmark_eval_needs_final(tmp_project, capsys):
    assert load_script("10_benchmark").main(["--set", "eval"]) == 3
    assert "--final" in capsys.readouterr().out


def test_benchmark_without_any_result(tmp_project, capsys):
    assert load_script("10_benchmark").main(["--set", "dev"]) == 1
    assert "04, 06, 08" in capsys.readouterr().out


def test_benchmark_eval_final(tmp_project):
    populate(role="eval")
    assert load_script("10_benchmark").main(["--set", "eval", "--final"]) == 0
    assert (tmp_project / "outputs" / "benchmark" / "benchmark_eval.md").exists()


def test_loose_company_counts_space_only_errors(tmp_project):
    populate()
    path = config.RULES_OUTPUT_DIR / "dev_rapidocr_sroie.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["fields"]["company"].update(correct=0, n_gt=10)
    path.write_text(json.dumps(data), encoding="utf-8")
    errors = path.with_name("dev_rapidocr_sroie_erreurs.csv")
    errors.write_text("doc_id,champ,statut,verite,prediction\n"
                      "a,company,wrong,KEDAI PAPAN,KEDAIPAPAN\n"
                      "b,company,wrong,GARDENIA (KL) SDN,GARDENIA(KL)SDN.\n"
                      "c,company,wrong,MYDIN,TRI SHAAS\n"
                      "d,company,missed,ABC,\n", encoding="utf-8")
    others = [config.LLM_OUTPUT_DIR / "dev_qwen2.5_3b_rapidocr_verif_sroie_erreurs.csv",
              config.LLM_OUTPUT_DIR / "dev_qwen2.5_7b_rapidocr_verif_sroie_erreurs.csv",
              config.DONUT_OUTPUT_DIR / "dev_sroie_local_erreurs.csv"]
    for other in others:
        other.write_text("doc_id,champ,statut,verite,prediction\n", encoding="utf-8")
    load_script("10_benchmark").main(["--set", "dev"])
    text = (tmp_project / "outputs" / "benchmark" / "benchmark_dev.md").read_text(encoding="utf-8")
    row = [ln for ln in text.splitlines() if ln.startswith("| *company, sans espaces")][0]
    assert row.split("|")[2].strip() == "20.0"           # 2 sur 10 retrouvees
