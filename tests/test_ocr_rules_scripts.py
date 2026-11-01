"""Scripts de l'etape 2 de bout en bout, avec un faux moteur OCR (sans Tesseract, sans reseau)."""
import csv
import json

import pytest
from conftest import fake_cord, fake_sroie, load_script

from docextract import ocr
from docextract.ocr import OcrResult, Segment


@pytest.fixture
def project(tmp_project, monkeypatch):
    """Projet prepare (faux jeux) + moteur « fake » qui lit « SHOP SDN BHD / TOTAL 9.00 »."""
    prepare = load_script("00_prepare_data")
    monkeypatch.setattr(prepare, "load_split",
                        lambda repo, split, limit: fake_sroie(130) if "SROIE" in repo else fake_cord(4))
    assert prepare.main([]) == 0

    def fake_engine(path, **params):
        return OcrResult("fake", [Segment("SHOP SDN BHD", 0, 0, 90, 10), Segment("TOTAL", 0, 30, 40, 40),
                                  Segment("9.00", 80, 31, 110, 41)], ms=7.0, width=100, height=300)

    monkeypatch.setitem(ocr.ENGINES, "fake", fake_engine)
    return tmp_project


def test_run_ocr_fills_then_reuses_cache(project, capsys):
    script = load_script("03_run_ocr")
    assert script.main(["--engine", "fake", "--set", "dev", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "5 calcules, 0 depuis le cache" in out and "Temps OCR moyen : 7 ms" in out
    cache = project / "data" / "ocr_cache" / "fake"
    assert len(list(cache.glob("*.json"))) == 5 + 4          # 5 SROIE (limite) + 4 CORD validation

    assert script.main(["--engine", "fake", "--set", "dev", "--limit", "5"]) == 0
    assert "0 calcules, 5 depuis le cache" in capsys.readouterr().out


def test_run_ocr_reports_missing_engine(project, monkeypatch, capsys):
    def broken(path, **params):
        raise RuntimeError("Tesseract introuvable. Installe-le")

    monkeypatch.setitem(ocr.ENGINES, "fake", broken)
    assert load_script("03_run_ocr").main(["--engine", "fake", "--limit", "1"]) == 2
    assert "ERREUR : Tesseract introuvable" in capsys.readouterr().out


def test_eval_rules_scores_cached_ocr(project, capsys):
    load_script("03_run_ocr").main(["--engine", "fake", "--set", "dev"])
    capsys.readouterr()
    ev = load_script("04_eval_rules")
    assert ev.main(["--engine", "fake", "--set", "dev"]) == 0
    out = capsys.readouterr().out
    assert "Temps : OCR 7 ms" in out and "total:keyword" in out
    summary = json.loads((project / "outputs" / "rules" / "dev_fake_sroie.json").read_text(encoding="utf-8"))
    assert summary["n_docs"] == 100 and summary["ocr_ms_moyen"] == 7.0
    assert summary["fields"]["total"]["correct"] >= 99       # 9.00 lu sur chaque recu
    assert summary["fields"]["company"]["correct"] == 0      # « SHOP SDN BHD » != « SHOP i SDN BHD »


def test_eval_rules_without_cache(project, capsys):
    assert load_script("04_eval_rules").main(["--engine", "tesseract", "--set", "dev"]) == 1
    assert "aucun OCR en cache" in capsys.readouterr().out


def test_eval_set_needs_final_flag(project, capsys):
    ev = load_script("04_eval_rules")
    assert ev.main(["--engine", "gt", "--set", "eval"]) == 3
    assert "reserve au score final" in capsys.readouterr().out
    assert ev.main(["--engine", "gt", "--set", "eval", "--final"]) == 0


def test_eval_rules_with_annotated_words_writes_error_csv(project, capsys):
    assert load_script("04_eval_rules").main(["--engine", "gt", "--set", "dev"]) == 0
    out = capsys.readouterr().out
    assert "sroie (train)" in out and "cord (validation)" in out and "Par regle" in out
    out_dir = project / "outputs" / "rules"
    rows = list(csv.DictReader((out_dir / "dev_gt_sroie_erreurs.csv").open(encoding="utf-8")))
    assert rows and {"doc_id", "champ", "regle", "ligne_source", "debut_ocr"} <= set(rows[0])


def test_explain_errors_writes_readable_report(project, capsys):
    explain = load_script("05_explain_errors")
    assert explain.main(["--engine", "gt", "--fields", "company", "total", "--max", "3"]) == 0
    out = capsys.readouterr().out
    assert "[sroie] erreurs :" in out and "erreur(s) detaillee(s)" in out
    text = (project / "outputs" / "rules" / "explication_dev_gt.txt").read_text(encoding="utf-8")
    assert "verite     :" in text and "lignes OCR :" in text and "regle :" in text
    assert text.count("champ=company") <= 6          # 3 par jeu au maximum (CORD n'a pas de fournisseur)


def test_explain_errors_without_cache(project, capsys):
    assert load_script("05_explain_errors").main(["--engine", "tesseract"]) == 0
    assert "aucun OCR en cache" in capsys.readouterr().out


def test_relevant_lines_fall_back_to_top():
    from docextract.ocr import Line
    explain = load_script("05_explain_errors")
    lines = [Line(f"ligne {i}", 0, i * 20, 50, i * 20 + 10) for i in range(10)]
    assert [n for n, _ in explain.relevant_lines("company", lines)] == [1, 2, 3, 4, 5, 6]
    assert [n for n, _ in explain.relevant_lines("tax", lines)] == [1, 2, 3, 4, 5, 6]     # aucune ligne typique
    lines[7].text = "GST 6% 1.20"
    assert explain.relevant_lines("tax", lines) == [(8, "GST 6% 1.20")]
