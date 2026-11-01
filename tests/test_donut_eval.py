import json

from conftest import fake_cord, fake_sroie, load_script
from docextract import config
from docextract.donut import fields_from_prediction, load_predictions, timing, unreadable
from docextract.samples import get_docs

# formes reelles vues dans les predictions Colab (dev)
CORD_FULL = {"menu": [{"nm": "REAL GANACHE", "cnt": "1", "price": "16,500"}],
             "sub_total": {"subtotal_price": "18,181", "tax_price": "1,818"},
             "total": {"total_price": "20,000", "cashprice": "100,000", "changeprice": "80,000"}}
CORD_NO_SUBTOTAL = {"total": {"total_price": "23.000", "cashprice": "50.000"}, "sub_total": None}


def test_sroie_fields_come_straight_from_the_json():
    assert fields_from_prediction("sroie", {"company": "TEO HENG", "date": "17/01/2018", "total": "26.00"}) == {
        "company": "TEO HENG", "date": "17/01/2018", "total": "26.00", "tax": None}


def test_unreadable_sequence_gives_no_field():
    assert set(fields_from_prediction("sroie", {"text_sequence": "abc"}).values()) == {None}
    assert set(fields_from_prediction("cord", {"_error": "x"}).values()) == {None}


def test_cord_total_and_tax_follow_the_public_schema():
    assert fields_from_prediction("cord", CORD_FULL) == {"company": None, "date": None, "total": "20,000", "tax": "1,818"}
    assert fields_from_prediction("cord", CORD_NO_SUBTOTAL)["tax"] is None          # pas de sub_total : taxe manquee


def test_lists_and_empty_values_are_handled():
    assert fields_from_prediction("cord", {"total": [{"total_price": "5,000"}], "sub_total": [{"tax_price": ["", "500"]}]}) == {
        "company": None, "date": None, "total": "5,000", "tax": "500"}
    assert fields_from_prediction("sroie", {"company": "  ", "date": None, "total": {"x": 1}})["company"] is None


def test_timing_and_unreadable_counts():
    preds = {"a": {"parsed": {"company": "X"}, "ms": 100}, "b": {"parsed": {"text_sequence": "?"}, "ms": 300}}
    assert timing(preds) == {"ms_moyen": 200.0, "ms_median": 200.0, "ms_max": 300.0} and unreadable(preds) == 1


# ------------------------------------------------------------------ script
def project(tmp_project, monkeypatch):
    prepare = load_script("00_prepare_data")
    monkeypatch.setattr(prepare, "load_split",
                        lambda repo, split, limit: fake_sroie(130) if "SROIE" in repo else fake_cord(4))
    assert prepare.main([]) == 0
    return tmp_project


def write_preds(dataset, role, parsed_of):
    docs = get_docs(dataset, role)
    path = config.DONUT_PRED_DIR / f"predictions_{dataset}_{role}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = {d["doc_id"]: {"parsed": parsed_of(i), "sequence": "<s_x>", "ms": 200.0 + i} for i, d in enumerate(docs)}
    path.write_text(json.dumps({"meta": {"gpu": "Tesla T4"}, "predictions": preds}), encoding="utf-8")


def test_script_scores_dev_and_writes_outputs(tmp_project, monkeypatch, capsys):
    project(tmp_project, monkeypatch)
    write_preds("sroie", "dev", lambda i: {"company": f"SHOP {i} SDN BHD", "date": "25/12/2018", "total": "9.00"})
    write_preds("cord", "dev", lambda i: {"total": {"total_price": "63,800"}, "sub_total": {"tax_price": "5,800"}})
    assert load_script("08_eval_donut").main(["--set", "dev"]) == 0
    out = capsys.readouterr().out
    assert "=== sroie (train) | Donut" in out and "=== cord (validation)" in out
    assert "| date | 100.0%" in out and "Tesla T4" in out and "Sequences illisibles : 0" in out
    d = tmp_project / "outputs" / "donut"
    summary = json.loads((d / "dev_sroie.json").read_text(encoding="utf-8"))
    assert summary["set"] == "dev" and summary["temps"]["ms_median"] > 0
    assert (d / "dev_cord_erreurs.csv").exists() and (d / "explication_dev_sroie.txt").exists()


def test_script_missing_file_explains_where_to_copy_it(tmp_project, monkeypatch, capsys):
    project(tmp_project, monkeypatch)
    assert load_script("08_eval_donut").main(["--set", "dev"]) == 1
    assert "donut_preds" in capsys.readouterr().out


def test_script_eval_needs_final(tmp_project, monkeypatch, capsys):
    project(tmp_project, monkeypatch)
    assert load_script("08_eval_donut").main(["--set", "eval"]) == 3
    assert "--final" in capsys.readouterr().out


def test_missing_predictions_count_as_missed(tmp_project, monkeypatch, capsys):
    project(tmp_project, monkeypatch)
    write_preds("sroie", "dev", lambda i: {"company": "X", "date": "25/12/2018", "total": "9.00"})
    path = config.DONUT_PRED_DIR / "predictions_sroie_dev.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for k in list(data["predictions"])[:10]:
        del data["predictions"][k]
    path.write_text(json.dumps(data), encoding="utf-8")
    load_script("08_eval_donut").main(["--set", "dev"])
    assert "10 recu(s) sans prediction" in capsys.readouterr().out
