import datetime as dt
import json
from decimal import Decimal

import pytest

from docextract.evaluation import classify, evaluate, load_ground_truth, token_f1
from docextract.schema import ExtractionResult, ReceiptFields

FIELDS = ("company", "date", "total")

GT = {
    "d1": {"company": "PETRON BKT LANJAN SB", "date": "01/02/2018", "total": "4.90"},
    "d2": {"company": "OLD TOWN KOPITAM SDN BHD", "date": "22 MAR 18", "total": "30.25"},
    "d3": {"company": "TEO HENG STATIONERY", "date": "27/01/2018", "total": None},
}


def test_perfect_prediction_gives_100():
    rep = evaluate(GT, GT, FIELDS)
    assert rep.field_rate == 1.0 and rep.doc_rate == 1.0
    assert rep.fields["total"].n_gt == 2


def test_formats_differ_but_normalized_equal():
    pred = {"d1": {"company": "Petron Bkt Lanjan S.B.", "date": "2018-02-01", "total": "RM 4.9"}}
    rep = evaluate(pred, {"d1": GT["d1"]}, FIELDS)
    assert rep.field_rate == 1.0


def test_counts_wrong_missed_extra():
    pred = {
        "d1": {"company": "PETRON", "date": "01/02/2018", "total": "4.90"},   # company faux
        "d2": {"company": None, "date": "22/03/2018", "total": "30.24"},      # manque + faux
        "d3": {"company": "TEO HENG STATIONERY", "date": "27/01/2018", "total": "9.99"},  # extra
    }
    rep = evaluate(pred, GT, FIELDS)
    c, t = rep.fields["company"], rep.fields["total"]
    assert (c.correct, c.wrong, c.missed, c.extra) == (1, 1, 1, 0)
    assert (t.correct, t.wrong, t.missed, t.extra) == (1, 1, 0, 1)
    assert rep.fields["date"].exact_match == 1.0
    # total : precision 1/3, rappel 1/2
    assert t.precision == pytest.approx(1 / 3) and t.recall == pytest.approx(0.5)
    assert t.f1 == pytest.approx(0.4)
    # 8 champs avec verite, 5 corrects
    assert rep.field_rate == pytest.approx(5 / 8)
    assert rep.docs_all_correct == 0
    assert 0 < c.mean_token_f1 < 1          # "PETRON" partiellement juste


def test_missing_document_counts_as_missed():
    rep = evaluate({}, GT, FIELDS)
    assert rep.field_rate == 0.0
    assert rep.fields["date"].missed == 3


def test_rows_and_markdown():
    rep = evaluate({"d1": {"total": "5.00"}}, {"d1": GT["d1"]}, FIELDS)
    row = next(r for r in rep.rows if r["field"] == "total")
    assert row["status"] == "wrong" and row["gt_norm"] == "4.90" and row["pred_norm"] == "5.00"
    md = rep.to_markdown()
    assert "| total |" in md and "Champs corrects" in md


def test_classify_and_token_f1():
    assert classify(None, None) == "empty"
    assert classify("A", None) == "extra"
    assert token_f1("TEO HENG", "TEO HENG BOOKS") == pytest.approx(0.8)
    assert token_f1(None, "X") == 0.0


def test_schema_normalizes_on_creation():
    f = ReceiptFields(company="  Teo  Heng ", date="22 MAR 18", total="RM 30.25", tax="1,55")
    assert f.company == "Teo Heng"
    assert f.date == dt.date(2018, 3, 22)
    assert f.total == Decimal("30.25") and f.tax == Decimal("1.55")
    assert ReceiptFields(date="pas une date", total="?").date is None


def test_schema_rejects_negative_and_serializes():
    with pytest.raises(ValueError):
        ReceiptFields(total=-5)
    res = ExtractionResult(method="rules", fields=ReceiptFields(total="4.90"), latency_ms=12.3)
    data = json.loads(res.model_dump_json())
    assert data["fields"]["total"] == "4.90" and data["method"] == "rules"
    # le resultat du schema s'evalue directement
    rep = evaluate({"d1": res.fields.model_dump()}, {"d1": {"total": "4.90"}}, ("total",))
    assert rep.field_rate == 1.0


def test_load_ground_truth_uses_sample(tmp_project):
    from docextract import config
    config.PROCESSED_DIR.mkdir(parents=True)
    rows = [{"doc_id": f"sroie_test_{i}", "gt": {"total": str(i)}} for i in range(5)]
    config.jsonl_path("sroie", "test").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    config.EVAL_SAMPLE_FILE.write_text(json.dumps({"sroie_test": ["sroie_test_1", "sroie_test_3"]}))
    assert set(load_ground_truth("sroie")) == {"sroie_test_1", "sroie_test_3"}
    assert len(load_ground_truth("sroie", sample_only=False)) == 5
