from conftest import fake_cord, fake_sroie

from docextract.datasets_io import (export_split, make_eval_sample, parse_cord_row,
                                    parse_sroie_row, read_jsonl)


def test_parse_sroie_row():
    row = fake_sroie(2)[1]
    rec = parse_sroie_row(row, "test", 1)
    assert rec["doc_id"] == "sroie_test_X51000000001"
    assert rec["gt"]["company"] == "SHOP 1 SDN BHD"
    assert rec["gt"]["date"] == "25/12/2018"
    assert rec["gt"]["total"] == "9.00"
    assert rec["gt"]["tax"] is None
    assert rec["words"] == ["SHOP", "TOTAL", "9.00"]
    assert rec["bboxes"][0] == [1, 2, 30, 12]


def test_parse_sroie_empty_total_becomes_none():
    rec = parse_sroie_row(fake_sroie(1)[0], "test", 0)
    assert rec["gt"]["total"] is None


def test_parse_cord_row_fields():
    rec = parse_cord_row(fake_cord(2)[0], "test", 0)
    gt = rec["gt"]
    assert gt["company"] is None and gt["date"] is None  # absents de CORD public
    assert gt["total"] == "63,800"
    assert gt["subtotal"] == "58,000"
    assert gt["tax"] == "5,800"
    assert gt["items"][0] == {"name": "NASI GORENG", "qty": "2", "price": "50,000"}
    assert rec["words"] == ["63,800"]
    assert rec["bboxes"] == [[10, 5, 50, 15]]


def test_parse_cord_menu_as_dict():
    rec = parse_cord_row(fake_cord(3)[2], "test", 2)
    assert rec["gt"]["items"] == [{"name": "KOPI", "qty": None, "price": "10"}]
    assert rec["gt"]["tax"] is None


def test_export_split_writes_images_and_jsonl(tmp_path):
    jsonl = tmp_path / "data" / "processed" / "sroie_test.jsonl"
    n = export_split(fake_sroie(3), "sroie", "test", root=tmp_path,
                     raw_dir=tmp_path / "data" / "raw", jsonl_file=jsonl)
    assert n == 3
    recs = list(read_jsonl(jsonl))
    assert len(recs) == 3
    assert recs[0]["image"] == "data/raw/sroie/test/X51000000000.jpg"
    assert (tmp_path / recs[0]["image"]).exists()
    assert (recs[0]["width"], recs[0]["height"]) == (100, 300)
    assert "file_stem" not in recs[0]


def test_eval_sample_is_deterministic():
    ids = [f"doc{i}" for i in range(500)]
    a = make_eval_sample(ids, 100, 42)
    b = make_eval_sample(list(reversed(ids)), 100, 42)
    assert a == b and len(a) == 100 and len(set(a)) == 100


def test_eval_sample_smaller_than_size():
    assert make_eval_sample(["b", "a"], 100, 42) == ["a", "b"]
