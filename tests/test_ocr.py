import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from docextract import ocr
from docextract.ocr import (OcrResult, Segment, box_to_xyxy, build_lines, cached_ocr,
                            find_tesseract, prepare_image, run_ocr, segments_from_gt)


def seg(text, x0, y0, x1, y1, conf=0.9):
    return Segment(text, x0, y0, x1, y1, conf)


# ------------------------------------------------------------------ lignes
def test_build_lines_merges_label_and_amount():
    # etiquette a gauche, montant a droite, centres verticaux a 2 px d'ecart
    segs = [seg("30.25", 300, 102, 360, 122), seg("Amount Due", 20, 100, 150, 120),
            seg("THANK YOU", 100, 160, 250, 180)]
    lines = build_lines(segs)
    assert [l.text for l in lines] == ["Amount Due 30.25", "THANK YOU"]
    assert lines[0].x0 == 20 and lines[0].x1 == 360


def test_build_lines_orders_top_to_bottom_and_ignores_blank():
    segs = [seg("B", 0, 100, 20, 120), seg("  ", 0, 50, 20, 70), seg("A", 0, 10, 20, 30)]
    assert [l.text for l in build_lines(segs)] == ["A", "B"]


def test_build_lines_keeps_close_lines_apart():
    segs = [seg("Sub Total", 0, 100, 80, 120), seg("Tax", 0, 126, 40, 146)]   # lignes de 20 px, pas de 26
    assert len(build_lines(segs)) == 2


# ------------------------------------------------------------------ verite terrain comme OCR
@pytest.mark.parametrize("box, expected", [
    ([10, 20, 50, 40], (10, 20, 50, 40)),
    ([50, 20, 10, 40], (10, 20, 50, 40)),                         # ordre inverse
    ([10, 20, 50, 20, 50, 40, 10, 40], (10, 20, 50, 40)),          # 4 coins a plat
    ([[10, 20], [50, 20], [50, 40], [10, 40]], (10, 20, 50, 40)),  # 4 paires
])
def test_box_to_xyxy(box, expected):
    assert box_to_xyxy(box) == expected


def test_box_to_xyxy_invalid():
    assert box_to_xyxy(None) is None
    assert box_to_xyxy([1, 2, 3]) is None


def test_segments_from_gt_builds_lines():
    words = ["TOTAL", "9.00", "", "SHOP"]
    boxes = [[10, 100, 60, 120], [200, 102, 260, 122], [0, 0, 1, 1], [10, 10, 60, 30]]
    lines = build_lines(segments_from_gt(words, boxes))
    assert [l.text for l in lines] == ["SHOP", "TOTAL 9.00"]


# ------------------------------------------------------------------ resultat et cache
def test_ocr_result_roundtrip():
    r = OcrResult("tesseract", [seg("A", 1, 2, 3, 4, 0.5), seg("B", 1, 20, 3, 30)], ms=12.34,
                  width=100, height=200, params={"psm": 6})
    back = OcrResult.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back.engine == "tesseract" and back.ms == 12.3 and back.params == {"psm": 6}
    assert [s.text for s in back.segments] == ["A", "B"]
    assert back.text() == "A\nB"


def test_cached_ocr_computes_once(tmp_project, monkeypatch):
    calls = []

    def fake_engine(path, **params):
        calls.append(path)
        return OcrResult("fake", [seg("TOTAL", 0, 0, 50, 10)], ms=5.0)

    monkeypatch.setitem(ocr.ENGINES, "fake", fake_engine)
    img = tmp_project / "a.png"
    Image.new("RGB", (50, 50), "white").save(img)

    r1, hit1 = cached_ocr("doc1", img, "fake")
    r2, hit2 = cached_ocr("doc1", img, "fake")
    assert (hit1, hit2) == (False, True) and len(calls) == 1
    assert r2.segments[0].text == "TOTAL" and r2.ms == 5.0
    assert ocr.load_cached("doc1", "fake") is not None
    assert ocr.load_cached("absent", "fake") is None
    _, hit3 = cached_ocr("doc1", img, "fake", force=True)
    assert hit3 is False and len(calls) == 2


def test_unknown_engine():
    with pytest.raises(ValueError):
        run_ocr("x.png", "inconnu")


def test_prepare_image_resizes(tmp_path):
    big, small = tmp_path / "big.png", tmp_path / "small.png"
    Image.new("RGB", (4000, 8000), "white").save(big)
    Image.new("RGB", (500, 1000), "white").save(small)
    img, scale = prepare_image(big, max_side=2000)
    assert max(img.size) == 2000 and scale == pytest.approx(0.25)
    img, scale = prepare_image(small, min_width=1000, max_side=3000)
    assert img.width == 1000 and scale == pytest.approx(2.0)
    img, scale = prepare_image(small)
    assert img.size == (500, 1000) and scale == 1.0


# ------------------------------------------------------------------ vrais moteurs (ignores si absents)
def _font():
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
              "/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, 34)
    return None


def _receipt_png(path):
    font = _font()
    if font is None:
        pytest.skip("aucune police TrueType disponible")
    img = Image.new("RGB", (700, 420), "white")
    d = ImageDraw.Draw(img)
    for y, text in ((30, "TEO HENG STATIONERY"), (110, "SUB TOTAL"), (170, "TOTAL"), (230, "CASH")):
        d.text((30, y), text, fill="black", font=font)
    for y, amount in ((110, "25.80"), (170, "27.35"), (230, "30.00")):
        d.text((470, y), amount, fill="black", font=font)
    img.save(path)


@pytest.mark.skipif(find_tesseract() is None, reason="Tesseract non installe")
def test_real_tesseract_reads_total_line(tmp_path):
    png = tmp_path / "r.png"
    _receipt_png(png)
    result = run_ocr(png, "tesseract", psm=6)
    text = result.text().upper()
    assert "TEO HENG" in text
    assert any("TOTAL" in l.text.upper() and "27.35" in l.text and "SUB" not in l.text.upper()
               for l in result.lines())
    assert result.ms > 0 and result.width == 700 and result.height == 420


def test_real_rapidocr_reads_total_line(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    png = tmp_path / "r.png"
    _receipt_png(png)
    result = run_ocr(png, "rapidocr")
    assert any("27.35" in l.text for l in result.lines())
    assert result.engine == "rapidocr"


def test_find_tesseract_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "tesseract.exe"
    fake.write_text("x")
    monkeypatch.setenv("TESSERACT_CMD", str(fake))
    assert find_tesseract() == str(fake)
    monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "absent"))   # chemin faux : on l'ignore
    assert find_tesseract() != str(tmp_path / "absent")
