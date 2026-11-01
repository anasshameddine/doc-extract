"""Verification des reponses du LLM contre le texte OCR : cas reels vus sur le jeu dev (etape 3)."""
import pytest

from docextract.grounding import verify_amount, verify_company, verify_date, verify_fields
from docextract.ocr import Line


def L(*texts):
    return [Line(t, 0, i * 10, 100, i * 10 + 9) for i, t in enumerate(texts)]


@pytest.mark.parametrize("value, lines, expected, status", [
    ("23.00", ["Total 23.000", "Tunai 50.000"], "23.000", "fixed"),                    # milliers reecrits en decimales
    ("379.50", ["Sub Total 345.000", "TOTAL 379,500"], "379,500", "fixed"),
    ("53.00", ["SUBTOTAL 53.000", "GRANDTOTAL 53.000"], "53.000", "fixed"),
    ("41.00", ["Total. PayCash50.000 Change:9.000 41,000"], "41,000", "fixed"),          # ligne cash mais avec Total
    ("20,000", ["Total: 20,000", "Cash: 100,000"], "20,000", "kept"),                  # deja bon : intact
    ("80000", ["Total: 20,000", "Cash: 100,000", "CHANGE: 80000"], None, "rejected"),   # monnaie rendue
    ("3500", ["TOTAL 16,500", "CASH 20,000 3,500", "Change"], None, "rejected"),
    ("100.00", ["SubTotal: 92929", "cashs 100,000", "Change: 40,500"], None, "rejected"),
    ("1234", ["TOTAL 99"], None, "rejected"),                                         # absent du texte
    (None, ["TOTAL 99"], None, "none"),
])
def test_verify_total(value, lines, expected, status):
    assert verify_amount(value, L(*lines), is_total=True) == (expected, status)


def test_tax_is_checked_without_the_cash_rule():
    assert verify_amount("20.45", L("TAX Rp20,446"), is_total=False) == (None, "rejected")   # 20.45 != 20,446
    assert verify_amount("34.50", L("Tax 34,500"), is_total=False) == ("34,500", "fixed")
    assert verify_amount("7.400", L("TAX 7. 400"), is_total=False) == ("7.400", "kept")      # espace apres le point


@pytest.mark.parametrize("value, lines, ok", [
    ("PASARAYA BORONG PINTAR SDN BHD", ["PASARAYABORONGPINTAR SDNBHD", "NO 5 JALAN"], True),   # espaces restaures
    ("GARDENIA BAKERIES (KL) SDN BHD", ["GARDENIA BAKERIES (KL) SDN BHD (139386 X)"], True),
    ("Darepes", ["TOTAL 59,500", "Cash 100,000"], False),                                       # invente
    ("Gojek", ["Sub Total 345.000", "TOTAL 379,500"], False),
    ("AB", ["AB CAFE"], False),                                                                 # trop court
    # cas reels du 7b (dev) : nom coupe sur plusieurs lignes ou OCR fautif -> ne plus rejeter une bonne reponse
    ("THREE STOOGES BISTRO&CAFE", ["THREE STOOGES", "BISTRO&CAFE", "109,SS21/1A,"], True),
    ("PASARAYA BORONG PINTAR SDN BHD", ["PASARAYABORONGPINTAR", "SDNBHD", "BRNo.:(124525-H)"], True),
    ("Nando's Chickenland Malaysia Sdn Bhd", ["Nando's", "PERi-PERICHICKEN", "Nando's Chickenland Malaysia", "Sdn Bhd"], True),
    ("Timeless Kitchenette SDN BHD", ["organfleld's", "Home of Sicky Baea", "Timeless Kitchenette Sdin Bhd"], True),
    ("TEO HENG STATIONERY & BOOKS", ["TEO HENGSTATIONERYSBOOKS", "(001451637-M)"], True),
    ("POPULAR BOOK CO. (M) SDN BHD", ["POFULAR BOOK.", "CO. (M) SDN BHD", "(Company No.113825-w)"], True),
    # inventions : le texte du recu ne contient rien de comparable, meme avec des fenetres de 3 lignes
    ("YHM Aeon Tebrau City", ["Order No. 100021911/03/2018 06:20 PM", "Receipt No.:003-1220815 Pax:", "FG-A00019 1.50 1.50 SR"], False),
    ("SPEEDMART SDN BHD", ["TOTAL 11.40", "CASH 20.00", "CHANGE 8.60"], False),
])
def test_verify_company(value, lines, ok):
    got, status = verify_company(value, L(*lines))
    assert (got is not None) is ok and status == ("kept" if ok else "rejected")


def test_verify_date_needs_a_real_date():
    assert verify_date("12/03/2019", L("x"))[1] == "kept"
    assert verify_date("Junlah Itep", L("x")) == (None, "rejected")
    assert verify_date("3 Items,", L("x")) == (None, "rejected")


def test_verify_fields_summary():
    lines = L("TOKO MAJU", "Total 23.000", "Tunai 50.000")
    out, v = verify_fields({"company": "Darepes", "date": None, "total": "23.00", "tax": None}, lines)
    assert out == {"company": None, "date": None, "total": "23.000", "tax": None}
    assert v.rejected == ["company"] and v.fixed == ["total"] and v.kept == []
