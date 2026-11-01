import json
from decimal import Decimal
from pathlib import Path

import pytest

from docextract.ocr import Line
from docextract.rules import (amounts_in, extract_company, extract_date, extract_rules,
                              extract_tax, extract_total)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "ocr_lines.json").read_text(encoding="utf-8"))


def L(*texts):
    """Lignes de test, de haut en bas."""
    return [Line(t, 0, 30 * i, 100, 30 * i + 20) for i, t in enumerate(texts)]


# ------------------------------------------------------------------ montants
@pytest.mark.parametrize("text, expected", [
    ("Total RM inc.GST: 4.90", ["4.90"]),
    ("Total 4 .90", ["4.90"]),                    # espace parasite de l'OCR
    ("GST 6.00% 0.28", ["0.28"]),                 # le pourcentage n'est pas un montant
    ("Sub Total 58,000 Tax 5,800", ["58000.00", "5800.00"]),
    ("GST No: 000584089600", []),                 # numero (0 initial) : pas un montant
    ("TOTAL TAX 7:20", ["7.20"]),                 # ":" lu a la place de "." sur une ligne a mot-cle
])
def test_amounts_in(text, expected):
    assert [str(a) for a in amounts_in(text, allow_int=True)] == expected


def test_amounts_in_integer_only_when_allowed():
    assert amounts_in("Tel 03-6156 8757") == []   # telephone : jamais un montant hors ligne a mot-cle
    assert amounts_in("Closed 12:44:00") == []    # une heure n'est pas un montant
    assert amounts_in("TOTAL 108000") == []
    assert amounts_in("TOTAL 108000", allow_int=True) == [Decimal("108000.00")]


# ------------------------------------------------------------------ total
def test_total_amount_due_beats_total_and_rounding():
    lines = L("Subtotal: 25.94", "Total: 30.24", "Rounding 0.01", "Amount Due 30.25", "CASH 50.00")
    t = extract_total(lines)
    assert t.value == Decimal("30.25") and t.rule == "total:strong-keyword"


def test_total_skips_subtotal_tax_qty_and_summary():
    lines = L("Total Qty : 8", "SUB-TOTAL (EX) 25.80", "TOTAL TAX 1.55", "TOTAL 27.35", "CASH 27.35",
              "GST SUMMARY", "TOTAL: 25.80 1.55")
    assert extract_total(lines).value == Decimal("27.35")


def test_total_works_when_ocr_glues_words():
    lines = L("SUB-TOTAL(EX) 25.80", "TOTALTAX 1.56", "TOTAL 27.35")
    assert extract_total(lines).value == Decimal("27.35")


def test_total_inclusive_gst_is_allowed():
    t = extract_total(L("GST RM: 0.28", "Total RM inc.GST: 4.90", "Cash 5.00", "Change 0.10"))
    assert t.value == Decimal("4.90")


def test_total_falls_back_to_cash_minus_change():
    t = extract_total(L("ITEM A 3.00", "ITEM B 1.90", "Cash 5.00", "Change 0.10"))
    assert t.value == Decimal("4.90") and t.rule == "total:cash-minus-change"


def test_total_last_resort_is_largest_decimal_amount_not_an_id():
    t = extract_total(L("GST No: 000584089600", "ITEM 155.66", "ITEM 16.79", "XYZ 278.80"))
    assert t.value == Decimal("278.80") and t.rule == "total:max-amount"


def test_total_rupiah():
    lines = L("Sub Total 58,000", "Tax 10% 5,800", "TOTAL 63,800", "CASH 100,000", "CHANGE 36,200")
    assert extract_total(lines).value == Decimal("63800.00")


def test_total_none_when_no_amount():
    assert extract_total(L("HELLO", "WORLD")) is None


# ------------------------------------------------------------------ TVA
def test_tax_rules():
    assert extract_tax(L("TOTAL TAX 1.55", "TOTAL 27.35")).value == Decimal("1.55")
    assert extract_tax(L("GST RM: 0.28", "Total 4.90")).value == Decimal("0.28")
    assert extract_tax(L("Tax 10% 5,800", "TOTAL 63,800")).value == Decimal("5800.00")
    assert extract_tax(L("GST ID No: 001210736640", "TAX INVOICE", "TOTAL 4.90")) is None
    assert extract_tax(L("GST 6% included in total")) is None


# ------------------------------------------------------------------ date
def test_date_prefers_date_keyword_and_skips_due():
    lines = L("Closed 05/01/2018", "Due Date: 30/01/2018", "Bill Date: 20/11/2017")
    assert extract_date(lines).value == "2017-11-20"


def test_date_repairs_letters_in_digits():
    assert extract_date(L("Date: 2O/11/2017")).value == "2017-11-20"


def test_date_textual_month_and_none():
    assert extract_date(L("22 Mar 18 12:44:00 PM")).value == "2018-03-22"
    assert extract_date(L("no date here", "TOTAL 4.90")) is None


# ------------------------------------------------------------------ fournisseur
def test_company_prefers_legal_suffix_over_logo_line():
    lines = L("Sy lee.", "OLDTOWN WHITE COFFEE", "Old Town Kopitiam Sdn Bhd (716269-x)", "Sri Rampai", "Tel: 03-4148 1876")
    c = extract_company(lines)
    assert c.value == "Old Town Kopitiam Sdn Bhd"          # numero d'enregistrement retire


def test_company_skips_headers_and_addresses():
    lines = L("TAX INVOICE", "Tel: 03-1234 5678", "TEO HENG STATIONERY & BOOKS", "NO 85, JALAN BESAR")
    assert extract_company(lines).value == "TEO HENG STATIONERY & BOOKS"


def test_company_none_on_empty():
    assert extract_company([]) is None
    assert extract_company(L("12345", "03-1234 5678")) is None


# ------------------------------------------------------------------ vrais textes OCR
@pytest.mark.parametrize("name", list(FIXTURES))
def test_real_ocr_outputs(name):
    case = FIXTURES[name]
    out = extract_rules(L(*case["lines"]))
    exp = case["expected"]
    assert str(out.fields.total) == exp["total"], out.evidence("total")
    if exp["date"]:
        assert str(out.fields.date) == exp["date"], out.evidence("date")
    if exp["tax"]:
        assert str(out.fields.tax) == exp["tax"], out.evidence("tax")
    if exp["company"]:
        assert out.fields.company.upper().replace(" ", "")[:14] == exp["company"].replace(" ", "")[:14]


def test_extract_rules_returns_fields_and_evidence():
    out = extract_rules(L("ACME SDN BHD", "Date: 01/02/2018", "GST RM: 0.28", "Total RM inc.GST: 4.90"))
    assert out.fields.company == "ACME SDN BHD" and str(out.fields.date) == "2018-02-01"
    assert out.fields.total == Decimal("4.90") and out.fields.tax == Decimal("0.28")
    assert out.found["total"].rule == "total:keyword" and "4.90" in out.evidence("total")
    assert out.ms >= 0
    empty = extract_rules([])
    assert empty.fields.total is None and empty.evidence("total") == ""
