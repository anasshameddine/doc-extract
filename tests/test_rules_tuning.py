"""Etape 2b : cas reels tires des erreurs du jeu dev (lignes OCR collees dans l'analyse d'erreurs).

Chaque cas est une erreur qu'on a vue sur les vrais recus ; la verite est celle du jeu de donnees.
Sert de test de non-regression : un changement de regle ne doit pas les casser.
"""
from decimal import Decimal

import pytest

from docextract.normalize import normalize_field
from docextract.ocr import Line
from docextract.rules import extract_company, extract_date, extract_tax, extract_total


def L(*texts):
    return [Line(t, 0, 30 * i, 100, 30 * i + 20) for i, t in enumerate(texts)]


def same_company(found, gt):
    assert found is not None
    assert normalize_field("company", found.value) == normalize_field("company", gt), found.value


# ------------------------------------------------------------------ fournisseur
@pytest.mark.parametrize("lines, gt", [
    # nom coupe sur deux lignes : la ligne "SDN BHD" est rattachee a celle du dessus
    (["NANDO'S", "PERI-PERI CHICKEN", "NANDO'S CHICKENLAND MALAYSIA", "SDN BHD", "NANDO'S 1 MONT KIARA",
      "UNIT G-13"], "NANDO'S CHICKENLAND MALAYSIA SDN BHD"),
    (["PASARAYA BORONG PINTAR", "SDN BHD", "BR NO. :(124525-H)", "NO 19-G& 19-1& 19-2 JALAN TASIK UTAMA 4",
      "MEDAN NIAGA TASIK DAMAI", "016-5498845"], "PASARAYA BORONG PINTAR SDN BHD"),
    (["sii — ee", "POPULAR BOOK.", "» CO. (M) SDN BHD", "{Company No. L13825~bh}", ":",
      "(G57 Reg No. 001492992000}"], "POPULAR BOOK CO. (M) SDN BHD"),
    (["AIK HUAT HARDWARE", "ENTERPRISE (SETIA", "ALAM) SDN BHD", "822737-X", "NO. 17-G", "(X) U13/X"],
     "AIK HUAT HARDWARE ENTERPRISE (SETIA ALAM) SDN BHD"),
    # numero d'enregistrement ou sigle colle au nom
    (["GARDENIA BAKERIES (KL) SDN BHD (139386 X)", "LOT 3", "40300 SHAH ALAM", "TEL: 03- 55423228 FAX:03- 55423213",
      "GST ID: 000381399040", "TAX INVOICE / ADJUSTMENT NOTE"], "GARDENIA BAKERIES (KL) SDN BHD"),
    (["KING'S CONFECTIONERY S/B 273500-U (KSB)", "NO.41", "TAMAN SRI BINTANG", "52100 KUALA LUMPUR.",
      "TEL NO:- 012 - 602 8832", "(GST ID : 000294060032)"], "KING'S CONFECTIONERY S/B"),
    (["|", "99 SPEED MART S/B (5i9537-¥)", "LOT P.T, 2811, JALAN ANGSA,", "TANAN BERKELEY",
      "41150 KLANG, SELANGOR", '"'], "99 SPEED MART S/B"),
    # sous-titre entre parentheses : pas le nom
    (["TONYMOLY VIVA CITY", "(TONYMOLY SHOP (95874))", "UNIT L1-044B", "JALAN WAN ALWI", "93350 KUCHING",
      "GST ID : 000349356032"], "TONYMOLY VIVA CITY"),
])
def test_company_real_cases(lines, gt):
    same_company(extract_company(L(*lines)), gt)


def test_company_postcode_line_is_never_the_name():
    found = extract_company(L("99 SPEED MART 9/8 (519537: X)", "roT oT, 28ii, JALAN ANGSA,", "TAMAN BERKELEY",
                              "41150 KLANG, SCLANGGR", "1317 TRILLIUM 56 BES!"))
    assert found.value.startswith("99 SPEED MART")


def test_company_multi_line_keeps_evidence_of_both_lines():
    found = extract_company(L("PASARAYA BORONG PINTAR", "SDN BHD", "BR NO. :(124525-H)"))
    assert found.rule == "company:multi-line" and "PASARAYA" in found.evidence and "SDN BHD" in found.evidence


def test_company_single_line_is_untouched():
    found = extract_company(L("SUN WONG KUT SDN BHD", "Company No 20965-Y", "Site: 1046"))
    assert found.value == "SUN WONG KUT SDN BHD" and found.rule == "company:top-line-score"


def test_company_name_ending_with_legal_word_is_not_merged_upward():
    found = extract_company(L("BISTRO CAFE", "MY SHOP SDN BHD", "NO 1 JALAN ABC"))
    assert found.value == "MY SHOP SDN BHD"


# ------------------------------------------------------------------ total
@pytest.mark.parametrize("lines, gt", [
    # le total imprime apres l'arrondi, pas "Total (Inclusive of GST)"
    (["Total (Excluding GST): 56.20", "GST Payable: 3.71", "Total (Inclusive of GST): 65.53",
      "Rounding Adj: 0.02", "TOTAL : 65.55", "GST Summary Amount(RM) Tax(RM)"], "65.55"),
    (["Total RM Incl. of GsT 31.47", "Rounding Adj -0.02", "Total Rn 31.45", "CASH -50.00", "CHANGE 18.55"], "31.45"),
    (["TOTAL (INCLUSIVE OF GST): 67.98", "ROUNDING ADJ : 0.02", "TOTAL : 68.00"], "68.00"),
    # arrondi imprime seulement comme ajustement : total + ajustement
    (["Payment Type: Cash", "Total(InclGST,0%) RM15.39", "Rounding RM0.01", "Total Paid RM50.00",
      "Change RM34.60"], "15.40"),
    (["TOTAL SALES (INCLUSIVE GST) RM 72.74", "ROUNDING ADJUSTMENT RM .01", "ROUNDING RM 72.75",
      "CASH RM 100.00", "CHANGE RM 27.25", "GST SUMMARY AMOUNT(RM) TAX(RM)"], "72.75"),
    (["TOTAL SALES INCLUSIVE GST @ 0.00% 28.31", "ROUNDING ADJUSTMENT: -0.01", "TOTAL : 28.30",
      "CASH 50.00", "CHANGE : 21.70"], "28.30"),
    (["Sub-total 46.54", "Total Sales Incl GST 46.54", "Rounding Adj 6.01", "Total After Adj Incl GST 46.55",
      "CASH 100.00", "Item Count 2 Change Ant 53.45"], "46.55"),
    # DUE / PAY, et la ligne "(TOTAL INCLUDED GST @ 6% : 1.42)" n'est pas un total
    (["Rounding Adj - 0.00", "DUE 25.15", "PAY 25.15", "CHANGE 0.00", "(TOTAL INCLUDED GST @ 6% : 1.42)"], "25.15"),
    (["Rounding Adj 0.00", "Due 7.50", "Pay 20.00", "Change 12.50", "GOODS SOLD NOT RETURNABLE"], "7.50"),
    (["Total 0% supplies: 26.61", "Total 6% suppl ies (excl. GST): 33.96", "Total 6% supplies (Inc. GST): 35.99",
      "Total 0% supplies: 26.61", "Total Payable: 62. 60"], "62.60"),
    # fautes d'OCR sur "total", point decimal separe
    (["Uieteail : CASH", "Iotal After Rounding 100. 60", "SI summary Amount(RM) Tax(RM)"], "100.60"),
    (["lotal After Rounding 48. 95", "4.) summary Amount(RM) Tax(RM)"], "48.95"),
    (["Total RM = 88. 17", "Total Gross C RMN 86.1?"], "88.17"),
    # entier sans point (8400.) : une valeur lue ailleurs avec decimales gagne
    (["6 Type: 2 Total 8400.", "Rounding Adj 0.00", "Total Inclusive GST: 83.00", "CASH 85.00", "Change 2.00"], "83.00"),
    (["TOTAL 1160", "(ExcludedGST)SubTotal 1094", "Total GST 0.66", "TOTAL 11.60", "Total: 10.94 0.66"], "11.60"),
    # la monnaie rendue donne le vrai total quand le mot-cle trompe
    (["AMOUNT INCL. GST 5.00 RM", "ACCEPTED TOTAL 10.00 RM", "CHANGE 5.00 RM"], "5.00"),
    # CORD : Net Total n'est pas le total, SURTOTAL est un sous-total mal lu, separateurs de milliers bizarres
    (["Net Total: Rp 194,726", "TOTAL: Rp 224,908"], "224908"),
    (["SURTOTAL 156,364", "TOTAL 172,000"], "172000"),
    (["NETI ANT 70273", "TOTAL 8:000", "CASH 00048"], "8000"),
    (["SUBTOTAL 240.000", "TOTAL 287-760"], "287760"),
])
def test_total_real_cases(lines, gt):
    found = extract_total(L(*lines))
    assert found is not None and found.value == Decimal(gt), (found.value, found.rule)


def test_total_summary_row_with_several_amounts_is_last_resort():
    # "Total 43.90 2.64" (net, taxe) apres le vrai total
    found = extract_total(L("Total 46.55", "CASH 100.00", "CHANGE 53.45", "Code Net Amt GST", "Total 43.90 2.64"))
    assert found.value == Decimal("46.55")


def test_total_max_amount_skips_cash_and_change_lines():
    found = extract_total(L("ITEM A 12.50", "ITEM B 8.00", "CASH 100.00"))
    assert found.rule != "total:max-amount" or found.value == Decimal("12.50")


def test_total_rounding_not_applied_twice():
    # le total imprime apres l'arrondi est deja arrondi
    found = extract_total(L("Subtotal 30.24", "Rounding 0.01", "Total 30.25"))
    assert found.value == Decimal("30.25")


# ------------------------------------------------------------------ date
@pytest.mark.parametrize("line", ["HD03-04-06 - 5/40/160", "MI11/3-21 - 12/480", "KH12 - 10/500"])
def test_date_ignores_product_codes(line):
    assert extract_date(L("TOILET BRUSH 47-2311 *S", line, "9067860 1 X 2.70 2.70")) is None


def test_date_keeps_real_dates_next_to_words():
    assert extract_date(L("Date: 19-09-2017 12:44")).value == "2017-09-19"
    assert extract_date(L("Date12/01/2018")).value == "2018-01-12"
    assert extract_date(L("INV 100  05/02/2018")).value == "2018-02-05"


def test_date_implausible_year_is_rejected():
    assert extract_date(L("Ref 12/01/1975")) is None


# ------------------------------------------------------------------ TVA (CORD)
@pytest.mark.parametrize("line, gt", [
    ("Pajak Resto 13636", "13636"),
    ("P.Rest 10 5,592", "5592"),
    ("P. Resto 10% 2,700", "2700"),
    ("Pajak REST0 10% 15.908", "15908"),
    ("Pajak Resto (10%) 750", "750"),
    ("Pajak Rest 10> 16 170", "16170"),
    ("TAX AMT 727", "727"),
    ("TAX 7. 400", "7400"),
    ("PB 1 10% 8.100", "8100"),
    ("PB1: 0", "0"),
    ("Total Tax Included: Rp 5272.73", "5272.73"),
])
def test_tax_indonesian_keywords(line, gt):
    found = extract_tax(L("Sub Total 55,908", line, "Total 61,500"))
    assert found is not None and found.value == Decimal(gt), (found and (found.value, found.rule))


def test_tax_without_amount_is_none():
    assert extract_tax(L("PB1:", "70%", "Total 28,000")) is None


# ------------------------------------------------------------------ date collee a l'heure (RapidOCR sans espaces)
@pytest.mark.parametrize("line, iso", [
    ("Order No. 100021911/03/2018 06:20 PM", "2018-03-11"),
    ("Date:25-03-201809:42:19AM", "2018-03-25"),
    ("09/03/201821:28 0001", "2018-03-09"),
    ("14/03/20185:57:07PM 100/50", "2018-03-14"),
    ("Date: 20/03/201813:01", "2018-03-20"),
    ("Date:19-03-201818:08:38", "2018-03-19"),
    ("Date : 30/05/201710:18:02AM", "2017-05-30"),
    ("18/04/20185:18:08p 5ession:11444", "2018-04-18"),
    ("Date:2016-07-3113:48:37", "2016-07-31"),
    ("28Mar1818:32:36", "2018-03-28"),
    ("18Apr201818:34", "2018-04-18"),
    ("16May201818:30", "2018-05-16"),
    ("InVNo:KSB80051978 Date11/03/1821:30", "2018-03-11"),
    ("21/02/1811:1940278 07", "2018-02-21"),
    ("28-03-1818:05 SH01ZJ42 T2R000246362", "2018-03-28"),
    ("22-05-1612:18SH01 ZJ19 T2R000170536", "2016-05-22"),
])
def test_date_glued_to_time(line, iso):
    assert extract_date(L("TEL 03-1234 5678", line, "Total 5.00")).value == iso


def test_date_glue_split_does_not_break_clean_dates():
    assert extract_date(L("Date: 11/03/2018 06:20 PM")).value == "2018-03-11"
    assert extract_date(L("12/01/2018")).value == "2018-01-12"
    assert extract_date(L("DATE/TIME12/03/2018 10:05")).value == "2018-03-12"      # "TIME" n'est pas un code produit
