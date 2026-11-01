from decimal import Decimal

import pytest

from docextract.normalize import normalize_amount, normalize_date, normalize_text


@pytest.mark.parametrize("raw, expected", [
    ("25/12/2018", "2018-12-25"),
    ("01/02/2018", "2018-02-01"),            # jour en premier (Malaisie)
    ("27-01-18", "2018-01-27"),
    ("27.01.2018", "2018-01-27"),
    ("2018-01-27", "2018-01-27"),
    ("2018/1/7", "2018-01-07"),
    ("22 MAR 18", "2018-03-22"),
    ("22 Mar 2018", "2018-03-22"),
    ("22-MAR-2018", "2018-03-22"),
    ("22MAR18", "2018-03-22"),
    ("MAR 22, 2018", "2018-03-22"),
    ("05 OGOS 2017", "2017-08-05"),          # mois malais
    ("Bill Date : 20/11/2017 03:57:01 PM", "2017-11-20"),
    ("01/27/2018", "2018-01-27"),            # mois > 12 -> format US
])
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "abc", "99/99/9999", "TOTAL 4.90", "12:30:45"])
def test_normalize_date_invalid(raw):
    assert normalize_date(raw) is None


@pytest.mark.parametrize("raw, expected", [
    ("9.00", "9.00"),
    ("RM 4.90", "4.90"),
    ("RM4.9", "4.90"),
    ("1,234.50", "1234.50"),
    ("1.234,50", "1234.50"),
    ("63,800", "63800.00"),                  # CORD : virgule = milliers
    ("1.234.567", "1234567.00"),
    ("9,50", "9.50"),
    ("30.25 ", "30.25"),
    ("278.80*", "278.80"),
    (" 127.20", "127.20"),
    ("0.285", "0.29"),                        # arrondi au centime
    (12.5, "12.50"),
    (Decimal("7"), "7.00"),
])
def test_normalize_amount(raw, expected):
    assert normalize_amount(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", [None, "", "TOTAL", "RM", True])
def test_normalize_amount_invalid(raw):
    assert normalize_amount(raw) is None


def test_normalize_text():
    assert normalize_text("  Teo Heng Stationery & Books ") == "TEO HENG STATIONERY & BOOKS"
    assert normalize_text("SDN. BHD.") == normalize_text("SDN BHD")
    assert normalize_text("...") is None


@pytest.mark.parametrize("raw, expected", [
    ("20180304", "2018-03-04"),      # YYYYMMDD
    ("25032018", "2018-03-25"),      # DDMMYYYY
    ("20180428", "2018-04-28"),
])
def test_normalize_date_compact(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw", ["12345678", "99999999", "Tel 03123456", "ID 20180304 X"])
def test_compact_date_only_when_whole_value(raw):
    assert normalize_date(raw) is None
