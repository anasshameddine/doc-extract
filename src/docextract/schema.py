"""Schema de sortie commun aux 3 approches (valide par Pydantic).

Chaque approche renvoie un ExtractionResult : les champs sont normalises a la
creation, donc deux approches qui trouvent "22 MAR 18" et "22/03/2018" produisent
la meme valeur "2018-03-22".
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .normalize import normalize_amount, normalize_date

Method = Literal["rules", "llm", "docai"]


class ReceiptFields(BaseModel):
    """Champs cles d'un recu. None = non trouve."""
    model_config = ConfigDict(extra="ignore")

    company: str | None = Field(None, description="Fournisseur / enseigne")
    date: dt.date | None = Field(None, description="Date du recu (ISO)")
    total: Decimal | None = Field(None, ge=0, description="Montant total paye")
    tax: Decimal | None = Field(None, ge=0, description="Montant de TVA / GST")

    @field_validator("company", mode="before")
    @classmethod
    def _company(cls, v):
        if v is None:
            return None
        v = " ".join(str(v).split())
        return v or None

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v):
        if v is None or isinstance(v, dt.date):
            return v
        return normalize_date(str(v))   # illisible -> None (pas d'exception)

    @field_validator("total", "tax", mode="before")
    @classmethod
    def _amount(cls, v):
        return normalize_amount(v)


class ExtractionResult(BaseModel):
    """Sortie d'une approche pour un document (ce que renverra l'API)."""
    doc_id: str | None = None
    method: Method
    fields: ReceiptFields
    latency_ms: float = Field(..., ge=0)
    error: str | None = None          # message si l'approche a echoue
