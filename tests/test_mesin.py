"""Uji mesin retrieval: tokenisasi, metrik, RRF, dan MRR M1/M2."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mesin  # noqa: E402

DATA = ROOT / "data"
KORPUS = json.loads((DATA / "korpus.json").read_text(encoding="utf-8"))
KUERI = json.loads((DATA / "kueri.json").read_text(encoding="utf-8"))
QRELS = json.loads((DATA / "qrels.json").read_text(encoding="utf-8"))
PAPER = {"M1": 0.390, "M2": 0.397}


def mrr(fn) -> float:
    nilai = []
    for q in KUERI:
        if q["qid"] not in QRELS:
            continue
        urut = fn(q["teks"], 10)
        nilai.append(mesin.rr(urut, set(QRELS[q["qid"]])))
    return sum(nilai) / len(nilai)


def test_data_sesuai_naskah():
    assert len(KORPUS) == 153
    assert len(KUERI) == 100
    assert len(QRELS) == 88
    assert sum(len(v) for v in QRELS.values()) == 281


def test_tokenisasi_sama_dengan_harness():
    """Tokenisasi dipatok pada keluaran harness evaluasi naskah."""
    contoh = json.loads((ROOT / "tests" / "data" / "token_contoh.json").read_text(encoding="utf-8"))
    for c in contoh:
        assert mesin.tokenize(c["teks"]) == c["token"], c["id"]


def test_m1_dan_m2_sesuai_naskah():
    m1 = mesin.TFIDF(KORPUS)
    m2 = mesin.BM25(KORPUS, k1=1.2, b=0.75)
    assert abs(mrr(lambda t, k: [d for d, _ in m1.cari(t, k)]) - PAPER["M1"]) <= 0.002
    assert abs(mrr(lambda t, k: [d for d, _ in m2.cari(t, k)]) - PAPER["M2"]) <= 0.002


def test_rrf_deterministik_dan_mengutamakan_yang_muncul_di_dua_daftar():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    assert mesin.rrf([a, b], k=60) == mesin.rrf([a, b], k=60)
    assert mesin.rrf([a, b], k=60)[0] in {"x", "y"}
    assert mesin.rrf([a], k=60) == a


def test_metrik_pada_contoh_kecil():
    urut = ["a", "b", "c", "d"]
    relevan = {"b", "d"}
    assert mesin.p_at_k(urut, relevan, 2) == 0.5
    assert mesin.r_at_k(urut, relevan, 4) == 1.0
    assert mesin.rr(urut, relevan) == 0.5
    assert mesin.rr(["x", "y"], relevan) == 0.0
    assert mesin.r_at_k(urut, set(), 4) == 0.0
