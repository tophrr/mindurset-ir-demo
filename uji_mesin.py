#!/usr/bin/env python3
"""Verifikasi MRR keempat konfigurasi terhadap angka naskah.

Tiga cara menjalankan:
    python uji_mesin.py                  # vektor acuan naskah (qid -> vektor tersimpan)
    python uji_mesin.py --backend onnx   # pengodean langsung dengan ONNX int8
    python uji_mesin.py --backend torch  # pengodean langsung dengan torch

Mode "acuan" memakai data/embedding/emb_kueri_cls.npy, yaitu vektor kueri yang
dipakai saat angka naskah dihitung, sehingga MRR M3/M4 seharusnya sama persis.
Mode "onnx"/"torch" mengodekan ulang dan akan bergeser beberapa persen.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

import encoder
from mesin import BM25, TFIDF, cari_dense_vektor, rr, rrf

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
PAPER = {"M1": 0.390, "M2": 0.397, "M3": 0.420, "M4": 0.500}


def hitung(backend: str, emb_path: pathlib.Path) -> dict[str, float]:
    korpus = json.loads((DATA / "korpus.json").read_text(encoding="utf-8"))
    kueri = json.loads((DATA / "kueri.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATA / "qrels.json").read_text(encoding="utf-8"))
    ids = [d["id"] for d in korpus]
    emb = np.load(emb_path)

    m1 = TFIDF(korpus)
    m2 = BM25(korpus, k1=1.2, b=0.75)
    acuan = encoder.vektor_kueri_acuan() if backend == "acuan" else {}
    enc = None if backend == "acuan" else encoder.muat(backend)
    kand = [q for q in kueri if q["qid"] in qrels]

    def vektor(q: dict) -> np.ndarray:
        if q["qid"] in acuan:
            return acuan[q["qid"]]
        return enc.kodekan(q["teks"])

    def m3(q: dict, k: int = 10) -> list[str]:
        return cari_dense_vektor(emb, ids, vektor(q), k)

    def m4(q: dict, k: int = 10) -> list[str]:
        return rrf([[d for d, _ in m2.cari(q["teks"], len(ids))], m3(q, len(ids))], k=60)[:k]

    hasil = {}
    hasil["M1"] = sum(rr([d for d, _ in m1.cari(q["teks"], 10)], set(qrels[q["qid"]]))
                      for q in kand) / len(kand)
    hasil["M2"] = sum(rr([d for d, _ in m2.cari(q["teks"], 10)], set(qrels[q["qid"]]))
                      for q in kand) / len(kand)
    hasil["M3"] = sum(rr(m3(q, 10), set(qrels[q["qid"]])) for q in kand) / len(kand)
    hasil["M4"] = sum(rr(m4(q, 10), set(qrels[q["qid"]])) for q in kand) / len(kand)
    return hasil


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["acuan", "onnx", "torch"], default="acuan")
    ap.add_argument("--emb", default=str(DATA / "embedding" / "emb_chunk_cls.npy"),
                    help="matriks vektor korpus; vektor naskah ada di data/embedding/")
    ap.add_argument("--toleransi", type=float, default=0.002)
    a = ap.parse_args()

    try:
        hasil = hitung(a.backend, pathlib.Path(a.emb))
    except (FileNotFoundError, KeyError) as e:
        print(f"gagal: {e}", file=sys.stderr)
        return 2

    print(f"backend: {a.backend} | vektor korpus: {a.emb}\n")
    beda = []
    for nama in ("M1", "M2", "M3", "M4"):
        d = hasil[nama] - PAPER[nama]
        tanda = "cocok" if abs(d) <= a.toleransi else ("dekat" if abs(d) < 0.03 else "BEDA")
        if tanda == "BEDA":
            beda.append(nama)
        print(f"  {nama}: MRR {hasil[nama]:.4f} vs naskah {PAPER[nama]:.3f}  selisih {d:+.4f}  {tanda}")

    if a.backend != "acuan":
        print("\nCatatan: pengodean ulang bergeser dari vektor naskah (kosinus ~0,98), "
              "jadi M3/M4 wajar berbeda.")
    if beda:
        print(f"\nGAGAL: {', '.join(beda)} menyimpang lebih dari 0,03 dari angka naskah.")
        return 1
    print("\nSemua konfigurasi dalam toleransi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
