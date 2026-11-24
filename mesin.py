#!/usr/bin/env python3
"""Mesin retrieval untuk demo Mind-Ur-Set.

Berisi empat konfigurasi yang dibandingkan pada paper:
    M1  VSM TF-IDF + cosine similarity
    M2  Okapi BM25 (k1=1.2, b=0.75)
    M3  dense embedding BAAI/bge-m3 (dijalankan terpisah, butuh torch)
    M4  Reciprocal Rank Fusion atas peringkat M2 dan M3

Modul ini berdiri sendiri: hanya memakai pustaka standar Python untuk M1, M2,
dan M4. Model dense dimuat malas (lazy) supaya M1/M2 tetap bisa dipakai tanpa
torch terpasang.

Implementasi TF-IDF, BM25, dan RRF di bawah ini ditulis ulang dari nol agar
tidak bergantung pada berkas di luar repositori. Rumusnya mengikuti paper
(Bagian III.D), sehingga angka MRR dapat direproduksi.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict


# --------------------------------------------------------------------------
# tokenisasi
# --------------------------------------------------------------------------
# Daftar stopword dan pola token disamakan dengan harness evaluasi paper
# (scripts/retrieval_eval.py) supaya angka MRR dapat dibandingkan langsung.
STOPWORDS = set("""
a an the and or of to in on for with at by from as is are was were be been being
this that these those it its he she they them his her their our your my me i you we
not no do does did have has had will would can could should may might must
dan atau yang di ke dari untuk pada dengan dalam adalah ini itu akan tidak bukan
juga saja sudah telah para kami kita mereka dia ia aku kamu anda nya
""".split())

KATA = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(teks: str) -> list[str]:
    """Token alfanumerik huruf kecil tanpa stopword dan tanpa token satu huruf."""
    return [t for t in KATA.findall(teks.lower()) if t not in STOPWORDS and len(t) > 1]


# --------------------------------------------------------------------------
# M1: VSM TF-IDF + cosine
# --------------------------------------------------------------------------
class TFIDF:
    """VSM dengan pembobotan TF-IDF dan kemiripan kosinus.

    Bobot istilah: tf(t,d) x log(N/df(t)), lalu vektor dinormalkan panjang
    satuan sehingga cosine similarity cukup berupa hasil kali titik.
    """

    def __init__(self, korpus: list[dict]):
        self.ids = [d["id"] for d in korpus]
        docs = [Counter(tokenize(d["teks"])) for d in korpus]
        self.N = len(docs)
        df: Counter = Counter()
        for tf in docs:
            df.update(tf.keys())
        # idf tidak pernah nol: istilah yang muncul di semua dokumen diberi
        # bobot sangat kecil, bukan nol, agar tetap dapat membedakan peringkat.
        self.idf = {t: math.log(self.N / c) for t, c in df.items()}
        self.vecs = [self._bobot(tf) for tf in docs]

    def _bobot(self, tf: Counter) -> dict[str, float]:
        # log-TF meredam dominasi istilah yang muncul berkali-kali
        v = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items() if t in self.idf}
        norma = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norma for t, x in v.items()}

    def cari(self, kueri: str, K: int = 10) -> list[tuple[str, float]]:
        q = self._bobot(Counter(tokenize(kueri)))
        skor = [sum(w * q.get(t, 0.0) for t, w in v.items()) for v in self.vecs]
        urut = sorted(zip(self.ids, skor), key=lambda x: (-x[1], x[0]))
        return urut[:K]


# --------------------------------------------------------------------------
# M2: Okapi BM25
# --------------------------------------------------------------------------
class BM25:
    """Okapi BM25 dengan saturasi frekuensi dan normalisasi panjang dokumen."""

    def __init__(self, korpus: list[dict], k1: float = 1.2, b: float = 0.75):
        self.ids = [d["id"] for d in korpus]
        self.docs = [Counter(tokenize(d["teks"])) for d in korpus]
        self.N = len(self.docs)
        self.k1, self.b = k1, b
        self.panjang = [sum(tf.values()) for tf in self.docs]
        self.avgdl = (sum(self.panjang) / self.N) if self.N else 1.0

        df: Counter = Counter()
        for tf in self.docs:
            df.update(tf.keys())
        self.idf = {
            t: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def cari(self, kueri: str, K: int = 10) -> list[tuple[str, float]]:
        qt = tokenize(kueri)
        skor: list[float] = []
        for i, tf in enumerate(self.docs):
            s = 0.0
            norma = self.k1 * (1 - self.b + self.b * self.panjang[i] / self.avgdl)
            for t in qt:
                f = tf.get(t)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + norma)
            skor.append(s)
        urut = sorted(zip(self.ids, skor), key=lambda x: (-x[1], x[0]))
        return urut[:K]


# --------------------------------------------------------------------------
# M3: dense (lazy, butuh torch + sentence-transformers)
# --------------------------------------------------------------------------
def pilih_device() -> str:
    """Pilih perangkat tercepat yang tersedia, lintas sistem operasi."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def muat_model_dense(nama: str = "BAAI/bge-m3"):
    """Muat encoder dense. Impor torch ditunda sampai benar-benar dipakai."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nama, device=pilih_device())


def cari_dense_vektor(emb_korpus, ids: list[str], vektor_kueri, K: int = 10) -> list[str]:
    """Peringkat dense dari vektor kueri yang sudah dihitung (ONNX atau torch)."""
    import numpy as np

    v = np.asarray(vektor_kueri, dtype=np.float32)
    skor = emb_korpus @ (v / (np.linalg.norm(v) or 1.0))
    return [ids[i] for i in np.argsort(-skor)[:K]]


def cari_dense(model, emb_korpus, ids: list[str], kueri: str, K: int = 10,
               cache_kueri: dict | None = None) -> list[str]:
    """Peringkat dense dari embedding korpus yang sudah dihitung luring."""
    import numpy as np

    if cache_kueri is not None and kueri in cache_kueri:
        q = cache_kueri[kueri]
    else:
        q = model.encode([kueri], normalize_embeddings=True, convert_to_numpy=True)[0]
        if cache_kueri is not None:
            cache_kueri[kueri] = q
    skor = emb_korpus @ q.astype(np.float32)
    return [ids[i] for i in np.argsort(-skor)[:K]]


# --------------------------------------------------------------------------
# M4: Reciprocal Rank Fusion
# --------------------------------------------------------------------------
def rrf(daftar: list[list[str]], k: int = 60) -> list[str]:
    """Gabungkan beberapa daftar peringkat berdasarkan posisi, bukan skor.

    Skor dokumen = jumlah 1/(k + posisi) atas semua daftar tempat ia muncul.
    """
    skor: dict[str, float] = defaultdict(float)
    for hasil in daftar:
        for posisi, doc_id in enumerate(hasil, start=1):
            skor[doc_id] += 1.0 / (k + posisi)
    return sorted(skor, key=lambda d: (-skor[d], d))


# --------------------------------------------------------------------------
# metrik evaluasi
# --------------------------------------------------------------------------
def p_at_k(urut: list[str], relevan: set[str], k: int) -> float:
    return sum(1 for d in urut[:k] if d in relevan) / k


def r_at_k(urut: list[str], relevan: set[str], k: int) -> float:
    if not relevan:
        return 0.0
    return sum(1 for d in urut[:k] if d in relevan) / len(relevan)


def rr(urut: list[str], relevan: set[str]) -> float:
    """Reciprocal rank dokumen relevan pertama."""
    for i, d in enumerate(urut, start=1):
        if d in relevan:
            return 1.0 / i
    return 0.0
