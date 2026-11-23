#!/usr/bin/env python3
"""Demo IR Mind-Ur-Set — M1..M4 tanpa LLM.

Sistem hanya mengembalikan peringkat chunk dari korpus tertutup beserta
halaman sumbernya. Tidak ada model generatif yang terhubung, sehingga tidak
ada risiko halusinasi pada keluaran.

Jalankan:
    streamlit run app.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import streamlit as st

from mesin import BM25, TFIDF, cari_dense, muat_model_dense, pilih_device, rrf, tokenize

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="Mind-Ur-Set · IR Demo", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------
# pemuatan artefak
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def muat_korpus():
    korpus = json.loads((DATA / "korpus.json").read_text(encoding="utf-8"))
    kueri = json.loads((DATA / "kueri.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATA / "qrels.json").read_text(encoding="utf-8"))
    emb = np.load(DATA / "emb_chunk.npy")
    return korpus, kueri, qrels, emb


@st.cache_resource(show_spinner=False)
def muat_leksikal(_jumlah: int):
    """M1 dan M2 dibangun di sini; biayanya ~0,03 detik untuk 153 chunk."""
    return TFIDF(korpus), BM25(korpus, k1=1.2, b=0.75)


@st.cache_resource(show_spinner=False)
def muat_dense():
    return muat_model_dense()


korpus, kueri_set, qrels, EMB = muat_korpus()
M1, M2 = muat_leksikal(len(korpus))
IDS = [d["id"] for d in korpus]
KOR = {d["id"]: d for d in korpus}
_cache_kueri: dict = {}


# --------------------------------------------------------------------------
# pencarian
# --------------------------------------------------------------------------
def cari_m1(teks: str, k: int = 10) -> list[str]:
    return [d for d, _ in M1.cari(teks, k)]


def cari_m2(teks: str, k: int = 10) -> list[str]:
    return [d for d, _ in M2.cari(teks, k)]


def cari_m3(teks: str, k: int = 10) -> list[str]:
    return cari_dense(muat_dense(), EMB, IDS, teks, k, _cache_kueri)


def cari_m4(teks: str, k: int = 10) -> list[str]:
    n = len(IDS)
    return rrf([cari_m2(teks, n), cari_m3(teks, n)], k=60)[:k]


METODE = {
    "M1 · VSM TF-IDF": cari_m1,
    "M2 · Okapi BM25": cari_m2,
    "M3 · BGE-M3 dense": cari_m3,
    "M4 · RRF hibrida": cari_m4,
}


def tumpang_tindih(kueri: str, doc_id: str) -> tuple[list[str], float]:
    qt = set(tokenize(kueri))
    dt = set(tokenize(KOR[doc_id]["teks"]))
    sama = sorted(qt & dt)
    return sama, (len(sama) / len(qt) if qt else 0.0)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🔎 Mind-Ur-Set · Demo Retrieval")
st.caption(
    "Korpus psikoedukasi tertutup 153 chunk · 100 kueri uji · tanpa LLM. "
    "Sistem hanya mengembalikan peringkat chunk, tidak menghasilkan teks."
)

with st.sidebar:
    st.header("Kontrol")
    mode = st.radio(
        "Mode",
        ["Kueri bebas", "Kueri uji (ada qrels)"],
        help="Kueri uji punya kunci relevansi, jadi ketepatan bisa dihitung.",
    )
    metode_pilih = st.multiselect(
        "Konfigurasi",
        list(METODE),
        default=["M2 · Okapi BM25", "M3 · BGE-M3 dense", "M4 · RRF hibrida"],
    )
    topk = st.slider("Kedalaman peringkat", 3, 20, 10)
    bahasa = st.radio("Saringan bahasa dokumen", ["Semua", "Indonesia", "Inggris"])
    tampil_teks = st.slider("Panjang teks ditampilkan (karakter)", 200, 1200, 400, step=100)
    st.divider()
    st.caption(
        "M1 VSM TF-IDF · M2 Okapi BM25 (k1=1,2 b=0,75) · "
        "M3 BGE-M3 1024-d · M4 RRF k=60"
    )
    st.caption(f"Perangkat dense: `{pilih_device()}`")

# ---------- input ----------
if mode == "Kueri bebas":
    teks_kueri = st.text_area(
        "Tulis keluhan / pertanyaan",
        value="saya tidak bisa tidur, pikiran saya terus berputar",
        height=80,
    )
    qid = None
else:
    opsi = {f"{q['qid']} · {q['teks'][:70]}": q for q in kueri_set if q["qid"] in qrels}
    pilih = st.selectbox("Pilih kueri uji", list(opsi))
    q = opsi[pilih]
    teks_kueri, qid = q["teks"], q["qid"]
    st.info(
        f"**{q['qid']}** ({q['bahasa']}) — "
        f"{len(qrels[qid])} chunk relevan menurut qrels final"
    )

jalan = st.button("Cari", type="primary", use_container_width=True)

if not jalan:
    st.stop()
if not teks_kueri.strip():
    st.warning("Kueri kosong.")
    st.stop()


# ---------- saringan bahasa ----------
def lolos(doc_id: str) -> bool:
    b = KOR[doc_id].get("bahasa")
    if bahasa == "Indonesia":
        return b == "id"
    if bahasa == "Inggris":
        return b == "en"
    return True


# ---------- eksekusi ----------
# Ambil seluruh peringkat lalu saring; kalau hanya top-K, saringan bahasa
# bisa mengosongkan hasil karena K teratas didominasi satu bahasa.
with st.spinner("Menghitung peringkat…"):
    N_SARING = len(IDS)
    hasil = {}
    for nama in metode_pilih:
        hasil[nama] = [d for d in METODE[nama](teks_kueri, N_SARING) if lolos(d)][:topk]

relevan = set(qrels[qid]) if qid else set()

# ---------- ringkasan ----------
if qid:
    st.subheader("Ringkasan ketepatan")
    kol = st.columns(len(hasil))
    for c, (nama, urut) in zip(kol, hasil.items()):
        hit = len([d for d in urut if d in relevan])
        p_at = hit / max(len(urut), 1)
        rr = next((1.0 / i for i, d in enumerate(urut, 1) if d in relevan), 0.0)
        c.metric(nama.split(" · ")[0], f"P@K {p_at:.2f}", f"RR {rr:.3f}", delta_color="off")
    st.caption("P@K = proporsi chunk relevan pada K teratas. RR = 1/peringkat relevan pertama.")

# ---------- hasil per metode ----------
for nama, urut in hasil.items():
    st.subheader(nama)
    if not urut:
        st.write("_tidak ada hasil_")
        continue

    for pos, doc_id in enumerate(urut, start=1):
        d = KOR[doc_id]
        benar = doc_id in relevan
        tanda = "✅" if benar else ("·" if qid else "")
        hal = d.get("halaman") or "—"
        judul = f"{tanda} **{pos}.** `{doc_id}` · {d.get('sumber','?')} · hal. {hal}"
        with st.expander(judul, expanded=(pos <= 2)):
            teks = d["teks"]
            st.write(teks[:tampil_teks] + ("…" if len(teks) > tampil_teks else ""))
            sama, cakupan = tumpang_tindih(teks_kueri, doc_id)
            st.caption(
                f"{d.get('token','?')} token · kemiripan istilah {cakupan:.0%} · "
                f"istilah sama: {', '.join(sama[:12]) if sama else '—'}"
            )
            st.caption(f"Lisensi: {d.get('lisensi','?')}")

# ---------- tabel perbandingan ----------
if len(hasil) > 1:
    st.subheader("Perbandingan peringkat")
    baris = []
    for pos in range(topk):
        row = {"#": pos + 1}
        for nama, urut in hasil.items():
            row[nama.split(" · ")[0]] = urut[pos] if pos < len(urut) else "—"
        baris.append(row)
    st.dataframe(baris, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Demo ini hanya menjalankan retrieval. Tidak ada model generatif yang terhubung, "
    "sehingga saran selalu berupa chunk asli dari korpus beserta halaman sumbernya."
)
