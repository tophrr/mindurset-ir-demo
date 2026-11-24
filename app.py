#!/usr/bin/env python3
"""Demo IR Mind-Ur-Set — M1..M4 tanpa LLM.

Sistem hanya mengembalikan peringkat chunk dari korpus tertutup beserta halaman
sumbernya. Tidak ada model generatif yang terhubung, sehingga tidak ada risiko
halusinasi pada keluaran.

Dua tab: "Pencarian" untuk mencoba kueri, "Evaluasi" untuk menghitung ulang MRR
88 kueri uji dan membandingkannya dengan angka naskah.

Jalankan:
    streamlit run app.py
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import streamlit as st

import encoder
from mesin import BM25, TFIDF, cari_dense_vektor, rrf, tokenize

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
EMB = DATA / "embedding" / "emb_chunk_cls.npy"
PAPER = {"M1": 0.390, "M2": 0.397, "M3": 0.420, "M4": 0.500}
MAKS_KARAKTER = 500

FRASA_RISIKO = (
    "bunuh diri", "mengakhiri hidup", "mengakhiri semuanya", "menyakiti diri",
    "tidak ingin hidup", "pengen mati", "ingin mati", "mati saja",
    "suicide", "kill myself", "end my life", "self-harm", "self harm", "hurt myself",
    "better off dead", "want to die",
)

st.set_page_config(page_title="Mind-Ur-Set · IR Demo", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------
# pemuatan artefak
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def muat_data():
    korpus = json.loads((DATA / "korpus.json").read_text(encoding="utf-8"))
    kueri = json.loads((DATA / "kueri.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATA / "qrels.json").read_text(encoding="utf-8"))
    emb = np.load(EMB) if EMB.exists() else None
    return korpus, kueri, qrels, emb


@st.cache_resource(show_spinner=False)
def muat_leksikal(_jumlah: int):
    """M1 dan M2 dibangun di sini; biayanya ~0,03 detik untuk 153 chunk."""
    return TFIDF(KORPUS), BM25(KORPUS, k1=1.2, b=0.75)


@st.cache_resource(show_spinner=False)
def muat_encoder(backend: str):
    return encoder.muat(backend)


KORPUS, KUERI, QRELS, EMBED = muat_data()
IDS = [d["id"] for d in KORPUS]
KOR = {d["id"]: d for d in KORPUS}
M1, M2 = muat_leksikal(len(KORPUS))

BACKEND_TERSEDIA = [b for b in ("onnx", "torch") if encoder.tersedia(b)]
NAMA_BACKEND = {"onnx": "ONNX int8 (angka naskah)", "torch": "torch fp32 (pembanding)"}


def deteksi_risiko(teks: str) -> list[str]:
    kecil = teks.lower()
    return [f for f in FRASA_RISIKO if f in kecil]


def tumpang_tindih(kueri: str, doc_id: str) -> tuple[list[str], float]:
    qt = set(tokenize(kueri))
    dt = set(tokenize(KOR[doc_id]["teks"]))
    sama = sorted(qt & dt)
    return sama, (len(sama) / len(qt) if qt else 0.0)


def pencari(backend: str | None):
    """Kembalikan kamus fungsi pencarian; M3/M4 hanya bila encoder tersedia."""
    def m1(teks: str, k: int = 10) -> list[str]:
        return [d for d, _ in M1.cari(teks, k)]

    def m2(teks: str, k: int = 10) -> list[str]:
        return [d for d, _ in M2.cari(teks, k)]

    fungsi = {"M1 · VSM TF-IDF": m1, "M2 · Okapi BM25": m2}
    if backend and EMBED is not None:
        def m3(teks: str, k: int = 10) -> list[str]:
            return cari_dense_vektor(EMBED, IDS, muat_encoder(backend).kodekan(teks), k)

        def m4(teks: str, k: int = 10) -> list[str]:
            return rrf([m2(teks, len(IDS)), m3(teks, len(IDS))], k=60)[:k]

        fungsi["M3 · BGE-M3 dense"] = m3
        fungsi["M4 · RRF hibrida"] = m4
    return fungsi


# --------------------------------------------------------------------------
# kepala halaman
# --------------------------------------------------------------------------
st.title("🔎 Mind-Ur-Set · Demo Retrieval")
st.warning(
    "**Demo penelitian, bukan nasihat klinis.** Sistem hanya menampilkan kutipan dari korpus "
    "psikoedukasi tertutup dan tidak menegakkan diagnosis. Bila Anda atau orang di sekitar Anda "
    "dalam bahaya, hubungi layanan darurat atau **119 ext. 8 (SEJIWA)**.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Kontrol")
    if BACKEND_TERSEDIA:
        backend = st.selectbox(
            "Backend dense",
            BACKEND_TERSEDIA,
            index=0,
            help="ONNX int8 memakai vektor yang sama dengan naskah, sehingga M3/M4 cocok "
                 "dengan angka paper. torch hanya pembanding.",
        )
        st.caption(f"Backend aktif: **{NAMA_BACKEND[backend]}**")
    else:
        backend = None
        st.caption("Encoder dense tidak tersedia — hanya M1 dan M2 yang bisa dipakai.")
    st.divider()
    st.caption("M1 VSM TF-IDF · M2 Okapi BM25 (k1=1,2 b=0,75) · M3 BGE-M3 1024-d · M4 RRF k=60")
    st.caption(f"Korpus: {len(KORPUS)} chunk · kueri: {len(KUERI)} · ber-qrels: {len(QRELS)}")

tab_cari, tab_eval = st.tabs(["Pencarian", "Evaluasi"])

# ==========================================================================
# TAB 1 — pencarian
# ==========================================================================
with tab_cari:
    fungsi = pencari(backend)
    tersedia_nama = list(fungsi)

    kol_kiri, kol_kanan = st.columns([2, 1])
    with kol_kiri:
        mode = st.radio(
            "Mode",
            ["Kueri bebas", "Kueri uji (ada qrels)"],
            horizontal=True,
            help="Kueri uji punya kunci relevansi, jadi ketepatan bisa dihitung.",
        )
        if mode == "Kueri bebas":
            teks_kueri = st.text_area(
                "Tulis keluhan / pertanyaan",
                value="saya tidak bisa tidur, pikiran saya terus berputar",
                height=90,
                max_chars=MAKS_KARAKTER,
            )
            qid = None
        else:
            opsi = {f"{q['qid']} · {q['teks'][:70]}": q for q in KUERI if q["qid"] in QRELS}
            pilih = st.selectbox("Pilih kueri uji", list(opsi))
            q = opsi[pilih]
            teks_kueri, qid = q["teks"], q["qid"]
            st.info(f"**{qid}** ({q['bahasa']}) — {len(QRELS[qid])} chunk relevan menurut qrels final")
    with kol_kanan:
        bawaan = [n for n in tersedia_nama if n.startswith(("M2", "M3", "M4"))] or tersedia_nama[:2]
        metode_pilih = st.multiselect("Konfigurasi", tersedia_nama, default=bawaan)
        topk = st.slider("Kedalaman peringkat", 3, 20, 10)
        bahasa = st.radio("Saringan bahasa dokumen", ["Semua", "Indonesia", "Inggris"])
        tampil_teks = st.slider("Panjang teks ditampilkan (karakter)", 200, 1200, 400, step=100)

    jalan = st.button("Cari", type="primary", width="stretch")

    if jalan and not teks_kueri.strip():
        st.warning("Kueri kosong.")

    if jalan and teks_kueri.strip():
        frasa = deteksi_risiko(teks_kueri)
        if frasa:
            st.error(
                "**Kueri ini menyentuh tema risiko tinggi.** Demo ini tidak dapat menilai risiko "
                "dan tidak memberikan pertolongan. Hubungi **119 ext. 8 (SEJIWA)**, layanan darurat "
                "setempat, atau orang terdekat sekarang. Hasil di bawah hanya bahan bacaan.",
                icon="🚨",
            )

        def lolos(doc_id: str) -> bool:
            b = KOR[doc_id].get("bahasa")
            if bahasa == "Indonesia":
                return b == "id"
            if bahasa == "Inggris":
                return b == "en"
            return True

        if not metode_pilih:
            st.info("Pilih sekurangnya satu konfigurasi.")
        else:
            with st.spinner("Menghitung peringkat…"):
                if backend:
                    muat_encoder(backend)   # pemanasan; biaya muat model tidak dihitung
                awal = time.perf_counter()
                hasil = {nama: [d for d in fungsi[nama](teks_kueri, len(IDS)) if lolos(d)][:topk]
                         for nama in metode_pilih}
                lama = (time.perf_counter() - awal) * 1000

            relevan = set(QRELS.get(qid, [])) if qid else set()

            if qid:
                st.subheader("Ringkasan ketepatan")
                kol = st.columns(len(hasil))
                for c, (nama, urut) in zip(kol, hasil.items()):
                    hit = len([d for d in urut if d in relevan])
                    p_at = hit / max(len(urut), 1)
                    rr = next((1.0 / i for i, d in enumerate(urut, 1) if d in relevan), 0.0)
                    c.metric(nama.split(" · ")[0], f"P@K {p_at:.2f}", f"RR {rr:.3f}",
                             delta_color="off")
                st.caption("P@K = proporsi chunk relevan pada K teratas. RR = 1/peringkat relevan pertama.")

            st.caption(
                f"Latensi pencarian: **{lama:.0f} ms** untuk {len(hasil)} konfigurasi "
                f"(kedalaman {topk}, backend {backend or 'leksikal saja'})."
            )

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
                    judul = f"{tanda} **{pos}.** `{doc_id}` · {d.get('sumber', '?')} · hal. {hal}"
                    with st.expander(judul, expanded=(pos <= 2)):
                        teks = d["teks"]
                        st.write(teks[:tampil_teks] + ("…" if len(teks) > tampil_teks else ""))
                        sama, cakupan = tumpang_tindih(teks_kueri, doc_id)
                        st.caption(
                            f"{d.get('token', '?')} token · kemiripan istilah {cakupan:.0%} · "
                            f"istilah sama: {', '.join(sama[:12]) if sama else '—'}"
                        )
                        st.caption(f"Lisensi: {d.get('lisensi', '?')}")

            if len(hasil) > 1:
                st.subheader("Perbandingan peringkat")
                baris = [{"#": pos + 1,
                          **{nama.split(" · ")[0]: (urut[pos] if pos < len(urut) else "—")
                             for nama, urut in hasil.items()}}
                         for pos in range(topk)]
                st.dataframe(baris, width="stretch", hide_index=True)

# ==========================================================================
# TAB 2 — evaluasi
# ==========================================================================
with tab_eval:
    st.subheader("Hitung ulang MRR pada 88 kueri uji")
    st.caption(
        "Menjalankan konfigurasi yang sama dengan tab Pencarian pada seluruh kueri yang punya "
        "qrels, lalu membandingkan hasilnya dengan Tabel III naskah."
    )
    if not BACKEND_TERSEDIA:
        st.info("Encoder dense tidak tersedia; evaluasi M3/M4 dilewati.")
    jalan_eval = st.button("Hitung MRR", type="primary")

    if jalan_eval:
        acuan = encoder.vektor_kueri_acuan()
        enc = muat_encoder(backend) if (backend and EMBED is not None) else None
        fungsi = pencari(backend)

        def vektor_uji(q: dict):
            """Vektor acuan naskah bila ada; kalau tidak, kodekan dengan encoder aktif."""
            if q["qid"] in acuan:
                return acuan[q["qid"]]
            if enc is None:
                raise RuntimeError("encoder dense tidak tersedia untuk kueri ini")
            return enc.kodekan(q["teks"])
        kand = [q for q in KUERI if q["qid"] in QRELS]
        progress = st.progress(0.0, text="menghitung…")
        hasil, lama = {}, {}
        for i, nama in enumerate(fungsi):
            awal = time.perf_counter()
            nilai = []
            for q in kand:
                relevan = set(QRELS[q["qid"]])
                if nama.startswith("M3") or nama.startswith("M4"):
                    v = vektor_uji(q)
                    if nama.startswith("M4"):
                        urut = rrf([[d for d, _ in M2.cari(q["teks"], len(IDS))],
                                    cari_dense_vektor(EMBED, IDS, v, len(IDS))], k=60)[:10]
                    else:
                        urut = cari_dense_vektor(EMBED, IDS, v, 10)
                else:
                    urut = fungsi[nama](q["teks"], 10)
                nilai.append(next((1.0 / r for r, d in enumerate(urut, 1) if d in relevan), 0.0))
            hasil[nama] = sum(nilai) / len(nilai)
            lama[nama] = time.perf_counter() - awal
            progress.progress((i + 1) / len(fungsi), text=f"{nama} selesai")
        progress.empty()
        st.caption(
            "M3/M4 memakai vektor kueri acuan naskah (data/embedding/emb_kueri_cls.npy) bila "
            "tersedia, sehingga angkanya sama dengan Tabel III."
        )

        baris = []
        for nama, nilai in hasil.items():
            kode = nama.split(" · ")[0]
            acuan = PAPER.get(kode)
            baris.append({
                "Konfigurasi": nama,
                "MRR": round(nilai, 3),
                "Naskah": acuan,
                "Selisih": None if acuan is None else round(nilai - acuan, 3),
                "Waktu (s)": round(lama[nama], 1),
            })
        st.dataframe(baris, width="stretch", hide_index=True)
        st.caption(
            f"{len(kand)} kueri ber-qrels. Selisih diharapkan 0,000 untuk M1/M2 dan ≤0,002 untuk "
            "M3/M4 bila backend ONNX dan vektor naskah dipakai."
        )

st.divider()
st.caption(
    "Demo ini hanya menjalankan retrieval. Tidak ada model generatif yang terhubung, sehingga "
    "saran selalu berupa chunk asli dari korpus beserta halaman sumbernya."
)
