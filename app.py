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
import re
import time

import numpy as np
import streamlit as st

import encoder
import ui_tema
from mesin import BM25, TFIDF, cari_dense_vektor, rrf, tokenize

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
EMB = DATA / "embedding" / "emb_chunk_cls.npy"
PAPER = {"M1": 0.390, "M2": 0.397, "M3": 0.420, "M4": 0.500}
RINGKAS = {"M1": "VSM", "M2": "BM25", "M3": "BGE-M3", "M4": "RRF"}
MAKS_KARAKTER = 500

NAMA_SUMBER = {
    "who-dwm-2020-eng": "WHO DWM (EN)",
    "who-dwm-2020-ind": "WHO DWM (ID)",
    "who-pm-18-5-eng": "WHO PM+ (EN)",
    "who-pm-18-5-id": "WHO PM+ (ID)",
    "mindurset-cards-v1": "Kartu koping",
    "modul-thought-record": "Thought record",
}

# Pola risiko: teks dinormalisasi dulu (huruf kecil, tanda baca jadi spasi) lalu dicocokkan
# sebagai kata, supaya variasi seperti "bundir", "mau mati", "gak kuat hidup" ikut tertangkap.
POLA_RISIKO = (
    r"\bbunuh\s*diri\b", r"\bbundir\b", r"\bsuicid\w*\b", r"\bkill\s+my\s?self\b",
    r"\bend\s+my\s+life\b", r"\bwant\s+to\s+die\b", r"\bbetter\s+off\s+dead\b",
    r"\b(mau|pengen|ingin|pingin)\s+mati\b", r"\bmati\s+aja\b",
    r"\b(tidak|gak|nggak|enggak)\s+(ingin|mau)\s+hidup\b", r"\bcapek\s+hidup\b",
    r"\bgak\s+kuat\s+hidup\b", r"\blebih\s+baik\s+mati\b", r"\bakhiri\s+hidup\w*\b",
    r"\bmengakhiri\s+semuanya\b", r"\bmenyakiti\s+diri\b", r"\bnyakitin\s+diri\b",
    r"\bmelukai\s+diri\b", r"\bsayat\b", r"\bself[\s-]?harm\b", r"\bself[\s-]?injur\w*\b",
    r"\bhurt\s+my\s?self\b", r"\boverdos\w*\b", r"\bgantung\s+diri\b",
)
_RE_RISIKO = re.compile("|".join(POLA_RISIKO), re.IGNORECASE)

st.set_page_config(page_title="Mind-Ur-Set Information Retrieval", page_icon="🔎", layout="wide")
ui_tema.pasang()


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

# Di hosting tanpa penyimpanan tetap (Streamlit Community Cloud meng-clone ulang repo setiap
# reboot) model dense belum ada. Unduh sekali di awal agar M3/M4 tetap bisa dipakai; bila
# gagal, aplikasi tetap hidup dengan M1 dan M2 saja.
if not BACKEND_TERSEDIA and encoder.bisa_unduh("onnx"):
    try:
        import unduh_model

        with st.spinner("Mengunduh model BGE-M3 int8 (±559 MB) — sekali per sesi…"):
            unduh_model.pastikan(senyap=True)
        BACKEND_TERSEDIA = ["onnx"]
    except Exception as e:  # noqa: BLE001
        st.caption(f"Model dense gagal disiapkan ({e}); demo berjalan dengan M1 dan M2.")


def deteksi_risiko(teks: str) -> list[str]:
    rapih = " ".join(re.sub(r"[^\w\s]", " ", teks.lower()).replace("_", " ").split())
    return sorted({m.group(0) for m in _RE_RISIKO.finditer(rapih)})


def rentang_halaman(halaman) -> str:
    """[77, 78, 79, 82] -> '77–79, 82'."""
    if not halaman:
        return ""
    if isinstance(halaman, int):
        return str(halaman)
    angka = sorted({int(h) for h in halaman})
    hasil, awal = [], angka[0]
    for i, a in enumerate(angka):
        b = angka[i + 1] if i + 1 < len(angka) else None
        if b is None or b != a + 1:
            hasil.append(f"{awal}" if awal == a else f"{awal}–{a}")
            if b is not None:
                awal = b
    return ", ".join(hasil)


def lokasi(d: dict) -> str:
    """'WHO DWM (ID) · hal. 77–80' atau 'Kartu koping · tanpa nomor halaman'."""
    sumber = NAMA_SUMBER.get(d.get("sumber", ""), d.get("sumber", "?"))
    hal = rentang_halaman(d.get("halaman"))
    return f"{sumber} · hal. {hal}" if hal else f"{sumber} · tanpa nomor halaman"


def tumpang_tindih(kueri: str, doc_id: str) -> tuple[list[str], float]:
    qt = set(tokenize(kueri))
    dt = set(tokenize(KOR[doc_id]["teks"]))
    sama = sorted(qt & dt)
    return sama, (len(sama) / len(qt) if qt else 0.0)


def pencari(backend: str | None):
    """Kamus fungsi pencarian; M3/M4 hanya bila encoder tersedia."""
    def m1(teks: str, k: int = 10) -> list[str]:
        return [d for d, s in M1.cari(teks, k) if s > 0]

    def m2(teks: str, k: int = 10) -> list[str]:
        return [d for d, s in M2.cari(teks, k) if s > 0]

    fungsi = {"M1 · VSM TF-IDF": m1, "M2 · Okapi BM25": m2}
    if backend and EMBED is not None:
        def m3(teks: str, k: int = 10) -> list[str]:
            return cari_dense_vektor(EMBED, IDS, muat_encoder(backend).kodekan(teks), k)

        def m4(teks: str, k: int = 10) -> list[str]:
            return rrf([m2(teks, len(IDS)), m3(teks, len(IDS))], k=60)[:k]

        fungsi["M3 · BGE-M3 dense"] = m3
        fungsi["M4 · RRF hibrida"] = m4
    return fungsi


def tabel_perbandingan(hasil: dict[str, list[str]], relevan: set[str], topk: int) -> str:
    """Tabel HTML: satu baris per peringkat, satu kolom per konfigurasi."""
    kode = [n.split(" · ")[0] for n in hasil]
    kepala = "".join(
        f'<th><span class="kode">{k}</span><span class="metode">{n.split(" · ")[1]}</span></th>'
        for k, n in zip(kode, hasil)
    )
    baris = []
    for pos in range(topk):
        sel = [f'<td class="peringkat">{pos + 1}</td>']
        for nama in hasil:
            urut = hasil[nama]
            if pos >= len(urut):
                sel.append('<td class="kosong">—</td>')
                continue
            doc_id = urut[pos]
            d = KOR[doc_id]
            kelas = "sel relevan" if doc_id in relevan else "sel"
            tanda = '<span class="tanda">✓</span>' if doc_id in relevan else ""
            sel.append(
                f'<td class="{kelas}"><span class="id">{doc_id}</span>{tanda}'
                f'<span class="ket">{lokasi(d)}</span></td>'
            )
        baris.append("<tr>" + "".join(sel) + "</tr>")

    return f"""
<style>
.tabel-ir {{ width:100%; border-collapse:separate; border-spacing:0; font-size:0.86rem;
             border:1px solid #D3DDF0; border-radius:10px; overflow:hidden; background:#fff; }}
.tabel-ir th {{ background:#203080; color:#fff; text-align:left; padding:8px 10px; font-weight:600;
                vertical-align:bottom; }}
.tabel-ir th .kode {{ display:block; font-size:0.95rem; letter-spacing:0.02em; }}
.tabel-ir th .metode {{ display:block; font-size:0.72rem; font-weight:400; opacity:0.85; }}
.tabel-ir td {{ padding:7px 10px; border-top:1px solid #D3DDF0; vertical-align:top; color:#16225C; }}
.tabel-ir tr:nth-child(even) td {{ background:#F7FAFF; }}
.tabel-ir td.peringkat {{ width:38px; color:#565E70; font-variant-numeric:tabular-nums; }}
.tabel-ir td.kosong {{ color:#9AA3B2; text-align:center; }}
.tabel-ir .id {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:0.8rem;
                 color:#16225C; }}
.tabel-ir .ket {{ display:block; font-size:0.72rem; color:#565E70; margin-top:2px; }}
.tabel-ir .tanda {{ color:#2E7D32; font-weight:700; margin-left:6px; }}
.tabel-ir td.relevan, .tabel-ir tr:nth-child(even) td.relevan {{
    background:#E3F2E7; box-shadow:inset 3px 0 0 #2E7D32; }}
</style>
<table class="tabel-ir">
  <thead><tr><th class="peringkat">#</th>{kepala}</tr></thead>
  <tbody>{''.join(baris)}</tbody>
</table>
"""


def kartu_bantuan() -> None:
    st.error(
        "**Kueri ini menyentuh tema risiko tinggi.** Demo ini tidak dapat menilai risiko dan tidak "
        "memberikan pertolongan. Hubungi **119 ext. 8 (SEJIWA)**, **112**, atau IGD terdekat "
        "sekarang. Di luar Indonesia: [findahelpline.com](https://findahelpline.com).",
        icon="🚨",
    )
    st.caption(
        "Kutipan di bawah hanya bahan bacaan psikoedukasi, bukan penanganan krisis, dan sengaja "
        "tidak dibuka otomatis."
    )


# --------------------------------------------------------------------------
# kepala halaman
# --------------------------------------------------------------------------
ui_tema.hero(
    "Mind-Ur-Set Information Retrieval",
    "Sistem hanya mengembalikan kutipan dari korpus psikoedukasi tertutup beserta halaman "
    "sumbernya; tidak ada model generatif yang menyusun jawaban.",
    [f"{len(KORPUS)} chunk", f"{len(KUERI)} kueri uji", "M1–M4", "M4 hybrid terbaik"],
)

with st.sidebar:
    st.header("Kontrol")
    if BACKEND_TERSEDIA:
        backend = st.selectbox(
            "Backend dense",
            BACKEND_TERSEDIA,
            index=0,
            format_func=lambda b: NAMA_BACKEND.get(b, b),
            help="ONNX int8 memakai vektor yang sama dengan naskah, sehingga M3/M4 cocok "
                 "dengan angka paper. torch hanya pembanding.",
        )
        st.caption(f"Backend aktif: **{NAMA_BACKEND[backend]}**")
    else:
        backend = None
        st.caption("Encoder dense tidak tersedia — hanya M1 dan M2 yang bisa dipakai.")
    st.divider()
    st.caption("M1 VSM TF-IDF · M2 Okapi BM25 (k1=1,2 b=0,75) · M3 BGE-M3 1024-d · M4 RRF k=60")
    st.caption(f"Korpus {len(KORPUS)} chunk · kueri {len(KUERI)} · ber-qrels {len(QRELS)}")

tab_cari, tab_eval = st.tabs(["Pencarian", "Evaluasi"])

# ==========================================================================
# TAB 1 — pencarian
# ==========================================================================
with tab_cari:
    fungsi = pencari(backend)
    tersedia_nama = list(fungsi)

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
            height=80,
            max_chars=MAKS_KARAKTER,
        )
        qid = None
    else:
        opsi = {f"{q['qid']} · {q['teks'][:60]}…": q for q in KUERI if q["qid"] in QRELS}
        pilih = st.selectbox("Pilih kueri uji", list(opsi))
        q = opsi[pilih]
        teks_kueri, qid = q["teks"], q["qid"]
        st.caption(f"{qid} ({q['bahasa']}) · {len(QRELS[qid])} chunk relevan menurut qrels")

    baris = st.columns([3, 1], vertical_alignment="bottom")
    with baris[0]:
        bawaan = [n for n in tersedia_nama if n.startswith(("M2", "M3", "M4"))] or tersedia_nama[:2]
        pilih_kode = st.pills(
            "Konfigurasi",
            options=tersedia_nama,
            default=bawaan,
            selection_mode="multi",
            format_func=lambda n: n.split(" · ")[0],
            help="Pilih satu atau beberapa konfigurasi untuk dibandingkan.",
        )
        metode_pilih = pilih_kode or []
    with baris[1]:
        jalan = st.button("Cari", type="primary", width="stretch")

    with st.expander("Opsi lanjutan", expanded=False):
        kol_opsi = st.columns(3)
        topk = kol_opsi[0].slider("Kedalaman peringkat", 3, 20, 10)
        bahasa = kol_opsi[1].radio("Saringan bahasa dokumen", ["Semua", "Indonesia", "Inggris"])
        tampil_teks = kol_opsi[2].slider(
            "Panjang teks ditampilkan (karakter)", 200, 1200, 400, step=100
        )

    if jalan:
        if not teks_kueri.strip():
            st.warning("Kueri kosong.")
        elif not tokenize(teks_kueri):
            st.warning(
                "Kueri tidak memuat kata yang bisa dicari (hanya tanda baca atau kata umum). "
                "Tulis keluhan dengan kata benda atau kata kerja, misalnya “sulit tidur”."
            )
        elif not metode_pilih:
            st.warning("Pilih sekurangnya satu konfigurasi.")
        else:
            frasa = deteksi_risiko(teks_kueri)

            def lolos(doc_id: str) -> bool:
                b = KOR[doc_id].get("bahasa")
                if bahasa == "Indonesia":
                    return b == "id"
                if bahasa == "Inggris":
                    return b == "en"
                return True

            with st.spinner("Menghitung peringkat…"):
                if backend and any(n.startswith(("M3", "M4")) for n in metode_pilih):
                    muat_encoder(backend)   # pemanasan; biaya muat model tidak dihitung
                awal = time.perf_counter()
                hasil = {nama: [d for d in fungsi[nama](teks_kueri, len(IDS)) if lolos(d)][:topk]
                         for nama in metode_pilih}
                lama = (time.perf_counter() - awal) * 1000
            st.session_state["hasil_cari"] = {
                "teks": teks_kueri, "qid": qid, "hasil": hasil, "lama": lama,
                "risiko": frasa, "topk": topk,
            }

    simpan = st.session_state.get("hasil_cari")
    if simpan:
        hasil = simpan["hasil"]
        relevan = set(QRELS.get(simpan["qid"], [])) if simpan["qid"] else set()
        if simpan["risiko"]:
            kartu_bantuan()

        if simpan["qid"]:
            kol = st.columns(len(hasil))
            for c, (nama, urut) in zip(kol, hasil.items()):
                hit = len([d for d in urut if d in relevan])
                k = len(urut) or 1
                rr = next((1.0 / i for i, d in enumerate(urut, 1) if d in relevan), 0.0)
                c.metric(nama.split(" · ")[0], f"P@{k} {hit / k:.2f}", f"RR {rr:.3f}")

        st.caption(
            f"{simpan['lama']:.0f} ms untuk {len(hasil)} konfigurasi · kedalaman {simpan['topk']} · "
            f"backend {backend or 'leksikal saja'} · P@K = proporsi relevan di hasil teratas, "
            "RR = 1/peringkat relevan pertama."
        )

        urut_konfig = {n: hasil[n] for n in sorted(hasil)}
        if len(urut_konfig) > 1:
            st.caption("Kolom diurutkan M1 → M4.")
        kolom = st.columns(len(urut_konfig), gap="small")
        for c, (nama, urut) in zip(kolom, urut_konfig.items()):
            with c:
                kode = nama.split(" · ")[0]
                st.subheader(f"{kode} · {RINGKAS.get(kode, nama.split(' · ')[1])}")
                if not urut:
                    ui_tema.kosong(
                        "Tidak ada hasil",
                        "Tidak ada kata kueri yang cocok di konfigurasi ini.",
                    )
                    continue
                kartu_html = []
                potong = min(tampil_teks, 320)
                for pos, doc_id in enumerate(urut, start=1):
                    d = KOR[doc_id]
                    teks = d["teks"]
                    sama, cakupan = tumpang_tindih(simpan["teks"], doc_id)
                    kartu_html.append(
                        ui_tema.kartu(
                            pos=pos,
                            doc_id=doc_id,
                            lokasi=lokasi(d),
                            teks=teks[:potong] + ("…" if len(teks) > potong else ""),
                            relevan=doc_id in relevan,
                            catatan=[
                                f"{d.get('token', '?')} token",
                                f"kemiripan {cakupan:.0%}",
                                f"istilah: {', '.join(sama[:8]) if sama else '—'}",
                                f"{d.get('lisensi', '?')}",
                            ],
                            buka=(pos <= 2 and not simpan["risiko"]),
                            ringkas=True,
                        )
                    )
                st.markdown("".join(kartu_html), unsafe_allow_html=True)

        if len(hasil) > 1:
            with st.expander("Perbandingan peringkat (tabel)"):
                st.markdown(
                    tabel_perbandingan(hasil, relevan if simpan["qid"] else set(), simpan["topk"]),
                    unsafe_allow_html=True,
                )
                if simpan["qid"]:
                    st.caption("Sel berlatar hijau menandai chunk relevan menurut qrels final.")
    else:
        ui_tema.kosong(
            "Belum ada hasil",
            "Tulis keluhan pada kotak di atas lalu tekan Cari — atau pilih mode “Kueri uji” "
            "untuk mencoba kueri yang sudah punya kunci relevansi. Contoh: “sulit tidur karena "
            "pikiran berputar”, “cara menenangkan diri saat panik”, “how to stop avoiding people”.",
        )

# ==========================================================================
# TAB 2 — evaluasi
# ==========================================================================
with tab_eval:
    st.subheader("Hitung ulang MRR pada 88 kueri uji")
    st.caption(
        "Memakai konfigurasi yang tersedia dan seluruh kueri yang punya qrels, lalu "
        "membandingkan hasilnya dengan Tabel III naskah. Bila vektor acuan tersedia, M3/M4 "
        "memakainya sehingga angkanya sama dengan naskah."
    )
    if not BACKEND_TERSEDIA:
        st.info("Encoder dense tidak terpasang; M3/M4 tetap dihitung dari vektor acuan bila ada.")
    jalan_eval = st.button("Hitung MRR", type="primary")

    if jalan_eval:
        acuan = encoder.vektor_kueri_acuan()
        kand = [q for q in KUERI if q["qid"] in QRELS]
        perlu_encoder = EMBED is not None and any(q["qid"] not in acuan for q in kand)
        enc = muat_encoder(backend) if (perlu_encoder and backend) else None
        fungsi = pencari(backend)
        progress = st.progress(0.0, text="menghitung…")
        hasil, lama = {}, {}

        def vektor_uji(q: dict):
            if q["qid"] in acuan:
                return acuan[q["qid"]]
            if enc is None:
                raise RuntimeError("vektor acuan tidak lengkap dan encoder tidak tersedia")
            return enc.kodekan(q["teks"])

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
        st.session_state["hasil_eval"] = {"hasil": hasil, "lama": lama, "n": len(kand)}

    simpan_eval = st.session_state.get("hasil_eval")
    if simpan_eval:
        baris = []
        for nama, nilai in simpan_eval["hasil"].items():
            acuan_mrr = PAPER.get(nama.split(" · ")[0])
            baris.append({
                "Konfigurasi": nama,
                "MRR": nilai,
                "Naskah": acuan_mrr,
                "Selisih": None if acuan_mrr is None else nilai - acuan_mrr,
                "Waktu (s)": simpan_eval["lama"][nama],
            })
        st.dataframe(
            baris, width="stretch", hide_index=True,
            column_config={
                "MRR": st.column_config.NumberColumn(format="%.4f"),
                "Naskah": st.column_config.NumberColumn(format="%.3f"),
                "Selisih": st.column_config.NumberColumn(format="%+.4f"),
                "Waktu (s)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(
            f"{simpan_eval['n']} kueri ber-qrels. Selisih diharapkan 0,0000 untuk M1/M2 dan "
            "≤0,002 untuk M3/M4 bila vektor acuan dipakai."
        )

st.divider()
st.caption(
    "Demo ini hanya menjalankan retrieval. Tidak ada model generatif yang terhubung, sehingga "
    "saran selalu berupa chunk asli dari korpus beserta halaman sumbernya."
)
