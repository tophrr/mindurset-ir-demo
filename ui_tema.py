"""Skin visual Mind-Ur-Set untuk Streamlit.

Token warna diambil apa adanya dari aplikasi Android Mind-Ur-Set
(`app/src/main/java/com/mindurset/app/ui/theme/Color.kt`), sehingga demo
retrieval ini tampil satu keluarga dengan prototipe produknya.

Semua yang ada di sini murni presentasi: tidak ada logika retrieval, tidak ada
ketergantungan pada Streamlit selain `st.markdown`.
"""
from __future__ import annotations

import html

import streamlit as st

NAVY = "#203080"
INK = "#16225C"
CANVAS = "#EEF4FF"
PALE = "#E3ECFB"
LAVENDER = "#E0D8E8"
MUTED = "#565E70"
OUTLINE = "#D3DDF0"
HIJAU = "#2E7D32"
HIJAU_MUDA = "#E3F2E7"
MERAH = "#B3261E"

_CSS = f"""
<style>
.stApp {{ background: {CANVAS}; }}
html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    color: {INK};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 0.5rem; }}
.block-container {{ padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1240px; }}

/* ---- hero ---- */
.hero {{
    background: linear-gradient(135deg, {NAVY} 0%, #2C3F9A 100%);
    border-radius: 20px; padding: 22px 26px; color: #fff;
    box-shadow: 0 1px 2px rgba(27,42,94,.10), 0 10px 28px rgba(27,42,94,.12);
    margin-bottom: 18px;
}}
.hero h1 {{ margin: 0; font-size: 1.5rem; font-weight: 700; letter-spacing: -.01em; color: #fff; }}
.hero p {{ margin: 7px 0 0; color: #D8E0FA; font-size: .88rem; }}
.hero .deret {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
.hero .deret span {{
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
    color: #fff; border-radius: 999px; padding: 3px 11px; font-size: .74rem;
}}

/* ---- kartu ---- */
.kartu {{
    background: #fff; border: 1px solid {OUTLINE}; border-radius: 16px;
    box-shadow: 0 1px 2px rgba(27,42,94,.08), 0 6px 18px rgba(27,42,94,.06);
    margin-bottom: 10px; overflow: hidden;
}}
.kartu summary {{
    list-style: none; cursor: pointer; padding: 12px 16px; display: flex;
    gap: 10px; align-items: baseline; flex-wrap: wrap;
}}
.kartu summary::-webkit-details-marker {{ display: none; }}
.kartu summary:hover {{ background: #F7FAFF; }}
.kartu .pos {{ color: {MUTED}; font-variant-numeric: tabular-nums; min-width: 1.6em; }}
.kartu .id {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-weight: 700; color: {NAVY}; font-size: .9rem; }}
.kartu .lokasi {{ color: {MUTED}; font-size: .76rem; }}
.kartu .tanda {{ color: {HIJAU}; font-weight: 700; }}
.kartu .badan {{ padding: 0 16px 14px 16px; border-top: 1px solid #EDF2FB; }}
.kartu .badan p {{ margin: 12px 0 8px; font-size: .88rem; line-height: 1.55; color: {INK}; }}
.kartu .meta {{ color: {MUTED}; font-size: .74rem; }}
.kartu.relevan {{ border-left: 3px solid {HIJAU}; }}
.kartu.relevan summary {{ background: {HIJAU_MUDA}; }}

/* ---- lencana ---- */
.lencana {{
    display: inline-block; border-radius: 999px; padding: 2px 10px;
    font-size: .72rem; font-weight: 600; background: {PALE}; color: {NAVY};
    margin-right: 6px;
}}
.lencana.hijau {{ background: {HIJAU_MUDA}; color: {HIJAU}; }}
.lencana.lavender {{ background: {LAVENDER}; color: #3C2F55; }}
.lencana.merah {{ background: #FBE9E7; color: {MERAH}; }}

/* ---- widget ---- */
.stButton button, .stDownloadButton button {{
    border-radius: 12px; font-weight: 600; border: 1px solid {OUTLINE};
}}
.stButton button[kind="primary"] {{ background: {NAVY}; border-color: {NAVY}; }}
[data-baseweb="input"], [data-baseweb="textarea"], .stTextArea textarea, .stTextInput input {{
    border-radius: 12px !important; background: #fff !important;
}}
button[data-variant="pills"], button[data-variant="segmented_control"] {{
    border-radius: 999px !important; font-weight: 600;
}}
button[data-variant="pills"][data-selected="true"],
button[data-variant="segmented_control"][data-selected="true"] {{
    background: {NAVY} !important; border-color: {NAVY} !important;
}}
button[data-variant="pills"][data-selected="true"] p,
button[data-variant="pills"][data-selected="true"] span,
button[data-variant="segmented_control"][data-selected="true"] p,
button[data-variant="segmented_control"][data-selected="true"] span {{
    color: #fff !important;
}}
button[data-variant="pills"] {{ padding: 4px 14px !important; }}
[data-testid="stSidebar"] {{
    background: {PALE}; border-right: 1px solid {OUTLINE};
}}
[data-testid="stSidebar"] h2 {{ font-size: 1rem; }}
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [role="tab"], .stTabs [data-baseweb="tab"] {{
    padding: 8px 16px; border-radius: 10px 10px 0 0; color: {MUTED};
    transition: background .15s ease, color .15s ease;
}}
.stTabs [role="tab"] p, .stTabs [data-baseweb="tab"] p {{ font-weight: 500; letter-spacing: .01em; }}
.stTabs [role="tab"][data-selected="true"], .stTabs [aria-selected="true"] {{
    background: #fff; color: {NAVY};
    box-shadow: inset 0 1px 0 {OUTLINE}, inset 1px 0 0 {OUTLINE}, inset -1px 0 0 {OUTLINE};
}}
.stTabs [role="tab"][data-selected="true"] p {{ color: {NAVY}; font-weight: 600; }}
.stTabs .react-aria-SelectionIndicator {{ height: 3px; border-radius: 2px 2px 0 0; background: {NAVY}; }}
.stTabs [data-baseweb="tab-highlight"] {{ height: 3px; background: {NAVY}; }}
h3 {{ font-size: 1.02rem; color: {INK}; margin-top: 1.1rem; }}
[data-testid="stColumn"] h3 {{ font-size: .95rem; line-height: 1.25; min-height: 2.5em;
    margin-top: .6rem; margin-bottom: .4rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.25rem; color: {NAVY}; }}
[data-testid="stMetricDelta"] {{ font-size: .75rem; }}
.kartu.ringkas summary {{ padding: 9px 11px; gap: 6px; }}
.kartu.ringkas .id {{ font-size: .82rem; }}
.kartu.ringkas .lokasi {{ display: block; width: 100%; font-size: .7rem; }}
.kartu.ringkas .badan {{ padding: 0 11px 10px; }}
.kartu.ringkas .badan p {{ font-size: .82rem; line-height: 1.5; }}
.kartu.ringkas .meta {{ font-size: .7rem; }}
@media (max-width: 640px) {{
    .block-container {{ padding-top: .5rem; }}
    .hero {{ padding: 16px 18px; border-radius: 16px; }}
    .hero h1 {{ font-size: 1.12rem; }}
    .hero p {{ font-size: .78rem; }}
    .hero .deret span {{ font-size: .68rem; padding: 2px 8px; }}
    [data-testid="stColumn"] h3 {{ min-height: 0; }}
}}
.kosong {{
    background: #fff; border: 1px dashed {OUTLINE}; border-radius: 16px;
    padding: 22px 24px; color: {MUTED}; font-size: .88rem;
}}
.kosong strong {{ color: {INK}; }}
</style>
"""


def pasang() -> None:
    """Sisipkan CSS global. Panggil sekali, setelah `st.set_page_config`."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(judul: str, subjudul: str, lencana: list[str] | None = None) -> None:
    deret = "".join(f"<span>{html.escape(x)}</span>" for x in (lencana or []))
    st.markdown(
        f'<div class="hero"><h1>{html.escape(judul)}</h1>'
        f'<p>{html.escape(subjudul)}</p>'
        f'{f"<div class=deret>{deret}</div>" if deret else ""}</div>',
        unsafe_allow_html=True,
    )


def lencana(teks: str, jenis: str = "") -> str:
    kelas = f"lencana {jenis}".strip()
    return f'<span class="{kelas}">{html.escape(str(teks))}</span>'


def kartu(
    pos: int,
    doc_id: str,
    lokasi: str,
    teks: str,
    relevan: bool,
    catatan: list[str] | None = None,
    buka: bool = False,
    ringkas: bool = False,
) -> str:
    """Satu kartu hasil. Memakai <details> agar bisa dibuka tanpa st.expander."""
    tanda = ('<span class="tanda">✓</span>' if ringkas else '<span class="tanda">✓ relevan</span>') \
        if relevan else ""
    kelas = "kartu relevan" if relevan else "kartu"
    if ringkas:
        kelas += " ringkas"
    meta = " · ".join(html.escape(c) for c in (catatan or []))
    return (
        f'<details class="{kelas}"{" open" if buka else ""}>'
        f'<summary><span class="pos">{pos}.</span>'
        f'<span class="id">{html.escape(doc_id)}</span>'
        f'<span class="lokasi">{html.escape(lokasi)}</span>{tanda}</summary>'
        f'<div class="badan"><p>{html.escape(teks)}</p>'
        f'<div class="meta">{meta}</div></div></details>'
    )


def kosong(judul: str, isi: str) -> None:
    st.markdown(
        f'<div class="kosong"><strong>{html.escape(judul)}</strong><br>{html.escape(isi)}</div>',
        unsafe_allow_html=True,
    )
