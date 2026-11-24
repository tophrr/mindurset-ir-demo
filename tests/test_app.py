"""Uji antarmuka dengan AppTest: memuat, mencari, dan lapis keselamatan."""
from __future__ import annotations

import pathlib
import sys

from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APP = str(ROOT / "app.py")


def test_app_memuat_tanpa_exception():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value.startswith("🔎")


def test_pencarian_menampilkan_hasil():
    at = AppTest.from_file(APP, default_timeout=120).run()
    # pilih M1 dan M2 saja agar tidak perlu encoder dense
    at.multiselect[0].set_value(["M1 · VSM TF-IDF", "M2 · Okapi BM25"]).run()
    at.button[0].click().run()
    assert not at.exception
    label = [s.value for s in at.subheader]
    assert any("VSM" in s for s in label) and any("BM25" in s for s in label)


def test_kueri_berisiko_memunculkan_peringatan():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.multiselect[0].set_value(["M1 · VSM TF-IDF"]).run()
    at.text_area[0].set_value("aku sudah tidak ingin hidup, mau mengakhiri semuanya").run()
    at.button[0].click().run()
    assert not at.exception
    pesan = " ".join(e.value for e in at.error)
    assert "119" in pesan and "SEJIWA" in pesan

def test_kueri_tanpa_kata_ditolak():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.multiselect[0].set_value(["M1 · VSM TF-IDF"]).run()
    at.text_area[0].set_value("???").run()
    at.button[0].click().run()
    assert not at.exception
    pesan = " ".join(w.value for w in at.warning)
    assert "tidak memuat kata" in pesan


def test_hasil_bertahan_saat_slider_diubah():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.multiselect[0].set_value(["M1 · VSM TF-IDF"]).run()
    at.button[0].click().run()
    jumlah_awal = len(at.subheader)
    at.slider[-1].set_value(700).run()
    assert len(at.subheader) == jumlah_awal


def test_frasa_risiko_varian_terdeteksi():
    for frasa in ("mau mati aja", "bundir", "suicidal", "gak kuat hidup", "capek hidup"):
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.multiselect[0].set_value(["M1 · VSM TF-IDF"]).run()
        at.text_area[0].set_value(frasa).run()
        at.button[0].click().run()
        pesan = " ".join(e.value for e in at.error)
        assert "119" in pesan, frasa
