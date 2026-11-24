---
title: Mind-Ur-Set IR Demo
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Perbandingan empat konfigurasi retrieval pada korpus psikoedukasi
---

# Mind-Ur-Set · Demo Retrieval

Prototipe information retrieval tanpa LLM: membandingkan VSM TF-IDF, Okapi BM25, BGE-M3 *dense*,
dan fusi RRF pada korpus psikoedukasi 153 *chunk* berbahasa Indonesia dan Inggris.

Korpus dan angka acuan berasal dari repo artefak `who-ir`. Dokumentasi lengkap ada di berkas
`README.md` repo GitHub; berkas ini hanya metadata Space.

Model ONNX int8 (568 MB) diunduh saat build:

```bash
python unduh_model.py
```

Bila model tidak tersedia, aplikasi tetap berjalan dengan M1 dan M2.
