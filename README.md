# Demo Retrieval Mind-Ur-Set

Prototipe **information retrieval murni** untuk korpus psikoedukasi Mind-Ur-Set. Tidak ada LLM
dan tidak ada model generatif: sistem hanya mengembalikan peringkat *chunk* dari korpus tertutup
beserta halaman sumbernya, sehingga keluarannya tidak dapat berhalusinasi.

Melengkapi artikel **"Perbandingan Model Retrieval Leksikal, Dense, dan Hibrida pada Korpus
Psikoedukasi Mikro Tertutup untuk Aplikasi Mind-Ur-Set"** (Kelompok 5, Kecerdasan Komputasional).

## Ringkas

| | |
|---|---|
| Korpus | 153 *chunk* psikoedukasi (77 Inggris, 76 Indonesia) |
| Kueri uji | 100 (50 Indonesia, 50 Inggris) |
| Kunci relevansi | 88 kueri, 281 pasangan positif |
| Konfigurasi | M1 VSM TF-IDF · M2 Okapi BM25 · M3 BGE-M3 · M4 RRF |
| Model generatif | tidak ada |
| Backend dense | ONNX Runtime int8 (568 MB), cadangan torch |

## Menjalankan

Butuh Python 3.10+ (diuji pada 3.14 lokal dan 3.12 di CI).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Buka <http://localhost:8501>.

M3 dan M4 memerlukan model BGE-M3 int8. Unduh lima berkas dari
<https://huggingface.co/Xenova/bge-m3/tree/main/onnx> (`model_int8.onnx`, `tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`, `config.json`) ke `models/bge-m3-onnx/`,
atau set `BGE_MODEL_DIR`. Tanpa model, M1 dan M2 tetap berjalan dan aplikasi menandainya sendiri
di bilah samping.

Model torch (≈2 GB) hanya diperlukan bila ingin membandingkan backend:

```bash
pip install -r requirements-torch.txt
```

## Isi antarmuka

| Bagian | Keterangan |
|---|---|
| Hero dan tema | Warna dan tipografi memakai token aplikasi Mind-Ur-Set (`Color.kt`): navy #203080, kanvas #EEF4FF, tinta #16225C |
| Kartu hasil | Satu kartu per *chunk*, dapat dibuka; *chunk* relevan bergaris dan berlatar hijau |
| Kartu keselamatan | Muncul hanya untuk kueri berfrasa risiko, dengan rujukan 119 ext. 8 (SEJIWA), 112, IGD, dan findahelpline.com |
| Mode kueri bebas | Tulis keluhan apa pun, lihat peringkat *chunk* |
| Mode kueri uji | Pilih dari 88 kueri uji yang punya kunci relevansi |
| Pemilih konfigurasi | Bandingkan M1–M4 berdampingan |
| Saringan bahasa | Semua / Indonesia / Inggris |
| Bukti relevansi | Tanda ✅ pada *chunk* yang ada di qrels, P@K dan RR per metode |
| Penjelasan | Istilah kueri yang bertumpang tindih dengan *chunk*, panjang token, lisensi |
| Latensi | Waktu pencarian per permintaan (ms) |
| Tab Evaluasi | Hitung ulang MRR 88 kueri dan bandingkan dengan Tabel III naskah |

Kueri yang memuat frasa risiko (mis. *ingin mati*, *menyakiti diri*) memunculkan kartu bantuan
**sebelum** hasil pencarian. Penyaring itu tidak memblokir pencarian; ia hanya mengubah urutan
tampilan agar rujukan bantuan terbaca lebih dulu.

## Konfigurasi

| Kode | Metode | Parameter |
|---|---|---|
| M1 | VSM TF-IDF + *cosine similarity* | log-TF · idf = log(N/df) · vektor ternormalisasi |
| M2 | Okapi BM25 | k₁ = 1,2 · b = 0,75 |
| M3 | BGE-M3 *dense* | 1024 dimensi · CLS pooling · vektor ternormalisasi |
| M4 | RRF(M2, M3) | k = 60 |

Tokenisasi menyamakan harness evaluasi naskah: huruf kecil, pola `[^\W_]+`, stopword Inggris dan
Indonesia dibuang, token satu huruf dibuang. Kesamaan itu dipatok oleh uji
`tests/test_mesin.py::test_tokenisasi_sama_dengan_harness`.

## Kesesuaian dengan naskah

```bash
python uji_mesin.py              # memakai vektor acuan naskah
python uji_mesin.py --backend onnx   # mengodekan ulang
```

| | Demo | Naskah | Selisih |
|---|---|---|---|
| M1 | 0,3904 | 0,390 | +0,0004 |
| M2 | 0,3966 | 0,397 | −0,0004 |
| M3 | 0,4201 | 0,420 | +0,0001 |
| M4 | 0,4997 | 0,500 | −0,0003 |

Angka itu hanya tercapai bila **vektor acuan** dipakai. Pengodean ulang dengan model yang sama
tetap bergeser (kosinus ≈0,98 terhadap vektor naskah) karena int8 tidak deterministik antar konfigurasi
*batch*; akibatnya MRR M3 bisa bergeser sampai ±0,02. Karena itu:

- `data/embedding/emb_chunk_cls.npy` dan `emb_kueri_cls.npy` disertakan — vektor yang menghasilkan
  angka naskah.
- Tab **Evaluasi** memakai vektor kueri acuan tersebut, sehingga angkanya sama dengan Tabel III.
- Pencarian bebas mengodekan kueri langsung dengan ONNX; hasilnya tunduk pada pergeseran di atas.

## Struktur berkas

```
mindurset-ir-demo/
├── app.py                 antarmuka Streamlit (tab Pencarian dan Evaluasi)
├── ui_tema.py             skin visual Mind-Ur-Set: CSS, hero, kartu, lencana
├── mesin.py               M1, M2, M3, M4 + metrik (tanpa Streamlit)
├── encoder.py             pengodean kueri ONNX int8, cadangan torch
├── siapkan_data.py        sinkron data dari repo who-ir, periksa, bangun ulang vektor
├── uji_mesin.py           hitung MRR, bandingkan dengan naskah
├── tests/                 pytest: mesin, tokenisasi, AppTest
├── data/
│   ├── korpus.json        153 chunk psikoedukasi
│   ├── kueri.json         100 kueri uji
│   ├── qrels.json         kunci relevansi (88 kueri, 281 positif)
│   └── embedding/         vektor acuan naskah (korpus + kueri)
└── models/                model ONNX int8 (tidak di-commit)
```

## Data

Sumber kebenaran ada di repo artefak `who-ir`. Jangan menyunting `data/*.json` langsung.

```bash
python siapkan_data.py sinkron --dari ~/dev/who-ir   # salin korpus, kueri, qrels, vektor acuan
python siapkan_data.py periksa                      # cetak jumlah + checksum, gagal bila tidak cocok
python siapkan_data.py embedding --paksa            # bangun ulang vektor dengan torch (opsional)
```

`periksa` dijalankan juga di CI, sehingga data yang menyimpang dari naskah menggagalkan build.

## Uji

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tiga belas uji: kesesuaian jumlah data, tokenisasi vs harness, MRR M1/M2, determinisme RRF, metrik
pada contoh kecil, aplikasi memuat, hero dan *empty state*, pencarian menampilkan hasil, kartu hasil
ter-render, kartu risiko, kueri tanpa kata ditolak, hasil bertahan saat slider diubah, dan lima
varian frasa risiko terdeteksi.

## Deploy

Target: **Hugging Face Spaces** dengan **SDK Docker** (berkas `Dockerfile` sudah disertakan).
Model ONNX int8 (568 MB) diunduh saat *build* lewat `python unduh_model.py`, bukan saat permintaan
pertama, sehingga M3/M4 langsung aktif setelah Space selesai dibangun.

1. Buat Space dengan SDK **Docker**, publik.
2. Salin isi repo ini; front-matter Space sudah ada di `space/README.md` (gunakan berkas itu
   sebagai `README.md` Space).
3. `Dockerfile` menjalankan `python unduh_model.py` pada tahap build; `BGE_MODEL_DIR` sudah di-set
   ke `/app/models/bge-m3-onnx` di dalam image.
4. Bila ingin menghindari model di dalam image, unduh dari *dataset* Hugging Face pada startup dan
   set `BGE_MODEL_DIR` sesuai lokasinya.

Tanpa model, Space tetap hidup dengan M1/M2; tab Evaluasi hanya menghitung dua konfigurasi itu.

## Lisensi dan atribusi

| Sumber | Lisensi |
|---|---|
| *Doing What Matters in Times of Stress* (WHO 2020) | CC BY-NC-SA 3.0 IGO |
| *Problem Management Plus* (WHO/MSD/MER/18.5) | CC BY-NC-SA 3.0 IGO |
| Kerangka *thought record* tujuh kolom (Beck dkk. 1979) | ditulis ulang, karya sendiri |
| Pustaka kartu *coping skills* Mind-Ur-Set | karya sendiri (MIT) |

Setiap *chunk* menyimpan `sumber`, `halaman`, dan `lisensi` agar atribusi dapat ditelusuri sampai
halaman asalnya. Lisensi CC BY-NC-SA 3.0 IGO bersifat non-komersial dan berbagi serupa; distribusi
di luar konteks akademik perlu memperhatikan kedua syarat itu. Rincian ada di [`NOTICE.md`](NOTICE.md).

Kode dan data karya sendiri dilisensikan MIT (lihat [`LICENSE`](LICENSE)). Lisensi MIT **tidak**
mencakup teks WHO di dalam `data/korpus.json`.

Repositori tidak memuat kredensial, kunci API, maupun data pribadi. Semua kueri uji adalah adaptasi
dari ESConv, bukan tulisan pengguna sungguhan.

## Sitasi

```bibtex
@misc{mindurset-ir-demo,
  title  = {Demo Retrieval Mind-Ur-Set},
  author = {Gijoh, Christopher M. M. and Oroh, Christian J. H. and
            Laluan, Evan and Alvaro, Tora},
  year   = {2026},
  note   = {Korpus psikoedukasi mikro tertutup, 153 chunk, 100 kueri},
}
```
