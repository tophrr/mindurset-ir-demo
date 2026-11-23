# Demo Retrieval Mind-Ur-Set

Demo **information retrieval murni** untuk korpus psikoedukasi Mind-Ur-Set.
Tidak ada LLM, tidak ada model generatif. Sistem hanya mengembalikan peringkat
*chunk* dari korpus tertutup beserta halaman sumbernya, sehingga keluaran tidak
dapat berhalusinasi.

Dibuat untuk melengkapi artikel **"Perbandingan Model Retrieval Leksikal, Dense,
dan Hibrida pada Korpus Psikoedukasi Mikro Tertutup untuk Aplikasi Mind-Ur-Set"**.

## Ringkas

| | |
|---|---|
| Korpus | 153 *chunk* psikoedukasi (77 Inggris, 76 Indonesia) |
| Kueri uji | 100 (50 Indonesia, 50 Inggris) |
| Kunci relevansi | 1.875 pasangan, 281 positif |
| Konfigurasi | M1 VSM TF-IDF · M2 Okapi BM25 · M3 BGE-M3 · M4 RRF |
| Model generatif | tidak ada |

## Menjalankan

Butuh **Python 3.10+** (diuji pada 3.11). Perintah di bawah berlaku sama di
macOS, Linux, dan Windows.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat

pip install -r requirements.txt
streamlit run app.py
```

Buka <http://localhost:8501>.

Saat pertama dibuka, aplikasi memuat `BAAI/bge-m3` (~2,2 GB) ke cache
HuggingFace. Unduhan hanya sekali. Kalau ingin melewati tahap itu, M1 dan M2
tetap dapat dipakai tanpa model dense — cukup hapus centang M3 dan M4 di bilah
samping.

## Isi antarmuka

| Bagian | Keterangan |
|---|---|
| Mode kueri bebas | Tulis keluhan apa pun, lihat peringkat *chunk* |
| Mode kueri uji | Pilih dari 88 kueri uji yang punya kunci relevansi |
| Pemilih konfigurasi | Bandingkan M1–M4 berdampingan |
| Saringan bahasa | Semua / Indonesia / Inggris |
| Bukti relevansi | Tanda ✅ pada *chunk* yang ada di qrels |
| Penjelasan | Istilah kueri yang bertumpang tindih dengan *chunk* |
| Tabel perbandingan | Peringkat 1..K keempat metode berdampingan |

## Konfigurasi

| Kode | Metode | Parameter |
|---|---|---|
| M1 | VSM TF-IDF + *cosine similarity* | log-TF · idf = log(N/df) · vektor ternormalisasi |
| M2 | Okapi BM25 | k₁ = 1,2 · b = 0,75 |
| M3 | BGE-M3 *dense* | 1024 dimensi · CLS pooling · vektor ternormalisasi |
| M4 | RRF(M2, M3) | k = 60 |

Tokenisasi menyamakan harness evaluasi paper: huruf kecil, pola `[^\W_]+`,
stopword Inggris dan Indonesia dibuang, token satu huruf dibuang.

## Kesesuaian dengan paper

MRR pada 88 kueri gabungan (`python uji_mesin.py`):

| | Demo | Paper | Selisih |
|---|---|---|---|
| M1 | 0,3904 | 0,390 | +0,0004 |
| M2 | 0,3966 | 0,397 | −0,0004 |
| M3 | 0,3989 | 0,420 | −0,0211 |
| M4 | 0,5005 | 0,500 | +0,0005 |

M1, M2, dan M4 **cocok sampai empat angka desimal**. M3 berselisih 0,021 karena
perbedaan *backend*: paper memakai ONNX Runtime int8, demo memakai PyTorch
(fp16/fp32) di MPS/CUDA/CPU. *Pooling* keduanya sama, yaitu CLS.

## Struktur berkas

```
mindurset-demo/
├── app.py              antarmuka Streamlit
├── mesin.py            M1, M2, M3, M4 + metrik (tanpa Streamlit)
├── siapkan_data.py     hitung ulang embedding korpus
├── uji_mesin.py        hitung MRR, bandingkan dengan paper
├── requirements.txt
├── data/
│   ├── korpus.json     153 chunk psikoedukasi
│   ├── kueri.json      100 kueri uji
│   ├── qrels.json      kunci relevansi (88 kueri)
│   └── emb_chunk.npy   embedding BGE-M3 (153 × 1024, 612 KB)
└── .venv/              lingkungan lokal, tidak di-commit
```

`mesin.py` sengaja tidak mengimpor Streamlit, sehingga dapat dipakai ulang di
skrip, *notebook*, atau layanan lain.

## Membangun ulang embedding

Berkas `data/emb_chunk.npy` sudah disertakan agar demo langsung jalan. Untuk
membuatnya ulang (misalnya ganti model atau presisi):

```bash
python siapkan_data.py --paksa
```

Encoding 153 *chunk* memakan 15 detik di GPU Apple M1 Pro, sekitar 1 menit di
CPU. Perangkat dipilih otomatis: CUDA → MPS → CPU.

## Catatan kompatibilitas

| Sistem | Status | Catatan |
|---|---|---|
| macOS Apple Silicon | diuji | MPS aktif otomatis |
| macOS Intel | seharusnya jalan | CPU, encoding lebih lambat |
| Linux + NVIDIA | seharusnya jalan | CUDA aktif otomatis |
| Windows 10/11 | seharusnya jalan | CPU; `torch` versi CPU terpasang otomatis |

Tidak ada *path* absolut, tidak ada *shell script*, dan tidak ada dependensi
khusus sistem operasi. Yang perlu diperhatikan hanya `torch`: di Windows,
`pip install -r requirements.txt` akan mengambil *wheel* CPU. Untuk dukungan
CUDA di Linux/Windows, pasang `torch` dari <https://pytorch.org> lebih dulu.

## Lisensi dan atribusi

Korpus disusun dari empat sumber:

| Sumber | Lisensi |
|---|---|
| *Doing What Matters in Times of Stress* (WHO 2020) | CC BY-NC-SA 3.0 IGO |
| *Problem Management Plus* (WHO/MSD/MER/18.5) | CC BY-NC-SA 3.0 IGO |
| Kerangka *thought record* tujuh kolom (Beck dkk. 1979) | ditulis ulang, karya sendiri |
| Pustaka kartu *coping skills* Mind-Ur-Set | karya sendiri (MIT) |

Setiap *chunk* menyimpan medan `sumber`, `halaman`, dan `lisensi` agar atribusi
dapat ditelusuri sampai halaman asalnya. Lisensi CC BY-NC-SA 3.0 IGO bersifat
**non-komersial** dan **berbagi serupa**; distribusi di luar konteks akademik
perlu memperhatikan kedua syarat itu. Rincian lengkap ada di [`NOTICE.md`](NOTICE.md).

Kode dan data karya sendiri dilisensikan MIT (lihat [`LICENSE`](LICENSE)).
Lisensi MIT **tidak** mencakup teks WHO di dalam `data/korpus.json`.

Repositori ini tidak memuat kredensial, kunci API, maupun data pribadi. Semua
kueri uji adalah adaptasi dari ESConv, bukan tulisan pengguna sungguhan.

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
