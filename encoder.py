#!/usr/bin/env python3
"""Pengodean kueri untuk demo: ONNX int8 (bawaan) atau torch (cadangan).

Backend ONNX memakai berkas model yang sama dengan repo artefak (`Xenova/bge-m3`
int8, CLS pooling, L2 normalisasi), sehingga skor M3 dan M4 identik dengan
angka naskah selama matriks korpus yang dipakai adalah `data/embedding/
emb_chunk_cls.npy`. Backend torch memakai sentence-transformers dan hanya untuk
eksplorasi; hasilnya akan bergeser dari angka naskah.

Letak model ditentukan berurutan: argumen `model_dir`, variabel lingkungan
`BGE_MODEL_DIR`, lalu `models/bge-m3-onnx` di dalam repo.
"""
from __future__ import annotations

import os
import pathlib
from functools import lru_cache

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
BERKAS_MODEL = "model_int8.onnx"
MAKS_CACHE = 256      # jumlah vektor kueri yang disimpan; menjaga memori tetap datar


def letak_model(nilai: str | pathlib.Path | None = None) -> pathlib.Path:
    kandidat = pathlib.Path(nilai) if nilai else pathlib.Path(
        os.environ.get("BGE_MODEL_DIR", ROOT / "models" / "bge-m3-onnx")
    )
    if not (kandidat / BERKAS_MODEL).exists():
        raise FileNotFoundError(
            f"model ONNX tidak ditemukan di {kandidat}. Unduh {BERKAS_MODEL}, tokenizer.json, "
            "tokenizer_config.json, special_tokens_map.json, dan config.json dari "
            "https://huggingface.co/Xenova/bge-m3/tree/main/onnx ke folder itu, "
            "atau set BGE_MODEL_DIR."
        )
    return kandidat


class EncoderOnnx:
    """Encoder BGE-M3 int8 dengan ONNX Runtime (CLS pooling)."""

    backend = "onnx"

    def __init__(self, model_dir: str | pathlib.Path | None = None,
                 threads: int = min(4, os.cpu_count() or 2), maks_panjang: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.dir = letak_model(model_dir)
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=maks_panjang)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sesi = ort.InferenceSession(str(self.dir / BERKAS_MODEL), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.masuk = [i.name for i in self.sesi.get_inputs()]
        self._cache: dict[str, np.ndarray] = {}   # dibatasi MAKS_CACHE agar memori stabil

    def kodekan(self, teks: str) -> np.ndarray:
        if teks in self._cache:
            return self._cache[teks]
        e = self.tok.encode(teks)
        ids = np.array([e.ids], dtype=np.int64)
        mask = np.array([e.attention_mask], dtype=np.int64)
        masuk = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.masuk:
            masuk["token_type_ids"] = np.zeros_like(ids)
        h = self.sesi.run(None, {k: v for k, v in masuk.items() if k in self.masuk})[0]
        v = h[:, 0, :][0]
        v = v / (np.linalg.norm(v) or 1.0)
        v = v.astype(np.float32)
        if len(self._cache) >= MAKS_CACHE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[teks] = v
        return v


class EncoderTorch:
    """Encoder cadangan memakai sentence-transformers (butuh torch)."""

    backend = "torch"

    def __init__(self, nama: str = "BAAI/bge-m3", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        from mesin import pilih_device

        self.model = SentenceTransformer(nama, device=device or pilih_device())
        self._cache: dict[str, np.ndarray] = {}

    def kodekan(self, teks: str) -> np.ndarray:
        if teks in self._cache:
            return self._cache[teks]
        v = self.model.encode([teks], normalize_embeddings=True, convert_to_numpy=True)[0]
        v = v.astype(np.float32)
        if len(self._cache) >= MAKS_CACHE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[teks] = v
        return v


@lru_cache(maxsize=1)
def vektor_kueri_acuan(dir_emb: str | pathlib.Path | None = None) -> dict[str, np.ndarray]:
    """Vektor kueri uji yang dipakai di naskah (qid -> vektor 1024-d).

    Dipakai tab Evaluasi agar angka M3/M4 sama dengan naskah tanpa bergantung pada
    pengodean ulang, yang bergeser beberapa persen meski modelnya sama.
    """
    import json

    folder = pathlib.Path(dir_emb) if dir_emb else ROOT / "data" / "embedding"
    matriks = folder / "emb_kueri_cls.npy"
    peta = folder / "emb_kueri_cls.npy.idx.json"
    if not (matriks.exists() and peta.exists()):
        return {}
    emb = np.load(matriks)
    idx = json.loads(peta.read_text(encoding="utf-8"))
    return {qid: emb[i].astype(np.float32) for qid, i in idx.items()}


@lru_cache(maxsize=2)
def muat(backend: str = "onnx", model_dir: str | None = None):
    """Muat encoder sekali; backend 'onnx' atau 'torch'."""
    if backend == "torch":
        return EncoderTorch()
    return EncoderOnnx(model_dir)


def tersedia(backend: str = "onnx") -> bool:
    """Apakah backend bisa dipakai di mesin ini (tanpa memuat model)."""
    if backend == "torch":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    try:
        berkas = letak_model() / BERKAS_MODEL
    except FileNotFoundError:
        return False
    # model int8 yang utuh berukuran ~540 MB; berkas kecil berarti unduhan belum selesai
    return berkas.stat().st_size > 100 * 1024 * 1024
