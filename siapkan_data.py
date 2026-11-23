#!/usr/bin/env python3
"""Hitung ulang embedding korpus untuk demo.

Hanya perlu dijalankan bila data/emb_chunk.npy belum ada atau ingin dibuat
ulang dengan model lain. Tanpa berkas ini, M1, M2, dan M4 leksikal tetap
jalan, tetapi M3 (dense) tidak bisa dipakai.

Pemakaian:
    python siapkan_data.py
    python siapkan_data.py --model BAAI/bge-m3 --batch 16
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--paksa", action="store_true", help="timpa embedding yang ada")
    a = ap.parse_args()

    korpus = json.loads((DATA / "korpus.json").read_text(encoding="utf-8"))
    print(f"korpus: {len(korpus)} chunk")

    emb_path = DATA / "emb_chunk.npy"
    if emb_path.exists() and not a.paksa:
        import numpy as np
        emb = np.load(emb_path)
        print(f"embedding sudah ada: {emb.shape} — dilewati (pakai --paksa untuk menimpa)")
        return 0

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from mesin import pilih_device

    device = pilih_device()
    print(f"perangkat: {device}")
    print(f"memuat {a.model} (unduhan pertama bisa beberapa menit)…")
    t0 = time.time()
    model = SentenceTransformer(a.model, device=device)
    print(f"model siap dalam {time.time()-t0:.1f}s")

    t1 = time.time()
    emb = model.encode(
        [d["teks"] for d in korpus],
        normalize_embeddings=True,
        batch_size=a.batch,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"encoding {len(korpus)} chunk: {time.time()-t1:.1f}s, shape {emb.shape}")
    np.save(emb_path, emb)
    print(f"tersimpan: {emb_path} ({emb_path.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
