#!/usr/bin/env python3
"""Unduh model BGE-M3 int8 (ONNX) ke models/bge-m3-onnx/.

Dipakai saat build Space atau saat menyiapkan mesin baru.

    python unduh_model.py            # unduh bila belum ada
    python unduh_model.py --paksa    # unduh ulang
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.request

BASE = "https://huggingface.co/Xenova/bge-m3/resolve/main"
BERKAS = {
    "tokenizer.json": f"{BASE}/tokenizer.json",
    "tokenizer_config.json": f"{BASE}/tokenizer_config.json",
    "special_tokens_map.json": f"{BASE}/special_tokens_map.json",
    "config.json": f"{BASE}/config.json",
    "model_int8.onnx": f"{BASE}/onnx/model_int8.onnx",
}
TUJUAN = pathlib.Path(__file__).resolve().parent / "models" / "bge-m3-onnx"


def unduh(url: str, jalur: pathlib.Path, percobaan: int = 3, timeout: int = 120) -> None:
    """Unduh dengan timeout dan percobaan ulang; berkas sementara diberi akhiran .part."""
    print(f"→ {jalur.name}", flush=True)
    sementara = jalur.with_suffix(jalur.suffix + ".part")
    for i in range(1, percobaan + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, sementara.open("wb") as f:
                while potong := r.read(1 << 20):
                    f.write(potong)
            sementara.replace(jalur)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  percobaan {i}/{percobaan} gagal: {e}", file=sys.stderr)
            time.sleep(2 * i)
    raise RuntimeError(f"gagal mengunduh {jalur.name} setelah {percobaan} percobaan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paksa", action="store_true")
    a = ap.parse_args()
    TUJUAN.mkdir(parents=True, exist_ok=True)
    for nama, url in BERKAS.items():
        jalur = TUJUAN / nama
        if jalur.exists() and jalur.stat().st_size > 1024 and not a.paksa:
            print(f"= {nama} sudah ada ({jalur.stat().st_size / 1e6:.1f} MB)")
            continue
        try:
            unduh(url, jalur)
        except Exception as e:  # noqa: BLE001
            print(f"gagal mengunduh {nama}: {e}", file=sys.stderr)
            return 1
    total = sum(f.stat().st_size for f in TUJUAN.iterdir())
    print(f"selesai: {TUJUAN} ({total / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
