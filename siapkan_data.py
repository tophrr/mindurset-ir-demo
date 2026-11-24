#!/usr/bin/env python3
"""Siapkan data demo dari repo artefak kanonik (~/dev/who-ir).

Subperintah:
    python siapkan_data.py sinkron [--dari ~/dev/who-ir]
        Menyalin korpus, kueri, qrels (hanya label 1), dan vektor acuan naskah
        ke data/. Tidak mengubah berkas di repo sumber.
    python siapkan_data.py periksa
        Mencetak jumlah baris, jumlah pasangan positif, dan checksum isi data.
    python siapkan_data.py embedding [--model BAAI/bge-m3] [--paksa]
        Membangun ulang vektor dengan torch (opsional; angka naskah memakai
        vektor yang disalin pada langkah sinkron).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
EMB = DATA / "embedding"
BAWAAN_SUMBER = pathlib.Path.home() / "dev" / "who-ir"

BERKAS_WAJIB = (
    "data/korpus.jsonl",
    "data/kueri.jsonl",
    "data/qrels_final.jsonl",
    "hasil/embedding/emb_chunk_cls.npy",
    "hasil/embedding/emb_chunk_cls.npy.idx.json",
    "hasil/embedding/emb_kueri_cls.npy",
    "hasil/embedding/emb_kueri_cls.npy.idx.json",
)


def baca_jsonl(jalur: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in jalur.read_text(encoding="utf-8").splitlines() if l.strip()]


def tulis_json(jalur: pathlib.Path, isi) -> None:
    jalur.parent.mkdir(parents=True, exist_ok=True)
    jalur.write_text(json.dumps(isi, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def sidik(jalur: pathlib.Path) -> str:
    return hashlib.sha256(jalur.read_bytes()).hexdigest()[:16]


def sinkron(sumber: pathlib.Path) -> int:
    hilang = [b for b in BERKAS_WAJIB if not (sumber / b).exists()]
    if hilang:
        print("berkas sumber tidak lengkap:", ", ".join(hilang), file=sys.stderr)
        print("jalankan dulu di repo who-ir: python3 scripts/qrels/gabung_qrels.py", file=sys.stderr)
        return 2

    korpus = baca_jsonl(sumber / "data/korpus.jsonl")
    kueri = baca_jsonl(sumber / "data/kueri.jsonl")
    qrels = baca_jsonl(sumber / "data/qrels_final.jsonl")

    tulis_json(DATA / "korpus.json", korpus)
    tulis_json(DATA / "kueri.json", kueri)

    per_kueri: dict[str, list[str]] = {}
    for baris in qrels:
        if int(baris.get("label", 0)) == 1:
            per_kueri.setdefault(baris["qid"], []).append(baris["doc_id"])
    tulis_json(DATA / "qrels.json", {k: sorted(v) for k, v in sorted(per_kueri.items())})

    EMB.mkdir(parents=True, exist_ok=True)
    for nama in ("emb_chunk_cls.npy", "emb_chunk_cls.npy.idx.json",
                 "emb_kueri_cls.npy", "emb_kueri_cls.npy.idx.json"):
        shutil.copyfile(sumber / "hasil/embedding" / nama, EMB / nama)

    print(f"korpus  : {len(korpus)} chunk")
    print(f"kueri   : {len(kueri)}")
    print(f"qrels   : {len(per_kueri)} kueri, {sum(len(v) for v in per_kueri.values())} pasangan positif")
    print(f"vektor  : {EMB / 'emb_chunk_cls.npy'} ({sidik(EMB / 'emb_chunk_cls.npy')})")
    return 0


def periksa() -> int:
    import numpy as np

    korpus = json.loads((DATA / "korpus.json").read_text(encoding="utf-8"))
    kueri = json.loads((DATA / "kueri.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATA / "qrels.json").read_text(encoding="utf-8"))
    vektor = EMB / "emb_chunk_cls.npy"
    emb = np.load(vektor) if vektor.exists() else None
    positif = sum(len(v) for v in qrels.values())

    print(f"korpus            : {len(korpus)} chunk")
    print(f"kueri             : {len(kueri)}")
    print(f"kueri ber-qrels   : {len(qrels)}")
    print(f"pasangan positif  : {positif}")
    print(f"vektor            : {None if emb is None else emb.shape}")
    print(f"checksum korpus   : {sidik(DATA / 'korpus.json')}")
    print(f"checksum qrels    : {sidik(DATA / 'qrels.json')}")
    sah = (len(korpus), len(kueri), len(qrels), positif) == (153, 100, 88, 281)
    print("status            :", "cocok dengan naskah" if sah else "TIDAK cocok dengan naskah")
    return 0 if sah else 1


def embedding(nama_model: str, paksa: bool, batch: int) -> int:
    import numpy as np

    korpus = json.loads((DATA / "korpus.json").read_text(encoding="utf-8"))
    tujuan = EMB / "emb_chunk_cls.npy"
    if tujuan.exists() and not paksa:
        print(f"vektor sudah ada: {tujuan} — dilewati (pakai --paksa untuk menimpa)")
        return 0
    from sentence_transformers import SentenceTransformer

    from mesin import pilih_device

    model = SentenceTransformer(nama_model, device=pilih_device())
    emb = model.encode([d["teks"] for d in korpus], normalize_embeddings=True,
                       batch_size=batch, show_progress_bar=True,
                       convert_to_numpy=True).astype(np.float32)
    EMB.mkdir(parents=True, exist_ok=True)
    np.save(tujuan, emb)
    print(f"tersimpan: {tujuan} {emb.shape} ({tujuan.stat().st_size / 1e6:.2f} MB)")
    print("PERINGATAN: vektor ini bukan vektor naskah; M3/M4 akan berbeda dari 0,420/0,500.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="perintah", required=True)
    s = sub.add_parser("sinkron", help="salin data dari repo who-ir")
    s.add_argument("--dari", default=str(BAWAAN_SUMBER))
    sub.add_parser("periksa", help="cetak jumlah dan checksum data")
    e = sub.add_parser("embedding", help="bangun ulang vektor dengan torch")
    e.add_argument("--model", default="BAAI/bge-m3")
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--paksa", action="store_true")
    a = ap.parse_args()
    if a.perintah == "sinkron":
        return sinkron(pathlib.Path(a.dari).expanduser())
    if a.perintah == "periksa":
        return periksa()
    return embedding(a.model, a.paksa, a.batch)


if __name__ == "__main__":
    raise SystemExit(main())
