"""Export chunkweaver chunks to individual files for ragtune --pre-chunked ingestion.

Naming: {stem}_chunk_{n:04d}.txt
ragtune normalizeSource strips the directory (filepath.Base) then prefix-matches
the stem against relevant_docs, so rfc7519_jwt_chunk_0001.txt matches rfc7519_jwt.txt.
"""

import os
import sys
import re
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from chunkweaver import Chunker
from chunkweaver.presets import RFC, LEGAL_EU, PLAIN

CORPUS = Path(__file__).parent / "../benchmarks/corpus"
OUT = Path(__file__).parent / "chunks-chunkweaver"


def preset_for(filename: str):
    if filename.startswith("rfc"):
        return RFC
    if "gdpr" in filename or "ai_act" in filename or "ccpa" in filename:
        return LEGAL_EU
    return PLAIN


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    total_chunks = 0
    for doc_path in sorted(CORPUS.glob("*.txt")):
        stem = doc_path.stem
        boundaries = preset_for(doc_path.name)
        chunker = Chunker(
            target_size=600,
            min_size=100,
            overlap=2,
            overlap_unit="sentence",
            boundaries=boundaries,
        )
        text = doc_path.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            out_file = OUT / f"{stem}_chunk_{i:04d}.txt"
            out_file.write_text(chunk, encoding="utf-8")
        print(f"  {doc_path.name}: {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"\nTotal: {total_chunks} chunks → {OUT}")


if __name__ == "__main__":
    main()
