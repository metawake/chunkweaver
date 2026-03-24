"""Export LangChain RecursiveCharacterTextSplitter chunks for baseline comparison.

Uses the same corpus as export_chunks.py but with LangChain's paragraph-aware
splitter — the most common chunking approach in production RAG systems.
"""

import os
import shutil
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS = Path(__file__).parent / "../../ragtune/benchmarks/hierarchical/corpus"
OUT = Path(__file__).parent / "chunks-langchain"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_chunks = 0
    for doc_path in sorted(CORPUS.glob("*.txt")):
        stem = doc_path.stem
        text = doc_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            out_file = OUT / f"{stem}_chunk_{i:04d}.txt"
            out_file.write_text(chunk, encoding="utf-8")
        print(f"  {doc_path.name}: {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"\nTotal: {total_chunks} chunks -> {OUT}")


if __name__ == "__main__":
    main()
