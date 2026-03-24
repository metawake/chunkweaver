"""Export chunkweaver + LangChain RCTS chunks for legal contract corpus."""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_US

CORPUS = Path(__file__).parent / "corpus-legal"
CW_OUT = Path(__file__).parent / "chunks-legal-cw"
LC_OUT = Path(__file__).parent / "chunks-legal-langchain"

CONTRACT_BOUNDARIES = LEGAL_US + [
    r"^ARTICLE\s+\d+",
    r"^Annex\s+",
    r"^Schedule\s+",
    r"^Appendix\s+",
    r"^Exhibit\s+",
    r"^\d+\.\d+\.\d+\s",
]


def export_chunkweaver():
    if CW_OUT.exists():
        shutil.rmtree(CW_OUT)
    CW_OUT.mkdir(parents=True)

    chunker = Chunker(
        target_size=600,
        min_size=100,
        overlap=2,
        overlap_unit="sentence",
        boundaries=CONTRACT_BOUNDARIES,
    )

    total = 0
    for doc in sorted(CORPUS.glob("*.txt")):
        text = doc.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            (CW_OUT / f"{doc.stem}_chunk_{i:04d}.txt").write_text(chunk, encoding="utf-8")
        print(f"  CW  {doc.name}: {len(chunks)} chunks")
        total += len(chunks)
    print(f"  CW total: {total} chunks -> {CW_OUT}\n")


def export_langchain():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if LC_OUT.exists():
        shutil.rmtree(LC_OUT)
    LC_OUT.mkdir(parents=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total = 0
    for doc in sorted(CORPUS.glob("*.txt")):
        text = doc.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            (LC_OUT / f"{doc.stem}_chunk_{i:04d}.txt").write_text(chunk, encoding="utf-8")
        print(f"  LC  {doc.name}: {len(chunks)} chunks")
        total += len(chunks)
    print(f"  LC total: {total} chunks -> {LC_OUT}\n")


if __name__ == "__main__":
    print("=== chunkweaver (contract boundaries) ===")
    export_chunkweaver()
    print("=== LangChain RCTS ===")
    export_langchain()
