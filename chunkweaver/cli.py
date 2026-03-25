"""Command-line interface for chunkweaver."""

from __future__ import annotations

import argparse
import json
import sys

from chunkweaver.chunker import Chunker
from chunkweaver.models import Chunk
from chunkweaver.presets import PRESETS, get_preset


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chunkweaver",
        description="Structure-aware text chunking for RAG.",
    )
    p.add_argument(
        "file",
        nargs="?",
        help="Input file (reads from stdin if omitted).",
    )
    p.add_argument(
        "--size",
        "-s",
        type=int,
        default=1024,
        dest="target_size",
        help="Target chunk size in characters (default: 1024).",
    )
    p.add_argument(
        "--overlap",
        "-o",
        type=int,
        default=2,
        help="Number of overlap units from previous chunk (default: 2).",
    )
    p.add_argument(
        "--overlap-unit",
        choices=("sentence", "paragraph", "chars"),
        default="sentence",
        help="Overlap unit (default: sentence).",
    )
    p.add_argument(
        "--boundaries",
        "-b",
        nargs="*",
        default=[],
        help="Regex patterns that mark section starts.",
    )
    p.add_argument(
        "--preset",
        "-p",
        choices=sorted(PRESETS),
        help="Use a built-in boundary preset.",
    )
    p.add_argument(
        "--fallback",
        choices=("paragraph", "sentence", "word"),
        default="paragraph",
        help="Fallback split strategy (default: paragraph).",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=200,
        help="Minimum chunk size in characters (default: 200).",
    )
    p.add_argument(
        "--format",
        "-f",
        choices=("text", "json", "jsonl"),
        default="text",
        dest="output_format",
        help="Output format (default: text).",
    )
    p.add_argument(
        "--detect-boundaries",
        action="store_true",
        help="Preview boundary detection without chunking.",
    )
    p.add_argument(
        "--recommend",
        action="store_true",
        help="Analyze the document and suggest configuration.",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Chunk the document, then analyze chunk quality and suggest improvements.",
    )
    p.add_argument(
        "--llm-audit",
        action="store_true",
        help="Add LLM coherence audit to --inspect (requires OPENAI_API_KEY env var).",
    )
    return p


def _read_input(path: str | None) -> str:
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def _chunk_to_dict(c: Chunk) -> dict:
    return {
        "index": c.index,
        "text": c.text,
        "start": c.start,
        "end": c.end,
        "boundary_type": c.boundary_type,
        "overlap_text": c.overlap_text,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — parse args, read input, chunk or analyze, and print results."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    boundaries: list[str] = list(args.boundaries)
    if args.preset:
        boundaries = get_preset(args.preset) + boundaries

    text = _read_input(args.file)

    if args.recommend:
        from chunkweaver.recommend import recommend

        rec = recommend(text)
        print(rec.report())
        return

    if args.detect_boundaries:
        from chunkweaver.boundaries import detect_boundaries

        matches = detect_boundaries(text, boundaries)
        if not matches:
            print("No boundaries detected.", file=sys.stderr)
            return
        for m in matches:
            print(f"line {m.line_number + 1}: [{m.pattern}] {m.matched_text!r}")
        return

    chunker = Chunker(
        target_size=args.target_size,
        overlap=args.overlap,
        overlap_unit=args.overlap_unit,
        boundaries=boundaries,
        fallback=args.fallback,
        min_size=args.min_size,
    )

    chunks = chunker.chunk_with_metadata(text)

    if args.inspect:
        import os

        from chunkweaver.inspect import audit_coherence, inspect_chunks

        report = inspect_chunks(
            chunks,
            text,
            target_size=args.target_size,
            boundaries=boundaries,
        )

        if args.llm_audit:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print(
                    "ERROR: --llm-audit requires OPENAI_API_KEY env var.",
                    file=sys.stderr,
                )
                sys.exit(1)
            ratings, summary = audit_coherence(chunks, api_key=api_key)
            report.coherence_ratings = ratings
            report.coherence_summary = summary

        print(report.report())
        return

    if args.output_format == "text":
        for i, c in enumerate(chunks):
            if i > 0:
                print("\n" + "=" * 60 + "\n")
            print(c.text)
    elif args.output_format == "json":
        print(json.dumps([_chunk_to_dict(c) for c in chunks], indent=2))
    elif args.output_format == "jsonl":
        for c in chunks:
            print(json.dumps(_chunk_to_dict(c)))


if __name__ == "__main__":
    main()
