# Changelog

All notable changes to chunkweaver are documented here.

## [0.3.0] — 2025-06-XX

### Added

- **Hierarchical boundaries** — assign levels to boundary patterns; the chunker
  splits only as deep as needed (`LEGAL_EU_LEVELED`, `SEC_10K_LEVELED`, etc.).
- **Annotation ingestion** — accept pre-computed `SplitPoint` and
  `KeepTogetherRegion` from upstream extractors (Unstructured, Docling, Azure DI).
- **`--inspect` CLI** — post-chunking diagnostics: size distribution, fallback
  ratio, orphan headings, near-miss patterns, and fix suggestions.
- **`--llm-audit`** — optional GPT-4o-mini coherence scoring per chunk
  (requires `OPENAI_API_KEY`).
- **Concurrent detectors** — `concurrent=True` fans out multiple detectors via
  `ThreadPoolExecutor`.
- **OCR damage detection** in `--recommend` — detects letterspacing artifacts
  and suggests ML heading detector.
- **Multilingual cookbook section** — guidance for Cyrillic, accented Latin,
  and other non-English scripts.
- 7 leveled preset variants (`LEGAL_EU_LEVELED`, `LEGAL_US_LEVELED`,
  `RFC_LEVELED`, `MARKDOWN_LEVELED`, `FINANCIAL_LEVELED`, `SEC_10K_LEVELED`,
  `FDA_LABEL_LEVELED`).
- Cross-domain benchmark validation on GDPR, EU AI Act, CCPA, 8 RFCs.
- GitHub Actions CI with ruff lint + pytest-cov on Python 3.9–3.12.

### Changed

- Development status bumped to **Beta**.
- `cli` extra no longer declares `click` (CLI uses `argparse`).
- All code modernized with ruff: PEP 585/604 annotations, sorted imports.

### Fixed

- README: corrected benchmark document count (11, not 14).
- README: corrected `_LEVELED` claim ("most presets", not "all").
- README: `boundary_type` list now includes `"keep_together"`.
- `Chunk` docstring: replaced phantom `"size"` boundary type with
  `"keep_together"`.
- `recommend` snippet no longer emits an unimportable
  `MLOCRHeadingDetector()` call.

## [0.2.2] — 2025-05-XX

### Added

- `chunkweaver --recommend` CLI command — analyzes a document and suggests
  presets, detectors, and target size with a ready-to-paste Python snippet.
- Improved preset scoring: density normalization, pattern coverage, combo
  detection (`FINANCIAL + FINANCIAL_TABLE`).
- Dry-run validation in recommendations with oversized/undersized warnings.

## [0.2.1] — 2025-05-XX

### Added

- LlamaIndex `ChunkWeaverNodeParser` integration.

## [0.2.0] — 2025-04-XX

### Added

- `BoundaryDetector` ABC — plug in custom structure detection.
- `HeadingDetector` — heuristic heading detection from casing, whitespace,
  length signals.
- `TableDetector` — numeric run detection for financial table keep-together.
- ML detector examples (`examples/ml-detectors/`).

## [0.1.0] — 2025-03-XX

### Added

- Initial release: `Chunker` with regex boundaries, paragraph/sentence/word
  fallback, sentence overlap, LangChain integration, CLI.
- 11 built-in presets: `LEGAL_EU`, `LEGAL_US`, `RFC`, `MARKDOWN`, `CHAT`,
  `CLINICAL`, `FINANCIAL`, `FINANCIAL_TABLE`, `SEC_10K`, `FDA_LABEL`, `PLAIN`.
- LLM-as-judge benchmark framework.
