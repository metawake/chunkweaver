# API Reference

## `Chunker`

```python
Chunker(target_size, overlap, overlap_unit, boundaries, fallback,
        min_size, sentence_pattern, keep_together, detectors,
        annotations, concurrent)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_size` | `int` | `1024` | Target chunk size in characters |
| `overlap` | `int` | `2` | Number of overlap units from previous chunk |
| `overlap_unit` | `str` | `"sentence"` | `"sentence"`, `"paragraph"`, or `"chars"` |
| `boundaries` | `list[BoundarySpec]` | `[]` | Regex patterns or `(regex, level)` tuples marking section starts |
| `fallback` | `str` | `"paragraph"` | Sub-split strategy: `"paragraph"`, `"sentence"`, `"word"` |
| `min_size` | `int` | `200` | Minimum chunk size (merge smaller segments) |
| `sentence_pattern` | `str \| Pattern \| None` | `None` | Custom regex for sentence detection (default: English). Built-in alternatives: `SENTENCE_END_CJK`, `SENTENCE_END_PERMISSIVE`. See [cookbook](cookbook.md#cyrillic-accented-latin-and-other-non-english-scripts). |
| `keep_together` | `list[str] \| None` | `None` | Patterns for lines that must stay with next segment |
| `detectors` | `list[BoundaryDetector] \| None` | `None` | Heuristic detectors for structure discovery |
| `annotations` | `list[Annotation] \| None` | `None` | Pre-computed `SplitPoint` / `KeepTogetherRegion` from extractors |
| `concurrent` | `bool` | `False` | Run detectors in parallel via `ThreadPoolExecutor` |

### `Chunker.chunk(text) -> list[str]`

Returns a list of chunk strings.

### `Chunker.chunk_with_metadata(text) -> list[Chunk]`

Returns a list of `Chunk` objects with full metadata.

## `Chunk`

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Full chunk content (including overlap) |
| `start` | `int` | Start offset in original text (excluding overlap) |
| `end` | `int` | End offset in original text |
| `index` | `int` | Zero-based chunk index |
| `boundary_type` | `str` | What triggered the split |
| `boundary_level` | `int` | Hierarchy level of the boundary (0 = strongest) |
| `overlap_text` | `str` | The overlap prefix |
| `content_text` | `str` | Text without overlap (property) |

## How it works

1. **Run detectors + merge annotations** — heuristic detectors and pre-computed annotations produce split points + keep-together regions
2. **Detect boundaries** — scan each line against your regex patterns, merge with detector/annotation split points, suppress splits inside keep-together regions
3. **Hierarchical split** — split at level-0 boundaries first; for oversized segments, descend to level-1 boundaries, then level-2, etc. (flat when all boundaries share level 0)
4. **Isolate protected regions** — carve keep-together regions (tables) into their own segments
5. **Sub-split oversized segments** — break large sections at paragraph -> sentence -> word boundaries; allow protected regions to overshoot `target_size`
6. **Merge undersized segments** — combine tiny segments (like standalone headings) with their body text; hierarchy-aware (chapter headings merge into first article)
7. **Add overlap** — prepend the last N sentences/paragraphs/chars from the previous chunk
8. **Return** — chunks with full metadata (offsets, boundary type, hierarchy level, overlap tracking)

## Architecture

```
chunkweaver/
├── __init__.py            # Public API: Chunker, Chunk, detectors, sentence patterns
├── chunker.py             # Core algorithm: hierarchical split + detector + annotation merge
├── detectors.py           # BoundaryDetector ABC, SplitPoint, KeepTogetherRegion
├── detector_heading.py    # HeadingDetector — heuristic heading detection
├── detector_table.py      # TableDetector — financial table keep-together
├── models.py              # Chunk dataclass (with boundary_level)
├── boundaries.py          # Regex boundary detection engine (with BoundarySpec levels)
├── sentences.py           # Configurable sentence splitting (regex, no NLP)
├── presets.py             # 11 flat presets + 7 leveled presets
├── recommend.py           # Document analysis and config recommendations
├── inspect.py             # Post-chunking diagnostics and LLM audit
├── cli.py                 # CLI entry point
└── integrations/
    ├── langchain.py       # LangChain TextSplitter wrapper
    └── llamaindex.py      # LlamaIndex NodeParser wrapper
```

**Design principles:**
- Each module has a single responsibility
- No deeply nested conditionals — small, testable functions
- All decisions are logged/exposed via chunk metadata
- Zero dependencies for core; optional extras for CLI and integrations
- Detectors and annotations are composable — stack any combination without conflicts
- Hierarchical boundaries degrade gracefully to flat splitting when all levels are equal
