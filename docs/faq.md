# FAQ

| Question | Answer |
|----------|--------|
| **...chunk PDFs or DOCX files?** | Yes — run your file through any extractor that outputs text/markdown (marker-pdf, docling, pdfminer, Azure Document Intelligence), then feed the result to chunkweaver. It operates on text, not file formats. |
| **...keep tables from being split?** | Yes — `TableDetector()` marks tables as keep-together regions. The chunker won't cut inside them. See [Financial documents](cookbook.md#financial-documents-tables--headings). |
| **...handle OCR-damaged headings like `D E F I N I T I O N S`?** | Yes — `chunkweaver --recommend` detects letterspacing artifacts and suggests `MLOCRHeadingDetector`. See [examples/ml-detectors/](../examples/ml-detectors/). |
| **...split on custom section markers?** | Yes — pass any regex as `boundaries=[r"^Article\s+\d+"]`. See [Custom boundaries](cookbook.md#custom-boundaries-for-any-domain). |
| **...keep chapters intact but split oversized ones at articles?** | Yes — use leveled presets like `LEGAL_EU_LEVELED`. Level 0 always splits; deeper levels only split when needed. See [Hierarchical boundaries](cookbook.md#hierarchical-boundaries). |
| **...pass structure from my PDF extractor?** | Yes — pass `SplitPoint` and `KeepTogetherRegion` annotations directly via the `annotations` parameter. See [Annotation ingestion](cookbook.md#annotation-ingestion-from-extractors). |
| **...auto-detect the right config for my document?** | Yes — `chunkweaver --recommend myfile.txt` analyzes structure and suggests presets, detectors, and target size. |
| **...use it with LangChain?** | Yes — `ChunkWeaverSplitter` is a drop-in `TextSplitter`. `pip install chunkweaver[langchain]`. |
| **...use it with LlamaIndex?** | Yes — `ChunkWeaverNodeParser` is a drop-in `NodeParser`. `pip install chunkweaver[llamaindex]`. |
| **...chunk Chinese / Japanese / Korean text?** | Yes — use `SENTENCE_END_CJK` for sentence splitting. See [CJK text](cookbook.md#chinese--japanese--korean-text). |
| **...chunk Cyrillic, Spanish, or other non-English text?** | Yes — use `SENTENCE_END_PERMISSIVE` for sentence splitting and write custom `boundaries` for your document structure. See [Non-English scripts](cookbook.md#cyrillic-accented-latin-and-other-non-english-scripts). |
| **...chunk chat logs or support transcripts?** | Yes — `CHAT` preset splits on speaker turns and timestamps. See [Chat logs](cookbook.md#chat-logs--customer-support-transcripts). |
| **...chunk clinical notes?** | Yes — `CLINICAL` preset recognizes `HPI:`, `ASSESSMENT:`, `PLAN:`, etc. See [Clinical notes](cookbook.md#healthcare--clinical-notes). |
| **...chunk FDA drug labels?** | Yes — `FDA_LABEL` / `FDA_LABEL_LEVELED` presets split on numbered sections and subsections. See [FDA drug labels](cookbook.md#fda-drug-labels). |
| **...chunk SEC 10-K filings?** | Yes — `SEC_10K` / `SEC_10K_LEVELED` presets split on `PART`/`Item` hierarchy plus ALL-CAPS sub-headings. See [SEC filings](cookbook.md#sec-filings-10-k). |
| **...write my own structure detector?** | Yes — subclass `BoundaryDetector` and return `SplitPoint` / `KeepTogetherRegion`. See [Custom detectors](cookbook.md#custom-detectors) (example in the README). |
| **...get token counts instead of character counts?** | Not directly — `target_size` is in characters. Estimate `tokens ≈ chars / 4`. |
| **...check if my chunking config is good?** | Yes — `chunkweaver --inspect myfile.txt` analyzes chunk quality, flags problems (oversized chunks, high fallback ratio, orphan headings), and suggests fixes. |
| **...get LLM-based quality feedback on chunks?** | Yes — `chunkweaver --inspect --llm-audit myfile.txt` rates each chunk's semantic coherence via GPT-4o-mini (requires `OPENAI_API_KEY`). |
| **...call a remote ML server or API from a detector?** | Yes — `BoundaryDetector.detect()` can do anything internally (HTTP, gRPC, etc.). Use `concurrent=True` on the Chunker to fan out multiple detectors in parallel. |
| **...install any ML or NLP dependencies?** | No — core is stdlib-only. ML detectors are optional examples, not requirements. |
