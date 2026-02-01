# Architecture

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLI (argparse)                                                              │
│  --data_dir | --zip_path | --extract_dir | --query | --top_k | --summarize   │
└──────────────────────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Data Resolve (resolve_data_dir)                                             │
│  • If --data_dir: use it                                                     │
│  • Else: extract --zip_path to --extract_dir                                 │
│  • Detect single top-level dir in zip → use that as data_dir                 │
└──────────────────────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Loader + Preprocess (build_index)                                           │
│  • Load .txt and .pdf files                                                  │
│  • clean_text: strip, normalize whitespace, remove null/non-printing chars   │
│  • PDF: chunk with overlap (extract_pdf_chunks)                              │
│  • TXT: one chunk per file (load_txt_file)                                   │
│  • Validate: min 100 documents                                               │
└──────────────────────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Embed Corpus (SentenceTransformer)                                          │
│  • Model: sentence-transformers/all-MiniLM-L6-v2                             │
│  • normalize_embeddings=True                                                 │
│  • Embeddings stored in-memory (numpy array)                                 │
└──────────────────────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Embed Query + Cosine Similarity (search)                                    │
│  • Embed query with same model                                               │
│  • cosine_similarity(query_emb, corpus_emb)                                  │
│  • argsort descending → top-K indices                                        │
└──────────────────────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Top-K Results (print_results)                                               │
│  rank, score, doc_id, snippet (~200 chars)                                   │
└──────────────────────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼ (if --summarize)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Summarizer (summarizer.summarize_results)                                   │
│  • Combine top-K full text                                                   │
│  • Extractive summary (first_sentences)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Functions and Classes

### semantic_search_engine.py

| Function | Role |
|----------|------|
| `resolve_data_dir()` | Chooses data directory from `--data_dir` or by extracting zip to `--extract_dir`; handles single top-level dir in zip |
| `load_txt_file()` | Loads one .txt file as one chunk; applies `clean_text()` |
| `extract_pdf_chunks()` | Chunks PDF with overlap; applies `clean_text()` |
| `clean_text()` | Strip, normalize whitespace, remove null bytes and non-printable chars |
| `build_index()` | Loads docs, validates min 100, creates embeddings; returns chunks, embeddings, model |
| `search()` | Embeds query, computes cosine similarity, returns top-K with rank, score, doc_id, snippet |
| `print_results()` | Prints results in rubric format |
| `main()` | Parses args, resolves data, builds index, runs search, optionally summarizer |

### summarizer.py

| Function | Role |
|----------|------|
| `first_sentences()` | Extractive summary: first N sentences up to max_chars |
| `summarize_results()` | Combines top-K full text, prints summary and sources |
| `main()` | Standalone interactive loop (load index, prompt for queries, summarize) |

## Data Flow

- **Chunks**: List of dicts with `chunk_id`, `pdf_file` (doc filename), `page`, `text`.
- **Embeddings**: `np.float32` array, shape `(n_chunks, dim)`, in memory. No disk cache by default.
- **Results**: List of dicts with `rank`, `score`, `doc_id`, `chunk_id`, `page`, `preview` (snippet).

## Limitations and Extensions

- **Embeddings**: Recomputed every run; no caching. Possible extension: cache embeddings in `./cache/embeddings.npz` and `./cache/metadata.json` keyed by hash of data directory.
- **Chunking**: PDFs chunked with overlap; TXT treated as single chunk. Possible extension: chunk long TXT files.
- **UI**: CLI only. Possible extension: Streamlit or web UI.
- **Summarizer**: Extractive only. Possible extension: LLM-based summarization with API key.
