# Semantic Search Engine (BANA 275)

Semantic search engine over the article corpus in `files.zip`. Indexes PDF and TXT documents, creates embeddings with SentenceTransformer, and retrieves top-K results via cosine similarity.

## Requirements Mapping

| Requirement | Implementation |
|-------------|----------------|
| Domain | Article/document corpus |
| Min 100 documents | Validated in `build_index()`; exits with error if fewer |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Cosine similarity top-K | `sklearn.metrics.pairwise.cosine_similarity` in `search()` |
| Summarization option | `--summarize` invokes `summarizer.py` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

### Default zip workflow

```bash
python semantic_search_engine.py --query "Risks of AI" --top_k 5
```

### Custom zip path + extract dir

```bash
python semantic_search_engine.py --zip_path path/to/files.zip --extract_dir ./data --query "machine learning" --top_k 10
```

### Custom directory (no zip)

```bash
python semantic_search_engine.py --data_dir path/to/articles --query "climate change" --top_k 5
```

### Summarization

```bash
python semantic_search_engine.py --query "Risks of AI" --top_k 5 --summarize
```

### Interactive mode

```bash
python semantic_search_engine.py --interactive
# Or with summarization:
python semantic_search_engine.py --interactive --summarize
```

**Note:** The summarizer uses extractive summarization (first sentences). No API key is required. If a future version uses an LLM, set `OPENAI_API_KEY` (or the relevant env var) and the script will produce a friendly error if it is missing.

## Example Output

```
#1  score=0.4523
document id: article_42.pdf
snippet: Machine learning models have become increasingly prevalent in risk assessment...

#2  score=0.4012
document id: report_15.txt
snippet: AI systems introduce new challenges for governance and transparency...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Zip not found | Use `--data_dir` to point to an existing directory, or place `files.zip` in the current directory |
| Fewer than 100 docs | Add more .txt or .pdf files to meet the rubric requirement (minimum 100 documents) |
| Missing API key | Current summarizer does not require one; if using an LLM-based summarizer, set the appropriate env var |
| Embedding model download fails | Check network connection; first run downloads the model |
| No results | Ensure query is non-empty and corpus has readable text |
