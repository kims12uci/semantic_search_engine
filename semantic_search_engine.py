# -*- coding: utf-8 -*-
"""
Semantic Search Engine - BANA 275
Indexes documents (PDF, TXT), creates embeddings, and retrieves top-K results via cosine similarity.
"""

import argparse
import os
import re
import zipfile

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 300
OVERLAP = 75
BATCH_SIZE = 64
MIN_DOCUMENTS = 100
SNIPPET_LEN = 200


def clean_text(s: str) -> str:
    """Strip, normalize whitespace, remove null bytes and non-printing chars."""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\x00", "")
    s = "".join(c for c in s if c.isprintable() or c in "\n\t\r")
    s = s.replace("-\n", "").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_txt_file(path: str) -> list[dict]:
    """Load a single .txt file as one chunk. Returns list of one chunk dict."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = clean_text(text)
    if not text.strip():
        return []
    filename = os.path.basename(path)
    return [{
        "chunk_id": filename,
        "pdf_file": filename,
        "page": 0,
        "text": text,
    }]


def extract_pdf_chunks(pdf_path: str, max_chars: int = 900, overlap: int = 120) -> list[dict]:
    """Extract overlapping text chunks from a PDF."""
    reader = PdfReader(pdf_path)
    pdf_file = os.path.basename(pdf_path)
    chunks = []

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if not text.strip():
            continue

        start = 0
        page_num = i + 1
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"{pdf_file}::p{page_num}::c{len(chunks)}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "pdf_file": pdf_file,
                    "page": page_num,
                    "text": chunk_text,
                })
            if end == len(text):
                break
            start = max(0, end - overlap)

    return chunks


def resolve_data_dir(
    data_dir: str | None,
    zip_path: str | None,
    extract_dir: str,
) -> str:
    """
    Resolve data directory:
    - If data_dir provided: use it.
    - Else: extract zip_path to extract_dir, detect single top-level dir, return path.
    """
    if data_dir:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        return os.path.abspath(data_dir)

    if not zip_path or not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Zip file not found: {zip_path}. "
            "Use --data_dir to specify a document directory, or ensure files.zip exists."
        )

    extract_path = os.path.abspath(extract_dir)
    os.makedirs(extract_path, exist_ok=True)

    # Check if already extracted (non-empty)
    entries = [e for e in os.listdir(extract_path) if not e.startswith(".")]
    if not entries:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)

    entries = [e for e in os.listdir(extract_path) if not e.startswith(".")]
    if not entries:
        raise ValueError(f"Zip extracted to {extract_path} but folder is empty.")

    # Single top-level directory: use it as data_dir
    if len(entries) == 1:
        single = os.path.join(extract_path, entries[0])
        if os.path.isdir(single):
            return os.path.abspath(single)
    return extract_path


def build_index(data_dir: str) -> tuple[list[dict], np.ndarray, SentenceTransformer]:
    """Load documents, preprocess, and build embeddings index."""
    txt_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".txt")
    ])
    pdf_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ])
    doc_count = len(txt_files) + len(pdf_files)

    if doc_count < MIN_DOCUMENTS:
        raise ValueError(
            f"Corpus has {doc_count} documents. Rubric requires at least {MIN_DOCUMENTS} documents. "
            f"Add more .txt or .pdf files to {data_dir}."
        )

    all_chunks = []
    for path in txt_files:
        try:
            chunks = load_txt_file(path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}", file=__import__("sys").stderr)

    for path in pdf_files:
        try:
            chunks = extract_pdf_chunks(path, MAX_CHARS, OVERLAP)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}", file=__import__("sys").stderr)

    if not all_chunks:
        raise ValueError(
            f"No readable text found in {data_dir}. "
            "Ensure .txt files are UTF-8 and .pdf files contain extractable text."
        )

    print(f"Loaded {doc_count} documents, {len(all_chunks)} chunks.")

    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{MODEL_NAME}'. "
            "Check network connection and sentence-transformers installation."
        ) from e

    texts = [c["text"] for c in all_chunks]
    embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        emb = model.encode(batch, normalize_embeddings=True)
        embs.append(emb)
    embeddings = np.vstack(embs).astype(np.float32)
    return all_chunks, embeddings, model


def search(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """Run semantic search and return top-K results with rank, score, doc_id, snippet."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if not chunks or embeddings.size == 0:
        raise ValueError("Corpus is empty. Load documents first.")

    q_emb = model.encode([query.strip()], normalize_embeddings=True).astype(np.float32)
    scores = cosine_similarity(q_emb, embeddings)[0]
    idx = np.argsort(-scores)[:top_k]

    results = []
    for rank, j in enumerate(idx, start=1):
        c = chunks[j]
        text = c["text"]
        snippet = (text[:SNIPPET_LEN] + "...") if len(text) > SNIPPET_LEN else text
        results.append({
            "rank": rank,
            "score": float(scores[j]),
            "doc_id": c["pdf_file"],
            "pdf_file": c["pdf_file"],
            "chunk_id": c["chunk_id"],
            "page": c["page"],
            "preview": snippet,
        })
    return results


def print_results(results: list[dict]) -> None:
    """Print results in rubric format: rank, score, doc_id, snippet."""
    for r in results:
        print(f"\n#{r['rank']}  score={r['score']:.4f}")
        print(f"document id: {r['doc_id']}")
        print(f"snippet: {r['preview']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search engine over document corpus (PDF, TXT)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing documents (.txt, .pdf). If provided, used directly; --zip_path ignored.",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default="./files.zip",
        help="Path to corpus zip file (default: ./files.zip). Used when --data_dir is not provided.",
    )
    parser.add_argument(
        "--extract_dir",
        type=str,
        default="./data",
        help="Where to extract zip (default: ./data). Used when loading from --zip_path.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query (required unless --interactive).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return (default: 5, must be > 0).",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize the retrieved top-K results using summarizer.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode: prompt for queries until empty input.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be > 0.")

    if not args.query and not args.interactive:
        parser.print_help()
        print("\nUsage: Provide --query \"your search\" or use --interactive for prompt mode.")
        raise SystemExit(1)

    try:
        data_dir = resolve_data_dir(args.data_dir, args.zip_path, args.extract_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        raise SystemExit(1) from e

    try:
        chunks, embeddings, model = build_index(data_dir)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        raise SystemExit(1) from e

    def run_query(q: str) -> list[dict]:
        try:
            return search(q, chunks, embeddings, model, top_k=args.top_k)
        except ValueError as e:
            print(f"Error: {e}", file=__import__("sys").stderr)
            return []

    def do_summarize(res: list[dict]) -> None:
        from summarizer import summarize_results
        summarize_results(res, chunks)

    if args.interactive:
        while True:
            q = input("Enter search query (or press Enter to quit): ").strip()
            if not q:
                print("Goodbye.")
                break
            results = run_query(q)
            if not results:
                print("No results.")
                continue
            print_results(results)
            if args.summarize:
                do_summarize(results)
        return

    results = run_query(args.query)
    if not results:
        raise SystemExit(1)
    print_results(results)
    if args.summarize:
        do_summarize(results)


if __name__ == "__main__":
    main()
