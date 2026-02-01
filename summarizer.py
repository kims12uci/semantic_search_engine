# -*- coding: utf-8 -*-
"""
summarizer.py - Summarize the top-K search results.
Uses semantic_search_engine for indexing and search.
"""

import re
import sys

from semantic_search_engine import build_index, resolve_data_dir, search


def first_sentences(text: str, max_sentences: int = 10, max_chars: int = 600) -> str:
    """Extract the first few sentences as an extractive summary."""
    text = text.strip()
    if not text:
        return ""
    # Split on sentence boundaries (simple: . ! ? followed by space or end)
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    length = 0
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if length + len(s) + 1 > max_chars and out:
            break
        out.append(s)
        length += len(s) + 1
        if len(out) >= max_sentences:
            break
    return " ".join(out).strip() or text[:max_chars].strip()


def summarize_results(results: list[dict], chunks: list[dict]) -> None:
    """Combine the top 5 results into one text and print a single combined summary."""
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    combined_parts = []
    for r in results:
        full_text = chunk_by_id.get(r["chunk_id"], {}).get("text", r.get("preview", ""))
        if full_text.strip():
            combined_parts.append(full_text.strip())
    combined_text = " ".join(combined_parts)

    k = len(results)
    print("\n" + "=" * 60)
    print(f"COMBINED SUMMARY (TOP {k} RESULTS)")
    print("=" * 60)
    print("\nSummary:\n")
    print(first_sentences(combined_text))
    print("\nSources:")
    for r in results:
        print(f"  • {r['pdf_file']} (page {r['page']})")
    print("=" * 60)


def main():
    print("Loading index (this may take a moment)...")
    try:
        data_dir = resolve_data_dir(None, "./files.zip", "./data")
        chunks, embeddings, model = build_index(data_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("Index ready.\n")
    while True:
        query = input("Enter search query (or press Enter to quit): ").strip()
        if not query:
            print("Goodbye.")
            break
        results = search(query, chunks, embeddings, model, top_k=5)
        if not results:
            print("No results found.\n")
            continue
        summarize_results(results, chunks)
        print()


if __name__ == "__main__":
    main()
