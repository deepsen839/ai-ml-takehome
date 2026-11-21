import json
import re
from pathlib import Path

import pymupdf
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assignment_results" / "task2"
PDF = OUT / "nist_ai_rmf.pdf"
QUESTIONS_FILE = OUT / "test_questions_and_gold_passages.json"

# Physical PDF page numbers, not the printed page numbers.
GOLD_PAGES = {
    1: 25,
    2: 26,
    3: 29,
    4: 33,
    5: 36,
    6: 17,
    7: 20,
    8: 21,
    9: 20,
    10: 21,
    11: 45,
    12: 23,
    13: 7,
    14: 26,
    15: 7,
}

doc = pymupdf.open(PDF)
questions = json.loads(QUESTIONS_FILE.read_text())

assert len(questions) == 15, f"Expected 15 questions, found {len(questions)}"

for i, item in enumerate(questions, start=1):
    page_no = GOLD_PAGES[i]
    page_text = re.sub(r"\s+", " ", doc[page_no - 1].get_text("text")).strip()

    if not page_text:
        raise RuntimeError(f"No text extracted from PDF page {page_no}")

    item["gold_page"] = page_no
    item["gold_passage"] = page_text[:3500]

    print(f"Q{i}: gold page {page_no} | {item['q']}")

QUESTIONS_FILE.write_text(
    json.dumps(questions, indent=2, ensure_ascii=False)
)

# Import the existing chunking and retrieval implementation.
import sys
sys.path.insert(0, str(ROOT / "task2_rag"))
from rag import extract_pages, make_chunks, run_retrieval, PDF as RAG_PDF

pages = extract_pages(PDF)
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

reports = []

for size in (150, 300):
    chunks = make_chunks(pages, size, overlap_words=40)
    result = run_retrieval(chunks, questions, embedder, size)

    result_path = OUT / f"retrieval_{size}_corrected.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    reports.append(result)

    print(f"\nChunk size: {size}")
    print(f"Recall@5: {result['hits']}/{result['total']} = "
          f"{result['recall_at_5']:.2%}")

    for row in result["rows"]:
        if not row["recall_at_5_hit"]:
            print("MISS:", row["question"])
            print("  Gold page:", row["gold_page"])
            print("  Retrieved:", row["top5_pages"])

comparison = [
    {
        "chunk_words": r["chunk_words"],
        "recall_at_5": r["recall_at_5"],
        "hits": r["hits"],
        "total": r["total"],
    }
    for r in reports
]

pd.DataFrame(comparison).to_csv(
    OUT / "retrieval_comparison_corrected.csv", index=False
)

best = max(reports, key=lambda r: r["recall_at_5"])
failures = [row for row in best["rows"] if not row["recall_at_5_hit"]]

(OUT / "retrieval_failures_corrected.json").write_text(
    json.dumps(failures, indent=2, ensure_ascii=False)
)

print("\nCorrected comparison:")
print(pd.DataFrame(comparison).to_string(index=False))
print("\nCorrected outputs saved in:", OUT)
