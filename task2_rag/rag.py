import os, re, json, argparse, urllib.request
from pathlib import Path
import fitz
import numpy as np
import pandas as pd
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception as exc:
    # NumPy exact cosine/IP search is a reliable fallback if the Kaggle image
    # cannot import a FAISS wheel for its Python/NumPy combination.
    faiss = None
    FAISS_AVAILABLE = False
    print(f"FAISS unavailable ({exc}); using NumPy vector search fallback.")
from sentence_transformers import SentenceTransformer

PDF_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
RESULTS_ROOT = Path(os.environ.get("ASSIGNMENT_RESULTS_DIR", str(Path(__file__).resolve().parents[1] / "assignment_results")))
OUT = RESULTS_ROOT / "task2"
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT/"nist_ai_rmf.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# User-authored evaluation questions. Anchor phrases identify the exact source page;
# the matching source text is saved as the gold passage for auditability.
QUESTIONS = [
 {"q":"What are the four functions in the NIST AI Risk Management Framework?","anchor":"GOVERN, MAP, MEASURE, and MANAGE"},
 {"q":"What is the purpose of the Govern function?","anchor":"GOVERN"},
 {"q":"What does the Map function establish about an AI system?","anchor":"MAP"},
 {"q":"What is the purpose of measuring AI risks?","anchor":"MEASURE"},
 {"q":"What does the Manage function focus on?","anchor":"MANAGE"},
 {"q":"What does trustworthy AI mean in the framework?","anchor":"valid and reliable"},
 {"q":"Which characteristic concerns resilience to unexpected events?","anchor":"resilient"},
 {"q":"What is meant by explainable and interpretable AI?","anchor":"explainable"},
 {"q":"Why is transparency important for AI systems?","anchor":"transparent"},
 {"q":"What does accountability mean in AI risk management?","anchor":"accountable"},
 {"q":"What is the role of human-AI interaction?","anchor":"human-AI"},
 {"q":"How does the framework describe harmful bias?","anchor":"harmful bias"},
 {"q":"What is an AI actor?","anchor":"AI actors"},
 {"q":"What is the purpose of the AI RMF Playbook?","anchor":"Playbook"},
 {"q":"How should organizations use the AI RMF?","anchor":"voluntary"}
]

def download_pdf():
    import urllib.request
    from pathlib import Path

    pdf_path = Path(PDF)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse only a real PDF; remove truncated/HTML error responses.
    if pdf_path.exists():
        try:
            if pdf_path.read_bytes()[:5] == b"%PDF-":
                print(f"Using cached PDF: {pdf_path}")
                return pdf_path
        except Exception:
            pass
        pdf_path.unlink(missing_ok=True)

    candidates = [
        PDF_URL,
        "https://www.govinfo.gov/content/pkg/GOVPUB-C13-ef6882c6bd970bc733de394f469ea3f3/pdf/GOVPUB-C13-ef6882c6bd970bc733de394f469ea3f3.pdf",
        "https://rosap.ntl.bts.gov/view/dot/93262/dot_93262_DS1.pdf",
    ]
    errors = []
    for candidate_url in candidates:
        try:
            request = urllib.request.Request(
                candidate_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
                    "Accept": "application/pdf,*/*",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if not payload.startswith(b"%PDF-") or len(payload) < 100_000:
                raise ValueError("Downloaded response is not a valid-sized PDF")
            pdf_path.write_bytes(payload)
            print(f"Downloaded PDF from {candidate_url} ({len(payload):,} bytes)")
            return pdf_path
        except Exception as exc:
            errors.append(f"{candidate_url}: {type(exc).__name__}: {exc}")
            print("PDF source failed:", errors[-1])

    raise RuntimeError(
        "Could not download the NIST AI RMF PDF from any configured source. "
        "Check Kaggle Internet access. Details: " + " | ".join(errors)
    )

def extract_pages(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        txt = re.sub(r"\s+", " ", page.get_text("text")).strip()
        if txt: pages.append({"page": i+1, "text": txt})
    if len(pages) < 30:
        raise RuntimeError(f"Expected at least 30 pages; extracted {len(pages)}")
    return pages

def make_chunks(pages, chunk_words, overlap_words=40):
    chunks = []
    for p in pages:
        words = p["text"].split()
        step = max(1, chunk_words-overlap_words)
        for start in range(0, len(words), step):
            part = words[start:start+chunk_words]
            if not part: continue
            chunks.append({"page": p["page"], "text": " ".join(part),
                           "start_word": start, "end_word": start+len(part)})
            if start+chunk_words >= len(words): break
    return chunks

def gold_passages(pages):
    output = []
    for item in QUESTIONS:
        needle = item["anchor"].lower()
        matches = [(p["page"], p["text"]) for p in pages if needle in p["text"].lower()]
        if not matches:
            # Robust fallback: find page with most anchor-token overlap.
            toks = [t for t in re.findall(r"[a-z0-9-]+", needle) if len(t)>2]
            scored = [(sum(t in p["text"].lower() for t in toks), p) for p in pages]
            score, page = max(scored, key=lambda x:x[0])
            if score == 0: raise RuntimeError(f"Could not locate gold source for: {item['q']}")
            matches = [(page["page"], page["text"])]
        page_no, passage = matches[0]
        output.append({**item, "gold_page": page_no, "gold_passage": passage[:1800]})
    return output

def run_retrieval(chunks, questions, embedder, chunk_words):
    texts = [c["text"] for c in chunks]
    emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                          show_progress_bar=True).astype("float32")
    qemb = embedder.encode([q["q"] for q in questions], convert_to_numpy=True,
                           normalize_embeddings=True).astype("float32")
    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        scores, ids = index.search(qemb, min(5, len(chunks)))
        backend = "FAISS IndexFlatIP"
    else:
        # Embeddings are unit-normalized, so dot product is cosine similarity.
        similarities = qemb @ emb.T
        k = min(5, len(chunks))
        ids = np.argsort(-similarities, axis=1)[:, :k]
        scores = np.take_along_axis(similarities, ids, axis=1)
        backend = "NumPy exact inner-product search"
    hits, rows = 0, []
    for q, sc, ix in zip(questions, scores, ids):
        retrieved = [chunks[int(i)] for i in ix if i >= 0]
        hit = any(c["page"] == q["gold_page"] for c in retrieved)
        hits += int(hit)
        rows.append({"question": q["q"], "gold_page": q["gold_page"],
                     "gold_passage": q["gold_passage"], "recall_at_5_hit": hit,
                     "top5_pages": [c["page"] for c in retrieved],
                     "top1_passage": retrieved[0]["text"] if retrieved else ""})
    return {"chunk_words": chunk_words, "recall_at_5": hits/len(questions),
            "hits": hits, "total": len(questions), "search_backend": backend, "rows": rows}

def generate_answer(question, retrieved_text, tokenizer, model):
    prompt = (
        "Answer the question using only the supplied NIST AI RMF evidence. "
        "If the evidence is insufficient, say so. Do not invent facts.\\n"
        f"Evidence: {retrieved_text[:3500]}\\n"
        f"Question: {question}\\nAnswer:"
    )
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"question": question, "answer": answer, "evidence": retrieved_text,
            "mode": "Qwen2.5-0.5B-Instruct grounded generation"}

def main():
    download_pdf()
    pages = extract_pages(PDF)
    questions = gold_passages(pages)
    with open(OUT/"test_questions_and_gold_passages.json", "w") as f:
        json.dump(questions, f, indent=2)
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    reports = []
    for size in [150, 300]:
        chunks = make_chunks(pages, size)
        result = run_retrieval(chunks, questions, embedder, size)
        reports.append(result)
        with open(OUT/f"retrieval_{size}.json", "w") as f: json.dump(result, f, indent=2)
    comparison = [{"chunk_words": x["chunk_words"], "recall_at_5": x["recall_at_5"],
                   "hits": x["hits"], "total": x["total"]} for x in reports]
    pd.DataFrame(comparison).to_csv(OUT/"retrieval_comparison.csv", index=False)
    best = max(reports, key=lambda x:x["recall_at_5"])
    failures = [r for r in best["rows"] if not r["recall_at_5_hit"]]
    with open(OUT/"retrieval_failures.json", "w") as f: json.dump(failures[:2], f, indent=2)
    # Run a small open-weight LLM for grounded final answers over the best retrieved passages.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    qa_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if torch.cuda.is_available():
        qa_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        qa_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float32
        )
        qa_model.to("cpu")
    generated_answers = [
        generate_answer(r["question"], r["top1_passage"], qa_tokenizer, qa_model)
        for r in best["rows"]
    ]
    with open(OUT/"generated_answers.json", "w") as f:
        json.dump(generated_answers, f, indent=2)
    del qa_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Pages:", len(pages))
    print(json.dumps(comparison, indent=2))
    print("Task 2 outputs:", OUT)

if __name__ == "__main__": main()
