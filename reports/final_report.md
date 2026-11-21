# Applied AI/ML Assignment — Results Report

**Status:** Results package assembled from the supplied run artifacts. Task 2 gold-evidence audit and Task 3 manual quality review remain outstanding; Task 1 recorded held-out test has no failures.

## Task 1 — LoRA fine-tuning
Model: Qwen2.5-0.5B-Instruct. Calendar-request-to-JSON task. Split: 1,300 train / 100 validation / 100 held-out. LoRA: r=8, alpha=16, dropout=0.05, q/k/v/o and gate/up/down projections, learning rate 2e-4, 3 epochs.

| Metric | Base | Fine-tuned |
|---|---:|---:|
| Exact match | 0.00% | 100.00% |
| Valid JSON | 100.00% | 100.00% |

The fine-tuned failure file is empty; three failures must not be invented. A separate challenge set is required to probe edge cases.

**7B memory estimate:** FP16 weights 7B×2=14 GB. Full Adam fine-tuning rough total: 14 GB weights +14 GB gradients +28 GB FP32 master weights +56 GB Adam moments ≈112 GB before activations. LoRA: 14 GB frozen FP16 base plus adapters/optimizer/activations, roughly 16–24+ GB. QLoRA: 7B×0.5=3.5 GB raw 4-bit weights plus metadata/adapters/activations, roughly 6–12+ GB. Estimates vary by implementation, batch, sequence length and checkpointing.

## Task 2 — RAG
NIST AI RMF 1.0; PyMuPDF extraction; all-MiniLM-L6-v2 embeddings; FAISS inner product or NumPy fallback; 15 questions; 40-word overlap. No LangChain/LlamaIndex.

| Chunk size | Recall@5 |
|---|---:|
| 150 words | 2/15 = 13.33% |
| 300 words | 3/15 = 20.00% |

300-word chunks performed better in the recorded run, but the scores are provisional: broad-anchor matching selected front matter/table-of-contents pages for some gold labels. Correct the gold passages and rerun before claiming validated accuracy. Two observed failures: four-functions question (gold points to front matter); Govern-purpose question (gold points to table of contents). Fix by manually anchoring to the body text of the relevant NIST sections and rerunning both chunk sizes.

BM25 adds lexical matching for exact terms/acronyms/rare words; dense embeddings add semantic matching. A re-ranker jointly scores query-passage pairs from the shortlist to improve ordering.

## Task 3 — inference optimisation
Qwen2.5-0.5B-Instruct; same 10 prompts; FP16 vs 4-bit NF4.

| Metric | FP16 | 4-bit NF4 |
|---|---:|---:|
| Tokens generated | 1,175 | 1,190 |
| Tokens/sec | 25.33 | 21.56 |
| Peak allocated | 0.739 GB | 0.445 GB |
| Peak reserved | 0.760 GB | 0.463 GB |

FP16 was about 17.5% faster; NF4 reduced recorded peak allocated memory by about 39.7%. The supplied quality-note fields are blank; manual pairwise review is still required. CUDA peak figures are benchmark peaks, not total process memory.

KV cache stores attention keys/values for prior tokens to avoid recomputation; it speeds decoding but grows with context and batch size. Quantisation stores weights at fewer bits, reducing raw weight memory with metadata overhead and possible quality/speed trade-offs.

**20-user / 24 GB GPU plan:** 7B 4-bit model, continuous batching, paged KV cache, bounded context/output lengths, admission control and request queue, streaming, timeouts, monitoring and a 20-session load test. Reserve memory for runtime/activations; do not assume 20 long-context sequences fit simultaneously.

## Submission caveats
Task 2 needs gold-passage correction and rerun. Task 1 needs a separately labelled stress set if three failure examples are mandatory; the supplied held-out run has none. Task 3 needs human quality notes for the 10 prompt pairs. Hardware identity for the supplied run is not included in the result JSON, so report it from the actual machine log rather than guessing.
