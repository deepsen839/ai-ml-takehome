# AI/ML Engineering Assignment — Results Report

**Environment:** Kaggle [GPU type: ___], Python [___], CUDA [___]  
**Repository:** [GitHub URL]  
**Run date:** [___]

## Task 1 — LoRA fine-tuning

Task: natural-language calendar request → fixed JSON. Base model: Qwen2.5-0.5B-Instruct.

| Metric | Base | LoRA fine-tuned |
|---|---:|---:|
| Exact JSON match (100 examples) | [ ] | [ ] |
| Valid JSON rate | [ ] | [ ] |

Settings: rank 8; alpha 16; dropout 0.05; target modules q/k/v/o and MLP gate/up/down projections; learning rate 2e-4; [3] epochs.

Rationale: rank 8 limits trainable adapter size while providing useful adaptation capacity. Alpha 16 scales the adapter update at 2r. Attention projections learn input/output relationships; MLP projections add capacity for field and format mapping. A 2e-4 learning rate is a practical adapter-tuning starting point.

**Three observed failure cases (copy from `failure_cases.json`):**
1. Input / expected / predicted: [___]. Likely cause: [___].
2. Input / expected / predicted: [___]. Likely cause: [___].
3. Input / expected / predicted: [___]. Likely cause: [___].

## Task 2 — RAG from scratch

Source: NIST AI Risk Management Framework 1.0 (official PDF, extracted pages: [ ]).  
Embedding model: all-MiniLM-L6-v2; vector index: FAISS IndexFlatIP; overlap: 40 words.

| Chunk size | Recall@5 |
|---|---:|
| 150 words | [ ] |
| 300 words | [ ] |

Better setting: [___]. Reason: [___]. Note that gold relevance is page-level, not exact chunk-level.

**Two retrieval failures and fixes:**
1. Question / retrieved pages / expected page: [___]. Fix: [___].
2. Question / retrieved pages / expected page: [___]. Fix: [___].

**BM25:** BM25 adds lexical matching, especially for exact names, rare terms, IDs, and phrases that embedding similarity may overlook. Hybrid retrieval combines lexical and semantic evidence.

**Re-ranker:** A re-ranker jointly scores each query and candidate passage, improving the order of the shortlist after the faster initial retrieval stage.

## Task 3 — local inference and optimisation

Model: Qwen2.5-0.5B-Instruct; same 10 prompts; greedy decoding; max 128 new tokens.

| Setting | Aggregate tokens/sec | Peak allocated VRAM | Peak reserved VRAM |
|---|---:|---:|---:|
| FP16 | [ ] | [ ] GB | [ ] GB |
| 4-bit NF4 | [ ] | [ ] GB | [ ] GB |

Quality observations across the same prompts: [Discuss factuality, completeness, formatting, and any regressions. Refer to `prompt_outputs.csv`.]

**KV cache:** Stores attention keys and values for previous tokens so they do not need to be recomputed at every decoding step. It improves decoding speed but grows with context length and concurrent sequences.

**Quantisation:** Stores weights with fewer bits. FP16 uses 2 bytes per weight while 4-bit uses 0.5 bytes per weight before metadata, substantially reducing weight memory.

**Serving 20 concurrent users on one 24 GB GPU (back-of-envelope):**
- Assumptions: 7B model in 4-bit; 4–5 GB quantised weights plus runtime overhead; 20 users; max context 2,048 tokens; max output 256 tokens; continuous batching.
- Keep model weights resident once. Use a serving engine such as vLLM or a suitable llama.cpp server build, with continuous batching and paged KV cache.
- Set a bounded maximum number of active sequences based on measured KV-cache use; queue excess requests rather than promising all 20 will decode simultaneously.
- Use request timeouts, per-user rate limits, prompt-length limits, health checks, and metrics for queue time, time-to-first-token, tokens/sec, GPU memory, and errors.
- Load test with the actual model, quantisation, context lengths, and traffic distribution. A 24 GB GPU may host the quantised 7B weights, but 20 simultaneous long contexts can exceed KV-cache capacity. Concurrency is a measured capacity limit, not simply a user count.

## Final notes

All scores and speed/memory values above must be copied from the actual Kaggle run outputs. Do not report estimates as measured results.
