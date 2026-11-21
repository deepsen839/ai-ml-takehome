# Applied AI/ML Take-Home Assignment — Complete Project

This repository contains the implementation for all three tasks, the uploaded execution notebook, the NIST source PDF, the LoRA adapter/checkpoint, raw evaluation outputs, benchmark outputs, and a report based on the supplied run artifacts. The original Kaggle-oriented README is preserved as `README_KAGGLE_ORIGINAL.md`.

## Important status and honesty notes

- Task 1 recorded 100 held-out examples: base exact-match 0.00, fine-tuned exact-match 1.00; valid JSON 1.00 for both. The fine-tuned failure-case file is empty. **Do not invent three failures.** Run the included stress-test protocol and report actual failures if found; if none are found, explicitly disclose that the 100-example held-out set had no failures and ask the assessor whether a stress set is acceptable.
- Task 2's recorded recall@5 is 2/15 (13.33%) for 150-word chunks and 3/15 (20.00%) for 300-word chunks. The existing gold-page generation selects the first page containing a broad anchor and can select front matter/table-of-contents pages. Two examples in the failure file expose this issue. Treat these scores as **provisional, not a sound final evaluation** until gold passages are manually audited and retrieval is rerun.
- Task 3 recorded FP16 25.33 tokens/s and NF4 21.56 tokens/s. Peak CUDA allocated memory was 0.739 GB and 0.445 GB respectively. These are the recorded benchmark peak figures, not a claim about total physical GPU memory use. The CSV's `manual_quality_notes` are blank; fill them after reviewing the 10 matched prompt pairs.
- The supplied artifacts do not identify the local machine/GPU model in a way that proves Task 3 was run on your personal computer. Do not claim personal-machine execution unless you can document that environment.
- The code is portable now: result paths default to this repository's `assignment_results/`, and can be overridden with `ASSIGNMENT_RESULTS_DIR`. The supplied result bundle is preserved separately from reruns.

## Repository layout

```text
ai_ml_takehome_complete/
├── README.md
├── README_KAGGLE_ORIGINAL.md
├── requirements-local.txt
├── requirements-kaggle.txt
├── run_all.py
├── Kaggle_AI_ML_Assignment_ReImport_Fixed.ipynb
├── task1_lora/                 # training and evaluation implementation
├── task2_rag/                  # PDF ingestion, chunking, embeddings, retrieval, QA
├── task3_inference/            # FP16 vs 4-bit benchmark
├── assignment_results/         # all supplied measured results and model artifacts
│   ├── task1/                  # base/fine-tuned JSON, adapter, checkpoint
│   ├── task2/                  # PDF, 15 questions, retrievals, failures, answers
│   └── task3/                  # paired outputs and benchmark summaries
├── reports/                    # final report PDF and editable Markdown
└── source_archives_*.zip       # original uploaded archives for provenance
```

## Environment and installation

Python 3.10 or 3.11 is recommended for the broadest compatibility of PyTorch, Transformers, PEFT and bitsandbytes. A CUDA NVIDIA GPU is recommended for Tasks 1 and 3. Task 2 can run on CPU, but embedding and LLM generation will be slower.

### Linux / Ubuntu local setup

```bash
cd ai_ml_takehome_complete
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

Install a PyTorch build compatible with the CUDA driver installed on your machine. Use the official PyTorch selector at https://pytorch.org/get-started/locally/ rather than assuming a CUDA version. Then install the remaining packages:

```bash
pip install -r requirements-local.txt
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
```

If `bitsandbytes` cannot load on your OS/CUDA combination, Task 3's 4-bit mode will fail; use a supported Linux/CUDA environment or the included Kaggle notebook. Do not blindly downgrade Torch/CUDA packages in an existing environment.

### Run all tasks

```bash
python run_all.py
```

Or individually:

```bash
python task1_lora/train.py
python task2_rag/rag.py
python task3_inference/benchmark.py
```

The scripts write to `./assignment_results/` by default. To use another writable folder:

```bash
export ASSIGNMENT_RESULTS_DIR="$PWD/run_outputs"
python run_all.py
```

**Compute warning:** Running all three tasks from scratch downloads model weights and trains the adapter. Task 1 and Task 3 require an appropriate GPU for practical execution. Running Task 1 on CPU may be extremely slow. Use the existing saved results to inspect the completed run; do not rerun training merely to regenerate the report.

### Kaggle

The included notebook is the supplied Kaggle execution notebook. Enable Internet and GPU, attach the repository, copy it to `/kaggle/working`, then run the notebook top-to-bottom. The notebook is included as a reproducibility route, not evidence that every cell was re-executed after packaging.

## Task 1 — LoRA fine-tuning

- Base model: `Qwen/Qwen2.5-0.5B-Instruct` (0.5B parameters).
- Task: convert natural-language calendar requests to a fixed JSON schema: `action`, `date`, `time`, `location`, `attendees`.
- Dataset split: 1,300 train / 100 validation / 100 held-out test. Data is synthetic and generated deterministically by the script.
- Metric: exact JSON-object match and valid JSON rate.
- LoRA: rank 8, alpha 16, dropout 0.05, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, learning rate 2e-4, 3 epochs.

### Why these settings

Rank 8 is a small adapter that limits trainable parameters and VRAM. Alpha 16 gives an alpha/rank scaling factor of 2. Dropout 0.05 provides modest regularisation. Attention projections adapt token-to-token/context behaviour; MLP projections add capacity for mapping request wording to schema values. A 2e-4 adapter learning rate is a practical starting point for a small instruction model, and three epochs provide several passes over the synthetic training examples without making the run unnecessarily long.

### Recorded results

| Metric | Base | LoRA fine-tuned |
|---|---:|---:|
| Held-out examples | 100 | 100 |
| Exact-match accuracy | 0.00% | 100.00% |
| Valid JSON rate | 100.00% | 100.00% |

The base model emitted syntactically valid JSON but did not match the exact target object. The fine-tuned model matched the target on all 100 held-out synthetic examples. This is evidence for this narrow synthetic task only, not general calendar understanding.

### Required failure examples

`assignment_results/task1/failure_cases.json` is empty because the recorded fine-tuned held-out run had no exact-match failures. The project deliberately does not fabricate examples. To satisfy the spirit of the requirement, evaluate a separately labelled challenge set covering missing dates, ambiguous relative dates, multiple time zones, duplicate attendees, unusual names, malformed times, and requests missing required fields. Save actual failures with input, expected JSON, model output and diagnosis. If there are still no failures, report that transparently and state that the requested three failures were not observed.

### 7B memory estimate (decimal GB, approximate)

For 7 billion parameters:

- FP16/BF16 weights: 7B × 2 bytes = 14 GB.
- Full fine-tuning with Adam (rough estimate): weights 14 GB + gradients 14 GB + FP32 master weights 28 GB + two FP32 Adam moments 56 GB = **112 GB**, before activations, temporary buffers and allocator overhead. Implementations differ; this is a planning estimate.
- LoRA: frozen FP16 base weights ≈14 GB + small adapter weights, adapter gradients and optimizer states + activations. A rough practical planning range is **16–24+ GB**, highly dependent on sequence length, batch size, rank and checkpointing.
- QLoRA: 4-bit base weights raw storage = 7B × 0.5 bytes = 3.5 GB, plus scales/metadata, adapters, optimizer states and activations. A rough planning range is **6–12+ GB**.

LoRA does not remove the frozen base-model memory. QLoRA reduces base-weight storage by quantising the frozen model.

## Task 2 — RAG from scratch

- Public document: NIST AI Risk Management Framework 1.0 PDF (provided in `assignment_results/task2/`).
- PDF extraction: PyMuPDF. Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Search: FAISS `IndexFlatIP` over normalized embeddings when importable; exact NumPy inner-product fallback otherwise. No LangChain/LlamaIndex or equivalent orchestration framework.
- Chunk sizes: 150 and 300 words, 40-word overlap.
- Evaluation: 15 questions. Current recall@5 is page-level: a hit means any of the top five chunks comes from the gold page.

### Recorded retrieval results

| Chunk size | Hits | Recall@5 |
|---|---:|---:|
| 150 words | 2/15 | 13.33% |
| 300 words | 3/15 | 20.00% |

300-word chunks performed better in this run, likely because they retain more surrounding context. However, both scores are low and the current gold annotations need manual review. The code chooses the first page containing a broad anchor; the two recorded failures show that this can select a table of contents or list-of-figures page rather than the explanatory section. Therefore, do not present these recall values as a validated final benchmark until gold pages/passages are corrected and both chunk-size experiments are rerun.

### Two recorded failure cases and fixes

1. “What are the four functions in the NIST AI Risk Management Framework?” The current gold passage points to front matter/list-of-figures (page 5 in the extracted numbering), while a relevant passage appears in the Core section. Fix: manually anchor the gold passage to the section describing the four Core functions, and use exact page/chunk evidence.
2. “What is the purpose of the Govern function?” The current gold passage is the table of contents. Fix: anchor to the body of section 5.1 (Govern), not the contents entry; consider query expansion with “governance”, “policies”, “accountability” and “oversight”.

### BM25 and re-ranking

BM25 is lexical retrieval. It scores term overlap using term frequency, inverse document frequency and document-length normalisation. It is especially useful for exact names, acronyms, rare terms and identifiers. Dense embeddings capture semantic similarity when the query and passage use different wording. A hybrid retriever combines lexical and semantic signals.

A re-ranker takes the query and a shortlist of retrieved passages and scores each query–passage pair jointly, often with a cross-encoder. It can improve the ordering of the top candidates but costs more than the initial retrieval step. A common pipeline retrieves broadly with BM25 and/or vectors, then re-ranks the shortlist.

## Task 3 — inference optimisation

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Modes: FP16 and 4-bit NF4 with double quantisation.
- Same 10 prompts, greedy decoding, maximum 128 new tokens.

### Recorded benchmark

| Metric | FP16 | 4-bit NF4 |
|---|---:|---:|
| Generated tokens | 1,175 | 1,190 |
| Aggregate tokens/sec | 25.33 | 21.56 |
| Peak CUDA allocated | 0.739 GB | 0.445 GB |
| Peak CUDA reserved | 0.760 GB | 0.463 GB |

FP16 was about 17.5% faster in this run. NF4 reduced peak allocated memory by about 39.7%, but was slower. These CUDA peak values are those recorded by the benchmark and should not be mistaken for total GPU memory occupied by the full process.

### Output quality review

The supplied `prompt_outputs.csv` and JSON contain 10 paired prompts (20 rows), but all `manual_quality_notes` fields are blank. Review each FP16/4-bit pair for factual correctness, completeness, instruction following, formatting and truncation; fill the notes column before submission. The raw outputs are preserved in `assignment_results/task3/`. Do not claim one mode has better answer quality until this review is completed.

### KV cache and quantisation

The KV cache stores attention keys and values for previously processed tokens during autoregressive generation. Reusing them avoids recomputing those projections at every new token and speeds decoding. The cache consumes memory and grows with context length, number of layers, batch size and model dimensions.

Quantisation stores weights using fewer bits. FP16 uses 16 bits per weight while 4-bit quantisation uses roughly one quarter of that raw weight storage, with extra scales/metadata. This reduces memory use, though compute may use a higher-precision type and quality/speed trade-offs depend on hardware and implementation.

### Serving 20 concurrent users on one 24 GB GPU — back-of-envelope plan

Assumptions: a 7B model in 4-bit, short-to-medium prompts (about 2,048 input tokens), up to 256 generated tokens, and variable user arrival times. Load the quantised model once; use a serving engine with continuous batching and paged KV cache; cap context/output lengths; set a bounded queue and per-user timeouts; use admission control when KV-cache capacity is reached; stream tokens; monitor GPU memory, queue depth, latency and tokens/sec; and load-test with 20 simultaneous sessions. Keep a safety margin for runtime/activations. Twenty connected users do not necessarily mean twenty long-context sequences can decode at full speed simultaneously; reduce batch concurrency or context limits if memory/latency exceed the budget.

