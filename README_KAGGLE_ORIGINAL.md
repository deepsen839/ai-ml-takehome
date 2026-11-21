# Applied AI/ML Take-Home Assignment — Kaggle Implementation

This repository implements all three tasks. It is designed for Kaggle notebooks with a GPU accelerator (T4 16 GB recommended). No local setup is required.

## 1. Kaggle setup (important: use a writable working copy)

1. Create a Kaggle Notebook, enable **GPU** and **Internet**.
2. Attach this repository ZIP/dataset with **Add Input**.
3. Copy the repository from the read-only `/kaggle/input` mount into `/kaggle/working`, then run from that copy. If the attached dataset uses a different mount path, update `SOURCE` below.

```python
from pathlib import Path
import shutil, os

SOURCE = Path("/kaggle/input/datasets/deepsen839/kaggle-ai-ml-assignment1/kaggle_ai_ml_assignment")
TARGET = Path("/kaggle/working/kaggle_ai_ml_assignment")
assert SOURCE.exists(), f"Repository folder not found: {SOURCE}. Check !ls -R /kaggle/input"
shutil.copytree(SOURCE, TARGET, dirs_exist_ok=True)
os.chdir(TARGET)
print("Running from:", Path.cwd())
```

Do **not** run the old script from `/kaggle/input`; use `/kaggle/working/kaggle_ai_ml_assignment` so you run the patched files. Kaggle input is read-only; results are written to `/kaggle/working/assignment_results/`.

Check the environment before installing anything:

```python
import torch, transformers, peft
print("Python / Torch:", __import__('sys').version, torch.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("Transformers / PEFT:", transformers.__version__, peft.__version__)
```

Only install missing Task 2 packages; do not reinstall or downgrade PyTorch, CUDA, Triton, Transformers, or PEFT blindly:

```python
# Run only if imports fail:
# %pip install -q pymupdf sentence-transformers
# FAISS is optional in this patched version; NumPy exact search is used if FAISS cannot import.
# %pip install -q faiss-cpu
```

Run all tasks sequentially (Task 1 → Task 2 → Task 3):

```python
!python run_all.py
```

Or run one task at a time:

```python
!python task1_lora/train.py
!python task2_rag/rag.py
!python task3_inference/benchmark.py
```

The scripts use separate Python processes, so GPU memory is released between tasks. Do not start the three scripts concurrently on one T4.

## 2. Task 1 — LoRA fine-tuning

- Base model: `Qwen/Qwen2.5-0.5B-Instruct` (within the 1.5B limit).
- Task: convert natural-language calendar requests to a fixed JSON schema.
- Dataset: deterministic synthetic examples generated in code; 1,300 train / 100 validation / 100 held-out test.
- Metric: exact JSON-object match and valid-JSON rate, base vs fine-tuned.
- LoRA: rank 8, alpha 16, dropout 0.05, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, learning rate 2e-4, default 3 epochs.
- Rationale: small rank keeps adapter memory low; alpha 2r is a common stable scale; attention and MLP projections give the adapter capacity to learn both format and field mapping; 2e-4 is a practical starting learning rate for adapter tuning on a small model.

Outputs: `base_test_results.json`, `finetuned_test_results.json`, `summary.json`, `failure_cases.json`, adapter folder, checkpoints.

## 3. Task 2 — RAG from scratch

- Source: NIST AI Risk Management Framework 1.0 PDF (official NIST publication, 48 pages).
- No LangChain, LlamaIndex, Haystack, or equivalent orchestration framework.
- PDF extraction: PyMuPDF. Embeddings: `all-MiniLM-L6-v2`. Vector search: FAISS inner product over normalized embeddings, with an exact NumPy inner-product fallback if FAISS is not importable in the Kaggle Python image.
- Compare 150-word and 300-word chunks, each with 40-word overlap.
- 15 hand-written questions. For each, the script finds and stores the matching source page and its source passage as the gold evidence.
- Metric: recall@5, defined here as whether at least one of the top five retrieved chunks comes from the gold source page.

Outputs: `test_questions_and_gold_passages.json`, `retrieval_comparison.csv`, `retrieval_150.json`, `retrieval_300.json`, `retrieval_failures.json`, and `generated_answers.json`. Each retrieval JSON records which search backend was used.

**Metric note:** page-level gold evidence makes evaluation auditable, but it is less strict than exact chunk-level matching. State this limitation in your report. The script stores full gold passages so a reviewer can inspect them.

The script passes the top retrieved passage to Qwen2.5-0.5B-Instruct for a grounded final answer and saves the answer together with its evidence in `generated_answers.json`. The model is instructed to say when the evidence is insufficient.

## 4. Task 3 — inference optimisation

- Model: same Qwen2.5-0.5B-Instruct.
- Compare FP16 against 4-bit NF4 (double quantisation enabled).
- Same 10 prompts, greedy decoding, max 128 new tokens.
- Measure aggregate generated tokens/second, peak CUDA allocated memory, and peak reserved memory.
- Outputs include model outputs for manual quality review. Add short notes to `manual_quality_notes` in `prompt_outputs.csv` after comparing each pair.

## 5. Required written explanations

### BM25 vs embedding retrieval
BM25 is lexical retrieval: it scores documents using term frequency, inverse document frequency, and document length normalisation. It is good at exact names, rare terms, IDs, and wording matches. Embeddings retrieve semantically similar text even when the question uses different words. A hybrid system combines both signals, which often improves recall.

### Re-ranker
A re-ranker takes the query and a shortlist of retrieved passages and scores each query-passage pair jointly (often with a cross-encoder). It is slower than the initial vector search but can order the top candidates more accurately. Retrieve broadly first, then re-rank the shortlist.

### KV cache
The key-value cache stores attention keys and values from previous tokens during autoregressive generation. Without it, the model would recompute prior-token attention projections at every generation step. It speeds decoding but consumes GPU memory, growing with sequence length, layer count, batch size, and hidden dimensions.

### Why quantisation saves memory
Quantisation stores model weights with fewer bits (e.g. 4-bit instead of 16-bit). A 16-bit weight uses 2 bytes; a 4-bit weight uses 0.5 bytes, so raw weight storage is about four times smaller, plus some scale/metadata overhead. Compute may still use a higher-precision type.

### 7B memory estimate (rough)
Assume 7 billion parameters and decimal GB for readability:
- FP16/BF16 weights: 7B × 2 bytes = **14 GB**.
- Full fine-tuning with Adam: weights 14 GB + gradients about 14 GB + FP32 master weights about 28 GB + Adam moments about 56 GB = **about 112 GB**, before activations and temporary buffers. Mixed-precision implementations vary; activation memory and optimizer implementation change the total.
- LoRA: frozen FP16 base weights about 14 GB, plus small adapter weights, adapter gradients and optimizer states (typically well below 1–2 GB for a modest adapter), plus activations. Practical total often **16–24+ GB**, depending on sequence length, batch size, checkpointing and rank. LoRA does not make the frozen base weights disappear.
- QLoRA: 4-bit base weights raw storage 7B × 0.5 bytes = 3.5 GB, plus quantisation scales/metadata (roughly 0.3–1 GB), LoRA adapter/optimizer states and activations. Practical total often **6–12+ GB**, depending on sequence length and batch size.

These are planning estimates, not guaranteed peak VRAM figures.

## 6. Report and submission

Run all tasks first. Use `reports/report_template.md` to write up measured results. Do not submit fabricated metrics. The template includes spaces for the actual measured scores and three observed fine-tuning failures. Export the completed report to PDF and keep it to four pages maximum.

Suggested GitHub layout:
```
.
├── README.md
├── requirements-kaggle.txt
├── run_all.py
├── task1_lora/
├── task2_rag/
├── task3_inference/
└── reports/
```

## 7. Reproducibility and caveats

- Seeds are fixed where practical.
- Task 1 synthetic examples are generated locally at runtime, so no private dataset is needed.
- Task 2 requires Kaggle Internet to download the official PDF and embedding model.
- Task 3 requires a GPU and a working bitsandbytes CUDA build.
- Task 1 detects the installed Transformers `TrainingArguments` and `Trainer` signatures, including the `evaluation_strategy`/`eval_strategy` and `tokenizer`/`processing_class` naming differences. It does not require blindly downgrading the Kaggle PyTorch/CUDA stack.
