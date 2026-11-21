import os, json, time, gc, csv
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_ROOT = Path(os.environ.get("ASSIGNMENT_RESULTS_DIR", str(Path(__file__).resolve().parents[1] / "assignment_results")))
OUT = RESULTS_ROOT / "task3"
OUT.mkdir(parents=True, exist_ok=True)
PROMPTS = [
 "Explain the difference between supervised and unsupervised learning in three sentences.",
 "Write a Python function that returns the factorial of a non-negative integer.",
 "Give five practical ways to reduce overfitting in a machine learning model.",
 "Explain what a REST API is to a junior Python developer.",
 "Return valid JSON with keys name and role for a person named Asha who is an ML engineer.",
 "Describe the purpose of a vector database in a RAG application.",
 "What is the difference between precision and recall?",
 "Write a SQL query to count orders per customer from an orders table.",
 "Explain quantisation in large language models without using equations.",
 "List a safe deployment checklist for a small FastAPI machine-learning service."
]

def load_model(mode):
    if mode == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    else:
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=qconf, device_map="auto")
    return model

def run_mode(mode, tokenizer):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = load_model(mode)
    model.eval()
    rows, total_new, total_time = [], 0, 0.0
    for prompt in PROMPTS:
        messages = [{"role":"user", "content":prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        new_tokens = output.shape[1]-inputs["input_ids"].shape[1]
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        total_new += new_tokens; total_time += elapsed
        rows.append({"mode":mode, "prompt":prompt, "output":answer,
                     "generated_tokens":new_tokens, "latency_seconds":elapsed,
                     "tokens_per_second":new_tokens/max(elapsed,1e-9),
                     "manual_quality_notes":""})
    peak_alloc = torch.cuda.max_memory_allocated()/1024**3
    peak_reserved = torch.cuda.max_memory_reserved()/1024**3
    summary = {"mode":mode, "total_generated_tokens":total_new,
               "total_generation_seconds":total_time,
               "aggregate_tokens_per_second":total_new/max(total_time,1e-9),
               "peak_gpu_allocated_gb":peak_alloc, "peak_gpu_reserved_gb":peak_reserved}
    del model
    gc.collect(); torch.cuda.empty_cache()
    return rows, summary

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before running Task 3.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    all_rows, summaries = [], []
    for mode in ["fp16", "4bit_nf4"]:
        rows, summary = run_mode(mode, tokenizer)
        all_rows.extend(rows); summaries.append(summary)
    with open(OUT/"prompt_outputs.json", "w") as f: json.dump(all_rows, f, indent=2)
    with open(OUT/"benchmark_summary.json", "w") as f: json.dump(summaries, f, indent=2)
    import pandas as pd
    pd.DataFrame(all_rows).to_csv(OUT/"prompt_outputs.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT/"benchmark_summary.csv", index=False)
    print(json.dumps(summaries, indent=2))
    print("Task 3 outputs:", OUT)

if __name__ == "__main__": main()
