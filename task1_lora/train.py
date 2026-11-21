import os, json, random, re, argparse, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
import inspect
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_ROOT = Path(os.environ.get("ASSIGNMENT_RESULTS_DIR", str(Path(__file__).resolve().parents[1] / "assignment_results")))
OUT = RESULTS_ROOT / "task1"
OUT.mkdir(parents=True, exist_ok=True)
FIELDS = ["action", "date", "time", "location", "attendees"]

def make_example(rng):
    actions = ["schedule a meeting", "book a project review", "arrange a demo", "set up an interview",
               "organise a planning session", "schedule a team sync"]
    dates = ["2026-12-02", "2026-12-05", "2026-12-11", "2027-01-08", "2027-01-15",
             "2026-12-18", "2027-02-03", "2027-02-12"]
    times = ["09:00", "10:30", "11:00", "13:30", "14:00", "15:30", "16:00"]
    places = ["Google Meet", "Zoom", "Conference Room A", "Bengaluru office", "online", "Kolkata office"]
    people = ["Asha", "Ravi", "Mina", "John", "Priya", "Sam", "Neha", "Arjun", "Fatima", "Liam"]
    action = rng.choice(actions)
    date = rng.choice(dates)
    time_ = rng.choice(times)
    location = rng.choice(places)
    attendees = sorted(rng.sample(people, rng.randint(1, 4)))
    # Vary surface form while preserving a deterministic target.
    templates = [
        "Please {action} for {date} at {time} at {location}. Invite {attendees}.",
        "I need to {action} on {date}, starting {time}. It will be at {location}; participants: {attendees}.",
        "Arrange this: {action}; date={date}; time={time}; place={location}; people={attendees}.",
        "Can you {action} for {attendees} on {date} at {time}? Location is {location}.",
        "Set the following calendar details: {action}, {date}, {time}, {location}, attendees {attendees}."
    ]
    names = ", ".join(attendees)
    request = rng.choice(templates).format(action=action, date=date, time=time_, location=location, attendees=names)
    target = {"action": action, "date": date, "time": time_, "location": location, "attendees": attendees}
    return request, target

def build_splits(seed=42):
    rng = random.Random(seed)
    rows = [make_example(rng) for _ in range(1500)]
    # Independent deterministic split; held-out test has 100 examples.
    return rows[:1300], rows[1300:1400], rows[1400:]

def render_prompt(request):
    return ("You convert calendar requests into JSON. Return exactly one valid JSON object "
            "with keys action, date, time, location, attendees. Do not add markdown.\n"
            f"Request: {request}\nJSON:\n")

class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=256):
        self.items = []
        for request, target in rows:
            prompt = render_prompt(request)
            answer = json.dumps(target, ensure_ascii=False, separators=(",", ":")) + tokenizer.eos_token
            p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a = tokenizer(answer, add_special_tokens=False)["input_ids"]
            ids = (p + a)[:max_length]
            labels = ([-100] * len(p) + a)[:max_length]
            mask = [1] * len(ids)
            pad = max_length - len(ids)
            self.items.append({
                "input_ids": ids + [tokenizer.pad_token_id] * pad,
                "attention_mask": mask + [0] * pad,
                "labels": labels + [-100] * pad
            })
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return {k: torch.tensor(v) for k, v in self.items[i].items()}

def parse_json(text):
    try:
        match = re.search(r"\{.*\}", text, flags=re.S)
        return json.loads(match.group(0)) if match else None
    except Exception:
        return None

def evaluate(model, tokenizer, rows, label, max_new_tokens=120):
    model.eval()
    exact, valid = 0, 0
    predictions = []
    device = next(model.parameters()).device
    for request, gold in rows:
        prompt = render_prompt(request)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_json(decoded)
        valid += int(pred is not None)
        exact += int(pred == gold)
        predictions.append({"request": request, "gold": gold, "raw_output": decoded,
                            "parsed": pred, "exact_match": pred == gold})
    result = {"model": label, "examples": len(rows), "valid_json_rate": valid/len(rows),
              "exact_match_accuracy": exact/len(rows), "predictions": predictions}
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU: Settings > Accelerator > GPU.")
    train_rows, val_rows, test_rows = build_splits()
    with open(OUT/"dataset_splits.json", "w") as f:
        json.dump({"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)}, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    base.config.use_cache = False

    # Evaluate base model on the untouched 100-example test set before training.
    base_result = evaluate(base, tokenizer, test_rows, "base")
    with open(OUT/"base_test_results.json", "w") as f: json.dump(base_result, f, indent=2)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(base, lora_cfg)
    # Required by some gradient-checkpointing configurations when only adapters train.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    train_ds = SFTDataset(train_rows, tokenizer, args.max_length)
    val_ds = SFTDataset(val_rows, tokenizer, args.max_length)
    # Transformers has changed a few argument names across releases. Build a
    # version-aware kwargs dictionary instead of failing on an unsupported name.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = {
        "output_dir": str(OUT/"checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "logging_steps": 25,
        "evaluation_strategy": "epoch",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "fp16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch",
        "report_to": "none",
        "seed": 42,
        "data_seed": 42,
    }
    # Use whichever evaluation argument this Transformers version exposes.
    if "evaluation_strategy" in ta_params:
        ta_kwargs.pop("eval_strategy", None)
    elif "eval_strategy" in ta_params:
        ta_kwargs.pop("evaluation_strategy", None)
    # Older/newer builds may not expose warmup_ratio; use warmup_steps instead.
    if "warmup_ratio" not in ta_params and "warmup_steps" in ta_params:
        estimated_steps = max(1, (len(train_ds) + 4*4 - 1) // (4*4)) * args.epochs
        ta_kwargs.pop("warmup_ratio", None)
        ta_kwargs["warmup_steps"] = max(1, int(estimated_steps * 0.05))
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_params}
    train_args = TrainingArguments(**ta_kwargs)

    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model, "args": train_args,
        "train_dataset": train_ds, "eval_dataset": val_ds,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    adapter_dir = OUT/"lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    model.config.use_cache = True
    tuned_result = evaluate(model, tokenizer, test_rows, "lora_finetuned")
    with open(OUT/"finetuned_test_results.json", "w") as f: json.dump(tuned_result, f, indent=2)

    summary = {
        "base_exact_match": base_result["exact_match_accuracy"],
        "finetuned_exact_match": tuned_result["exact_match_accuracy"],
        "base_valid_json_rate": base_result["valid_json_rate"],
        "finetuned_valid_json_rate": tuned_result["valid_json_rate"],
        "lora": {"rank": 8, "alpha": 16, "target_modules": lora_cfg.target_modules,
                 "learning_rate": 2e-4, "epochs": args.epochs, "dropout": 0.05,
                 "train_examples": len(train_rows), "validation_examples": len(val_rows),
                 "held_out_test_examples": len(test_rows)}
    }
    with open(OUT/"summary.json", "w") as f: json.dump(summary, f, indent=2, default=lambda x: sorted(x) if isinstance(x, set) else str(x))
    # Three failure cases are selected from actual model errors, not fabricated.
    failures = [x for x in tuned_result["predictions"] if not x["exact_match"]][:3]
    with open(OUT/"failure_cases.json", "w") as f: json.dump(failures, f, indent=2)
    print(json.dumps(summary, indent=2, default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
    print(f"Task 1 outputs: {OUT}")

if __name__ == "__main__": main()
