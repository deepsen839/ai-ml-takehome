"""Create a human-review worksheet from the saved paired Task 3 generations."""
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"assignment_results/task3/prompt_outputs.csv"
OUT=ROOT/"assignment_results/task3/quality_review_template.csv"
rows=list(csv.DictReader(SRC.open(encoding="utf-8")))
by_prompt={}
for r in rows: by_prompt.setdefault(r["prompt"],{})[r["mode"]]=r
with OUT.open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=["prompt","fp16_output","nf4_output","fp16_quality","nf4_quality","preferred_mode","reason"])
 w.writeheader()
 for prompt,pair in by_prompt.items():
  w.writerow({"prompt":prompt,"fp16_output":pair.get("fp16",{}).get("output",""),"nf4_output":pair.get("4bit_nf4",{}).get("output",""),"fp16_quality":"","nf4_quality":"","preferred_mode":"","reason":""})
print(f"Wrote {OUT}")
