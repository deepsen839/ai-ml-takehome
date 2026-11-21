import subprocess, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT/"task1_lora"/"train.py",
    ROOT/"task2_rag"/"rag.py",
    ROOT/"task3_inference"/"benchmark.py"
]
for script in SCRIPTS:
    print("\n"+"="*80+"\nRUNNING", script, flush=True)
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(ROOT))
print("\nAll tasks completed. Results are in", Path(os.environ.get("ASSIGNMENT_RESULTS_DIR", str(ROOT / "assignment_results"))))
