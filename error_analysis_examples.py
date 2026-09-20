"""Print saved error examples from reports/evaluation.json.

Does not retrain. Run py -m src.evaluate first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / "reports" / "evaluation.json"
if not path.exists():
    sys.exit("Missing reports/evaluation.json. Run: py -m src.evaluate")

payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "completed":
    sys.exit("Evaluation did not complete; no error examples to print.")

print("Selected model:", payload["selected_model"]["label"])
print("\nNews videos the unweighted model approved for Kids & Family at 0.55")
for row in payload.get("news_leaks_unweighted_vs_selected", []):
    after = "APPROVED" if row["approved_after"] else "withheld"
    print(f"- {row['title'][:78]}")
    print(f"    p_allow {row['p_allow_before']:.2f} -> {row['p_allow_after']:.2f} ({after})")

print("\nRemaining excluded-category approvals (selected model, Kids & Family)")
for row in payload.get("error_examples", []):
    if row["kind"] != "incorrect_approval":
        continue
    print(f"- [{row['mapped_label']}] {row['title'][:70]}")
    print(f"    pred={row['predicted']} p_allow={row['p_allow']:.2f}")
