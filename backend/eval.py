from typing import List, Dict, Any
import json
import os
from .utils import ensure_dir, write_json

def load_eval_set(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def simple_accuracy(pred: str, expected: str) -> float:
    """
    Simple scoring: checks if expected keyword/phrase appears in prediction.
    This is fast + transparent for recruiter demos.
    """
    p = (pred or "").lower()
    e = (expected or "").lower().strip()
    if not e:
        return 0.0
    return 1.0 if e in p else 0.0

def run_eval(eval_items: List[Dict[str, Any]], ask_fn, out_dir: str) -> Dict[str, Any]:
    """
    eval_items: [{question, expected}]
    ask_fn(question)-> {answer, citations, ...}
    """
    ensure_dir(out_dir)
    results = []
    score_sum = 0.0

    for item in eval_items:
        q = item["question"]
        exp = item.get("expected", "")
        out = ask_fn(q)
        acc = simple_accuracy(out["answer"], exp)
        score_sum += acc
        results.append({
            "question": q,
            "expected": exp,
            "answer": out["answer"],
            "citations": out.get("citations", []),
            "score": acc
        })

    report = {
        "n": len(eval_items),
        "accuracy": (score_sum / max(1, len(eval_items))),
        "results": results
    }
    write_json(os.path.join(out_dir, "eval_report.json"), report)
    return report
