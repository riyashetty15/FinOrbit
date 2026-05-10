"""
CI Evaluation Gate — checks saved evaluation_report.json against quality thresholds.

Usage:
    python scripts/ci_eval_check.py --report tests/evaluation_report.json

Exits with code 1 (blocks deploy) if any threshold is breached.
Exits with code 0 (pass) if thresholds are met or no report exists yet.
"""

import argparse
import json
import sys
import os


# ── Quality thresholds ────────────────────────────────────────────────────────
THRESHOLDS = {
    "compliance_pass_rate":        0.90,   # 90% compliance required
    "routing_accuracy":            0.80,   # 80% routing accuracy
    "rag_decision_accuracy":       0.75,   # 75% RAG decision accuracy
    "grounding_accuracy":          0.80,   # 80% grounding accuracy
    "citation_precision":          0.70,   # 70% citation precision
    "evidence_coverage_accuracy":  0.70,   # 70% evidence coverage
    "rag_hit_rate":                0.70,   # 70% retrieval hit rate
}

# P99 latency hard cap (milliseconds)
MAX_P99_LATENCY_MS = 8000


def check(report_path: str) -> bool:
    """
    Read the evaluation report and check all thresholds.
    Returns True if all thresholds pass, False if any fail.
    """
    if not os.path.exists(report_path):
        print(f"[eval-gate] No report found at {report_path} — skipping gate (first run).")
        return True

    with open(report_path, "r") as f:
        report = json.load(f)

    summary = report.get("summary", {})
    failures = []

    # Check metric thresholds
    for metric, threshold in THRESHOLDS.items():
        value = summary.get(metric)
        if value is None:
            print(f"[eval-gate] WARNING: metric '{metric}' not found in report — skipping.")
            continue
        status = "PASS" if value >= threshold else "FAIL"
        print(f"[eval-gate] {status}  {metric}: {value:.2%}  (threshold: {threshold:.0%})")
        if value < threshold:
            failures.append(f"{metric} = {value:.2%} < {threshold:.0%}")

    # Check latency
    p99 = summary.get("p99_latency_ms", 0)
    latency_ok = p99 <= MAX_P99_LATENCY_MS
    print(
        f"[eval-gate] {'PASS' if latency_ok else 'FAIL'}  "
        f"p99_latency_ms: {p99:.0f}ms  (threshold: {MAX_P99_LATENCY_MS}ms)"
    )
    if not latency_ok:
        failures.append(f"p99_latency_ms = {p99:.0f}ms > {MAX_P99_LATENCY_MS}ms")

    # Print recommendations from report
    recommendations = report.get("recommendations", [])
    if recommendations:
        print("\n[eval-gate] Recommendations from last evaluation:")
        for r in recommendations:
            print(f"  • {r}")

    if failures:
        print(f"\n[eval-gate] BLOCKED — {len(failures)} threshold(s) failed:")
        for f in failures:
            print(f"  ✗ {f}")
        return False

    print(f"\n[eval-gate] All thresholds passed — deployment approved.")
    return True


def main():
    parser = argparse.ArgumentParser(description="FinOrbit CI evaluation gate")
    parser.add_argument(
        "--report",
        default="tests/evaluation_report.json",
        help="Path to evaluation_report.json",
    )
    args = parser.parse_args()

    passed = check(args.report)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
