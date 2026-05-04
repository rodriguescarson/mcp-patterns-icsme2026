#!/usr/bin/env python3
"""
MCP Server Pattern Classification (§5.1).

Loads 30 server descriptions from corpus.json and asks Claude Haiku 4.5 to
classify each into one of five architectural patterns. Reports overall and
per-pattern accuracy, latency, and a confusion matrix; writes raw outputs
to results/classification_N30.json and regenerates Figure 1.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 analyze.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_FILE = SCRIPT_DIR / "corpus.json"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PATTERNS = [
    "Resource Gateway",
    "Tool Orchestrator",
    "Stateful Session Server",
    "Proxy Aggregator",
    "Domain-Specific Adapter",
]

MODEL = "claude-haiku-4-5-20251001"


@dataclass
class ClassificationResult:
    name: str
    actual: str
    predicted: str
    correct: bool
    latency_ms: float
    reasoning: str = ""


def load_examples() -> list[dict]:
    with CORPUS_FILE.open() as f:
        corpus = json.load(f)
    examples = corpus["classification_examples"]["examples"]
    if not examples:
        raise RuntimeError(f"No classification examples found in {CORPUS_FILE}")
    return examples


def classify(client: anthropic.Anthropic, server: dict) -> ClassificationResult:
    payload = {k: server[k] for k in ("name", "tools", "resources", "description") if k in server}
    prompt = (
        "You are an expert in Model Context Protocol (MCP) server architecture.\n\n"
        "Classify this MCP server into exactly ONE of these five patterns:\n"
        "- Resource Gateway: exposes external data sources as MCP resources/tools.\n"
        "- Tool Orchestrator: wraps multiple external APIs/services as discrete tools.\n"
        "- Stateful Session Server: maintains session state across multiple tool calls.\n"
        "- Proxy Aggregator: aggregates multiple downstream MCP servers.\n"
        "- Domain-Specific Adapter: domain-specific wrapper with business logic/validation.\n\n"
        f"Server:\n{json.dumps(payload, indent=2)}\n\n"
        'Respond with JSON only: {"pattern": "<exact pattern name>", "reasoning": "<one sentence>"}'
    )

    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.lstrip().startswith("json"):
                raw = raw.lstrip()[4:]
        parsed = json.loads(raw)
        predicted = parsed.get("pattern", "Unknown").strip()
        reasoning = parsed.get("reasoning", "").strip()
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        predicted = "Unknown"
        reasoning = f"error: {e}"

    return ClassificationResult(
        name=server["name"],
        actual=server["pattern"],
        predicted=predicted,
        correct=(predicted == server["pattern"]),
        latency_ms=latency_ms,
        reasoning=reasoning,
    )


def confusion(results: list[ClassificationResult]) -> np.ndarray:
    n = len(PATTERNS)
    idx = {p: i for i, p in enumerate(PATTERNS)}
    m = np.zeros((n, n), dtype=int)
    for r in results:
        ai = idx.get(r.actual)
        pi = idx.get(r.predicted)
        if ai is not None and pi is not None:
            m[ai, pi] += 1
    return m


def plot_figure1(results: list[ClassificationResult], output: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    per_pattern = {p: {"correct": 0, "total": 0} for p in PATTERNS}
    for r in results:
        per_pattern[r.actual]["total"] += 1
        if r.correct:
            per_pattern[r.actual]["correct"] += 1

    accs = [per_pattern[p]["correct"] / per_pattern[p]["total"] * 100 if per_pattern[p]["total"] else 0
            for p in PATTERNS]
    colors = plt.cm.Set2(np.linspace(0, 1, len(PATTERNS)))
    bars = ax1.barh(PATTERNS, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Per-pattern Recall (%)", fontsize=11)
    ax1.set_title(f"Pattern Classification (N={len(results)})\nClaude Haiku 4.5", fontsize=12)
    ax1.set_xlim(0, 110)
    for bar, acc, p in zip(bars, accs, PATTERNS):
        n = per_pattern[p]["total"]
        ax1.text(acc + 1, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.0f}% (n={n})", va="center", fontsize=10, fontweight="bold")
    mean = sum(accs) / len(accs)
    ax1.axvline(mean, color="gray", linestyle="--", alpha=0.7, label=f"Mean: {mean:.0f}%")
    ax1.legend(fontsize=9)

    cm = confusion(results)
    im = ax2.imshow(cm, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(len(PATTERNS)))
    ax2.set_yticks(range(len(PATTERNS)))
    short = [p.split()[0] for p in PATTERNS]
    ax2.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax2.set_yticklabels(short, fontsize=9)
    ax2.set_xlabel("Predicted", fontsize=11)
    ax2.set_ylabel("Actual", fontsize=11)
    ax2.set_title("Confusion Matrix", fontsize=12)
    for i in range(len(PATTERNS)):
        for j in range(len(PATTERNS)):
            v = cm[i, j]
            if v:
                ax2.text(j, i, str(v), ha="center", va="center", fontsize=11, fontweight="bold",
                         color="white" if v >= cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax2, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on examples processed (for smoke tests).")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    random.seed(args.seed)
    np.random.seed(args.seed)

    examples = load_examples()
    if args.limit:
        examples = examples[: args.limit]
    random.shuffle(examples)

    print(f"Classifying {len(examples)} MCP server descriptions with {MODEL} (seed={args.seed})")
    results: list[ClassificationResult] = []
    for i, ex in enumerate(examples, 1):
        r = classify(client, ex)
        marker = "OK " if r.correct else "MISS"
        print(f"  [{i:2d}/{len(examples)}] {marker}  {r.name:32s}  actual={r.actual:24s}  predicted={r.predicted:24s}  ({r.latency_ms:5.0f} ms)")
        results.append(r)

    n = len(results)
    correct = sum(r.correct for r in results)
    acc = correct / n if n else 0.0
    lo, hi = wilson_ci(correct, n)
    p50 = statistics.median(r.latency_ms for r in results) if results else 0.0
    print()
    print(f"Overall accuracy: {correct}/{n} = {acc * 100:.1f}% (95% CI [{lo * 100:.1f}, {hi * 100:.1f}])")
    print(f"Median latency:   {p50:.0f} ms")
    print()
    print("Per-pattern:")
    for p in PATTERNS:
        sub = [r for r in results if r.actual == p]
        if not sub:
            continue
        c = sum(r.correct for r in sub)
        sub_lo, sub_hi = wilson_ci(c, len(sub))
        print(f"  {p:26s}  {c}/{len(sub)} = {c / len(sub) * 100:5.1f}%  [95% CI {sub_lo*100:.1f}, {sub_hi*100:.1f}]")

    out_json = RESULTS_DIR / "classification_N30.json"
    with out_json.open("w") as f:
        json.dump({
            "model": MODEL,
            "seed": args.seed,
            "n": n,
            "correct": correct,
            "accuracy": acc,
            "ci95_low": lo,
            "ci95_high": hi,
            "median_latency_ms": p50,
            "patterns": PATTERNS,
            "confusion_matrix": confusion(results).tolist(),
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"Wrote {out_json}")

    fig1 = RESULTS_DIR / "fig1_pattern_classification.pdf"
    plot_figure1(results, fig1)
    print(f"Wrote {fig1}")


if __name__ == "__main__":
    main()
