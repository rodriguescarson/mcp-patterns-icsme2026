# MCP Server Architecture Patterns — Replication Package

Code, corpus, and prompts for:

> **MCP Server Architecture Patterns for LLM-Integrated Applications.**
> Carson Rodrigues, Oysturn Vas. arXiv:2606.30317.

**Correction (October 2026).** Earlier versions of this repository included `tool_count_telemetry.csv`,
described as ANSYR production telemetry from Q1 2025. Those numbers could not be traced to production
records, and both models they name were released after that quarter. The file has been removed and the
analysis is withdrawn in version 2 of the paper. See `CORRECTION.md`.

## Contents

| File | Purpose |
|------|---------|
| `corpus.json` | Enumerated 15-server taxonomy corpus + 30 classification examples used in §5.1. |
| `analyze.py` | Pattern classification experiment (§5.1). N=30, deterministic seed, Claude Haiku 4.5 as classifier. |
| `transport_bench.py` | Transport latency benchmark (§5.2). Real measurements for `stdio` and `streamable-http`; modeled estimates for cross-host paths with explicit calibration source. |
| `kappa_eval.py` | Inter-rater reliability study: two independent LLM raters classify 54 held-out servers; writes `results_kappa.json`. Needs `ANTHROPIC_API_KEY`. |
| `results_kappa.json` | Per-server predictions of both raters, Cohen's kappa with its bootstrap CI, and each rater's agreement with the authors' labels (n = 54, kappa 0.76). |
| `requirements.txt` | Python dependencies. |
| `results/` | Output directory created on first run. JSON results + regenerated figures. |

## Reproduction

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. API key (only needed for analyze.py)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run experiments
python3 analyze.py --seed 42        # ~3 min, ~30 Claude API calls (~$0.05)
python3 transport_bench.py          # ~1 min, no external services

# 4. Outputs land in results/
ls results/
#   classification_N30.json
#   transport_measured.json
#   fig1_pattern_classification.pdf
#   fig2_transport_latency.pdf
```

## Determinism

* `analyze.py --seed 42` controls Python `random` and shuffling order. Claude API outputs are not bit-deterministic (sampling temperature 0 reduces but does not eliminate variance); expected accuracy variance is ±2 percentage points across runs.
* `transport_bench.py` runs N=100 round-trips per transport on `localhost`; absolute numbers vary by host, but the relative gap between transports is stable.

## Models used

* **Subject + judge** in §5.1 classification: `claude-haiku-4-5-20251001` (single-rater design — see Limitations in the paper).
* **Calibration** for §5.2 transport-latency table: same model used to baseline a representative remote-LLM round-trip; values then composed with measured stdio/HTTP transport overheads.

## Citation

```bibtex
@misc{rodrigues2026mcp,
  author        = {Rodrigues, Carson and Vas, Oysturn},
  title         = {{MCP Server Architecture Patterns for LLM-Integrated Applications}},
  year          = {2026},
  eprint        = {2606.30317},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2606.30317}
}
```

## License

MIT. See `LICENSE`.
