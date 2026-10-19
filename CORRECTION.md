# Correction (October 2026)

Version 1 of "MCP Server Architecture Patterns for LLM-Integrated Applications" (arXiv:2606.30317, June
2026) reported an observational analysis of tool count against tool-selection accuracy (Section VI-C and
Figure 2), and this repository shipped its numbers as `tool_count_telemetry.csv`, labelled "ANSYR production
Q1 2025, post-call human QA review".

Those numbers could not be traced to production records. They were constants in the analysis script, every
bucket had exactly 200 sessions, and both models named (Claude Haiku 4.5 and Claude Sonnet 4) were released
after the first quarter of 2025. We have removed the file and withdrawn the analysis, its figure and every
statement that rested on it (the 10-15-tool budget, the Proxy Aggregator threshold guidance, the practitioner
implications and the conclusion) in version 2 of the paper.

The pattern catalogue, the inter-rater study (`kappa_eval.py`, `results_kappa.json`) and the transport
measurements (`transport_bench.py`, `results/transport_measured.json`) are unchanged.
