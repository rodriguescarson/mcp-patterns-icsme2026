"""
Inter-rater reliability for the MCP pattern taxonomy (Paper 2 strengthening).

The ICSME submission explicitly listed "no Cohen's kappa" as a limitation
(coding was single-author with a verification pass). This script converts that
limitation into a reported result: two INDEPENDENT LLM raters (Claude Haiku and
Claude Sonnet), each given only the pattern definitions and a server
description, classify a 54-server corpus. We report:
  * Cohen's kappa between the two raters (taxonomy applicability / reliability),
  * each rater's accuracy against the authors' ground-truth labels.

Two independent LLM raters are a reproducible proxy for dual human coding; we
report it as such (human dual-coding remains future work). The corpus draws on
real servers from the official modelcontextprotocol/servers registry and popular
community servers, labeled by the authors.

Run:  python3 kappa_eval.py
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import anthropic
from sklearn.metrics import cohen_kappa_score, accuracy_score

PATTERNS = ["Resource Gateway", "Tool Orchestrator", "Stateful Session Server",
            "Proxy Aggregator", "Domain-Specific Adapter"]

# 54-server corpus (28 first, then 26 more; see below). Descriptions are NEUTRAL and function-focused (what the
# server does, paraphrased from real READMEs) and deliberately do NOT name the
# architecture, so agreement reflects genuine taxonomy applicability rather than
# keyword matching. Several entries are intentionally borderline (they blend two
# patterns); disagreement on these is the informative signal. Ground-truth is
# the authors' considered label.
CORPUS = [
    ("filesystem", "Read, write, move, and search files and directories on the local disk.", "Resource Gateway"),
    ("postgres", "Run SQL queries and inspect schemas against a PostgreSQL database.", "Resource Gateway"),
    ("sqlite", "Query and modify tables in a SQLite database file.", "Resource Gateway"),
    ("s3-objects", "Get, put, and list objects in Amazon S3 buckets.", "Resource Gateway"),
    ("gdrive", "Search for and read the contents of Google Drive documents.", "Resource Gateway"),
    ("memory-kv", "Store and retrieve values by key in a persistent store.", "Resource Gateway"),
    ("brave-search", "Search the web and return ranked results for a query.", "Tool Orchestrator"),
    ("code-runner", "Execute Python or JavaScript snippets and return their output.", "Tool Orchestrator"),
    ("slack", "Send messages, list channels, and read recent conversation history.", "Tool Orchestrator"),
    ("sentry", "Look up application errors and retrieve stack traces for an issue.", "Tool Orchestrator"),
    ("multi-api", "Get the weather, translate text, and summarize documents.", "Tool Orchestrator"),
    ("notion", "Create pages, query databases, and append content blocks.", "Tool Orchestrator"),
    ("puppeteer", "Open a web page, click elements, fill forms, and take screenshots.", "Stateful Session Server"),
    ("git", "Stage changes, commit, show diffs, and switch branches in a repository.", "Stateful Session Server"),
    ("playwright", "Navigate pages, interact with elements, and assert on page content.", "Stateful Session Server"),
    ("ssh-session", "Run commands on a remote host over an interactive shell.", "Stateful Session Server"),
    ("repl-server", "Evaluate expressions in a long-lived language interpreter.", "Stateful Session Server"),
    ("mcp-hub", "Discover and forward calls to other configured MCP servers.", "Proxy Aggregator"),
    ("mcp-router", "Direct each tool call to the appropriate backend server.", "Proxy Aggregator"),
    ("gateway-proxy", "Expose tools and resources collected from several servers.", "Proxy Aggregator"),
    ("federation", "Combine the capabilities of many servers under one endpoint.", "Proxy Aggregator"),
    ("medical-ehr", "Fetch patient records, look up ICD-10 codes, and check drug interactions.", "Domain-Specific Adapter"),
    ("ecommerce", "Search products, check inventory, place orders, and track shipments.", "Domain-Specific Adapter"),
    ("stripe", "Create charges, issue refunds, and manage customer subscriptions.", "Domain-Specific Adapter"),
    ("kubernetes", "Apply manifests, scale deployments, and fetch pod logs.", "Domain-Specific Adapter"),
    ("jira", "Create issues, transition their status, and run JQL searches.", "Domain-Specific Adapter"),
    ("salesforce", "Query and update CRM records and run reports.", "Domain-Specific Adapter"),
    ("twilio", "Send SMS messages and place phone calls to a number.", "Tool Orchestrator"),
    # --- expansion to N=54 (neutral descriptions, real registry/community servers) ---
    ("redis", "Get, set, and delete keys in a Redis instance.", "Resource Gateway"),
    ("mongodb", "Find, insert, and update documents in MongoDB collections.", "Resource Gateway"),
    ("elasticsearch", "Index documents and run search queries against an index.", "Resource Gateway"),
    ("google-sheets", "Read and update cell ranges in a spreadsheet.", "Resource Gateway"),
    ("airtable", "List, create, and update records in a base.", "Resource Gateway"),
    ("obsidian", "Read and write notes in a markdown vault.", "Resource Gateway"),
    ("github", "Create issues and pull requests, and search code in a repository.", "Tool Orchestrator"),
    ("gitlab", "Open merge requests, list pipelines, and comment on issues.", "Tool Orchestrator"),
    ("linear", "Create tickets, assign them, and update their status.", "Tool Orchestrator"),
    ("discord", "Post messages to channels and read recent messages.", "Tool Orchestrator"),
    ("wolfram", "Answer factual and computational questions from a query.", "Tool Orchestrator"),
    ("youtube-transcript", "Fetch the transcript and metadata for a video.", "Tool Orchestrator"),
    ("calendar", "Create events, list upcoming events, and check availability.", "Tool Orchestrator"),
    ("selenium", "Drive a browser through a sequence of navigation and input steps.", "Stateful Session Server"),
    ("docker-exec", "Start a container and run successive commands inside it.", "Stateful Session Server"),
    ("jupyter-kernel", "Execute notebook cells against a running kernel.", "Stateful Session Server"),
    ("terminal", "Run shell commands in a persistent working session.", "Stateful Session Server"),
    ("vnc-desktop", "Control a remote desktop with clicks and keystrokes over time.", "Stateful Session Server"),
    ("composio", "Expose tools from many connected apps through one endpoint.", "Proxy Aggregator"),
    ("mcp-proxy", "Forward requests to and from a remote MCP server.", "Proxy Aggregator"),
    ("toolhub", "Register backend servers and route calls to the right one.", "Proxy Aggregator"),
    ("shopify", "Manage products, orders, and customers in a store.", "Domain-Specific Adapter"),
    ("quickbooks", "Create invoices, record payments, and reconcile accounts.", "Domain-Specific Adapter"),
    ("hubspot", "Manage CRM contacts, deals, and marketing campaigns.", "Domain-Specific Adapter"),
    ("plaid", "Retrieve bank balances and categorize financial transactions.", "Domain-Specific Adapter"),
    ("fhir", "Read patient resources and submit clinical observations.", "Domain-Specific Adapter"),
]


def _key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("set ANTHROPIC_API_KEY")
    return key


def classify(client, model, name, desc):
    prompt = (
        "You are classifying a Model Context Protocol (MCP) server into exactly one "
        "architectural pattern.\n\nPatterns:\n"
        "- Resource Gateway: exposes a data source/store as addressable resources with CRUD.\n"
        "- Tool Orchestrator: exposes several distinct action tools with unified handling.\n"
        "- Stateful Session Server: holds server-side session state across tool calls.\n"
        "- Proxy Aggregator: fronts/aggregates multiple downstream MCP servers.\n"
        "- Domain-Specific Adapter: encodes domain business logic and validation.\n\n"
        f"Server '{name}': {desc}\n\n"
        'Respond with JSON only: {"pattern": "<one pattern name>"}')
    for _ in range(3):
        try:
            r = client.messages.create(model=model, max_tokens=40, timeout=30,
                                       temperature=0,  # deterministic, reproducible classification
                                       messages=[{"role": "user", "content": prompt}])
            m = re.search(r'"pattern"\s*:\s*"([^"]+)"', r.content[0].text)
            if m:
                # snap to nearest known pattern label
                p = m.group(1).strip()
                for cand in PATTERNS:
                    if cand.lower() in p.lower() or p.lower() in cand.lower():
                        return cand
                return p
        except Exception as e:
            print("   retry:", e); time.sleep(3)
    return "UNKNOWN"


def main():
    client = anthropic.Anthropic(api_key=_key())
    raters = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-4-6"}
    gold = [p for _, _, p in CORPUS]
    preds = {r: [] for r in raters}
    for name, desc, _ in CORPUS:
        for r, model in raters.items():
            preds[r].append(classify(client, model, name, desc))
    import numpy as np
    n = len(CORPUS)
    out = {"n": n}

    def boot_ci(fn, n_boot=10000, seed=42):
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            vals.append(fn(idx))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    h, s, g = np.array(preds["haiku"]), np.array(preds["sonnet"]), np.array(gold)
    out["cohen_kappa_haiku_sonnet"] = float(cohen_kappa_score(h, s))
    out["cohen_kappa_ci"] = boot_ci(lambda i: cohen_kappa_score(h[i], s[i]))
    out["raw_agreement"] = float((h == s).mean())
    for r in raters:
        p = np.array(preds[r])
        out[f"acc_{r}_vs_gold"] = float(accuracy_score(g, p))
        out[f"acc_{r}_vs_gold_ci"] = boot_ci(lambda i: accuracy_score(g[i], p[i]))
        out[f"kappa_{r}_vs_gold"] = float(cohen_kappa_score(g, p, labels=PATTERNS))
    out["preds"] = preds
    out["gold"] = gold
    Path(__file__).parent.joinpath("results_kappa.json").write_text(json.dumps(out, indent=2))
    print(f"N={out['n']}")
    print(f"Inter-rater (Haiku vs Sonnet): kappa={out['cohen_kappa_haiku_sonnet']:.3f}, "
          f"raw agreement={out['raw_agreement']*100:.1f}%")
    for r in raters:
        print(f"  {r}: acc vs gold={out[f'acc_{r}_vs_gold']*100:.1f}%, kappa={out[f'kappa_{r}_vs_gold']:.3f}")
    print("saved results_kappa.json")


if __name__ == "__main__":
    main()
