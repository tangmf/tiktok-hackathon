# Shopping Copilot — Prototype Notes

This implementation replaces `starter/agent.py` with a hybrid conversational
agent (original BM25 baseline preserved at `starter/agent_baseline.py`). It
runs entirely offline against the official local evaluator — no LLM API key
required.

## Results on the 200-session public set

| Metric | Baseline (BM25) | Prototype | Delta |
|---|---|---|---|
| Hit Rate@10 | 0.125 | **0.430** | +0.305 |
| MRR | 0.068 | **0.255** | +0.187 |
| MTTC (lower is better) | 9.81 | **7.24** | -2.57 turns |
| TechnicalScore | 0.107 | **0.367** | +0.260 |

Per-scenario Hit Rate@10: buying 0.36, browsing 0.48, intent_override 0.57,
boundary 0.20. Reproduce with:

```bash
python3 -m evaluator.local_evaluator --output results_hybrid.json
```

## How it maps to the four pillars

**I. Dual-track routing + hybrid pipeline** — Every turn runs two in-memory
FTS5 (BM25) retrievals: Route A restricts to hard constraint terms only
(material/color/size/budget/brand) for precision; Route B searches over the
full accumulated conversation for recall/diversity. The two ranked lists are
combined with reciprocal-rank fusion, weighted by a runtime `alpha` (see
pillar III). A local heuristic reranker (`_semantic_rerank`) then rescores
the fused candidates against distilled hard vs. soft terms — this is the
"LLM Semantic Ranking" stage, written so the scoring loop is a drop-in
replacement point for a real ranking prompt if a team wants to spend LLM
budget there.

**II. Multi-turn dialog state machine** — `SessionState.slots` is a
dict keyed by the contract's `ask_attribute` vocabulary
(material/color/size/style/brand/budget/feature/use_case), accumulated turn
over turn. Override cues ("actually", "ignore my earlier preference",
"instead", "never mind") trigger slot erasure of the softer attributes and a
rewrite from the new message. When the fused candidate pool is still large
and fewer than 2 hard constraints have been collected, the agent skips
guessing and asks a targeted clarification instead (`ask_attribute` set,
cycling through unseen attributes).

**III. Self-evolution / dynamic context programming** — `state.alpha`
(precision-vs-diversity weighting) starts from a profile-derived prior and
is nudged every turn based on how many hard slots have been collected —
i.e., the retrieval strategy re-orchestrates itself as the session
narrows, without any hardcoded turn-by-turn schedule.

**IV. Evaluation** — Uses the organizer's own `local_evaluator.py`
unmodified, scored against the frozen 50k-product catalog. No mock ASINs,
no catalog mutation.

## What's real vs. simplified

- Catalog (50,000 products), public sessions (200), and evaluator are the
  actual organizer-provided kit, cloned and run unmodified.
- "Dense retrieval" here is BM25 + heuristic term-overlap reranking, not a
  learned embedding model — this keeps the prototype dependency-free and
  fast (~a few seconds per full 200-session eval), and stays inside the
  "no paid LLM required" allowance. Swapping in a real embedding model or an
  LLM ranking call is a natural next step and only touches
  `_fts_search`/`_semantic_rerank`.
- Clarification phrasing is templated per attribute rather than
  LLM-generated, for the same reason.

## Next steps if extending this

1. Replace `_semantic_rerank` with a real LLM ranking call (batch the top-N
   fused candidates into one ranking prompt) and report token usage — the
   `usage` field in `respond()` is already wired for this.
2. Swap the term-overlap "dense" route for actual embeddings (e.g. a small
   local sentence-transformer) for genuine semantic recall on paraphrased
   browsing queries.
3. Tune `alpha`'s update rule and the over-generality threshold against the
   `boundary` and `buying` scenario buckets specifically — they're currently
   the weakest (0.20 and 0.36 Hit Rate@10 respectively).
