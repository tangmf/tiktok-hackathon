from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "actually", "ignore", "earlier", "preference", "need", "still", "just",
    "exploring", "not", "sure", "yet", "one", "requirement", "key",
}

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex",
             "silk", "rayon", "fabric", "denim", "linen", "suede", "mesh")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray",
          "grey", "purple", "yellow", "orange", "navy", "beige", "gold", "silver")
SIZES = ("size", "sizing", "width", "wide", "narrow", "small", "medium",
          "large", "xl", "xs", "petite", "plus")
USE_CASES = ("hiking", "running", "gym", "winter", "outdoor", "work", "wedding",
             "casual", "formal", "travel", "party", "summer", "beach", "sport")
STYLES = ("style", "fit", "sleeve", "neck", "slim", "relaxed", "vintage",
          "classic", "trendy")
BUDGET_RE = re.compile(r"(?:\$\s?\d+(?:\.\d+)?|under\s+\d+|budget|<=\s*\d+|around\s+\$?\d+)", re.I)
OVERRIDE_CUES = ("actually", "ignore my earlier", "instead", "changed my mind",
                  "forget that", "never mind", "on second thought")

ATTRIBUTE_ORDER = ["material", "color", "size", "style", "use_case", "budget", "brand", "feature"]


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def classify_term(term: str) -> str:
    """Map a raw token/phrase to one of the contract's ask_attribute buckets."""
    lowered = term.lower()
    if BUDGET_RE.search(lowered):
        return "budget"
    if any(m in lowered for m in MATERIALS):
        return "material"
    if any(c in lowered for c in COLORS):
        return "color"
    if any(s in lowered for s in SIZES):
        return "size"
    if any(u in lowered for u in USE_CASES):
        return "use_case"
    if any(s in lowered for s in STYLES):
        return "style"
    return "feature"


def is_override(message: str) -> bool:
    lowered = message.lower()
    return any(cue in lowered for cue in OVERRIDE_CUES)


def looks_like_browsing(message: str, slot_count: int) -> bool:
    lowered = message.lower()
    vague_cues = ("still exploring", "just looking", "not sure", "browsing", "open to")
    if any(cue in lowered for cue in vague_cues):
        return True
    return slot_count == 0 and not BUDGET_RE.search(lowered) and not any(
        m in lowered for m in (*MATERIALS, *COLORS, *SIZES)
    )


# ---------------------------------------------------------------------------
# Session state (dynamic context that gets distilled/updated every turn)
# ---------------------------------------------------------------------------

class SessionState:
    __slots__ = ("profile", "slots", "asked", "turn", "alpha", "history_terms")

    def __init__(self, profile: dict) -> None:
        self.profile = profile
        # attribute -> list of raw constraint phrases (accumulated, most-recent last)
        self.slots: dict[str, list[str]] = defaultdict(list)
        self.asked: set[str] = set()
        self.turn = 0
        # alpha = weight given to the precision ("buying") retrieval track.
        # starts biased by the user's long-term profile, then self-adjusts at runtime.
        tags = " ".join(profile.get("preference_tags") or [])
        self.alpha = 0.55 if tags else 0.45
        self.history_terms: list[str] = []


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """Hybrid conversational shopping agent.

    Pillars implemented:
      I.   Dual-track intent routing (precision filter vs diverse dense retrieval)
           fused via reciprocal-rank fusion over an in-memory FTS5 index.
      II.  A per-session slot-tracking state machine that accumulates
           constraints, handles abrupt intent override (slot erasure +
           rewrite), and triggers proactive clarification on over-generality.
      III. Runtime self-evolution: the retrieval-track weighting (alpha) and
           the clarification policy adapt each turn based on how the
           candidate pool is narrowing (dynamic context programming).
      IV.  Local heuristic "semantic ranking" stage rescoring the fused
           candidates against the full distilled constraint set -- pluggable
           with an external LLM call without changing the surrounding
           pipeline (see `_semantic_rerank`).
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.sessions: dict[str, SessionState] = {}
        self.products: dict[str, dict] = {}
        self._build_index()

    # -- indexing -----------------------------------------------------
    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.products[asin] = product
                batch.append((
                    asin,
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    # -- retrieval helpers ---------------------------------------------
    def _fts_search(self, terms: list[str], limit: int) -> list[tuple[str, float]]:
        unique_terms = list(dict.fromkeys(terms))[:40]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, limit),
        ).fetchall()
        # sqlite bm25 returns *lower is better*; flip sign so higher = better.
        return [(str(asin), -float(score)) for asin, score in rows]

    def _reciprocal_rank_fusion(
        self, ranked_lists: list[list[tuple[str, float]]], weights: list[float], k: int = 60
    ) -> list[str]:
        fused: dict[str, float] = defaultdict(float)
        for ranked, weight in zip(ranked_lists, weights):
            for position, (asin, _score) in enumerate(ranked):
                fused[asin] += weight * (1.0 / (k + position + 1))
        return [asin for asin, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]

    def _semantic_rerank(self, candidates: list[str], state: SessionState, limit: int) -> list[str]:
        """Local scoring stand-in for an LLM semantic-ranking stage.

        Scores each candidate against the *distilled* constraint set
        (hard slots weighted above soft/profile terms). Swappable for a
        real LLM call: replace the scoring loop with a single ranking
        prompt over `candidates` + `state.slots` + `state.profile`.
        """
        hard_terms = set()
        for attr in ("material", "color", "size", "budget"):
            for phrase in state.slots.get(attr, []):
                hard_terms.update(_terms(phrase))
        soft_terms = set(state.history_terms)
        soft_terms.update(_terms(" ".join(state.profile.get("preference_tags") or [])))

        scored: list[tuple[str, float]] = []
        for asin in candidates:
            product = self.products.get(asin)
            if not product:
                continue
            corpus_terms = set(_terms(_text(product.get("title")) + " " +
                                       _text(product.get("features")) + " " +
                                       _text(product.get("details"))))
            hard_hits = len(hard_terms & corpus_terms)
            soft_hits = len(soft_terms & corpus_terms)
            score = 3.0 * hard_hits + 1.0 * soft_hits
            scored.append((asin, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [asin for asin, _ in scored[:limit]]

    # -- dialog state machine -------------------------------------------
    def _update_slots(self, state: SessionState, message: str) -> None:
        override = is_override(message)
        terms = _terms(message)
        state.history_terms.extend(terms)

        if override:
            # Abrupt intent override: erase soft/style slots, keep category,
            # rewrite with whatever new constraint follows the override cue.
            for attr in ("style", "feature", "use_case"):
                state.slots.pop(attr, None)

        # group consecutive terms into short phrases for nicer constraint text
        phrase = " ".join(terms[:8]) if terms else ""
        if phrase:
            bucket = classify_term(phrase)
            if phrase not in state.slots[bucket]:
                state.slots[bucket].append(phrase)

    def _next_attribute_to_ask(self, state: SessionState) -> str | None:
        for attr in ATTRIBUTE_ORDER:
            if attr not in state.slots and attr not in state.asked:
                return attr
        return None

    # -- public API -------------------------------------------------------
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = SessionState(user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.turn = turn
        self._update_slots(state, user_message)

        buying_like = not looks_like_browsing(user_message, len(state.slots))

        # Route A: precision filter track -- hard constraints + category only.
        hard_terms: list[str] = []
        for attr in ("material", "color", "size", "budget", "brand"):
            for phrase in state.slots.get(attr, []):
                hard_terms.extend(_terms(phrase))
        route_a = self._fts_search(hard_terms or state.history_terms, limit=100)

        # Route B: diverse dense-ish track -- everything accumulated so far.
        route_b = self._fts_search(state.history_terms, limit=300)

        # Self-evolving orchestration: shift weight toward precision as more
        # hard slots accumulate; shift toward diversity while still vague.
        hard_slot_count = sum(1 for a in ("material", "color", "size", "budget") if state.slots.get(a))
        state.alpha = min(0.85, max(0.15, state.alpha + 0.1 * (hard_slot_count - (1 if buying_like else 0))))

        fused = self._reciprocal_rank_fusion(
            [route_a, route_b], weights=[state.alpha, 1.0 - state.alpha]
        )

        reranked = self._semantic_rerank(fused[:150], state, limit=top_k)

        # Over-generality check: fused pool huge & few hard constraints yet ->
        # cut retrieval short and ask a clarification instead of guessing.
        pool_size = len(fused)
        few_constraints = hard_slot_count < 2
        ask_attribute = None
        if (pool_size > 40 and few_constraints and turn < 8) or not reranked:
            ask_attribute = self._next_attribute_to_ask(state)

        if ask_attribute:
            state.asked.add(ask_attribute)
            phrasing = {
                "material": "Do you have a material preference?",
                "color": "Any particular color you'd like?",
                "size": "What size are you looking for?",
                "style": "What style or fit works best for you?",
                "use_case": "What will you mainly use this for?",
                "budget": "Do you have a target budget in mind?",
                "brand": "Is there a brand you prefer?",
                "feature": "Any specific feature that matters most to you?",
            }.get(ask_attribute, "Could you tell me a bit more about what you need?")
        else:
            phrasing = "Here are the closest matches I found."

        recommendations = [{"parent_asin": asin} for asin in reranked[:top_k]]
        return {
            "message": phrasing,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
