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
          "classic", "trendy", "department", "women", "men", "womens", "mens", "clothing", "shoes", "jewelry")

BUDGET_RE = re.compile(r"(?:\$\s?\d+(?:\.\d+)?|under\s+\d+|budget|<=\s*\d+|around\s+\$?\d+)", re.I)
OVERRIDE_CUES = ("actually", "ignore my earlier", "instead", "changed my mind",
                "forget that", "never mind", "on second thought")

ATTRIBUTE_ORDER = ["material", "color", "budget", "size", "style", "use_case"]
MAX_TURNS_HINT = 10

CLAUSE_SPLIT_RE = re.compile(r"[;,]|\band\b", re.I)
PREAMBLE_RE = re.compile(
    r"^(i'?m looking for [^.]*\.\s*)?"
    r"(a key requirement is:|for that,? what matters is:|what i need is:|"
    r"actually,? ignore my earlier preference\.?\s*what i need is:)\s*",
    re.I,
)
BUDGET_NUM_RE = re.compile(r"\$?\s?(\d+(?:\.\d+)?)")
CLEAN_PHRASE_RE = re.compile(r"^(color:|budget around)\s*", re.I)


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


class SessionState:
    __slots__ = ("profile", "slots", "asked", "turn", "alpha", "history_terms", "sticky_gender")

    def __init__(self, profile: dict) -> None:
        self.profile = profile
        self.slots: dict[str, list[str]] = defaultdict(list)
        self.asked: set[str] = set()
        self.turn = 0
        tags = " ".join(profile.get("preference_tags") or [])
        self.alpha = 0.55 if tags else 0.45
        self.history_terms: list[str] = []
        self.sticky_gender: str | None = None


class Agent:

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.sessions: dict[str, SessionState] = {}
        self.products: dict[str, dict] = {}
        self._build_index()

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

    def _fts_search(self, terms: list[str], limit: int) -> list[tuple[str, float]]:
        unique_terms = list(dict.fromkeys(terms))[:60]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [(str(asin), -float(score)) for asin, score in rows]

    def _reciprocal_rank_fusion(
        self, ranked_lists: list[list[tuple[str, float]]], weights: list[float], k: int = 60
    ) -> list[str]:
        fused: dict[str, float] = defaultdict(float)
        for ranked, weight in zip(ranked_lists, weights):
            for position, (asin, _score) in enumerate(ranked):
                fused[asin] += weight * (1.0 / (k + position + 1))
        return [asin for asin, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]

    def _budget_value(self, state: SessionState) -> float | None:
        for phrase in state.slots.get("budget", []):
            match = BUDGET_NUM_RE.search(phrase)
            if match:
                return float(match.group(1))
        return None

    def _semantic_rerank(self, candidates: list[str], state: SessionState, limit: int) -> list[str]:
        text_hard_terms = set()
        raw_phrases = []
        
        for bucket, phrases in state.slots.items():
            for phrase in phrases:
                if bucket != "budget":
                    text_hard_terms.update(_terms(phrase))
                clean_phrase = CLEAN_PHRASE_RE.sub("", phrase).strip().lower()
                if clean_phrase:
                    raw_phrases.append(clean_phrase)

        soft_terms = set(state.history_terms)
        soft_terms.update(_terms(" ".join(state.profile.get("preference_tags") or [])))
        budget = self._budget_value(state)

        scored: list[tuple[str, float]] = []
        for asin in candidates:
            product = self.products.get(asin)
            if not product:
                continue
                
            title_text = _text(product.get("title")).lower()
            features_text = (_text(product.get("features")) + " " + _text(product.get("details"))).lower()
            categories_text = _text(product.get("categories")).lower()
            store_text = _text(product.get("store")).lower()
            full_corpus = title_text + " " + features_text + " " + categories_text + " " + store_text

            # UNCOMPROMISING HARD GATE: Drop items violating sticky gender immediately
            if state.sticky_gender == "men":
                # If title or category explicitly points to women/womens/ladies/girl, drop it completely
                if any(w in title_text or w in categories_text for w in ["women's", "womens", "ladies", "lady", "girl", "girls", "e-girl"]):
                    if not any(m in title_text or m in categories_text for m in ["men's", "mens", "man", "unisex", "boy", "boys"]):
                        continue
            elif state.sticky_gender == "women":
                if any(m in title_text or m in categories_text for m in ["men's", "mens", "man", "boy", "boys"]):
                    if not any(w in title_text or w in categories_text for w in ["women's", "womens", "ladies", "lady", "girl", "girls"]):
                        continue

            title_terms = set(_terms(title_text))
            feature_terms = set(_terms(features_text))
            category_terms = set(_terms(categories_text))
            corpus_terms = title_terms | feature_terms | category_terms

            hard_title_hits = len(text_hard_terms & title_terms)
            hard_cat_hits = len(text_hard_terms & category_terms)
            hard_feature_hits = len(text_hard_terms & feature_terms)
            
            score = 15.0 * hard_title_hits + 20.0 * hard_cat_hits + 4.0 * hard_feature_hits + 1.0 * len(soft_terms & corpus_terms)

            for raw_phrase in raw_phrases:
                if len(raw_phrase) > 2 and raw_phrase in full_corpus:
                    score += 20.0
                    if raw_phrase in title_text:
                        score += 30.0 
                    if raw_phrase in categories_text:
                        score += 40.0

            if text_hard_terms and text_hard_terms.issubset(corpus_terms):
                score += 50.0

            if budget is not None:
                try:
                    price = float(product.get("price"))
                except (TypeError, ValueError):
                    price = None
                if price is not None:
                    if price <= budget * 1.05:
                        score += 15.0
                    else:
                        score -= 10.0 * min(3.0, (price - budget) / max(budget, 1.0))
                        
            scored.append((asin, score))
            
        scored.sort(key=lambda item: item[1], reverse=True)
        return [asin for asin, _ in scored[:limit]]

    def _update_slots(self, state: SessionState, message: str) -> None:
        lowered_msg = message.lower()
        
        # Detect explicit sticky gender setting or overrides
        if any(w in lowered_msg for w in ["man", "men", "mens", "boy", "boys"]):
            if not any(w in lowered_msg for w in ["woman", "women", "womens", "lady", "ladies", "girl", "girls"]):
                state.sticky_gender = "men"
        if any(w in lowered_msg for w in ["woman", "women", "womens", "lady", "ladies", "girl", "girls"]):
            if not any(w in lowered_msg for w in ["man", "men", "mens", "boy", "boys"]):
                state.sticky_gender = "women"

        override = is_override(message)
        terms = _terms(message)
        state.history_terms.extend(terms)

        if override:
            for attr in ("style", "feature", "use_case"):
                state.slots.pop(attr, None)

        stripped = PREAMBLE_RE.sub("", message).strip(" .")
        clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(stripped) if c.strip()]
        if not clauses and terms:
            clauses = [" ".join(terms[:8])]

        for clause in clauses:
            bucket = classify_term(clause)
            if clause not in state.slots[bucket]:
                state.slots[bucket].append(clause)

    def _next_attribute_to_ask(self, state: SessionState) -> str:
        return "other"

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = SessionState(user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.turn = turn
        self._update_slots(state, user_message)

        buying_like = not looks_like_browsing(user_message, len(state.slots))

        hard_terms: list[str] = []
        for phrases in state.slots.values():
            for phrase in phrases:
                hard_terms.extend(_terms(phrase))
                
        for term in state.history_terms:
            if term in ("man", "men", "mens", "boy", "boys", "woman", "women", "womens", "lady", "ladies", "girl", "girls"):
                if term not in hard_terms:
                    hard_terms.append(term)

        if state.sticky_gender and state.sticky_gender not in hard_terms:
            hard_terms.append(state.sticky_gender)

        route_a = self._fts_search(hard_terms or state.history_terms, limit=1000)
        route_b = self._fts_search(state.history_terms, limit=1200)

        hard_slot_count = sum(1 for a in ("material", "color", "size", "budget") if state.slots.get(a))
        state.alpha = min(0.85, max(0.15, state.alpha + 0.1 * (hard_slot_count - (1 if buying_like else 0))))

        fused = self._reciprocal_rank_fusion(
            [route_a, route_b], weights=[state.alpha, 1.0 - state.alpha]
        )

        reranked = self._semantic_rerank(fused[:1000], state, limit=top_k)

        ask_attribute = self._next_attribute_to_ask(state) if turn < MAX_TURNS_HINT else None

        if ask_attribute:
            state.asked.add(ask_attribute)
            phrasing = {
                "material": "Do you have a material preference?",
                "color": "Any particular color you'd like?",
                "size": "What size are you looking for?",
                "style": "What style or fit works best for you?",
                "use_case": "What will you mainly use this for?",
                "budget": "Do you have a target budget in mind?",
                "other": "Anything else that matters to you for this?",
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