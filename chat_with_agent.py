"""Interactive CLI to chat with the shopping agent yourself.

Run from the project root:
    python chat_with_agent.py

Type messages as if you were the customer. The agent will ask
clarification questions and/or show ranked product recommendations
each turn, for up to 10 turns.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from starter.agent import Agent

CATALOG_PATH = "data/catalog.jsonl"
MAX_TURNS = 10
TOP_K = 10


def load_product_titles(catalog_path: str) -> dict[str, str]:
    titles: dict[str, str] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            titles[str(product["parent_asin"])] = str(product.get("title", ""))[:80]
    return titles


def ask_profile() -> dict:
    print("First, a couple of quick questions about your shopping history")
    print("(press Enter to accept the default shown in brackets)\n")

    freq = input("How often do you buy clothing/shoes/jewelry online? "
                 "[3-4 prior purchases]: ").strip() or "3-4 prior purchases"
    rating = input("Roughly, average star rating you give purchases (1-5)? [4.5]: ").strip()
    rating_val = float(rating) if rating else 4.5
    style = input("Are your reviews usually positive or critical? "
                  "[usually positive]: ").strip() or "usually positive"
    tags_raw = input("A few things you care about, comma separated "
                     "[comfort, fit]: ").strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] or ["comfort", "fit"]

    return {
        "purchase_frequency": freq,
        "average_prior_rating": rating_val,
        "rating_style": style,
        "preference_tags": tags,
        "summary": f"Prior purchases emphasize {', '.join(tags)}; ratings are {style}.",
    }


def main() -> None:
    print("Loading catalog and building search index (first run may take a moment)...\n")
    agent = Agent(CATALOG_PATH)
    titles = load_product_titles(CATALOG_PATH)

    profile = ask_profile()
    session_id = f"cli_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, profile)

    print("\n" + "=" * 60)
    print("Chat with the shopping agent. Type 'quit' to exit early.")
    print("=" * 60 + "\n")

    for turn in range(1, MAX_TURNS + 1):
        message = input(f"[Turn {turn}] You: ").strip()
        if message.lower() in {"quit", "exit"}:
            print("Ending session.")
            break
        if not message:
            print("(please type something)")
            continue

        response = agent.respond(session_id, message, turn, TOP_K)

        print(f"\nAgent: {response['message']}")
        if response.get("ask_attribute"):
            print(f"       (asking about: {response['ask_attribute']})")

        recs = response.get("recommendations") or []
        if recs:
            print(f"\nTop {min(len(recs), 5)} recommendations:")
            for i, rec in enumerate(recs[:5], start=1):
                asin = rec["parent_asin"]
                title = titles.get(asin, "(title unavailable)")
                print(f"  {i}. [{asin}] {title}")
        else:
            print("\n(no recommendations yet)")
        print()

    else:
        print("Reached the 10-turn limit.")


if __name__ == "__main__":
    main()