# TechJam 2026 — Conversational E-Commerce Search Agent

A conversational product search agent built for the **TechJam Conversational E-Commerce Search Challenge**.

The goal is to identify a customer's hidden target product from a catalog of **50,000 Clothing, Shoes & Jewelry products** within a maximum of **10 conversational turns**.

Our solution improves upon the provided BM25 baseline by combining conversational memory, multi-route full-text retrieval, Reciprocal Rank Fusion (RRF), and constraint-aware reranking.

---

## 🏆 Current Performance

| Metric | Result |
|---|---:|
| Technical Score | **0.722** |
| Catalog Size | 50,000 products |
| Public Evaluation Sessions | 200 |
| Maximum Turns | 10 |
| Maximum Recommendations / Turn | 10 |

The technical score is calculated using:

```text
Technical Score =
0.50 × Hit Rate@10
+ 0.30 × MRR
+ 0.20 × Efficiency

Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

The provided weak BM25 starter achieves:

| Metric | BM25 Baseline |
|---|---:|
| Hit Rate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |

---

## 🎯 Problem

Traditional e-commerce search works well when users already know the correct keywords.

Real shopping conversations are different.

A customer may:

- Start with a vague request
- Add requirements progressively
- Express preferences naturally
- Change their mind
- Specify a budget later
- Describe product features instead of knowing the product name

Our agent therefore treats product search as a **multi-turn conversation** rather than a sequence of independent keyword searches.

---

## 💡 Our Approach

The overall recommendation pipeline is:

```text
Customer Message
       ↓
Conversation State Tracking
       ↓
Constraint Extraction
       ↓
Multi-Route FTS5 Retrieval
       ↓
Reciprocal Rank Fusion
       ↓
Constraint-Aware Reranking
       ↓
Top-K Recommendations
```

### 1. Conversational State

Each session maintains information such as:

- User preference profile
- Extracted constraints
- Previously requested attributes
- Conversation history
- Current turn
- Persistent gender intent

For example:

```text
User: I'm looking for men's running shoes.
User: I want them for outdoor trails.
User: I'd prefer ASICS.
```

The final request is not interpreted independently.

Instead, the agent retains the accumulated intent:

```text
Gender: Men's
Category: Running shoes
Use case: Trail running
Brand preference: ASICS
```

---

## 🔎 Product Retrieval

The 50,000-product catalog is indexed using **SQLite FTS5**.

Our search index considers multiple product fields:

- Title
- Categories
- Features
- Details
- Store
- Description

This allows the system to match a customer's description against information beyond the product title.

---

## 🔀 Multi-Route Retrieval

Rather than relying on a single search query, the agent performs multiple retrieval routes.

### Route A — Current Constraints

Searches using the user's currently identified requirements.

### Route B — Conversation History

Searches using terms accumulated throughout the conversation.

The ranked candidate lists are then combined using **Reciprocal Rank Fusion (RRF)**.

```text
Current Constraints ──→ FTS5 ──┐
                               ├──→ RRF ──→ Reranker
Conversation History ─→ FTS5 ──┘
```

This allows both recent constraints and earlier conversational information to influence the recommendations.

---

## 📊 Constraint-Aware Reranking

After candidate retrieval, products are reranked according to how well they satisfy the customer's requirements.

The reranker considers:

- Title matches
- Category matches
- Feature matches
- Exact phrase matches
- Hard-constraint coverage
- Budget compatibility
- User preferences

The final ranked list contains up to **10 `parent_asin` values**.

---

## 💰 Budget Handling

The agent detects numerical budget expressions from natural-language messages.

Example:

```text
User: My budget is around $80.
```

Products that satisfy the budget can receive a ranking bonus, while products substantially exceeding the budget are penalised.

---

## 🔄 Intent Override

Shopping requirements can change during a conversation.

Our agent recognises override expressions such as:

```text
actually
ignore my earlier
instead
changed my mind
forget that
never mind
on second thought
```

For example:

```text
User: I want black shoes.

User: Actually, ignore my earlier preference.
      I want white shoes instead.
```

The agent updates its conversational state so outdated preferences do not continue to dominate subsequent recommendations.

---

## 👤 Persistent Gender Intent

The agent also maintains persistent gender intent.

If a customer clearly requests men's products, explicitly women's-only results can be filtered unless the product is appropriate for both.

This helps prevent irrelevant recommendations caused by overlapping product keywords.

---

## 💬 Agent Interface

The implementation follows the provided TechJam Agent API.

```python
class Agent:

    def reset(
        self,
        session_id: str,
        user_profile: dict
    ) -> None:
        ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int
    ) -> dict:

        return {
            "message": "...",
            "ask_attribute": "...",
            "recommendations": [
                {"parent_asin": "..."}
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        }
```

On each turn, the agent may:

1. Ask a clarification question
2. Return ranked product recommendations
3. Do both

Valid clarification attributes include:

```text
category
material
color
size
style
brand
budget
feature
use_case
other
```

---

## 🧪 Interactive Demo

Run the conversational demo from the project root:

```bash
python chat_with_agent.py
```

The demo allows a user to interact directly with the agent and inspect the recommendations generated after each turn.

### Example Target

For demonstration purposes, we select a known product from the catalog:

**ASICS Men's Gel-Venture 6 MX Running Shoes**

```text
Target ASIN: B093R14VP1
```

We intentionally avoid giving the agent the exact product name.

Instead, the customer progressively reveals information:

```text
Turn 1
I'm looking for men's running shoes.

Turn 2
I want them specifically for trail running.

Turn 3
I'd prefer something made from synthetic material.

Turn 4
I want a rubber sole.

Turn 5
Good shock-absorbing cushioning is important to me.

Turn 6
I'd like something with rearfoot GEL cushioning.

Turn 7
I also want a removable sockliner.

Turn 8
I need good uphill and downhill traction on different types of terrain.

Turn 9
I'd prefer ASICS with a durable AHAR outsole.

Turn 10
My budget is around $65, and I'd like a low-top trail running shoe with multi-directional lugs.
```

After every turn, the system reranks the catalog.

This allows us to observe:

- When the target first appears
- The target's ranking
- How the ranking changes as constraints are added
- How quickly the system converges towards the intended product

---

## 📈 Evaluation

Run the official local evaluator:

```bash
python -m evaluator.local_evaluator --output results.json
```

The evaluator uses the provided **200 labeled public development sessions**.

An additional **800 sessions are private** and reserved by the organizers for final evaluation.

Sessions cover:

- Buying
- Browsing
- Intent Override
- Boundary behaviour

A session terminates when the exact target `parent_asin` enters the scored Top 10 or after Turn 10.

---

## 📏 Evaluation Metrics

### Hit Rate@10

Fraction of sessions where the exact target product appears in the Top 10 within 10 turns.

### Mean Reciprocal Rank (MRR)

Measures how highly the correct product is ranked.

For example:

```text
Rank 1 → 1/1 = 1.00
Rank 2 → 1/2 = 0.50
Rank 5 → 1/5 = 0.20
Miss   → 0
```

### Mean Turns to Convergence (MTTC)

Measures the first turn on which the target product appears.

A missed target is assigned Turn 11.

### Efficiency

```text
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

### Technical Score

```text
Technical Score =
0.50 × Hit Rate@10
+ 0.30 × MRR
+ 0.20 × Efficiency
```

Our current implementation achieves:

```text
Technical Score: 0.722
```

> The Technical Score is an objective input to the Technical Execution assessment and does not represent the entire judging score.

---

## ⚙️ Technology

The current solution uses:

- Python
- SQLite
- SQLite FTS5
- BM25-based full-text retrieval
- Reciprocal Rank Fusion
- Rule-based constraint extraction
- Constraint-aware reranking
- Session-based conversational state

The implementation uses the Python standard library and does not require an external LLM for retrieval.

---

## 💵 Model Choice & Cost

Our current retrieval implementation does not call an external LLM API.

| Item | Value |
|---|---|
| External LLM API | None |
| API Cost | $0 |
| Reported Prompt Tokens | 0 |
| Reported Completion Tokens | 0 |

This removes external API costs and network latency from the core recommendation pipeline.

---

## 📁 Project Structure

```text
.
├── data/
│   ├── catalog.jsonl
│   └── public_set.jsonl
│
├── docs/
│   ├── agent_api_contract.json
│   ├── baseline_results.json
│   ├── competition_specification.md
│   ├── evaluation_config.json
│   └── submission_rules.md
│
├── evaluator/
│   └── local_evaluator.py
│
├── starter/
│   └── agent.py
│
├── chat_with_agent.py
├── results.json
└── README.md
```

---

## 🚀 Getting Started

### Requirements

Python **3.10+** is recommended.

Check your version:

```bash
python --version
```

### 1. Clone the repository

```bash
git clone <repository-url>
cd tiktok-hackathon
```

### 2. Prepare the catalog

Download the competition `catalog.jsonl.gz` and decompress it into:

```text
data/catalog.jsonl
```

The expected catalog contains **50,000 products**.

### 3. Run the interactive agent

```bash
python chat_with_agent.py
```

### 4. Run evaluation

```bash
python -m evaluator.local_evaluator --output results.json
```

Evaluation results will be written to:

```text
results.json
```

---

## ⚠️ Limitations

The current system is primarily based on lexical retrieval and rule-based constraint matching.

Therefore:

- Semantically similar wording may not always match
- Product metadata quality affects retrieval quality
- Implicit preferences can be difficult to identify
- Similar products can outrank the intended target
- Clarification-question selection can be further optimized

---

## 🔮 Future Improvements

Potential improvements include:

- Semantic/vector retrieval
- Hybrid lexical + semantic search
- Better clarification-question selection
- Dynamic attribute prioritization
- Improved conflicting-constraint handling
- More sophisticated user-profile integration
- Learned reranking models

These improvements could increase Hit Rate@10 and MRR while reducing MTTC.

---

## 📚 Data

The catalog and evaluation sessions are derived from **Amazon Reviews 2023** by McAuley Lab, UCSD.

The challenge uses the `Clothing_Shoes_and_Jewelry` category.

Please refer to `DATA_ATTRIBUTION.md` for the required dataset attribution and redistribution information.

---

## Summary

Our conversational shopping agent extends the provided BM25 baseline with:

**Conversation Memory → Constraint Extraction → Multi-Route FTS5 Retrieval → Reciprocal Rank Fusion → Constraint-Aware Reranking**

The resulting system can progressively adapt to customer preferences across multiple turns while attempting to retrieve the hidden target product as early and as highly ranked as possible.

**Current Public Technical Score: 0.722**
