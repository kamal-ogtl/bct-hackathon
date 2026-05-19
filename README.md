# DSN × BCT LLM Agent Challenge

Submission for the **Data & AI Summit Hackathon 3.0** — DSN × BCT LLM Agent Challenge.

Two containerized FastAPI services that model user behaviour from Yelp review history and generate personalized reviews and recommendations.

---

## Architecture

```
Yelp Dataset (JSONL)
      │
      ▼
data_loader.py ──► businesses_subset.jsonl
                   reviews_subset.jsonl
                   users_subset.jsonl
                        │
                        ▼
                 ingest_data.py ──► ChromaDB (text-embedding-3-small)
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                     ReviewAgent               RecommendationAgent
                  (LangGraph 3-node)          (LangGraph 4-node)
                              │                     │
                              ▼                     ▼
                     Task A API :8000       Task B API :8001
```

**Stack:** Python · FastAPI · LangGraph · LangChain · OpenAI GPT-4o-mini · ChromaDB · Docker

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd bct-hackathon
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Prepare data

Download the [Yelp Academic Dataset](https://www.yelp.com/dataset) and place the JSON files in:

```
data/raw/Yelp-JSON/Yelp JSON/yelp_dataset/
```

Then run the preprocessing pipeline:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Sample ~100 users from the full dataset
python src/utils/data_loader.py

# Embed businesses into ChromaDB
python -m src.utils.ingest_data
```

### 3. Start services

```bash
docker compose up --build
```

Both services start automatically:

| Service | Base URL | Swagger UI | Health |
|---------|----------|------------|--------|
| Task A — Review Generation | `http://localhost:8000` | [`/docs`](http://localhost:8000/docs) | `/health` |
| Task B — Recommendations | `http://localhost:8001` | [`/docs`](http://localhost:8001/docs) | `/health` |

---

## API Reference

### Task A — Generate Review

**`POST /generate-review`**

```bash
curl -X POST http://localhost:8000/generate-review \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "business_id": "biz_456",
    "user_history": [
      {
        "business_name": "Suya Spot",
        "category": "Nigerian, BBQ",
        "stars_given": 5,
        "review_text": "Omo the suya here is too good. The pepper is on point."
      }
    ],
    "business_metadata": {
      "name": "Jollof Kitchen",
      "category": "Nigerian, West African",
      "avg_stars": 4.2,
      "price_range": "$$"
    },
    "nigerian_mode": true
  }'
```

**Response:**
```json
{
  "predicted_stars": 5,
  "generated_review": "Omo, Jollof Kitchen is where it's at! ...",
  "confidence": 0.95,
  "cold_start": false
}
```

**`nigerian_mode: true`** activates Nigerian English slang and cultural framing in the generated review.

---

### Task B — Recommendations

**`POST /recommend`**

```bash
curl -X POST http://localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "user_history": [
      {"business_name": "Mama Cass", "category": "Nigerian", "stars_given": 5}
    ],
    "context": {
      "time_of_day": "evening",
      "mood": "casual dining with friends",
      "location_preference": "nearby"
    },
    "num_recommendations": 5
  }'
```

**Response:**
```json
{
  "recommendations": [
    {
      "business_id": "abc123",
      "business_name": "Kumo Sushi",
      "category": "Japanese, Sushi Bars",
      "predicted_rating": 4.5,
      "reason": "Matches user preference for casual group dining",
      "confidence": 0.95
    }
  ],
  "cold_start": false,
  "reasoning_trace": "Profile built from history. -> Retrieved 10 candidates. -> Scored and ranked. -> Final recommendations generated."
}
```

---

## Cold-Start Handling

Both endpoints handle users with no review history. Pass an empty `user_history` array:

```json
{ "user_history": [] }
```

The system generates a balanced baseline persona and proceeds. The response includes `"cold_start": true` so callers can adjust downstream.

---

## Evaluation

Run evaluation scripts against the processed subset:

```bash
# Rating accuracy (RMSE, MAE, ±1 star accuracy)
python eval/eval_rmse.py --max-users 20

# Review quality (ROUGE-1, ROUGE-2, ROUGE-L)
python eval/eval_rouge.py --max-users 20 --save-examples 5

# With Nigerian mode
python eval/eval_rouge.py --max-users 20 --nigerian-mode --save-examples 5
```

**Results (n=20, leave-one-out evaluation):**

| Metric | Score |
|--------|-------|
| RMSE | 1.75 |
| MAE | 1.15 |
| Within ±1 star | 75% |
| ROUGE-1 F1 | 0.262 |
| ROUGE-2 F1 | 0.051 |
| ROUGE-L F1 | 0.144 |

---

## Project Structure

```
├── data/
│   ├── raw/            # Yelp JSON files (not committed)
│   ├── processed/      # Sampled subsets (JSONL)
│   └── chroma_db/      # ChromaDB vector store
├── eval/
│   ├── eval_rmse.py    # Rating accuracy evaluation
│   └── eval_rouge.py   # Review quality evaluation
├── src/
│   ├── agents/
│   │   ├── review_agent.py          # Task A: LangGraph 3-node agent
│   │   └── recommendation_agent.py  # Task B: LangGraph 4-node agent
│   ├── models/
│   │   └── user_profiler.py         # Persona extraction via GPT-4o-mini
│   ├── api/
│   │   ├── task_a_api.py            # FastAPI app, port 8000
│   │   └── task_b_api.py            # FastAPI app, port 8001
│   └── utils/
│       ├── data_loader.py           # Streaming JSONL loader + sampler
│       ├── vector_store.py          # ChromaDB manager
│       └── ingest_data.py           # Ingestion script
├── task_a/Dockerfile
├── task_b/Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `CHROMA_DB_PATH` | Path to ChromaDB storage (default: `./data/chroma_db`) |
| `PROCESSED_DATA_DIR` | Path to processed JSONL files (default: `./data/processed`) |
| `RAW_DATA_DIR` | Path to raw Yelp JSON files |
| `LOG_LEVEL` | Logging level (default: `INFO`) |

---

## Key Design Decisions

- **Streaming data loading** — 5GB review file read line-by-line, never loaded into memory
- **Cold-start as first class** — 85% of Yelp users have one review; the system never fails on empty history
- **Index-based LLM ranking** — Task B sends numbered candidates to the LLM instead of raw IDs, preventing ID corruption in LLM output
- **Nigerian cultural layer** — `nigerian_mode` flag adds authentic Nigerian English (Omo, Abeg, Chai) to review generation via targeted system prompt injection
- **Reasoning trace** — Task B exposes the agent's step-by-step reasoning for interpretability
