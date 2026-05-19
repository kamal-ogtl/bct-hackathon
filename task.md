You are a senior ML engineer building a RAG-based LLM agent system for a hackathon. The project is already scaffolded at bct-hackathon/ with this structure:

bct-hackathon/
├── data/raw/         # contains Yelp JSON dataset files
├── src/
│   ├── agents/       # LangGraph agent logic
│   ├── models/       # user modeling logic
│   ├── api/          # FastAPI endpoints
│   └── utils/        # shared helpers
├── task_a/           # Task A container
├── task_b/           # Task B container
├── docker-compose.yml
├── requirements.txt
└── README.md

Dataset: Yelp Academic Dataset. Five JSON files, one record per line (JSONL format):

yelp_academic_dataset_review.json — fields: review_id, user_id, business_id, stars, text, date, useful, funny, cool
yelp_academic_dataset_business.json — fields: business_id, name, address, city, state, categories, stars, review_count, attributes, hours
yelp_academic_dataset_user.json — fields: user_id, name, review_count, average_stars, friends, elite
yelp_academic_dataset_tip.json — fields: user_id, business_id, text, date, compliment_count
yelp_academic_dataset_checkin.json — fields: business_id, date

Key data facts you must design around:

85% of users have only 1 review — cold-start is the dominant scenario, not an edge case
Average review length is ~548 characters
Star distribution is skewed toward 4-5 stars
Full review file is 5GB — never load it entirely into memory. Always stream with line-by-line reading

Stack:

Python, FastAPI, LangGraph, LangChain, Google Gemini (gemini-1.5-flash) via langchain-google-genai
ChromaDB for vector store
Docker + Docker Compose for containerization
No finetuning — RAG + prompting only


TASK A — User Modeling Agent
Build a FastAPI endpoint: POST /generate-review
Input:

{
  "user_id": "string",
  "business_id": "string",
  "user_history": [
    {
      "business_name": "string",
      "category": "string",
      "stars_given": 4,
      "review_text": "string"
    }
  ],
  "business_metadata": {
    "name": "string",
    "category": "string",
    "avg_stars": 4.2,
    "price_range": "$$"
  }
}

Output:
{
  "predicted_stars": 4,
  "generated_review": "string",
  "confidence": 0.87
}

Agent logic:

Build a user persona from their review history — extract tone, vocabulary, rating patterns, topics they care about (service, food quality, ambiance, value)
Retrieve the 5 most similar businesses this user has reviewed using ChromaDB semantic search
Prompt Gemini to generate a review in that user's voice for the new business
For cold-start users (1 or no history): fall back to business category averages and generate a plausible persona. Do not fail — handle it gracefully and flag it in the response
Add a Nigerian cultural context layer in the system prompt — the agent should be able to generate reviews that sound authentically Nigerian when the nigerian_mode: true flag is passed


TASK B — Recommendation Agent
Build a FastAPI endpoint: POST /recommend
Input:
{
  "user_id": "string",
  "user_history": [
    {
      "business_name": "string",
      "category": "string",
      "stars_given": 4
    }
  ],
  "context": {
    "time_of_day": "evening",
    "mood": "casual dining",
    "location_preference": "nearby"
  },
  "num_recommendations": 10
}
Output:
{
  "recommendations": [
    {
      "business_id": "string",
      "business_name": "string",
      "category": "string",
      "predicted_rating": 4.3,
      "reason": "string",
      "confidence": 0.91
    }
  ],
  "cold_start": false,
  "reasoning_trace": "string"
}
Agent logic:

Build user preference profile from history
Use ChromaDB to retrieve candidate businesses semantically similar to what the user has liked
Use LangGraph to implement a reasoning loop: retrieve → score → filter → rank → explain
Cold-start handling: if user has no history, use context fields alone to retrieve and rank candidates. Never return an empty list
Cross-domain: if user history is from one category (e.g. Italian restaurants) but context suggests another (e.g. coffee shops), handle the domain shift explicitly in reasoning
Return a reasoning_trace showing what the agent considered — this is important for scoring

Data preprocessing — build this first as src/utils/data_loader.py:
# Must handle:
# 1. Streaming JSON files line by line (never load full file)
# 2. Building a user profile from their review history
# 3. Building a business profile from metadata + reviews
# 4. Sampling a working subset: users with 3+ reviews for evaluation,
#    plus cold-start users for cold-start testing
# 5. Saving processed subsets as smaller JSON files in data/processed/

ChromaDB setup — build as src/utils/vector_store.py:

One collection for businesses: embed name + category + top review snippets
One collection for user personas: embed summarized preference profile
Use text-embedding-004 from Google or all-MiniLM-L6-v2 from sentence-transformers

Docker:

task_a/ gets its own Dockerfile exposing port 8000
task_b/ gets its own Dockerfile exposing port 8001
docker-compose.yml orchestrates both with shared ChromaDB volume
Both must run with docker-compose up and have a /health endpoint

What to build in this order:

src/utils/data_loader.py — streaming data loader and preprocessor
src/utils/vector_store.py — ChromaDB setup and ingestion
src/models/user_profiler.py — persona builder
src/agents/review_agent.py — Task A LangGraph agent
src/agents/recommendation_agent.py — Task B LangGraph agent
src/api/task_a_api.py — FastAPI app for Task A
src/api/task_b_api.py — FastAPI app for Task B
Dockerfiles and docker-compose

Rules:

Every file must have docstrings and they should not look AI generated, they should look like how senior engineer will
No hardcoded paths — use environment variables via .env
Every endpoint must have error handling — never return a 500 with no message
Log what the agent is doing at each step
The cold-start path must be explicitly tested and must work

Start with src/utils/data_loader.py. Show me the complete file before moving to the next.


an