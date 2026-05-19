"""
FastAPI application for the Recommendation Agent (Task B).
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from src.agents.recommendation_agent import RecommendationAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TaskB-API")

app = FastAPI(
    title="Task B — Personalised Recommendation Agent",
    description=(
        "Returns ranked business recommendations tailored to the user's history and current context. "
        "Handles cold-start users, cross-domain shifts, and exposes a reasoning trace for interpretability."
    ),
    version="1.0.0",
)
agent = RecommendationAgent()


class UserHistoryItem(BaseModel):
    business_name: str = Field(..., example="Mama Cass")
    category: str = Field(..., example="Nigerian")
    stars_given: int = Field(..., ge=1, le=5, example=5)


class RecommendationContext(BaseModel):
    time_of_day: str = Field(..., example="evening")
    mood: str = Field(..., example="casual dining with friends")
    location_preference: str = Field(..., example="nearby")


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., example="user_abc123")
    user_history: List[UserHistoryItem] = Field(
        ...,
        description="User's past interactions. Pass an empty list for cold-start — context fields are used instead."
    )
    context: RecommendationContext
    num_recommendations: int = Field(5, ge=1, le=20, example=5)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Warm user — evening casual dining",
                    "value": {
                        "user_id": "user_abc123",
                        "user_history": [
                            {"business_name": "Suya Spot", "category": "Nigerian, BBQ", "stars_given": 5},
                            {"business_name": "Mama Cass", "category": "Nigerian", "stars_given": 4}
                        ],
                        "context": {
                            "time_of_day": "evening",
                            "mood": "casual dining with friends",
                            "location_preference": "nearby"
                        },
                        "num_recommendations": 5
                    }
                },
                {
                    "summary": "Cold-start user — context-only retrieval",
                    "value": {
                        "user_id": "new_user_999",
                        "user_history": [],
                        "context": {
                            "time_of_day": "morning",
                            "mood": "quick coffee and work",
                            "location_preference": "quiet"
                        },
                        "num_recommendations": 3
                    }
                },
                {
                    "summary": "Cross-domain — Italian history, coffee mood",
                    "value": {
                        "user_id": "user_xyz789",
                        "user_history": [
                            {"business_name": "Pasta Roma", "category": "Italian", "stars_given": 5},
                            {"business_name": "Bella Napoli", "category": "Italian, Pizza", "stars_given": 4}
                        ],
                        "context": {
                            "time_of_day": "morning",
                            "mood": "working alone, need coffee",
                            "location_preference": "quiet cafe"
                        },
                        "num_recommendations": 3
                    }
                }
            ]
        }
    }


class RecommendationItem(BaseModel):
    business_id: str = Field(..., example="abc123xyz")
    business_name: str = Field(..., example="Kumo Sushi & Asian Bistro")
    category: str = Field(..., example="Restaurants, Sushi Bars, Japanese")
    predicted_rating: float = Field(..., example=4.5)
    reason: str = Field(..., example="Matches user preference for casual group dining with high-rated Asian cuisine.")
    confidence: float = Field(..., example=0.95)


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    cold_start: bool = Field(..., description="True if the user had no review history")
    reasoning_trace: str = Field(..., description="Step-by-step agent reasoning log")


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "service": "task_b"}


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Task B"],
    summary="Get personalised business recommendations",
    response_description="Ranked list of businesses with predicted ratings, reasons, and confidence scores",
)
async def recommend(request: RecommendationRequest):
    """
    Runs a 4-node LangGraph agent:
    1. **build_profile** — extracts user persona from history
    2. **retrieve_candidates** — semantic search via ChromaDB using persona + context
    3. **score_and_rank** — GPT-4o-mini ranks candidates by user alignment
    4. **generate_explanations** — assembles final list with confidence decay by rank

    - **Cold-start**: no history → context fields drive retrieval. Never returns empty.
    - **Cross-domain**: history in one category + context in another → combined query handles the shift.
    - **reasoning_trace**: exposes what the agent considered at each step.
    """
    logger.info(f"Received recommendation request for user {request.user_id}")

    try:
        input_data = {
            "user_id": request.user_id,
            "user_history": [
                {
                    "business_name": item.business_name,
                    "category": item.category,
                    "stars": item.stars_given,
                }
                for item in request.user_history
            ],
            "context": request.context.model_dump(),
            "num_recommendations": request.num_recommendations,
        }

        result = agent.run(input_data)

        return RecommendationResponse(
            recommendations=result["recommendations"],
            cold_start=result["cold_start"],
            reasoning_trace=result["reasoning_trace"],
        )

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
