"""
FastAPI application for the User Modeling and Review Generation Agent (Task A).
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from src.agents.review_agent import ReviewAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TaskA-API")

app = FastAPI(
    title="Task A — User Modeling & Review Generation",
    description=(
        "Generates a personalized Yelp review in a user's voice based on their review history. "
        "Handles cold-start users (empty history) and supports Nigerian cultural mode."
    ),
    version="1.0.0",
)
agent = ReviewAgent()


class UserHistoryItem(BaseModel):
    business_name: str = Field(..., example="Suya Spot")
    category: str = Field(..., example="Nigerian, BBQ")
    stars_given: int = Field(..., ge=1, le=5, example=5)
    review_text: str = Field(..., example="Omo the suya here is too good. The pepper is on point, no be small thing.")


class BusinessMetadata(BaseModel):
    name: str = Field(..., example="Jollof Kitchen")
    category: str = Field(..., example="Nigerian, West African")
    avg_stars: float = Field(..., example=4.2)
    price_range: Optional[str] = Field(None, example="$$")


class ReviewRequest(BaseModel):
    user_id: str = Field(..., example="user_abc123")
    business_id: str = Field(..., example="biz_xyz456")
    user_history: List[UserHistoryItem] = Field(
        ...,
        description="User's past reviews. Pass an empty list for cold-start users."
    )
    business_metadata: BusinessMetadata
    nigerian_mode: bool = Field(
        False,
        description="When true, generates the review with Nigerian English slang and cultural nuance (Omo, Abeg, Chai, etc.)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Warm user — Nigerian mode on",
                    "value": {
                        "user_id": "user_abc123",
                        "business_id": "biz_xyz456",
                        "user_history": [
                            {
                                "business_name": "Suya Spot",
                                "category": "Nigerian, BBQ",
                                "stars_given": 5,
                                "review_text": "Omo the suya here is too good. The pepper is on point, no be small thing."
                            },
                            {
                                "business_name": "Chicken Republic",
                                "category": "Fast Food",
                                "stars_given": 4,
                                "review_text": "Fast service and fresh chicken. Will come back."
                            }
                        ],
                        "business_metadata": {
                            "name": "Jollof Kitchen",
                            "category": "Nigerian, West African",
                            "avg_stars": 4.2,
                            "price_range": "$$"
                        },
                        "nigerian_mode": True
                    }
                },
                {
                    "summary": "Cold-start user — no history",
                    "value": {
                        "user_id": "new_user_999",
                        "business_id": "biz_xyz789",
                        "user_history": [],
                        "business_metadata": {
                            "name": "The Burger Lab",
                            "category": "Burgers, American",
                            "avg_stars": 3.8,
                            "price_range": "$"
                        },
                        "nigerian_mode": False
                    }
                }
            ]
        }
    }


class ReviewResponse(BaseModel):
    predicted_stars: int = Field(..., example=4, description="Predicted star rating (1–5)")
    generated_review: str = Field(..., description="Generated review text in the user's voice")
    confidence: float = Field(..., example=0.87, description="Model confidence score (0.0–1.0)")
    cold_start: bool = Field(..., description="True if the user had no review history")


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "service": "task_a"}


@app.post(
    "/generate-review",
    response_model=ReviewResponse,
    tags=["Task A"],
    summary="Generate a personalized review",
    response_description="Predicted star rating and generated review text in the user's voice",
)
async def generate_review(request: ReviewRequest):
    """
    Builds a user persona from review history, retrieves similar business context from
    ChromaDB, then generates a review and star rating via a 3-node LangGraph agent.

    - **Cold-start**: empty `user_history` is handled gracefully — a baseline persona is used.
    - **nigerian_mode**: adds Nigerian English slang and cultural framing to the output.
    """
    logger.info(f"Received review generation request for user {request.user_id}")

    try:
        input_data = {
            "user_id": request.user_id,
            "business_id": request.business_id,
            "user_history": [
                {
                    "text": item.review_text,
                    "stars": item.stars_given,
                    "business_name": item.business_name,
                    "category": item.category,
                }
                for item in request.user_history
            ],
            "business_metadata": request.business_metadata.model_dump(),
            "nigerian_mode": request.nigerian_mode,
        }

        result = agent.run(input_data)

        return ReviewResponse(
            predicted_stars=result.get("predicted_stars", 4),
            generated_review=result.get("generated_review", ""),
            confidence=result.get("confidence", 0.8),
            cold_start=result.get("is_cold_start", False),
        )

    except Exception as e:
        logger.error(f"Error generating review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
