"""
User profiling logic using OpenAI GPT.

This module analyzes user review history to extract consistent personas,
preferences, and behavioral patterns to guide RAG-based generation.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class UserProfiler:
    """
    Analyzes user history to build descriptive personas.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the LLM for persona generation.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=0.2
        )

    def extract_rating_stats(self, review_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes numerical rating statistics from review history.
        These act as hard anchors for star prediction, keeping the LLM calibrated.
        """
        if not review_history:
            return {"avg_stars": 3.5, "category_avgs": {}, "min_stars": 1, "max_stars": 5}

        stars = [r.get("stars", 3) for r in review_history if r.get("stars") is not None]
        avg = round(sum(stars) / len(stars), 2) if stars else 3.5

        # Per-category averages — useful when target business matches a known category
        category_stars: Dict[str, List[float]] = {}
        for r in review_history:
            cat = r.get("category", "").lower().strip()
            s = r.get("stars")
            if cat and s is not None:
                category_stars.setdefault(cat, []).append(s)

        category_avgs = {
            cat: round(sum(vals) / len(vals), 2)
            for cat, vals in category_stars.items()
        }

        return {
            "avg_stars": avg,
            "category_avgs": category_avgs,
            "min_stars": min(stars) if stars else 1,
            "max_stars": max(stars) if stars else 5,
        }

    def generate_persona(self, user_id: str, review_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts a persona from a list of user reviews.

        Args:
            user_id: The ID of the user.
            review_history: List of dictionaries containing 'stars', 'text', 'business_name', etc.

        Returns:
            A dictionary containing the generated persona text, rating stats, and cold-start flag.
        """
        if not review_history:
            return self._generate_cold_start_persona(user_id)

        rating_stats = self.extract_rating_stats(review_history)

        history_text = ""
        for i, rev in enumerate(review_history[:10]):
            history_text += f"Review {i+1} ({rev.get('stars')} stars for {rev.get('business_name', 'Unknown')}):\n"
            history_text += f"\"{rev.get('text', '')}\"\n\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert behavioral analyst. Your task is to analyze a user's Yelp review history "
                "and build a detailed 'Reviewer Persona'. Focus on:\n"
                "1. Tone and Voice: Is it formal, casual, sarcastic, enthusiastic, or complainy?\n"
                "2. Vocabulary: Do they use specific slang, industry terms, or simple language?\n"
                "3. Priorities: What do they mention most? (Service, price, food quality, speed, ambiance)\n"
                "4. Rating Pattern: Are they a tough grader or easily impressed?\n"
                "\nReturn a concise 2-3 paragraph summary of this persona."
            )),
            ("user", "Here is the review history for user {user_id}:\n\n{history}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"user_id": user_id, "history": history_text})

        return {
            "user_id": user_id,
            "persona_summary": response.content,
            "rating_stats": rating_stats,
            "is_cold_start": False
        }

    def _generate_cold_start_persona(self, user_id: str) -> Dict[str, Any]:
        """
        Generates a baseline persona for a user with no history.
        """
        # Baseline persona: generic but polite, focuses on value and clarity.
        # In a real app, we might use user demographic info if available.
        persona = (
            "This is a new user with no review history. They appear to be a balanced reviewer "
            "who values clear communication and standard service quality. They are likely "
            "looking for reliable experiences and will provide straightforward feedback."
        )
        return {
            "user_id": user_id,
            "persona_summary": persona,
            "rating_stats": {"avg_stars": 3.5, "category_avgs": {}, "min_stars": 1, "max_stars": 5},
            "is_cold_start": True
        }

if __name__ == "__main__":
    # Test with dummy data
    try:
        profiler = UserProfiler()
        sample_reviews = [
            {"stars": 5, "text": "Absolutely loved the vibe here! The pasta was al dente and the server was so attentive.", "business_name": "Pasta Place"},
            {"stars": 2, "text": "Wait time was ridiculous. Food was okay but not worth 45 mins of standing.", "business_name": "Burger Joint"}
        ]
        result = profiler.generate_persona("test_user_123", sample_reviews)
        print(f"Persona: {result['persona_summary']}")
    except Exception as e:
        print(f"Error testing UserProfiler: {e}")
