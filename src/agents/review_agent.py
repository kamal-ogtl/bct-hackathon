"""
LangGraph agent for generating user-aligned reviews (Task A).

This module implements the logic to retrieve user personas and similar 
business context to generate authentic-sounding reviews.
"""

import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.models.user_profiler import UserProfiler
from src.utils.vector_store import VectorStoreManager
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    """
    State maintained across the agentic graph.
    """
    user_id: str
    business_id: str
    business_metadata: Dict[str, Any]
    user_history: List[Dict[str, Any]]
    nigerian_mode: bool
    # Internal working fields
    persona: str
    rating_stats: Dict[str, Any]
    similar_context: str
    predicted_stars: int
    generated_review: str
    confidence: float
    is_cold_start: bool


class ReviewAgent:
    """
    LangGraph agent for Task A: User Modeling and Review Generation.
    """
    
    def __init__(self):
        self.profiler = UserProfiler()
        self.vector_store = VectorStoreManager()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7
        )
        self._build_graph()

    def _build_graph(self):
        """
        Constructs the LangGraph for review generation.
        """
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("profile_user", self.profile_user)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_review", self.generate_review)

        # Define edges
        workflow.set_entry_point("profile_user")
        workflow.add_edge("profile_user", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_review")
        workflow.add_edge("generate_review", END)

        self.graph = workflow.compile()

    def profile_user(self, state: AgentState) -> Dict[str, Any]:
        """
        Node: Builds or retrieves the user persona.
        """
        print(f"[Agent] Profiling user: {state['user_id']}")
        result = self.profiler.generate_persona(state['user_id'], state['user_history'])
        return {
            "persona": result['persona_summary'],
            "rating_stats": result['rating_stats'],
            "is_cold_start": result['is_cold_start']
        }

    def retrieve_context(self, state: AgentState) -> Dict[str, Any]:
        """
        Node: Retrieves similar businesses to understand user's preferences in context.
        """
        print(f"[Agent] Retrieving context for business: {state['business_metadata'].get('name')}")
        query = f"Business category: {state['business_metadata'].get('category')}. "
        results = self.vector_store.query_businesses(query, n_results=5)
        
        context_str = "\n".join([r['document'] for r in results])
        return {"similar_context": context_str}

    def generate_review(self, state: AgentState) -> Dict[str, Any]:
        """
        Node: Final generation of the review.
        """
        print("[Agent] Generating final review...")
        
        nigerian_instruction = ""
        if state.get('nigerian_mode'):
            nigerian_instruction = (
                "\nADDITIONAL INSTRUCTION: Use Nigerian English nuances, slang, and cultural context. "
                "The review should sound like it's from a Nigerian living in the US or Nigeria, "
                "using terms like 'Omo', 'Abeg', 'Chai', or referring to service quality in a Nigerian way."
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AI that writes Yelp reviews in the specific voice of a user.\n"
                "USER PERSONA:\n{persona}\n\n"
                "SIMILAR BUSINESS CONTEXT:\n{context}\n\n"
                "TARGET BUSINESS:\nName: {target_name}\nCategory: {target_cat}\nAvg Stars: {target_stars}\n"
                "{nigerian_instruction}\n"
                "\nGenerate a review that matches the user's voice and typical rating pattern. "
                "Output JSON format with: 'predicted_stars' (1-5), 'generated_review', and 'confidence' (0.0-1.0)."
            )),
            ("user", "Write a review for {target_name}.")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "persona": state['persona'],
            "context": state['similar_context'],
            "target_name": state['business_metadata'].get('name'),
            "target_cat": state['business_metadata'].get('category'),
            "target_stars": state['business_metadata'].get('avg_stars'),
            "nigerian_instruction": nigerian_instruction
        })

        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()

        import json
        try:
            data = json.loads(content)
            return {
                "predicted_stars": data.get("predicted_stars", 4),
                "generated_review": data.get("generated_review", ""),
                "confidence": data.get("confidence", 0.8)
            }
        except json.JSONDecodeError:
            print("[Agent] Warning: LLM returned non-JSON. Using raw content as review text.")
            return {
                "predicted_stars": 4,
                "generated_review": content,
                "confidence": 0.5
            }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent graph.
        """
        # Initialize state
        initial_state: AgentState = {
            "user_id": input_data.get("user_id", "unknown"),
            "business_id": input_data.get("business_id", "unknown"),
            "business_metadata": input_data.get("business_metadata", {}),
            "user_history": input_data.get("user_history", []),
            "nigerian_mode": input_data.get("nigerian_mode", False),
            "persona": "",
            "rating_stats": {},
            "similar_context": "",
            "predicted_stars": 0,
            "generated_review": "",
            "confidence": 0.0,
            "is_cold_start": False
        }
        
        return self.graph.invoke(initial_state)

if __name__ == "__main__":
    # Test Task A
    agent = ReviewAgent()
    test_input = {
        "user_id": "u1",
        "business_id": "b1",
        "user_history": [{"text": "Great food!", "stars": 5}],
        "business_metadata": {"name": "Test Grill", "category": "Steakhouse", "avg_stars": 4.5},
        "nigerian_mode": True
    }
    result = agent.run(test_input)
    print(f"Generated Review: {result['generated_review']}")
