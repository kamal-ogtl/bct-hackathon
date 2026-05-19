"""
LangGraph agent for personalized business recommendations (Task B).

This agent implements a multi-step reasoning loop to retrieve, score, 
and rank recommendations based on user history and current context.
"""

import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.vector_store import VectorStoreManager
from src.models.user_profiler import UserProfiler
from dotenv import load_dotenv

load_dotenv()

class RecommendationState(TypedDict):
    """
    State for the recommendation reasoning loop.
    """
    user_id: str
    user_history: List[Dict[str, Any]]
    context: Dict[str, Any]
    num_recommendations: int
    # Internal
    persona: str
    candidates: List[Dict[str, Any]]
    scored_candidates: List[Dict[str, Any]]
    final_recommendations: List[Dict[str, Any]]
    reasoning_trace: List[str]
    cold_start: bool


class RecommendationAgent:
    """
    LangGraph agent for Task B: Personalized Recommendations.
    """
    
    def __init__(self):
        self.profiler = UserProfiler()
        self.vector_store = VectorStoreManager()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )
        self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RecommendationState)

        workflow.add_node("build_profile", self.build_profile)
        workflow.add_node("retrieve_candidates", self.retrieve_candidates)
        workflow.add_node("score_and_rank", self.score_and_rank)
        workflow.add_node("generate_explanations", self.generate_explanations)

        workflow.set_entry_point("build_profile")
        workflow.add_edge("build_profile", "retrieve_candidates")
        workflow.add_edge("retrieve_candidates", "score_and_rank")
        workflow.add_edge("score_and_rank", "generate_explanations")
        workflow.add_edge("generate_explanations", END)

        self.graph = workflow.compile()

    def build_profile(self, state: RecommendationState) -> Dict[str, Any]:
        """
        Extracts user preferences.
        """
        print(f"[RecAgent] Building profile for {state['user_id']}")
        result = self.profiler.generate_persona(state['user_id'], state['user_history'])
        return {
            "persona": result['persona_summary'],
            "cold_start": result['is_cold_start'],
            "reasoning_trace": ["Profile built from history."] if not result['is_cold_start'] else ["Cold start detected."]
        }

    def retrieve_candidates(self, state: RecommendationState) -> Dict[str, Any]:
        """
        Retrieves candidates from ChromaDB based on persona and context.
        """
        print(f"[RecAgent] Retrieving candidates for mood: {state['context'].get('mood')}")
        
        # Build query string combining history/persona and current mood/context
        query = f"User preferences: {state['persona']}. Current mood: {state['context'].get('mood', 'general')}. Location: {state['context'].get('location_preference', 'anywhere')}"
        
        results = self.vector_store.query_businesses(query, n_results=state['num_recommendations'] * 2)
        
        candidates = []
        for r in results:
            candidates.append({
                "business_id": r['id'],
                "name": r['metadata']['name'],
                "categories": r['metadata']['categories'],
                "avg_stars": r['metadata']['stars'],
                "description": r['document']
            })
            
        trace = state['reasoning_trace'] + [f"Retrieved {len(candidates)} candidates from vector store."]
        return {"candidates": candidates, "reasoning_trace": trace}

    def score_and_rank(self, state: RecommendationState) -> Dict[str, Any]:
        """
        Uses LLM to score and rank candidates based on user alignment.
        """
        print("[RecAgent] Scoring candidates...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a recommendation engine. Rank the following businesses for a user based on their persona and current context.\n"
                "USER PERSONA:\n{persona}\n\n"
                "CONTEXT:\nTime: {time}, Mood: {mood}, Location: {loc}\n\n"
                "CANDIDATES (use the number as 'index' in your response — do NOT modify it):\n{candidates_text}\n"
                "\nRank them by alignment. For each, output a JSON object with: "
                "'index' (the candidate number), 'predicted_rating' (0-5 float), 'reason' (one sentence). "
                "Output a JSON array only, no extra text."
            )),
        ])

        # Use 1-based index labels — LLM reliably echoes small integers
        cand_text = "\n".join([
            f"{i+1}. {c['name']} | Category: {c['categories']}"
            for i, c in enumerate(state['candidates'])
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "persona": state['persona'],
            "time": state['context'].get('time_of_day'),
            "mood": state['context'].get('mood'),
            "loc": state['context'].get('location_preference'),
            "candidates_text": cand_text
        })

        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()

        import json
        try:
            scores = json.loads(content)
            scores.sort(key=lambda x: x.get('predicted_rating', 0), reverse=True)
            # Resolve index → actual candidate
            resolved = []
            for s in scores[:state['num_recommendations']]:
                idx = s.get('index')
                if idx is not None and 1 <= idx <= len(state['candidates']):
                    c = state['candidates'][idx - 1]
                    resolved.append({
                        "business_id": c['business_id'],
                        "predicted_rating": s['predicted_rating'],
                        "reason": s['reason'],
                    })
            trace = state['reasoning_trace'] + ["Candidates scored and ranked via LLM reasoning."]
            return {"scored_candidates": resolved, "reasoning_trace": trace}
        except json.JSONDecodeError:
            print("[RecAgent] Warning: LLM returned non-JSON during scoring. Falling back to retrieval order.")
            trace = state['reasoning_trace'] + ["Scoring parse failed. Fell back to retrieval order."]
            fallback = [
                {"business_id": c['business_id'], "predicted_rating": c['avg_stars'], "reason": "Retrieved via semantic similarity."}
                for c in state['candidates'][:state['num_recommendations']]
            ]
            return {"scored_candidates": fallback, "reasoning_trace": trace}

    def generate_explanations(self, state: RecommendationState) -> Dict[str, Any]:
        """
        Finalizes the recommendation list.
        """
        print("[RecAgent] Finalizing recommendations...")
        
        cand_map = {c['business_id']: c for c in state['candidates']}
        total = len(state['scored_candidates'])

        final_recs = []
        for rank, s in enumerate(state['scored_candidates']):
            b_id = s['business_id']
            cand = cand_map.get(b_id)
            if not cand:
                continue
            confidence = round(0.95 - (rank / max(total - 1, 1)) * 0.35, 2) if total > 1 else 0.95
            final_recs.append({
                "business_id": b_id,
                "business_name": cand['name'],
                "category": cand['categories'],
                "predicted_rating": s['predicted_rating'],
                "reason": s['reason'],
                "confidence": confidence,
            })
        
        trace = state['reasoning_trace'] + ["Final recommendations generated."]
        return {"final_recommendations": final_recs, "reasoning_trace": trace}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the recommendation agent.
        """
        initial_state: RecommendationState = {
            "user_id": input_data.get("user_id", "unknown"),
            "user_history": input_data.get("user_history", []),
            "context": input_data.get("context", {}),
            "num_recommendations": input_data.get("num_recommendations", 5),
            "persona": "",
            "candidates": [],
            "scored_candidates": [],
            "final_recommendations": [],
            "reasoning_trace": [],
            "cold_start": False
        }
        
        result = self.graph.invoke(initial_state)
        return {
            "recommendations": result['final_recommendations'],
            "cold_start": result['cold_start'],
            "reasoning_trace": " -> ".join(result['reasoning_trace'])
        }

if __name__ == "__main__":
    # Test Task B
    agent = RecommendationAgent()
    test_input = {
        "user_id": "u2",
        "user_history": [{"business_name": "Cafe Roma", "category": "Italian", "stars_given": 5}],
        "context": {"time_of_day": "morning", "mood": "working", "location_preference": "quiet"},
        "num_recommendations": 3
    }
    result = agent.run(test_input)
    print(f"Recommendations: {len(result['recommendations'])}")
    print(f"Trace: {result['reasoning_trace']}")
