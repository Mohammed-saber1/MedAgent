from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.agent_graph import create_agent_graph
from src.session_manager import session_manager

app = FastAPI(title="Medical Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    sessionId: str
    message: str


# Global graph instance
graph = create_agent_graph()


def _build_initial_state(
    user_query: str,
    conversation_history: Optional[list] = None,
) -> dict:
    """Build the initial state dict for the agent graph."""
    return {
        "userQuery": user_query,
        "messages": [],
        "conversationHistory": conversation_history or [],
        "tasks": {},
        "medILlamaResponse": "",
        "webSearchResponse": "",
        "finalResponse": "",
        "iterationCount": 0,
        "qualityPassed": True,
        "requiredAgents": {
            "medILlama": False,
            "webSearch": False,
            "rag": False,
        },
        "isSimpleQuery": False,
    }


# ─────────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ─────────────────────────────────────────────
#  Chat  (single endpoint for all queries)
# ─────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Send a message within a session.
    Auto-creates the session if the given sessionId does not exist.
    """
    session = await session_manager.get_session(request.sessionId)

    # Auto-create session when it doesn't exist
    if session is None:
        session = await session_manager.create_session(request.sessionId)

    # Build conversation history for context
    history = [
        {"query": entry["query"], "response": entry["response"]}
        for entry in session.get("history", [])
    ]

    initial_state = _build_initial_state(
        user_query=request.message,
        conversation_history=history,
    )

    final_state = await graph.ainvoke(initial_state)

    final_response = final_state.get("finalResponse", "")
    is_simple = final_state.get("isSimpleQuery", False)

    # Record this turn in the session history
    await session_manager.add_to_history(
        session_id=request.sessionId,
        query=request.message,
        response=final_response,
        is_simple=is_simple,
    )

    return {
        "sessionId": request.sessionId,
        "response": final_response,
        "isSimpleQuery": is_simple,
        "qualityPassed": final_state.get("qualityPassed"),
    }


# ─────────────────────────────────────────────
#  Delete Session
# ─────────────────────────────────────────────

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its conversation history."""
    deleted = await session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail="Session not found"
        )
    return {"message": "Session deleted successfully"}
