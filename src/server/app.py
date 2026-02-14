from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.agent_graph import create_agent_graph
from src.session_manager import SessionManager
from src.config import logger

app = FastAPI(
    title="MedAgent API",
    description="Backend API for the MedAgent Medical Research Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.schemas.models import ChatRequest

# Global instances
# Note: creating graph and session manager at module level
graph = create_agent_graph()
session_manager = SessionManager()

def _build_initial_state(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
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
    """Health check endpoint."""
    return {"status": "ok"}


# ─────────────────────────────────────────────
#  Chat
# ─────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Send a message within a session.
    Auto-creates the session if the given sessionId does not exist.
    """
    logger.info(f"Received chat request for session {request.sessionId}")
    
    try:
        session = await session_manager.get_session(request.sessionId)
        if session is None:
            logger.info(f"Creating new session {request.sessionId}")
            await session_manager.create_session(request.sessionId)
            session = await session_manager.get_session(request.sessionId)

        # Build conversation history
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
        # Fallback if empty (shouldn't happen with proper agents)
        if not final_response and final_state.get("medILlamaResponse"):
             final_response = final_state.get("medILlamaResponse")

        is_simple = final_state.get("isSimpleQuery", False)

        # Record this turn
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
            "qualityPassed": final_state.get("qualityPassed", True),
        }

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  Session Management
# ─────────────────────────────────────────────

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its conversation history."""
    logger.info(f"Deleting session {session_id}")
    deleted = await session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail="Session not found"
        )
    return {"message": "Session deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
