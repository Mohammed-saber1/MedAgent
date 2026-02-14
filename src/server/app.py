import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.agent_graph import create_agent_graph
from src.schemas.state import GraphState
from src.session_manager import session_manager

app = FastAPI(title="Medical Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    userQuery: str


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
#  Stateless Query (no session)
# ─────────────────────────────────────────────

@app.post("/api/query")
async def run_query(request: QueryRequest):
    """Execute a one-off query without session tracking."""
    initial_state = _build_initial_state(request.userQuery)
    final_state = await graph.ainvoke(initial_state)
    return {
        "finalResponse": final_state.get("finalResponse"),
        "isSimpleQuery": final_state.get("isSimpleQuery"),
        "qualityPassed": final_state.get("qualityPassed"),
    }


# ─────────────────────────────────────────────
#  Session Management
# ─────────────────────────────────────────────

@app.post("/api/sessions", status_code=201)
async def create_session():
    """Create a new conversation session."""
    session = session_manager.create_session()
    return {
        "sessionId": session.id,
        "createdAt": session.created_at,
        "message": "Session created successfully",
    }


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = session_manager.list_sessions()
    return {
        "sessions": [
            {
                "sessionId": s.id,
                "createdAt": s.created_at,
                "updatedAt": s.updated_at,
                "messageCount": len(s.history),
            }
            for s in sessions
        ],
        "total": len(sessions),
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details and full conversation history."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "sessionId": session.id,
        "createdAt": session.created_at,
        "updatedAt": session.updated_at,
        "history": [
            {
                "query": entry.query,
                "response": entry.response,
                "isSimple": entry.is_simple,
                "timestamp": entry.timestamp,
            }
            for entry in session.history
        ],
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its conversation history."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


# ─────────────────────────────────────────────
#  Session-Based Query
# ─────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/query")
async def run_session_query(session_id: str, request: QueryRequest):
    """Execute a query within a session, preserving conversation history."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build conversation history for context
    history = [
        {"query": entry.query, "response": entry.response}
        for entry in session.history
    ]

    initial_state = _build_initial_state(
        user_query=request.userQuery,
        conversation_history=history,
    )

    final_state = await graph.ainvoke(initial_state)

    final_response = final_state.get("finalResponse", "")
    is_simple = final_state.get("isSimpleQuery", False)

    # Record this turn in the session history
    session_manager.add_to_history(
        session_id=session_id,
        query=request.userQuery,
        response=final_response,
        is_simple=is_simple,
    )

    return {
        "sessionId": session_id,
        "finalResponse": final_response,
        "isSimpleQuery": is_simple,
        "qualityPassed": final_state.get("qualityPassed"),
        "historyLength": len(session.history),
    }


# ─────────────────────────────────────────────
#  WebSocket (legacy, kept for compatibility)
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()

        try:
            payload = json.loads(data)
            user_query = payload.get("userQuery")
        except Exception:
            user_query = data

        if not user_query:
            await websocket.send_json(
                {"type": "error", "message": "No query provided"}
            )
            return

        initial_state = _build_initial_state(user_query)

        async for event in graph.astream(
            initial_state, stream_mode="updates"
        ):
            for node, update in event.items():
                await websocket.send_json(
                    {"type": "state_update", "node": node, "data": update}
                )

        await websocket.send_json(
            {"type": "end", "message": "Workflow complete"}
        )

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json(
                {"type": "error", "message": str(e)}
            )
        except Exception:
            pass
