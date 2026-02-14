import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ConversationEntry(BaseModel):
    """A single conversation turn (query + response)."""
    query: str
    response: str
    is_simple: bool = False
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Session(BaseModel):
    """Represents a user session with conversation history."""
    id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    history: List[ConversationEntry] = []


class SessionManager:
    """In-memory session store for managing conversation sessions."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self) -> Session:
        """Create a new session and return it."""
        session_id = str(uuid.uuid4())
        session = Session(id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID. Returns None if not found."""
        return self._sessions.get(session_id)

    def add_to_history(
        self,
        session_id: str,
        query: str,
        response: str,
        is_simple: bool = False,
    ) -> None:
        """Add a conversation turn to the session history."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")

        entry = ConversationEntry(
            query=query,
            response=response,
            is_simple=is_simple,
        )
        session.history.append(entry)
        session.updated_at = datetime.now(timezone.utc).isoformat()

    def get_history(
        self, session_id: str
    ) -> List[ConversationEntry]:
        """Get conversation history for a session."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        return session.history

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[Session]:
        """List all active sessions."""
        return list(self._sessions.values())


# Singleton instance
session_manager = SessionManager()
