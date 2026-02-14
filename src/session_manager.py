import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "medagent")


class ConversationEntry(BaseModel):
    """A single conversation turn (query + response)."""
    query: str
    response: str
    is_simple: bool = False
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SessionManager:
    """MongoDB-backed session store for conversation history."""

    def __init__(self):
        self._client = AsyncIOMotorClient(MONGODB_URI)
        self._db = self._client[MONGODB_DB_NAME]
        self._col = self._db["sessions"]

    async def create_session(
        self, session_id: Optional[str] = None
    ) -> dict:
        """Create a new session document in MongoDB."""
        import uuid
        if session_id is None:
            session_id = str(uuid.uuid4())

        now = datetime.now(timezone.utc).isoformat()
        doc = {
            "_id": session_id,
            "created_at": now,
            "updated_at": now,
            "history": [],
        }
        await self._col.insert_one(doc)
        return doc

    async def get_session(
        self, session_id: str
    ) -> Optional[dict]:
        """Get a session by ID. Returns None if not found."""
        return await self._col.find_one({"_id": session_id})

    async def add_to_history(
        self,
        session_id: str,
        query: str,
        response: str,
        is_simple: bool = False,
    ) -> None:
        """Append a conversation turn to the session history."""
        entry = {
            "query": query,
            "response": response,
            "is_simple": is_simple,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._col.update_one(
            {"_id": session_id},
            {
                "$push": {"history": entry},
                "$set": {
                    "updated_at": datetime.now(timezone.utc).isoformat()
                },
            },
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        result = await self._col.delete_one({"_id": session_id})
        return result.deleted_count > 0


# Singleton instance
session_manager = SessionManager()
