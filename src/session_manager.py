import uuid
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

from src.config import settings, logger

from src.schemas.models import ConversationEntry


class SessionManager:
    """MongoDB-backed session store for conversation history."""

    def __init__(self):
        self._client = AsyncIOMotorClient(settings.MONGODB_URI)
        self._db = self._client[settings.MONGODB_DB_NAME]
        self._col = self._db["sessions"]
        logger.info(f"SessionManager initialized with DB: {settings.MONGODB_DB_NAME}")

    async def create_session(
        self, session_id: Optional[str] = None
    ) -> dict:
        """Create a new session document in MongoDB."""
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
        logger.debug(f"Created new session: {session_id}")
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
        logger.debug(f"Added interaction to history for session: {session_id}")

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        result = await self._col.delete_one({"_id": session_id})
        return result.deleted_count > 0

    def close(self):
        """Close the MongoDB client."""
        self._client.close()


# Singleton instance
session_manager = SessionManager()
