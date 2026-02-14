from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# ─────────────────────────────────────────────
#  API Models
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    sessionId: str
    message: str

# ─────────────────────────────────────────────
#  Session Models
# ─────────────────────────────────────────────

class ConversationEntry(BaseModel):
    """A single conversation turn (query + response)."""
    query: str
    response: str
    is_simple: bool = False
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

# ─────────────────────────────────────────────
#  Agent Models
# ─────────────────────────────────────────────

class ReflectionOutput(BaseModel):
    qualityPassed: bool
    feedback: Optional[str] = None
