from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class Task(BaseModel):
    query: str = Field(..., description="The specific sub-query to be answered.")

class TasksByType(BaseModel):
    MedILlama: Optional[List[Task]] = Field(default=None, description="Tasks for MedILlama.")
    Web: Optional[List[Task]] = Field(default=None, description="Tasks for Web Search Agent.")
    # RAG: Optional[List[Task]] = Field(default=None, description="Tasks for RAG Database Search Agent.")

class RequiredAgents(BaseModel):
    medILlama: bool = Field(..., description="Whether MedILlama is required.")
    webSearch: bool = Field(..., description="Whether Web Search is required.")
    # rag: bool = Field(..., description="Whether RAG is required.")

class DecompositionOutput(BaseModel):
    tasks: TasksByType = Field(..., description="Tasks grouped by type.")
    requiredAgents: RequiredAgents = Field(..., description="Required agents for the query.")
