from typing import List, Optional, Any, Dict, TypedDict, Annotated
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

# Define Pydantic models for structured parts of the state

class WebSearchResultItem(BaseModel):
    url: str
    title: str
    content: Optional[str] = None

class WebSearchResult(BaseModel):
    query: str
    results: List[WebSearchResultItem]

class RequiredAgents(BaseModel):
    medILlama: bool = False
    webSearch: bool = False
    rag: bool = False

class OrchestrationData(BaseModel):
    requiredAgents: RequiredAgents
    reasoning: Optional[str] = None
    plan: Optional[str] = None

# Define the GraphState using TypedDict
# Annotations are used for reducer functions (like expanding lists)

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    userQuery: str
    conversationHistory: Optional[List[Dict[str, str]]]
    tasks: Dict[str, Any]
    medILlamaResponse: str
    webSearchResponse: str
    webSearchResults: Optional[List[WebSearchResult]]
    finalResponse: str
    isSimpleQuery: bool
    iterationCount: int
    reflectionFeedback: Optional[str]
    qualityPassed: bool
    requiredAgents: Optional[RequiredAgents]
    orchestrationData: Optional[OrchestrationData]
