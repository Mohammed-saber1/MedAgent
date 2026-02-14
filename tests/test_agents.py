import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.schemas.state import GraphState
from src.agents.evaluation_agent import evaluation_agent
from src.agents.medillama_agent import medillama_agent
from langchain_core.messages import AIMessage

@pytest.mark.asyncio
async def test_evaluation_agent_simple():
    state: GraphState = {"userQuery": "What is 2+2?", "tasks": {}}
    
    # Patch the PROMPT so when prompt | LLM is called, we return a mock chain
    with patch("src.agents.evaluation_agent.query_evaluation_prompt") as mock_prompt:
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = AIMessage(content="SIMPLE: This is simple.")
        
        # prompt | LLM calls prompt.__or__(LLM)
        mock_prompt.__or__.return_value = mock_chain
        
        result = await evaluation_agent(state)
        
        assert result["isSimpleQuery"] is True
        assert result["finalResponse"] == "This is simple."


@pytest.mark.asyncio
async def test_evaluation_agent_complex():
    state: GraphState = {"userQuery": "Complex medical case", "tasks": {}}
    
    with patch("src.agents.evaluation_agent.query_evaluation_prompt") as mock_prompt:
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = AIMessage(content="COMPLEX: Needing research.")
        mock_prompt.__or__.return_value = mock_chain
        
        result = await evaluation_agent(state)
        
        assert result["isSimpleQuery"] is False


@pytest.mark.asyncio
async def test_medillama_agent():
    state: GraphState = {
        "userQuery": "Diabetes info", 
        "tasks": {"MedILlama": [{"query": "Explain Diabetes"}]}
    }
    
    with patch("src.agents.medillama_agent.medillama_prompt") as mock_prompt:
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = AIMessage(content="Diabetes is high blood sugar.")
        mock_prompt.__or__.return_value = mock_chain
        
        result = await medillama_agent(state)
        
        assert "Diabetes is high blood sugar." in result["medILlamaResponse"]
        assert "Tasks:" in result["medILlamaResponse"]

