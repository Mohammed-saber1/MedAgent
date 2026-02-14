from src.schemas.state import GraphState
from src.utils.prompts import (
    compile_agent_prompt,
    compile_without_web_prompt,
    compile_refinement_prompt,
)
from src.config import LLM


async def compile_agent_stream(state: GraphState):
    """
    Streaming variant of the compile agent.
    Yields tokens as they are generated for real-time display.
    Returns the full response string at the end.
    """
    required_agents = state.get("requiredAgents", {})
    med_response = state.get("medILlamaResponse", "")
    web_response = state.get("webSearchResponse", "")
    rag_response = ""

    if state.get("reflectionFeedback"):
        chain = compile_refinement_prompt | LLM
        invoke_args = {
            "previousFinalResponse": state.get("finalResponse", ""),
            "medILlamaResponse": med_response,
            "webSearchResponse": web_response,
            "reflectionFeedback": state.get("reflectionFeedback"),
        }
    else:
        if required_agents.get("medILlama") or (
            web_response and len(web_response) > 50
        ):
            chain = compile_agent_prompt | LLM
        else:
            chain = compile_without_web_prompt | LLM

        invoke_args = {
            "userQuery": state["userQuery"],
            "medILlamaResponse": med_response,
            "webSearchResponse": web_response,
            "ragResponse": rag_response,
        }

    full_response = ""
    async for chunk in chain.astream(invoke_args):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full_response += token
            yield token

    return full_response
