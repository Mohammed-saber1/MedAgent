from src.schemas.state import GraphState
from src.utils.prompts import compile_agent_prompt, compile_without_web_prompt, compile_refinement_prompt
from src.config import LLM, logger

async def compile_agent(state: GraphState) -> dict:
    """
    Compiles the final answer from various agent responses.
    Selects the appropriate prompt based on available information (Web, RAG, etc.).
    """
    logger.info("📝 Compile Agent Started")
    
    required_agents = state.get("requiredAgents", {})
    
    med_response = state.get("medILlamaResponse", "")
    web_response = state.get("webSearchResponse", "")
    rag_response = state.get("ragResponse", "")
    
    try:
        if state.get("reflectionFeedback"):
            logger.info("Using Refinement Prompt")
            chain = compile_refinement_prompt | LLM
            response = await chain.ainvoke({
                "previousFinalResponse": state.get("finalResponse", ""),
                "medILlamaResponse": med_response,
                "webSearchResponse": web_response,
                "reflectionFeedback": state.get("reflectionFeedback")
            })
        else:
            # Choose prompt based on web search requirement/availability
            if required_agents.get("medILlama") or (web_response and len(web_response) > 50):
                 # Use web prompt if web search was required OR if we have substantial web response
                 logger.info("Using Standard Compile Prompt (with Web/Ext Sources)")
                 chain = compile_agent_prompt | LLM
            else:
                 logger.info("Using Compile Without Web Prompt")
                 chain = compile_without_web_prompt | LLM
            
            response = await chain.ainvoke({
                "userQuery": state["userQuery"],
                "medILlamaResponse": med_response,
                "webSearchResponse": web_response,
                "ragResponse": rag_response
            })
            
        logger.info("✅ Compilation Completed")
        return {
            "finalResponse": response.content
        }

    except Exception as e:
        logger.error(f"❌ Compile Error: {e}")
        return {
            "finalResponse": f"Error compiling response: {str(e)}"
        }
