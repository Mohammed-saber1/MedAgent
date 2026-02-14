from src.schemas.state import GraphState
from src.utils.prompts import compile_agent_prompt, compile_without_web_prompt, compile_refinement_prompt
from src.config import LLM

async def compile_agent(state: GraphState) -> dict:
    """
    Compiles the final answer from various agent responses.
    """
    print("\n📝 Compile Agent Started")
    
    required_agents = state.get("requiredAgents", {})
    
    # Check if we have all necessary responses (basic check)
    # LangGraph usually handles dependencies via edges, but we can double check content
    
    med_response = state.get("medILlamaResponse", "")
    web_response = state.get("webSearchResponse", "")
    rag_response = "" # Stub for now
    
    try:
        if state.get("reflectionFeedback"):
            print("Using Refinement Prompt")
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
                 print("Using Standard Compile Prompt (with Web/Ext Sources)")
                 chain = compile_agent_prompt | LLM
            else:
                 print("Using Compile Without Web Prompt")
                 chain = compile_without_web_prompt | LLM
            
            response = await chain.ainvoke({
                "userQuery": state["userQuery"],
                "medILlamaResponse": med_response,
                "webSearchResponse": web_response,
                "ragResponse": rag_response
            })
            
        print("\n✅ Compilation Completed")
        return {
            "finalResponse": response.content
        }

    except Exception as e:
        print(f"❌ Compile Error: {e}")
        return {
            "finalResponse": f"Error compiling response: {str(e)}"
        }
