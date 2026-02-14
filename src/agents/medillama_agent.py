from src.schemas.state import GraphState
from src.utils.prompts import medillama_prompt
from src.config import FINETUNED_MODEL, logger

async def medillama_agent(state: GraphState) -> dict:
    """
    Handles domain-specific medical analysis using the fine-tuned model.
    Consolidates multiple tasks into a single query for efficiency.
    """
    logger.info("🏥 MedILlama Agent Started")
    
    tasks = state.get("tasks", {}).get("MedILlama", [])
    
    combined_queries = ""
    for t in tasks:
        if hasattr(t, 'query'):
            combined_queries += t.query + "\n\n"
        elif isinstance(t, dict):
             combined_queries += t.get('query', '') + "\n\n"
        else:
            combined_queries += str(t) + "\n\n"
            
    combined_queries = combined_queries.strip()
    
    if not combined_queries:
        logger.warning("⚠️ No tasks for MedILlama")
        return {"medILlamaResponse": ""}

    try:
        chain = medillama_prompt | FINETUNED_MODEL
        result = await chain.ainvoke({"query": combined_queries})
        
        full_response = str(result.content)
        formatted_response = f"Tasks:\n{combined_queries}\n\nResponse:\n{full_response}"
        
        return {
            "medILlamaResponse": formatted_response
        }

    except Exception as e:
        logger.error(f"❌ MedILlama error: {e}")
        return {
            "medILlamaResponse": f"Error processing medical queries: {str(e)}"
        }
