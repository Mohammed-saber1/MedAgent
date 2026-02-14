from src.schemas.state import GraphState
from src.utils.prompts import medillama_prompt
from src.config import FINETUNED_MODEL

async def medillama_agent(state: GraphState) -> dict:
    """
    Handles domain-specific medical analysis using the fine-tuned model.
    """
    print("\n🏥 MedILlama Agent Started")
    
    tasks = state.get("tasks", {}).get("MedILlama", [])
    
    # Combine all task queries
    # tasks is a list of Task objects (from Pydantic schema) or dictionaries?
    # Using .with_structured_output usually returns Pydantic objects.
    
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
        print("⚠️ No tasks for MedILlama")
        return {"medILlamaResponse": ""}

    try:
        chain = medillama_prompt | FINETUNED_MODEL
        
        # Invoke (no streaming for now in this version, or standard invoke)
        result = await chain.ainvoke({"query": combined_queries})
        
        full_response = str(result.content)
        
        # Format response (mapping back to individual tasks logic is simplified here to one big block 
        # as per original implementation's core logic which joins them anyway)
        
        formatted_response = f"Tasks:\n{combined_queries}\n\nResponse:\n{full_response}"
        
        # return {"medILlamaResponse": formatted_response} 
        # Original TS implementation updates medILlamaResponse global variable AND state.
        # We will update state.
        
        return {
            "medILlamaResponse": formatted_response
        }

    except Exception as e:
        print(f"❌ MedILlama error: {e}")
        return {
            "medILlamaResponse": f"Error processing medical queries: {str(e)}"
        }
