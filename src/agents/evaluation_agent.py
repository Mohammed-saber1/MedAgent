from src.schemas.state import GraphState
from src.utils.prompts import query_evaluation_prompt
from src.config import LLM

async def evaluation_agent(state: GraphState) -> dict:
    """
    Evaluates if the query is simple or complex.
    """
    print("\n⚖️ Evaluation Agent Started")
    
    chain = query_evaluation_prompt | LLM
    evaluation = await chain.ainvoke({"userQuery": state["userQuery"]})
    response = evaluation.content
    
    if response.startswith("SIMPLE:"):
        print("✅ Query evaluated as SIMPLE")
        return {
            "finalResponse": response[7:].strip(),
            "isSimpleQuery": True
        }
    
    print("Example: Query evaluated as COMPLEX")
    return {
        "isSimpleQuery": False
    }
