from src.schemas.state import GraphState
from src.utils.prompts import query_evaluation_prompt
from src.config import LLM, logger

async def evaluation_agent(state: GraphState) -> dict:
    """
    Evaluates if the query is simple or complex.
    Simple queries are answered directly; complex ones go through full orchestration.
    """
    logger.info("⚖️ Evaluation Agent Started")
    
    chain = query_evaluation_prompt | LLM
    evaluation = await chain.ainvoke({"userQuery": state["userQuery"]})
    response = evaluation.content
    
    if response.startswith("SIMPLE:"):
        logger.info("✅ Query evaluated as SIMPLE")
        return {
            "finalResponse": response[7:].strip(),
            "isSimpleQuery": True
        }
    
    logger.info("Query evaluated as COMPLEX")
    return {
        "isSimpleQuery": False
    }
