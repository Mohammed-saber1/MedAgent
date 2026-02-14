from src.schemas.state import GraphState, OrchestrationData
from src.schemas.decomposition import DecompositionOutput
from src.utils.prompts import task_decomposition_prompt, improvement_prompt
from src.config import LLM, logger
# MAX_ITERATIONS used to be in config, now handling locally or via settings if needed. 
# Assuming settings.MAX_ITERATIONS or local logic.
# The original code imported MAX_ITERATIONS from config.
MAX_ITERATIONS = 3 # Hardcoding or retrieving from settings if available in config.py

async def orchestrate_query(state: GraphState) -> dict:
    """
    Orchestrates the query by decomposing it into tasks for other agents.
    Handles both initial decomposition and iterative improvement.
    """
    logger.info("🎵 Orchestration Agent Started")
    
    # Reset accumulated responses for a new iteration
    med_response = ""
    web_response = ""
    
    # Check for reflection feedback
    iteration_count = state.get("iterationCount", 0)
    quality_passed = state.get("qualityPassed", True)
    
    if not quality_passed and iteration_count <= MAX_ITERATIONS:
        logger.warning(f"⚠️ Quality check failed. Reflection feedback: {state.get('reflectionFeedback')}")
        
        # Use improvement prompt
        llm_with_structured = LLM.with_structured_output(DecompositionOutput)
        chain = improvement_prompt | llm_with_structured
        
        improved_decomposition = await chain.ainvoke({
            "previousResponse": state.get("finalResponse"),
            "improvementFeedback": state.get("reflectionFeedback"),
            "userQuery": state["userQuery"]
        })
        
        orchestration_data = OrchestrationData(
            requiredAgents=improved_decomposition.requiredAgents.model_dump(),
            reasoning=f"Improvement based on feedback: {state.get('reflectionFeedback')}",
            plan=f"Revised plan for iteration {iteration_count + 1}"
        )
        
        tasks_mapping = {
            "MedILlama": improved_decomposition.tasks.MedILlama or [],
            "WebSearch": improved_decomposition.tasks.Web or []
            # "RAG": improved_decomposition.tasks.RAG or []
        }
        
        return {
            "orchestrationData": orchestration_data,
            "tasks": tasks_mapping,
            "requiredAgents": improved_decomposition.requiredAgents.model_dump(),
            "medILlamaResponse": med_response, # Reset
            "webSearchResponse": web_response  # Reset
        }
        
    # Initial Decomposition
    llm_with_structured = LLM.with_structured_output(DecompositionOutput)
    chain = task_decomposition_prompt | llm_with_structured
    
    initial_decomposition = await chain.ainvoke({"userQuery": state["userQuery"]})
    
    orchestration_data = OrchestrationData(
        requiredAgents=initial_decomposition.requiredAgents.model_dump(),
        reasoning=f"Initial analysis of query: {state['userQuery']}",
        plan="Execution plan generated."
    )
    
    tasks_mapping = {
        "MedILlama": initial_decomposition.tasks.MedILlama or [],
        "WebSearch": initial_decomposition.tasks.Web or []
        # "RAG": initial_decomposition.tasks.RAG or []
    }
    
    return {
        "orchestrationData": orchestration_data,
        "tasks": tasks_mapping,
        "requiredAgents": initial_decomposition.requiredAgents.model_dump(),
        "medILlamaResponse": med_response, # Reset
        "webSearchResponse": web_response  # Reset
    }
