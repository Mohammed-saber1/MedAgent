from typing import Literal, List
from langgraph.graph import StateGraph, END

from src.schemas.state import GraphState
from src.agents.evaluation_agent import evaluation_agent
from src.agents.orchestration_agent import orchestrate_query
from src.agents.medillama_agent import medillama_agent
from src.agents.web_search_agent import web_search_agent
from src.agents.compile_agent import compile_agent
from src.agents.reflection_agent import reflection_agent
from src.agents.pubmed_rag_agent import pubmed_rag_agent
from src.config import MAX_ITERATIONS

def create_agent_graph():
    """
    Constructs the LangGraph workflow.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("evaluate", evaluation_agent)
    workflow.add_node("orchestrate", orchestrate_query)
    workflow.add_node("med_illama", medillama_agent)
    workflow.add_node("web_search", web_search_agent)
    workflow.add_node("pubmed_rag", pubmed_rag_agent)
    workflow.add_node("compile", compile_agent)
    workflow.add_node("reflect", reflection_agent)

    # Define edges
    workflow.set_entry_point("evaluate")

    # Conditional edge from Evaluate
    def evaluate_condition(state):
        if state.get("isSimpleQuery"):
            return "end"
        return "orchestrate"

    workflow.add_conditional_edges(
        "evaluate",
        evaluate_condition,
        {
            "end": END,
            "orchestrate": "orchestrate"
        }
    )

    # Conditional edge from Orchestrate (Fan-out)
    def orchestrate_condition(state) -> List[str]:
        # Determine which agents to call based on requirements
        # LangGraph allows returning a list of node names for parallel execution
        next_nodes = []
        required = state.get("requiredAgents", {})
        
        if required.get("medILlama"):
            next_nodes.append("med_illama")
        if required.get("webSearch"):
            next_nodes.append("web_search")
        if required.get("rag"):
            next_nodes.append("pubmed_rag")
            
        # Fallback if nothing required (shouldn't happen with correct prompt, but safe to go to compile)
        if not next_nodes:
            return ["compile"]
            
        return next_nodes

    workflow.add_conditional_edges(
        "orchestrate",
        orchestrate_condition,
        ["med_illama", "web_search", "pubmed_rag", "compile"]
    )

    # Fan-in to Compile
    workflow.add_edge("med_illama", "compile")
    workflow.add_edge("web_search", "compile")
    workflow.add_edge("pubmed_rag", "compile")

    # Compile -> Reflect
    workflow.add_edge("compile", "reflect")

    # Conditional edge from Reflect (Loop or End)
    def reflect_condition(state):
        quality_passed = state.get("qualityPassed", True)
        iteration_count = state.get("iterationCount", 0)
        
        if quality_passed or iteration_count >= MAX_ITERATIONS:
            return "end"
        return "orchestrate"

    workflow.add_conditional_edges(
        "reflect",
        reflect_condition,
        {
            "end": END,
            "orchestrate": "orchestrate"
        }
    )

    return workflow.compile()
