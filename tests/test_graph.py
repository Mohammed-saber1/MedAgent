import pytest
from src.agent_graph import create_agent_graph
from langgraph.graph import StateGraph

def test_graph_structure():
    graph = create_agent_graph()
    assert graph is not None
    # Verify it compiles
    # We can't easily inspect nodes on a compiled graph in older versions, 
    # but if it compiles without error, that's a good start.
    assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")
