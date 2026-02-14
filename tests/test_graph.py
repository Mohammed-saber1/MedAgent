import pytest
from src.agent_graph import create_agent_graph

def test_graph_creation():
    graph = create_agent_graph()
    assert graph is not None
    # We can check if nodes exist in the compiled graph
    # assert "evaluate" in graph.nodes
