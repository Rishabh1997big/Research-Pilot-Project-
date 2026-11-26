"""T06, T07, T11, T21 (Rishabh): research graph wiring.

The full flow will be:  plan -> [outline approval] -> search -> read -> write -> verify -> compile,
plus section-level change requests. Built so far:

    plan -> search -> read

    from app.agents.research.graph import build_research_graph
    from app.agents.research.state import initial_state
    from app.observability import run_config

    graph = build_research_graph()
    final = graph.invoke(initial_state("Quick commerce in India"), config=run_config("research", topic=...))

Pass a checkpointer (e.g. SqliteSaver) to save the state after every step; the outline approval pause (T11) needs it.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.research.nodes.planner import planner
from app.agents.research.nodes.sources import read_node, search_node
from app.agents.research.state import ResearchState


def build_research_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(ResearchState)
    builder.add_node("plan", planner)
    builder.add_node("search", search_node)
    builder.add_node("read", read_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "read")
    builder.add_edge("read", END)
    return builder.compile(checkpointer=checkpointer)
