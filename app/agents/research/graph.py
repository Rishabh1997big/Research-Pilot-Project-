"""T06, T07, T11, T21 (Rishabh): research graph wiring.

The full flow will be:  plan -> approve_outline -> search -> read -> write -> verify -> compile,
plus section-level change requests. Built so far:

    plan -> approve_outline -> search -> read

The graph pauses at approve_outline (T11) and needs a checkpointer, so it can be resumed with the approved outline:

    from langgraph.types import Command
    from app.agents.research.checkpoint import open_checkpointer
    from app.agents.research.graph import build_research_graph
    from app.agents.research.state import initial_state
    from app.observability import run_config

    with open_checkpointer() as saver:
        graph = build_research_graph(saver)
        config = run_config("research", topic=topic, thread_id=run_id)
        graph.invoke(initial_state(topic), config=config)          # plans, then stops at approve_outline
        # the outline to show the user is in graph.get_state(config).tasks[0].interrupts[0].value
        graph.invoke(Command(resume={"sections": [...]}), config=config)   # approved or edited outline; runs on
        # (approve as proposed: Command(resume={"sections": None}); an empty {} does NOT resume)

See nodes/approval.py for the resume format.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.research.nodes.approval import approve_outline
from app.agents.research.nodes.planner import planner
from app.agents.research.nodes.sources import read_node, search_node
from app.agents.research.state import ResearchState


def build_research_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    builder = StateGraph(ResearchState)
    builder.add_node("plan", planner)
    builder.add_node("approve_outline", approve_outline)
    builder.add_node("search", search_node)
    builder.add_node("read", read_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "approve_outline")
    builder.add_edge("approve_outline", "search")
    builder.add_edge("search", "read")
    builder.add_edge("read", END)
    return builder.compile(checkpointer=checkpointer)
