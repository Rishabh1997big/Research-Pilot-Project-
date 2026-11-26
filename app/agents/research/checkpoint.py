"""T11: the SQLite checkpointer for the research graph.

A run that pauses for outline approval has to survive the API process restarting, so its state is saved to a file
(CHECKPOINT_DB in .env, default data/checkpoints.sqlite). One run is one thread: use the run ID as `thread_id`.

    from app.agents.research.checkpoint import open_checkpointer

    with open_checkpointer() as saver:
        graph = build_research_graph(saver)
        graph.invoke(initial_state(topic), config=run_config("research", topic=topic, thread_id=run_id))
        # ... the run is now paused at "approve_outline"; later, even in another process:
        graph.invoke(Command(resume={"sections": [...]}), config=run_config("research", thread_id=run_id))

Open it once when the API starts and keep it open (for example in FastAPI's lifespan), not once per request: the saver
holds one SQLite connection and a lock, and a graph built from it can be shared by every request.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import settings


@contextmanager
def open_checkpointer(path: Path | str | None = None) -> Iterator[SqliteSaver]:
    """A SqliteSaver on `path` (default: settings.checkpoint_db). The folder is created if it does not exist.
    Pass ":memory:" for a throwaway saver in tests."""
    target = str(path if path is not None else settings.checkpoint_db)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(target) as saver:
        yield saver
