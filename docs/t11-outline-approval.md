# T11: Outline approval (pause and resume the graph)

**Goal (from the tracker):** pause the graph with `interrupt()` after planning, SQLite checkpointer, resume with the
approved or edited outline. Reviewer Deepanshu, depends on T06 (done), start 30-11-26.

**Why it exists:** the brief says the user approves or edits the outline before the report is written. T10 is the screen;
T11 is the graph side that waits for it.

### What was built

- `app/agents/research/nodes/approval.py`: the `approve_outline` node.
  - Pauses with `interrupt()` and shows `{"topic", "sections": [{id, title, subtopics}]}`.
  - Resumes with `{"sections": [{"id": "sec-1" | null, "title", "subtopics"}]}`. The order of the list is the report
    order, a section left out is removed, `id: null` is a new section (the ID is assigned here).
  - `{"sections": None}` approves the outline as proposed.
  - Edited sections keep what the planner gave them. A retitled or new section gets search queries made from its
    title and subtopics (no LLM call). A subtopic the user adds always gets its own query, even at the cap of 3.
  - Sets `section_order`, `sections_to_run` and `outline_approved = True`.
- `app/agents/research/checkpoint.py`: `open_checkpointer()` gives a SQLite saver on `CHECKPOINT_DB`
  (default `data/checkpoints.sqlite`) and creates the folder. Use the run ID as `thread_id`.
- `app/agents/research/graph.py`: the flow is now `plan -> approve_outline -> search -> read`. The checkpointer is
  now required (an interrupt cannot work without one).
- `tests/test_approval.py`: 45 tests (rules, queries, IDs, pause and resume, correction rounds, restart in the middle of a
  correction, two runs on one saver). The existing graph test in `tests/test_sources.py` now resumes past the pause.

### Rules the approval enforces (each with a user-facing message and a test)

| Rule | Why |
|---|---|
| At least 1 and at most 6 sections (`max_sections` is sent in the pause payload so the UI can show it) | Every section costs several web searches |
| Title up to 120 characters, up to 10 subtopics of up to 200 characters each | Titles and subtopics become search queries |
| No two sections with the same title (ignoring case and spacing) | One fact belongs in exactly one section |
| No "References", "Sources", "Bibliography" section | Nothing to search for; the reference list is built automatically |
| An unknown section ID, or the same ID twice, is rejected; a blank ID means a new section | Protects the state from a wrong or stale UI |
| Sections with a blank title are dropped; extra fields sent by the UI are ignored | Matches how the outline screen behaves |
| Only `None`/`{"sections": None}` means "approve as proposed". `[]`, `""`, `0` are errors | Garbage must never approve by accident |

The approval never changes the state it was given, and a new section's ID is never one that was used before in the run,
even if that section was removed in the same approval.

### Two things learned the hard way (both verified with small experiments)

1. **`Command(resume={})` does nothing.** LangGraph ignores an empty resume value (the run stays paused), and
   `resume=None` crashes. So "approve as proposed" is `{"sections": None}`. **Deepanshu's API (T24) must always send
   a non-empty dict.**
2. **Raising an error on a bad outline cannot be recovered.** LangGraph stores the bad resume value and replays it on
   every later resume, so the user would be stuck. Instead the node asks again: it calls `interrupt()` again with the
   outline plus an `"error"` message. **The UI (T10) should show `error` when it is present.**

### How it was done (workflow, in order)

1. **Read before writing.** Read the planner (`nodes/planner.py`), the search and read nodes, the state rules in
   `state.py`, the existing graph and tests, and `config.py` (`checkpoint_db` was already there).
2. **First version.** The approval node, the SQLite checkpointer helper, and the graph wiring
   `plan -> approve_outline -> search -> read`. The checkpointer became a required argument of
   `build_research_graph`, because `interrupt()` cannot work without one.
3. **First tests failed, and found two LangGraph behaviours.** Each was checked with a tiny stand-alone experiment
   before changing the design:
   - `Command(resume={})` does nothing and `resume=None` crashes, so "approve as proposed" became `{"sections": None}`.
   - Raising an error after a bad resume traps the run, because LangGraph replays the stored bad value. The node now
     asks again with an `error` message instead.
4. **Review of my own code.** I re-read the finished module looking for edge cases and found the gaps listed in the
   next section, then rewrote the module and added tests for each.
5. **Branch fix.** The T11 branch had been created from the T10 branch, so a PR would have carried the T10 files. The
   branch was moved onto `main` (nothing lost: T10 stays on its own branch), and these notes were moved out of the T10
   work log into this file.
6. **Final checks:** `ruff` clean, full test suite green (149 tests), restart and two-run scenarios tested.

### Review and hardening: what was found and fixed

| Problem found in the first version | Fix |
|---|---|
| `[]`, `""` or `0` as the answer silently approved the outline (`resume_value or {}`) | Only `None` or `{"sections": None}` means "as proposed". Any other invalid value is an error. |
| A blank `id` was rejected as unknown | A blank `id` now means a new section |
| Titles and subtopics could be any length | Limits: title 120 characters, 10 subtopics of up to 200 characters |
| A user could add a "References" section, which the planner itself bans | Rejected, using the planner's own `NOT_RESEARCHABLE` rule |
| The returned sections shared list objects with the old state | The update holds its own copies; a test checks the state is never changed |
| A subtopic added at the 3-query cap could be cut | New queries are placed first, and the oldest are dropped instead |
| A whitespace-only title change counted as a rename and threw away the planner's queries | Titles are compared after tidying, so queries are kept |
| The UI did not know the section limit | `max_sections` is part of the pause payload |
| Error messages were vague | They name the field (`sections > 0 > title`) and never repeat what the user typed |
| The checkpointer had no usage guidance | `checkpoint.py` says: open it once at API start and share it, not once per request |

### Tests (`tests/test_approval.py`, 45 tests)

- **Rules:** every rule in the table above, including the limits being inclusive (6 sections, 120 characters, 10
  subtopics are allowed; one more is not).
- **Queries and IDs:** edited, renamed, added and removed sections; a removed section's ID is never reused.
- **Safety:** the state is never changed; the update shares no lists with it; empty, falsy and wrong-typed answers are
  rejected; extra fields from the UI are ignored.
- **Graph:** it stops after planning and searches nothing before approval; an edited outline researches only the
  approved sections; an empty `{}` resume does not resume the run (so nobody relies on it by accident).
- **Correction rounds:** a bad answer asks again with the reason, any number of times, then a good answer works.
- **Persistence:** a paused run resumes in a new process (the folder is created if missing); a restart in the middle of a
  correction round loses nothing; two runs sharing one saver do not mix.

### Edge-case tests (`tests/test_approval_edges.py`, 309 tests)

A second file of nasty inputs. The rule it checks: whatever is sent, approval either returns a valid update or raises
`OutlineError` with a message, never anything else, and never changes the state.

- **Odd text:** Hindi, Chinese, emoji, HTML, SQL, newlines, tabs, non-breaking spaces, one-character and punctuation-only
  titles.
- **Wrong types:** 21 kinds of wrong answer (`None` entries, numbers, bare lists, a JSON string, `NaN`, bytes, an
  object, wrong types inside a section...). All give a readable `OutlineError`.
- **IDs:** wrong case, padding, path-like and injection-like IDs; gaps in the numbering; orphan sections.
- **Behaviour:** reversing the order; re-approving gives the same result; a section that already has research keeps
  its passages, sources and status; many subtopics never pass the 3-query cap.
- **Fuzz:** 300 seeded random outlines (random IDs, titles up to 130 characters, up to 12 subtopics, odd characters).
  Every one is checked against the invariants: 1 to 6 unique sections, unique titles, 1 to 3 distinct non-empty queries,
  no reused IDs, removed sections removed, state untouched.
- **Checkpointer and graph:** string paths and existing files, the default path from settings, a graph built without a
  checkpointer is refused, and 12 bad answers in a row followed by a good one.

**Do the tests really catch bugs?** Three deliberate breakages were tried and each was caught: falsy garbage approving,
reused section IDs, and duplicate titles allowed. The code was restored afterwards.

### What this means for the other tasks

| Who | What they need to know |
|---|---|
| T24 API (Deepanshu) | Keep one saver open for the whole API lifetime. Start: `invoke(initial_state)` with `thread_id = run_id`; read the outline from `get_state(config).tasks[0].interrupts[-1].value`. Approve: `invoke(Command(resume=<non-empty dict>))`. |
| T10 UI | Send `{"sections": [...]}`; if the response outline has an `error`, show it and let the user fix the outline. |
| T21 change requests | The same checkpointer and `thread_id` can be reused to resume work on one section. |

### Open items

- [ ] Commit, push, PR for Deepanshu (reviewer). Stage only the T11 files: `app/agents/research/`, `tests/test_approval.py`,
      `tests/test_sources.py`, and this file. The T10 UI files live on the T10 branch.
- [ ] Tell Deepanshu about the two points above before he writes the API.
- [ ] Update the tracker row for T11 after the merge.
