# UI and evaluation work log (Aasritha's tasks)

This file records, task by task, what was done, how it was done, why, and what is still open. Update it every time a
task moves forward. Newest work goes at the bottom of each task section.

Tasks covered: **T10, T15, T25** (UI) and **T26, T29** (evaluation).

## Status at a glance

| Task | What | Status | Branch / PR |
|---|---|---|---|
| T10 | Topic input and outline review screens | Done on mock data. Merged (PR #9). Real API waits for T11 and T24. | `feature/T10-outline-screen` |
| T15 | Report view, clickable citations, change-request box | Not started | |
| T25 | Analysis tab and chat window | Not started | |
| T26 | Chatbot metrics with RAGAS | Not started | |
| T29 | Full evaluation run (v1 vs v2) and report | Not started | |

A task is only "done" when its pull request is reviewed and merged, and its row in the tracker sheet is updated
(README, section 5).

---

## Setup work (before T10)

| Date | What | Result |
|---|---|---|
| 26-11-26 | Checked the repo against the tracker sheet | T01 to T07 merged, all by Rishabh. T08 onwards not started. See "Project state" below. |
| 26-11-26 | Wrote [aasritha-guide.md](aasritha-guide.md) | Plain-language guide to the project and to this person's five tasks |

### Project state when T10 started

- Merged: T01 scaffold, T02 LLM layer, T03 state, T04 search, T05 test set, T06 planner, T07 source registry.
- The research graph runs `plan -> search -> read` and then ends. Nothing writes a report yet.
- `app/api/main.py` is a 15-line shell with only `/health`. The real routes are T24 (Deepanshu).
- Outline approval (pause and resume the graph) is T11 (Rishabh) and does not exist yet.
- So the UI cannot talk to real research yet. That is why T10 uses mock data.

---

## T10: Topic input and outline review

**Goal (from the tracker):** Streamlit topic input screen and outline review screen (edit, add or remove sections,
approve). Owner Aasritha, reviewer Rishabh, depends on T03, start 27-11-26.

**Why it exists:** the project brief requires that the user can review and approve or edit the outline before the
full report is written (human-in-the-loop). This screen is where that happens.

### Workflow, step by step

1. **Created the branch** `feature/T10-outline-screen`, following the README branch naming rule.
2. **Read the data shape** in `app/agents/research/state.py`. An outline section has `id`, `title` and `subtopics`.
   A new section added in the UI has `id = None`, and the backend assigns the ID (`next_section_id`).
3. **Built a mock API layer**, `ui/api_client.py`:
   - Every backend call goes through this one file, so the screens never call HTTP directly.
   - `UI_USE_MOCK=1` (default) returns a fake four-section outline. `UI_USE_MOCK=0` calls the real API.
   - Functions: `start_research(topic)` and `approve_outline(run_id, sections)`.
   - Proposed endpoints: `POST /research` and `POST /research/{run_id}/approve`. **These need to be agreed with
     Deepanshu (T24) and Rishabh (T11).**
   - Failures raise `ApiError`, which the screens show as a message instead of crashing.
4. **Built the screens**, `ui/streamlit_app.py`. The current screen is stored in `st.session_state.page`:

   ```
   topic  ->  outline  ->  writing
   ```

   - **topic:** text box, four example-topic buttons, "Start research" (disabled while the topic is empty).
   - **outline:** one card per section with an editable title and a subtopics box (one per line), a Remove button,
     an "Add section" button and "Approve outline".
   - **writing:** a placeholder list of the approved sections. T15 replaces it with the real report view.
5. **Handled edge cases:**
   - Each section has a private `key` (random 8 characters). Streamlit widgets are matched by key; with index keys,
     removing section 2 would shift the text of every section after it.
   - On approve, sections with a blank title and blank subtopic lines are dropped, and the private key is removed,
     so the API receives only `id`, `title`, `subtopics`.
   - Approve is disabled when no section has a title.
   - Section titles and the topic are escaped with `html.escape` before being placed in HTML, so typed text cannot
     inject markup.
   - "Start a new topic" clears the page state, the outline and the topic box.
6. **Wrote tests**, `tests/test_ui_outline.py`, using Streamlit's `AppTest` (no browser, no network, no API keys):
   - Start is disabled without a topic.
   - A topic leads to the outline screen with four sections.
   - Removing a section keeps the other sections' values.
   - A blank new section is dropped on approval.
   - Edits to title and subtopics are kept on approval.
   - "Start a new topic" resets.
7. **Fixed problems found while testing:**
   - Import order and long lines failed `ruff`. Fixed with `ruff --fix` and by wrapping lines.
   - The test file could not find the app (relative path). Now built from `Path(__file__)`.
   - One test timed out because patching `time.sleep` for the whole process also broke Streamlit's wait loop.
     Removed the patch and raised the test timeout to 90 s.
8. **Checks:** `uv run ruff check .` passes. `uv run pytest tests/test_ui_outline.py` passes, 6 tests. The full
   suite passed (109 tests) before the final test timeout fix; only the UI test files were re-run afterwards.

### Design pass (visual)

Done after the first version, at the request for a "really good" look.

- [ui/styles.py](../ui/styles.py): all CSS in one place, plus small helpers `hero`, `page_header`, `step_tracker`.
- [.streamlit/config.toml](../.streamlit/config.toml): dark theme (violet primary colour).
- Look: navy background with violet, cyan and pink glows; animated gradient title; a five-step tracker
  (Topic, Outline, Report, Analysis, Chat); glass cards that lift and glow on hover; gradient primary button;
  numbered section badges; spinning loaders on the writing screen.
- `pyproject.toml`: ruff now ignores the line-length rule (`E501`) for `ui/styles.py` only, because CSS lines are long.
- The design has not been looked at in a browser by the assistant. A human needs to check it.

### Files added or changed

| File | Change |
|---|---|
| `ui/api_client.py` | New. Mock and real API calls |
| `ui/streamlit_app.py` | Rewritten. Topic, outline and writing screens |
| `ui/styles.py` | New. CSS and layout helpers |
| `.streamlit/config.toml` | New. Dark theme |
| `tests/test_ui_outline.py` | New. 6 tests |
| `pyproject.toml` | Per-file ruff ignore for CSS |
| `docs/aasritha-guide.md` | New. Guide for this person's tasks |
| `docs/ui-work-log.md` | New. This file |

### How to run and look at it

```bash
uv run streamlit run ui/streamlit_app.py     # then open http://localhost:8501
```

Try: enter a topic (or click an example), start research, edit a title, remove a section, add a blank section,
approve, then use "Start a new topic" in the sidebar. The sidebar says "API not reachable; mock data is used" in
mock mode. That is expected. If the page does not change after an edit, restart Streamlit: the theme file is read
only at start-up.

### How to analyse T10 (review checklist)

Use this to check the work before the pull request.

1. **Read the flow:** `ui/streamlit_app.py`, bottom of the file: `init_state()`, `sidebar()`, `step_tracker()`,
   then the screen chosen by `session_state.page`.
2. **Read the data path:** `topic_screen` calls `api_client.start_research`; `outline_screen` edits
   `session_state.sections`; `to_outline` cleans them; `approve_outline` sends them.
3. **Check against the task text:** topic input (yes), outline review (yes), edit (yes), add (yes), remove (yes),
   approve (yes).
4. **Check the weak points:**
   - Mock only: nothing is proven against the real API.
   - The "writing" screen is a placeholder and does not poll for the finished report.
   - Add and edit work, but the UI cannot reorder sections. The task does not ask for it. Decide if it is needed.
   - Design is unreviewed in a browser, and Streamlit's internal CSS selectors (for example
     `stVerticalBlockBorderWrapper`) may change in a Streamlit upgrade.
5. **Run:** `uv run ruff check .` and `uv run pytest tests/test_ui_outline.py`.

### Fix after review

Found in review after the merge: while the API was down, every click in the UI froze for 3 to 5 seconds, and the 6 UI
tests took about 3.5 minutes (the whole offline suite went from 16 seconds to 3.6 minutes).

- **Cause:** Streamlit re-runs the whole script on every click, and the sidebar called `/health` each time with a
  3-second timeout. On Windows a refused connection is retried, and `localhost` tries IPv6 and then IPv4, so a
  down API took 3 to 5 seconds to fail (measured: 4.99 s on `localhost`, 3.07 s on `127.0.0.1`).
- **Fix:**
  - Mock mode no longer checks the API, because it never uses it.
  - Otherwise the check (`api_provider()`) is cached for 30 seconds with `st.cache_data` and gives up after 1 second.
  - The mock delay is 0.3 seconds instead of 1 (`MOCK_DELAY`).
- **Guard:** `test_mock_mode_never_calls_the_api` fails if mock mode touches the network. It fails on the old code.
- **Result:** the 7 UI tests run in about 10 seconds, about 1 second each.

### Open items for T10

- [x] Commit, push, open the PR, assign Rishabh as reviewer. Merged as PR #9.
- [x] Fix the slow clicks while the API is down (see "Fix after review" below).
- [ ] Agree the endpoint paths and JSON with Deepanshu (T24) and Rishabh (T11). The approved outline sends only
  `id`, `title` and `subtopics`, so the backend must write search queries for new and edited sections.
- [ ] Switch the `UI_USE_MOCK` default to `0` once the real routes exist, so the UI cannot show fake data by mistake.
- [ ] Decide how the UI learns the report is ready (polling or a status endpoint). Needed for T15.
- [ ] Try it with `UI_USE_MOCK=0` once T11 and T24 are merged.
- [ ] Update the tracker row for T10 after the merge.

---

## T15: Report view with clickable citations and change requests

Not started. Plan: render each section in outline order, make citation numbers clickable, show the reference
list, add a change-request box that calls the API and refreshes only the changed section.
Depends on T10, and on the report format from T16 (Rishabh) and the change-request endpoint (T21, T24).

## T25: Analysis tab and chat window

Not started. Plan: analysis tab (overall, per-section and per-source sentiment, recommendations tagged with their
section) and chat window (ingest button, source tags on answers, "I don't know" style, session id for follow-ups).
Depends on T15 and T19, and on Sabrina's T18 and T23 and Deepanshu's T14, T19, T20.

## T26: Chatbot metrics with RAGAS

Not started. Plan: freeze 15 reports, fill `expected_answer` in `eval/qa_set.json`, then score faithfulness,
answer relevancy, context precision and recall, and "I don't know" accuracy. RAGAS is not yet in `pyproject.toml`
(`uv add ragas` first). Depends on T05, T19, T20.

## T29: Full evaluation run and report

Not started. Plan: run all metrics on the 15 topics for v1 (no verifier) and v2 (with verifier), add latency,
tokens and tool calls, take LangSmith screenshots, fill in `docs/evaluation.md`. Depends on T22 and T26, and needs
a switch to turn the verifier off (ask Rishabh).

---

## Change log

| Date | Task | What happened |
|---|---|---|
| 26-11-26 | Setup | Compared the repo with the tracker; wrote the guide |
| 26-11-26 | T10 | Branch created; mock API client, three screens and 6 tests written |
| 26-11-26 | T10 | Visual design added (CSS theme, step tracker, animations) |
| 26-11-26 | Docs | This work log created |
| 26-11-26 | T10 | Merged as PR #9 |
| 26-11-26 | T10 | Review fix (Rishabh): no health check in mock mode, cached 1-second check otherwise, mock delay 0.3 s |
