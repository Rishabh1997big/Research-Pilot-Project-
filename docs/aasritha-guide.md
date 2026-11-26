# Aasritha's guide: UI and Evaluation

This guide explains, in plain steps, what you build, why it exists, and how it connects to everyone else's work.
Read section 1 first. Then use sections 3 to 7 as a checklist for each task.

Your tasks: **T10, T15, T25** (the user interface) and **T26, T29** (evaluation).
You also **review** T07 (done), T17 (verifier) and T22 (report metrics) for Rishabh.

---

## 1. The whole project in five minutes

ResearchPilot is a research assistant. A user types a topic. The system then:

1. **Plans** an outline (sections and subtopics).
2. **Waits for the user to approve** the outline. This is the human-in-the-loop (HITL) step.
3. **Searches** the web and **reads** pages for every section.
4. **Writes** each section using only what it read, with citations like `[S1]`.
5. **Verifies** that the citations are real and support the claims.
6. **Compiles** the final report with numbered citations and a reference list.
7. Lets the user ask for **changes** ("make section 3 more detailed"). Only that section is redone.
8. Runs an **analysis** (sentiment and recommendations).
9. Lets the user **chat with the report** (a chatbot that answers only from the report).
10. **Evaluation** measures how good all of this is.

### Who builds which part

| Part | What it does | Owner | Tasks |
|---|---|---|---|
| Research agent | Steps 1 to 7 above | Rishabh | T06, T07, T11, T12, T16, T17, T21 |
| Search and reader tools | Find and fetch web pages | Rishabh, Sabrina | T04, T08 |
| Analysis agent | Sentiment and recommendations | Sabrina | T18, T23 |
| Knowledge base and chatbot | Store the report, answer questions | Deepanshu | T09, T14, T19, T20 |
| Backend API (FastAPI) | One door that connects everything | Deepanshu | T24 |
| **UI (Streamlit)** | **What the user sees** | **You** | **T10, T15, T25** |
| **Evaluation** | **Scores how good it is** | **You (chatbot, final run), Rishabh (report metrics)** | **T26, T29, T22** |

### The picture

```
          YOU: Streamlit UI  (ui/streamlit_app.py)
                    |
                    |  HTTP requests (httpx)
                    v
          Deepanshu: FastAPI backend (app/api/main.py)
            |              |                 |
            v              v                 v
   Research agent    Analysis agent    Knowledge base + chatbot
   (LangGraph)       (sentiment,       (chunk, embed, Chroma,
   Rishabh           recommendations)   retrieve, answer)
                     Sabrina            Deepanshu

          YOU: Evaluation (eval/)  reads the outputs of all of these
                                   and gives them scores
```

**Key rule for the UI:** the UI never imports the agent code. It only calls the API over HTTP.
That means you can build the whole UI before the agents are finished, using fake data.

---

## 2. What already exists that you can use

You do not start from zero.

- **`ui/streamlit_app.py`**: a small file that shows the title and checks that the API is running. You will grow this.
- **`app/config.py`**: gives you `settings.api_base_url`, the address of the API.
- **`app/agents/research/state.py`**: describes the data the research agent produces. Read this file. It tells you
  what a *section* and a *source* look like (see section 3).
- **`eval/topics.json`**: the 15 test topics, each with the "expected aspects" a good report should cover.
- **`eval/qa_set.json`**: 45 chatbot questions (30 answerable, 15 unanswerable).
- **`eval/datasets.py`**: loads those two files safely.
- **`docs/evaluation.md`**: an empty template for the evaluation report. You fill it in at T29.

To run the UI:

```
uv run uvicorn app.api.main:app --reload      # terminal 1: API
uv run streamlit run ui/streamlit_app.py      # terminal 2: UI
```

---

## 3. Understand the data first

Everything you show comes from the research state. These are the pieces you need:

**A section** (what you show in the outline screen and the report):

| Field | Meaning |
|---|---|
| `id` | `sec-1`, `sec-2`... never changes |
| `title` | Section heading |
| `subtopics` | Short list of what the section covers |
| `draft` | The written text, with `[S1]`-style citations |
| `status` | `planned`, `researching`, `drafted`, `verified` or `needs_revision` |

**A source** (what you show in the reference list):

| Field | Meaning |
|---|---|
| `id` | `S1`, `S2`... |
| `url` | Web address |
| `title` | Page title |
| `domain` | Website name (used for the source diversity metric) |

**The order** of sections is in `section_order` (a list of section ids). Show sections in that order.

**The final report** is a Markdown string in `report`. Citations are already numbered and the reference list is
included (made by Rishabh's T16).

> The exact JSON the API returns will be decided with Deepanshu (T24). **Agree on it in the first days** and write
> it down. Until then, build with a hard-coded Python dict that looks like the fields above.

---

## 4. UI tasks

### T10: Topic input and outline review (start 27-11)

**Goal:** the user enters a topic, sees the planned outline, edits it, and approves it.

**Why:** this is the HITL requirement of the project. The brief says the user must be able to "review and approve
(or edit) the outline before the full report is written."

**How it connects:**
1. Your UI sends the topic to the API (`start research`).
2. The API runs the **planner** (Rishabh, T06). The planner returns sections and subtopics.
3. The graph then **pauses** (`interrupt()`, Rishabh, T11). It waits.
4. Your UI shows the outline. The user edits it and presses Approve.
5. Your UI sends the approved outline to the API (`approve outline`). The graph **resumes** and writes the report.

**What to build, step by step:**
1. A text box for the topic and a "Start research" button.
2. A loading spinner while the planner runs.
3. An outline screen. For every section, show an editable title and an editable list of subtopics.
4. A button to **add** a section and a button to **remove** a section.
5. An **Approve outline** button that sends the final outline to the API.
6. A "researching..." screen after approval (see Tips for progress).

**Streamlit hints:** use `st.session_state` to remember the outline between clicks. Use `st.text_input`,
`st.text_area`, `st.button`, `st.form` and `st.rerun`.

**Done when:** with a mocked API you can type a topic, edit the outline, add and remove sections, and approve. The
approved outline is printed or sent in the right shape.

---

### T15: Report view with clickable citations and change requests (start 30-11)

**Goal:** show the finished report nicely and let the user ask for changes.

**Why:** the brief says the report needs inline citations and a reference list, and that the user can steer it
afterwards ("add a subtopic on government policy", "make section 3 more detailed", "replace the weak sources in
section 2").

**How it connects:**
- The report comes from the API (`get report`). It was written by the writer (T12), verified (T17) and compiled
  (T16).
- The change request goes to the API (`send change request`). Rishabh's T21 decides what kind of change it is and
  re-runs **only the affected section**. Then you re-fetch the report.

**What to build:**
1. Show each section with its heading and text, in outline order.
2. Make citation numbers clickable. A click opens the source URL (or jumps to the reference list entry).
3. Show the **reference list** at the bottom: number, title, link.
4. Show a small status tag per section if available (for example "verified" or "needs revision").
5. A text box "Ask for a change" and a Submit button. After submitting, show a spinner and then refresh the report.
6. Highlight which section(s) changed, so the user can see only part of the report was redone.

**Done when:** a sample report renders with working links, and submitting a change request calls the API and
refreshes the view.

---

### T25: Analysis tab and chat window (start 02-12)

**Goal:** two more screens: analysis results and the chatbot.

**Why:** these are Module 2 (sentiment and recommendations) and Module 3 (knowledge base chatbot) of the brief.

**Analysis tab:**
- A button "Run analysis" calls the API (`run analysis`). The API runs Sabrina's analysis agent.
- Show **overall sentiment** (positive, neutral or negative) with a short reason.
- Show **sentiment per section** and **per source** (a table or colored tags works well).
- Show **recommendations**: key takeaways, opportunities, risks, next steps. Each one is linked to a report
  section, so show which section it comes from.

**Chat window:**
- A button "Add report to knowledge base" calls the API (`ingest`). Deepanshu's code splits the report by section,
  embeds the chunks and stores them in Chroma (T09).
- A chat box using `st.chat_input` and `st.chat_message`.
- Each answer shows **source tags**: the section name and the source (URL) it came from. This is required by the
  brief ("show which source/section they came from").
- If the bot says it does not know, show that clearly. Do not hide it. It is the correct behavior (T19).
- Follow-up questions must work. The conversation memory (T20) is on the backend, but your UI must send the chat
  or session id so the backend knows which conversation it is.

**Done when:** you can ingest a report, ask a question, see the answer with source tags, ask a follow-up, and ask
an unanswerable question and see "I don't know".

---

## 5. Evaluation tasks

The brief says: "You must prove your system works, not just claim it." Evaluation does that with numbers on a fixed
test set.

### How the evaluation fits together

```
topics.json (15 topics)  ->  research agent  ->  15 reports  ->  report metrics   (Rishabh, T22)
                                                    |
                                                    +-> FROZEN copy of the 15 reports
                                                              |
qa_set.json (45 questions) -> chatbot reads frozen reports -> chatbot metrics     (YOU, T26)

report metrics + chatbot metrics + latency/tokens/calls -> evaluation report      (YOU, T29)
```

### T26: Chatbot metrics with RAGAS (start 02-12)

**Goal:** a script that scores the chatbot. Put it in `eval/metrics/chatbot_metrics.py`.

**Step 0: freeze the reports and fill in the expected answers.** This is the only manual part.
1. Generate the 15 reports once and save them. These are the "frozen reports".
2. For each **answerable** question in `qa_set.json`, write `expected_answer` (one or two sentences) from the
   topic's frozen report. It must come from the *report*, not from your general knowledge.
3. If the report cannot answer the question, rewrite the question to ask something the report does say. Do not
   just change `answerable` to false. The unanswerable questions are meant to be unanswerable for any report on the
   topic.
4. Check: `uv run pytest tests/test_eval_datasets.py` and `load_qa_set(require_answers=True)` must pass.

**Why freeze?** If the research agent changes later, the reports change. Then a chatbot score change could be caused
by the report, not by the chatbot. Frozen reports keep the chatbot test fair.

**The four metric groups:**

| Metric | Question it answers | Plain meaning |
|---|---|---|
| Faithfulness | Is the answer supported by the retrieved chunks? | Did it avoid making things up? |
| Answer relevancy | Does it answer what was asked? | Is it on topic? |
| Context precision and recall | Did retrieval fetch the right chunks? | Are the chunks it found the useful ones, and did it find all the useful ones? |
| "I don't know" accuracy | Does it refuse when it should? | Refuses the 15 unanswerable questions, answers the 30 answerable ones |

**How to do it:**
1. For each question, call the chatbot (through the API or Deepanshu's chatbot function). Save: the question, the
   answer, the retrieved chunks, and whether it refused.
2. Give RAGAS those four things plus the `expected_answer` as the reference.
3. For "I don't know" accuracy, no RAGAS needed. Just count: for each question, did the refusal match `answerable`?
   Report it as a simple percentage (and note false refusals and false answers separately).
4. Print or save results to a JSON file so T29 can use them.

**Tip:** RAGAS needs an LLM as the judge. Use the same swappable provider layer (`app/llm/provider.py`) so you do
not hit a rate limit with only one provider.

**Done when:** one command runs all 45 questions and prints the four scores.

---

### T29: Full evaluation run and report (start 03-12)

**Goal:** run everything on all 15 topics for two versions and write the evaluation report.

**The two versions:**
- **v1:** the research agent **without** the verifier.
- **v2:** the research agent **with** the verifier (T17).

The brief requires a comparison of at least two versions, with an explanation of what changed and what you learned.

**Steps:**
1. For each topic and each version, generate a report.
2. Run Rishabh's report metrics (T22): citation validity, groundedness (hallucination rate), coverage, source
   diversity, readability.
3. Run your chatbot metrics (T26) on the frozen reports.
4. Collect **system metrics**: latency per run, tokens (and estimated cost), number of tool and search calls. The
   run state already counts these (`metrics` in `state.py`). LangSmith has the rest.
5. Take **LangSmith screenshots** of a few representative runs (one good, one with a verifier catch, one chat).
6. Fill in `docs/evaluation.md`. Its headings are already there:
   - Test set (what the 15 topics are and why)
   - Report quality metrics (table, v1 and v2)
   - Chatbot quality metrics
   - System metrics
   - v1 vs v2 (what changed, how the numbers moved)
   - What we learned (honest conclusions, including what did not work)

**What makes a good report:** tables with real numbers, a clear statement like "the verifier lowered hallucination
rate from X% to Y% but raised latency by Z seconds", and honest limits ("only 15 topics, one judge model").

**Done when:** `docs/evaluation.md` is complete with numbers for both versions and screenshots are saved.

---

## 6. Reviewer duties

A reviewer reads the pull request, runs the tests, and checks it matches the task text.

| Task | What to check |
|---|---|
| T07 (done) | Every passage keeps its source ID; search and reader are wired into the graph |
| T17 verifier | Cited URLs are checked; unsupported claims are removed or flagged; results go into `Section.citations` |
| T22 report metrics | The five metrics are computed correctly; the code is usable by your T29 run |

Because your T29 depends on T22, review T22 carefully. It is the code you will run.

---

## 7. Suggested order and timing

| Date | What to do |
|---|---|
| Now to 27-11 | Read this guide, `state.py`, `eval/README.md`. Agree the API JSON shapes with Deepanshu. |
| 27-11 | **T10**: topic screen and outline review, using fake data |
| 30-11 | **T15**: report view and change-request box |
| 30-11 to 01-12 | Start **T26** early: the scoring script, tested on a few hand-made reports |
| 02-12 | **T25**: analysis tab and chat. **T26** finished. Review T22. |
| 03-12 | **T29**: full run and the report |

T29 is the most at risk because it needs everyone's work. Make the scripts work on a small sample before the full
pipeline exists.

---

## 8. How your work connects to the others (checklist)

| You need | From | For |
|---|---|---|
| The outline shape and a way to pause and resume | Rishabh (T06, T11) | T10 |
| The final report format and change-request endpoint | Rishabh (T16, T21), Deepanshu (T24) | T15 |
| Sentiment and recommendation JSON | Sabrina (T18, T23) | T25 |
| Ingest and chat endpoints, source tags, refusal flag | Deepanshu (T09, T14, T19, T20, T24) | T25, T26 |
| Report metrics code | Rishabh (T22) | T29 |
| Verifier on and off switch | Rishabh (T17) | T29 (v1 and v2) |
| The 15 topics and 45 questions | `eval/` (T05) | T26, T29 |

**Ask for these early**, especially the API JSON shapes and a switch to turn the verifier off. They are the most
common reasons integration (T28) goes wrong.

---

## 9. Tips

- **Mock first.** Write a small `mock_api.py` or hard-code dicts so you can build screens without waiting.
- **Progress feedback.** Research takes a while. Use `st.spinner` or `st.status`, and say what step is running.
- **Show errors kindly.** If the API is down or rate-limited, show a clear message, not a crash.
- **Never commit `.env` or API keys.**
- **Git:** branch `feature/T10-outline-screen`, commit messages start with the task id (`T10: ...`), open a pull
  request, and assign the reviewer from the tracker. Run `uv run pytest` and `uv run ruff check .` first.
- If something is unclear, ask the task's owner. The goal is that you can explain every part at the demo.
