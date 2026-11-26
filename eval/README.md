# Evaluation data

Owner: Aasritha (T05)

Load the files with `eval/datasets.py`; it checks them every time, so a typo fails loudly instead of skewing scores:

    from eval.datasets import load_topics, load_qa_set

    topics = load_topics()
    qa = load_qa_set(require_answers=True)   # use this in evaluation runs

## `topics.json`: 15 test topics

One object per topic. `expected_aspects` are what a good report on the topic must cover; the coverage metric checks
the report against them.

    {"id": "t01", "topic": "State of electric two-wheelers in India", "expected_aspects": ["sales volume and growth", "key manufacturers and market share", ...]}

The topics are spread across India-specific and global subjects, technology, policy, health, energy and business,
and across likely sentiment (mostly positive, mixed, mostly negative), so the metrics are not tuned to one kind of
topic.

## `qa_set.json`: 45 chatbot questions

Three per topic: two the report should answer, and one it should refuse ("I don't know" handling).

    {"id": "q01", "topic_id": "t01", "question": "Which companies lead ...?", "answerable": true, "aspect": "key manufacturers and market share", "expected_answer": null}
    {"id": "q03", "topic_id": "t01", "question": "What colour options does the Ather 450X come in?", "answerable": false, "reason": "Product trim details are outside a market-level report"}

- An **answerable** question names the expected aspect it checks, so a report that covers its aspects can answer it.
- An **unanswerable** question is close to the topic, so a careless chatbot is tempted to make something up, but its
  answer is something no report on the topic would contain. `reason` says why.

## Filling in the expected answers (after the reports are frozen)

The chatbot answers from the report, so an expected answer must come from the report, not from general knowledge.
Every `expected_answer` is null until then.

1. Generate the 15 reports once and save them (the frozen reports). Every chatbot evaluation uses these, so a change
   to the research agent does not change the chatbot scores.
2. For each answerable question, write `expected_answer` from its topic's frozen report, in one or two sentences.
3. If the report does not answer it, rewrite the question for something the report does say. Do not just flip
   `answerable` to false: unanswerable questions are meant to be unanswerable for every report on the topic.
4. Check: `uv run pytest tests/test_eval_datasets.py`, then `load_qa_set(require_answers=True)` must pass.
