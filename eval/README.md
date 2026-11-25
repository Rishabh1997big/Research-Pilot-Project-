# Evaluation data

Owner: Aasritha (T05)

`topics.json`: the 15 test topics, one object per topic:

    {"id": "t01", "topic": "State of electric two-wheelers in India", "expected_aspects": ["market size", "key players", "government policy"]}

`qa_set.json`: chatbot questions, one object per question. Set `answerable` to false for questions the report cannot answer:

    {"id": "q01", "topic_id": "t01", "question": "Which company has the largest market share?", "expected_answer": "...", "answerable": true}
