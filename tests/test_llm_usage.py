from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.llm.usage import track_usage


def fake_model(*token_counts: tuple[int, int]) -> GenericFakeChatModel:
    """A chat model whose replies report the given (input, output) token counts, one reply per call."""
    replies = [
        AIMessage(content="ok", usage_metadata={"input_tokens": i, "output_tokens": o, "total_tokens": i + o})
        for i, o in token_counts
    ]
    return GenericFakeChatModel(messages=iter(replies))


def test_counts_calls_and_tokens_inside_the_block():
    model = fake_model((10, 5), (20, 7), (99, 99))

    with track_usage() as usage:
        model.invoke("first")
        model.invoke("second")
    model.invoke("after the block, not counted")

    assert usage.as_metrics() == {"llm_calls": 2, "input_tokens": 30, "output_tokens": 12}


def test_nested_blocks_count_separately():
    model = fake_model((1, 1), (2, 2))

    with track_usage() as outer:
        model.invoke("outer")
        with track_usage() as inner:
            model.invoke("inner")

    assert outer.llm_calls == 1
    assert inner.input_tokens == 2
