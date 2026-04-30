"""Tests for the Engram ingestion loader's session-text parser."""

from query_agent_benchmarking.internal.adapters.database.engram_loader import (
    _parse_session_text,
)


class TestParseSessionText:
    def test_single_turn(self):
        text = "user: Hello there"
        result = _parse_session_text(text)
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello there"

    def test_two_turns(self):
        text = "user: What is 2+2?\nassistant: 4"
        result = _parse_session_text(text)
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "What is 2+2?"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "4"

    def test_multiline_content(self):
        text = (
            "user: Tell me a story\n"
            "assistant: Once upon a time,\n"
            "there was a cat.\n"
            "It liked to nap."
        )
        result = _parse_session_text(text)
        assert len(result.messages) == 2
        assert result.messages[1].content == (
            "Once upon a time,\nthere was a cat.\nIt liked to nap."
        )

    def test_multiple_exchanges(self):
        text = (
            "user: Hi\n"
            "assistant: Hello!\n"
            "user: How are you?\n"
            "assistant: I'm doing well, thanks."
        )
        result = _parse_session_text(text)
        assert len(result.messages) == 4
        roles = [m.role for m in result.messages]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_longmemeval_format(self):
        """Test with a realistic LongMemEval session snippet."""
        text = (
            "user: The farmer needs to transport a fox, a chicken, "
            "and some grain across a river.\n"
            "assistant: To solve this puzzle, the farmer can follow these steps:\n"
            "\n"
            "1. First, take the chicken across.\n"
            "2. Then go back and take the fox.\n"
            "3. Bring the chicken back.\n"
            "4. Finally, take the grain across."
        )
        result = _parse_session_text(text)
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert "farmer" in result.messages[0].content
        assert result.messages[1].role == "assistant"
        # The numbered list and blank line should be preserved
        assert "\n\n1." in result.messages[1].content

    def test_empty_string(self):
        result = _parse_session_text("")
        assert len(result.messages) == 0

    def test_empty_message_content_filtered(self):
        """Messages with empty content should be dropped (Engram rejects them)."""
        text = "user: \nassistant: Here is the answer"
        result = _parse_session_text(text)
        assert len(result.messages) == 1
        assert result.messages[0].role == "assistant"
        assert result.messages[0].content == "Here is the answer"

    def test_whitespace_only_message_filtered(self):
        """Messages with only whitespace should be dropped."""
        text = "user:   \nassistant: Response"
        result = _parse_session_text(text)
        assert len(result.messages) == 1
        assert result.messages[0].role == "assistant"
