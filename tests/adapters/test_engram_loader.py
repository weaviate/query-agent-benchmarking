"""Tests for the Engram ingestion loader's session-text parser."""

import pytest

from query_agent_benchmarking.internal.adapters.database.engram_loader import (
    _get_inputs_from_conversation,
    _parse_session_date,
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


class TestParseSessionDate:
    @pytest.mark.parametrize("session_date, expected", [
        ("2023/05/20 (Sat) 10:58", "2023-05-20T10:58:00Z"),
        ("2023/05/27 (Sat) 05:39", "2023-05-27T05:39:00Z"),
        ("2023/05/28 (Sun) 02:58", "2023-05-28T02:58:00Z"),
        ("2000/01/01 (Sat) 00:00", "2000-01-01T00:00:00Z"),
        ("",                        None),
        ("not a date",              None),
        ("2023-05-20",              None),
    ])
    def test_parse_session_date(self, session_date, expected):
        assert _parse_session_date(session_date) == expected


class TestGetInputsFromConversation:
    """Verify updated_at is propagated to all split ConversationInputs."""

    _SESSION = "user: Hi\nassistant: Hello!\nuser: How are you?\nassistant: Fine."
    _UPDATED_AT = "2023-05-20T10:58:00Z"

    def _make_conversation(self, updated_at=None):
        conv = _parse_session_text(self._SESSION)
        conv.updated_at = updated_at
        return conv

    def test_conversation_mode_preserves_updated_at(self):
        conv = self._make_conversation(self._UPDATED_AT)
        results = _get_inputs_from_conversation(conv, "conversation")
        assert len(results) == 1
        assert results[0].updated_at == self._UPDATED_AT

    def test_conversation_mode_none_updated_at(self):
        conv = self._make_conversation(None)
        results = _get_inputs_from_conversation(conv, "conversation")
        assert results[0].updated_at is None

    def test_user_messages_mode_propagates_updated_at(self):
        conv = self._make_conversation(self._UPDATED_AT)
        results = _get_inputs_from_conversation(conv, "user_messages")
        assert len(results) == 2
        assert all(r.updated_at == self._UPDATED_AT for r in results)

    def test_user_messages_mode_none_updated_at(self):
        conv = self._make_conversation(None)
        results = _get_inputs_from_conversation(conv, "user_messages")
        assert all(r.updated_at is None for r in results)

    def test_message_turn_mode_propagates_updated_at(self):
        conv = self._make_conversation(self._UPDATED_AT)
        results = _get_inputs_from_conversation(conv, "message_turn")
        assert len(results) == 2
        assert all(r.updated_at == self._UPDATED_AT for r in results)

    def test_message_turn_mode_none_updated_at(self):
        conv = self._make_conversation(None)
        results = _get_inputs_from_conversation(conv, "message_turn")
        assert all(r.updated_at is None for r in results)
