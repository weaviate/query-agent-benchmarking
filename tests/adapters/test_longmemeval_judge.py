"""Tests for the LongMemEval type-specific judge prompt builder."""

from query_agent_benchmarking.internal.adapters.metrics.longmemeval_judge import _build_prompt


class TestBuildPrompt:
    """Verify that the correct template is selected for each question type."""

    Q = "What did I have for lunch?"
    A = "Pizza"
    R = "You had pizza for lunch."

    def test_standard_types_use_standard_template(self):
        for qtype in ("single-session-user", "single-session-assistant", "multi-session"):
            prompt = _build_prompt(qtype, self.Q, self.A, self.R)
            assert "contains all the intermediate steps" in prompt
            assert "off-by-one" not in prompt
            assert "previous information along with an updated answer" not in prompt
            assert "rubric" not in prompt.lower()

    def test_temporal_reasoning_allows_off_by_one(self):
        prompt = _build_prompt("temporal-reasoning", self.Q, self.A, self.R)
        assert "off-by-one" in prompt

    def test_knowledge_update_allows_previous_info(self):
        prompt = _build_prompt("knowledge-update", self.Q, self.A, self.R)
        assert "previous information along with an updated answer" in prompt

    def test_single_session_preference_uses_rubric(self):
        prompt = _build_prompt("single-session-preference", self.Q, self.A, self.R)
        assert "rubric" in prompt.lower()
        assert "personal information" in prompt

    def test_abstention_detected_by_question_id(self):
        prompt = _build_prompt("multi-session", self.Q, self.A, self.R, question_id="abc_abs_123")
        assert "unanswerable" in prompt

    def test_abstention_overrides_type(self):
        # Even for temporal-reasoning, _abs in question_id should use abstention template
        prompt = _build_prompt("temporal-reasoning", self.Q, self.A, self.R, question_id="q1_abs")
        assert "unanswerable" in prompt
        assert "off-by-one" not in prompt

    def test_unknown_type_falls_back_to_standard(self):
        prompt = _build_prompt("some-unknown-type", self.Q, self.A, self.R)
        assert "contains all the intermediate steps" in prompt

    def test_prompt_contains_question_and_answer(self):
        prompt = _build_prompt("multi-session", self.Q, self.A, self.R)
        assert self.Q in prompt
        assert self.A in prompt
        assert self.R in prompt

    def test_all_prompts_end_with_yes_or_no(self):
        for qtype in ("single-session-user", "temporal-reasoning", "knowledge-update",
                       "single-session-preference"):
            prompt = _build_prompt(qtype, self.Q, self.A, self.R)
            assert prompt.strip().endswith("Answer yes or no only.")
