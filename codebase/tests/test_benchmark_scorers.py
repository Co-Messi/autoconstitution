"""Tests for CodingScorer and JudgeScorer."""

from __future__ import annotations

import pytest

from autoconstitution.benchmark.protocol import BenchCase, ScoreResult
from autoconstitution.benchmark.scorers import SCORERS, CodingScorer, JudgeScorer
from autoconstitution.providers.fake import FakeProvider

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_scorers_registry_has_both_entries() -> None:
    assert set(SCORERS) == {"coding", "judge"}
    assert SCORERS["coding"] is CodingScorer
    assert SCORERS["judge"] is JudgeScorer


# ---------------------------------------------------------------------------
# CodingScorer
# ---------------------------------------------------------------------------


_HIDDEN_ADD_TESTS = """
def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2
"""


def _add_case(tests: str = _HIDDEN_ADD_TESTS) -> BenchCase:
    return BenchCase(id="add", prompt="Implement add(a, b).", metadata={"hidden_tests": tests})


@pytest.mark.asyncio
async def test_coding_scorer_all_tests_pass_gives_score_one() -> None:
    scorer = CodingScorer()
    answer = "```python\ndef add(a, b):\n    return a + b\n```"
    result = await scorer.score(_add_case(), answer)
    assert result.score == 1.0
    assert result.passed is True
    assert "2/2" in result.detail


@pytest.mark.asyncio
async def test_coding_scorer_partial_pass_gives_partial_score() -> None:
    scorer = CodingScorer()
    # Works for positives but wrong for negatives.
    answer = "```python\ndef add(a, b):\n    return a + b if a > 0 else 0\n```"
    result = await scorer.score(_add_case(), answer)
    assert 0.0 < result.score < 1.0
    assert result.passed is False


@pytest.mark.asyncio
async def test_coding_scorer_all_fail_gives_score_zero() -> None:
    scorer = CodingScorer()
    answer = "```python\ndef add(a, b):\n    return a * b\n```"
    result = await scorer.score(_add_case(), answer)
    assert result.score == 0.0
    assert result.passed is False


@pytest.mark.asyncio
async def test_coding_scorer_missing_hidden_tests_returns_zero() -> None:
    scorer = CodingScorer()
    case = BenchCase(id="nope", prompt="...", metadata={})
    result = await scorer.score(case, "def f(): pass")
    assert result.score == 0.0
    assert result.passed is False
    assert "hidden_tests" in result.detail


@pytest.mark.asyncio
async def test_coding_scorer_empty_answer_returns_zero() -> None:
    scorer = CodingScorer()
    result = await scorer.score(_add_case(), "")
    assert result.score == 0.0
    assert "no Python code" in result.detail


@pytest.mark.asyncio
async def test_coding_scorer_extracts_from_markdown_fence() -> None:
    scorer = CodingScorer()
    answer = (
        "Here's my solution:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "Hope this helps."
    )
    result = await scorer.score(_add_case(), answer)
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_coding_scorer_uses_whole_answer_when_no_fence() -> None:
    scorer = CodingScorer()
    answer = "def add(a, b):\n    return a + b"  # no fences
    result = await scorer.score(_add_case(), answer)
    assert result.score == 1.0
    assert "no ```python fence" in result.detail


@pytest.mark.asyncio
async def test_coding_scorer_times_out_on_infinite_loop() -> None:
    scorer = CodingScorer(timeout_s=2.0)
    answer = "```python\ndef add(a, b):\n    while True: pass\n```"
    result = await scorer.score(_add_case(), answer)
    assert result.score == 0.0
    assert result.passed is False
    assert "timed out" in result.detail


@pytest.mark.asyncio
async def test_coding_scorer_handles_syntax_error() -> None:
    scorer = CodingScorer()
    answer = "```python\ndef add(a, b:\n    return a + b\n```"  # missing )
    result = await scorer.score(_add_case(), answer)
    assert result.score == 0.0
    assert result.passed is False


@pytest.mark.asyncio
async def test_coding_scorer_applies_setup_block() -> None:
    scorer = CodingScorer()
    case = BenchCase(
        id="import",
        prompt="Use math.floor.",
        metadata={
            "setup": "import math",
            "hidden_tests": "def test_floor():\n    assert my_floor(2.7) == 2",
        },
    )
    answer = "```python\ndef my_floor(x):\n    return math.floor(x)\n```"
    result = await scorer.score(case, answer)
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_coding_scorer_close_is_idempotent() -> None:
    scorer = CodingScorer()
    await scorer.close()
    await scorer.close()


@pytest.mark.asyncio
async def test_coding_scorer_strips_api_keys_from_subprocess_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Secrets must not leak into subprocess env the student code runs in."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-appear")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-also-not")

    scorer = CodingScorer()
    case = BenchCase(
        id="env_leak",
        prompt="...",
        metadata={
            "hidden_tests": (
                "import os\n"
                "def test_no_openai_key():\n"
                "    assert os.environ.get('OPENAI_API_KEY') is None\n"
                "def test_no_anthropic_key():\n"
                "    assert os.environ.get('ANTHROPIC_API_KEY') is None\n"
            ),
        },
    )
    answer = "```python\n# no code needed for env tests\n```"
    result = await scorer.score(case, answer)
    assert result.score == 1.0, result.detail


# ---------------------------------------------------------------------------
# JudgeScorer
# ---------------------------------------------------------------------------


def _judge_case() -> BenchCase:
    return BenchCase(
        id="tax",
        prompt="Explain capital gains tax briefly.",
        metadata={"rubric": "Accurate, concise, no throat-clearing."},
    )


@pytest.mark.asyncio
async def test_judge_scorer_parses_clean_json() -> None:
    provider = FakeProvider(responses=['{"score": 0.8, "reasoning": "concise and correct"}'])
    scorer = JudgeScorer(provider=provider)
    result = await scorer.score(_judge_case(), "Capital gains tax applies to...")
    assert result.score == 0.8
    assert result.passed is None
    assert "concise" in result.detail


@pytest.mark.asyncio
async def test_judge_scorer_clamps_scores_above_one() -> None:
    provider = FakeProvider(responses=['{"score": 1.4, "reasoning": "great"}'])
    scorer = JudgeScorer(provider=provider)
    result = await scorer.score(_judge_case(), "...")
    assert result.score == 1.0
    assert "clamped" in result.detail


@pytest.mark.asyncio
async def test_judge_scorer_clamps_scores_below_zero() -> None:
    provider = FakeProvider(responses=['{"score": -0.2, "reasoning": "terrible"}'])
    scorer = JudgeScorer(provider=provider)
    result = await scorer.score(_judge_case(), "...")
    assert result.score == 0.0
    assert "clamped" in result.detail


@pytest.mark.asyncio
async def test_judge_scorer_handles_markdown_fenced_json() -> None:
    provider = FakeProvider(
        responses=['```json\n{"score": 0.5, "reasoning": "ok"}\n```']
    )
    scorer = JudgeScorer(provider=provider)
    result = await scorer.score(_judge_case(), "...")
    assert result.score == 0.5


@pytest.mark.asyncio
async def test_judge_scorer_extracts_json_from_prose() -> None:
    provider = FakeProvider(
        responses=['Here is the grade: {"score": 0.3, "reasoning": "weak"} — thanks!']
    )
    scorer = JudgeScorer(provider=provider)
    result = await scorer.score(_judge_case(), "...")
    assert result.score == 0.3


@pytest.mark.asyncio
async def test_judge_scorer_returns_zero_on_parse_error() -> None:
    provider = FakeProvider(responses=["I cannot do that as an AI language model."])
    scorer = JudgeScorer(provider=provider)
    result = await scorer.score(_judge_case(), "...")
    assert result.score == 0.0
    assert "parse" in result.detail.lower() or "JSON" in result.detail


@pytest.mark.asyncio
async def test_judge_scorer_returns_zero_on_missing_score_field() -> None:
    provider = FakeProvider(responses=['{"reasoning": "no score key"}'])
    scorer = JudgeScorer(provider=provider)
    result = await scorer.score(_judge_case(), "...")
    assert result.score == 0.0
    assert "score" in result.detail


@pytest.mark.asyncio
async def test_judge_scorer_missing_rubric_returns_zero() -> None:
    scorer = JudgeScorer(provider=FakeProvider(responses=["unused"]))
    case = BenchCase(id="norubric", prompt="...", metadata={})
    result = await scorer.score(case, "...")
    assert result.score == 0.0
    assert "rubric" in result.detail


@pytest.mark.asyncio
async def test_judge_scorer_catches_provider_exception() -> None:
    class ExplodingProvider:
        async def complete(
            self,
            prompt: str,
            system: str | None = None,
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> str:
            raise RuntimeError("network go boom")

    scorer = JudgeScorer(provider=ExplodingProvider())
    result = await scorer.score(_judge_case(), "...")
    assert result.score == 0.0
    assert "RuntimeError" in result.detail
    assert "go boom" in result.detail


@pytest.mark.asyncio
async def test_judge_scorer_close_calls_provider_close_if_present() -> None:
    closed = {"called": False}

    class CloseableProvider:
        async def complete(self, *args: object, **kwargs: object) -> str:
            return '{"score": 1.0, "reasoning": "x"}'

        async def close(self) -> None:
            closed["called"] = True

    scorer = JudgeScorer(provider=CloseableProvider())
    await scorer.close()
    assert closed["called"] is True


@pytest.mark.asyncio
async def test_judge_scorer_close_is_safe_with_provider_without_close() -> None:
    scorer = JudgeScorer(provider=FakeProvider(responses=["x"]))
    await scorer.close()
    await scorer.close()


@pytest.mark.asyncio
async def test_judge_scorer_passed_is_always_none() -> None:
    """Judge is continuous — no hard pass/fail."""
    provider = FakeProvider(responses=['{"score": 0.9, "reasoning": "great"}'])
    scorer = JudgeScorer(provider=provider)
    result = await scorer.score(_judge_case(), "...")
    assert result.passed is None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_both_scorers_conform_to_scorer_protocol() -> None:
    from autoconstitution.benchmark.protocol import Scorer

    coding = CodingScorer()
    judge = JudgeScorer(provider=FakeProvider(responses=["x"]))
    assert isinstance(coding, Scorer)
    assert isinstance(judge, Scorer)


@pytest.mark.asyncio
async def test_scorer_score_results_are_proper_dataclass() -> None:
    result = ScoreResult(score=0.5, detail="test", passed=True)
    assert result.score == 0.5
    assert result.detail == "test"
    assert result.passed is True
