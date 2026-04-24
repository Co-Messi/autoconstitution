"""Shared fixtures for substrate tests.

Provides:
- ``tmp_manifold``: an in-memory SQLite Manifold (fresh per test).
- ``fake_provider``: a callable-mode FakeProvider whose responses are keyed on
  prompt substrings so they survive asyncio.gather reordering.
"""

from __future__ import annotations

import pytest

from autoconstitution.providers.fake import FakeProvider
from autoconstitution.substrate.manifold import Manifold


@pytest.fixture()
def tmp_manifold(tmp_path: "pytest.TempPathFactory") -> Manifold:
    """Fresh in-memory Manifold for each test."""
    db = tmp_path / "substrate_test.db"
    m = Manifold(db_path=db)
    yield m
    m.close()


@pytest.fixture()
def fake_provider() -> FakeProvider:
    """Deterministic callable-mode FakeProvider.

    Responses are keyed on recognisable substrings so tests don't depend on
    call ordering (important for asyncio.gather in counterfactuals).
    """

    def _respond(prompt: str) -> str:
        p = prompt.lower()
        if "critique" in p or "judge" in p or "compliant" in p:
            return '{"verdict": "needs_revision", "critiques": [{"principle": "P5", "quote": "...", "fix": "be concise", "severity": "minor"}]}'
        if "revise" in p or "improved answer" in p or "revision" in p:
            return "Revised answer: this is the improved version."
        if "synthesize" in p or "practice" in p or "curriculum" in p:
            return "Practice problem: implement a binary search function."
        if "extract" in p or "lesson" in p or "skill" in p:
            return "Lesson: use early returns to simplify logic."
        if "shadow" in p or "alternative" in p or "counterfactual" in p:
            return "Alternative approach: iterate with a while loop."
        return "Mock response from FakeProvider."

    return FakeProvider(responses=_respond)
