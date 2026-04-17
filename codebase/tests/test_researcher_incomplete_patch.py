"""Tests for the IncompletePatchError guard in agents.researcher.

Before this change, a proposed CodeChange without a real patch would silently
ship a Python comment string (``"# TODO: Generate patch"``) as its patch. The
guard raises :class:`IncompletePatchError` instead, letting the caller log the
dropped proposal rather than propagate a no-op fake patch downstream.
"""

from __future__ import annotations

import pytest

from autoconstitution.agents.researcher import (
    IncompletePatchError,
    _extract_patch,
)


def test_extract_patch_returns_real_patch() -> None:
    patch = _extract_patch(
        {"suggested_patch": "--- a/x.py\n+++ b/x.py\n+ real diff"},
        "suggested_patch",
        kind="optimization",
        source_agent="test_agent",
    )
    assert "real diff" in patch


def test_extract_patch_raises_when_key_missing() -> None:
    with pytest.raises(IncompletePatchError, match="no 'fix_patch' field"):
        _extract_patch(
            {"unrelated": "x"},
            "fix_patch",
            kind="bug-fix",
            source_agent="agent_a",
        )


def test_extract_patch_raises_when_value_is_empty_string() -> None:
    with pytest.raises(IncompletePatchError, match="no 'suggested_patch' field"):
        _extract_patch(
            {"suggested_patch": ""},
            "suggested_patch",
            kind="generic",
            source_agent="agent_b",
        )


def test_extract_patch_raises_when_value_is_whitespace() -> None:
    with pytest.raises(IncompletePatchError):
        _extract_patch(
            {"suggested_patch": "   \n\t  "},
            "suggested_patch",
            kind="generic",
            source_agent="agent_c",
        )


def test_extract_patch_raises_when_value_is_todo_placeholder() -> None:
    with pytest.raises(IncompletePatchError, match="TODO placeholder"):
        _extract_patch(
            {"config_patch": "# TODO: Generate config patch"},
            "config_patch",
            kind="configuration",
            source_agent="agent_d",
        )


def test_extract_patch_raises_when_value_is_todo_placeholder_with_leading_whitespace() -> None:
    with pytest.raises(IncompletePatchError, match="TODO placeholder"):
        _extract_patch(
            {"refactoring_patch": "   # TODO: nothing yet"},
            "refactoring_patch",
            kind="architecture",
            source_agent="agent_e",
        )


def test_extract_patch_raises_when_value_is_non_string() -> None:
    with pytest.raises(IncompletePatchError):
        _extract_patch(
            {"suggested_patch": 42},
            "suggested_patch",
            kind="optimization",
            source_agent="agent_f",
        )


def test_extract_patch_allows_comments_that_arent_todo_markers() -> None:
    """A real diff can start with regular comment lines — only '# TODO:' is blocked."""
    patch = _extract_patch(
        {"suggested_patch": "# Refactor per review\n--- a/x.py\n+++ b/x.py"},
        "suggested_patch",
        kind="optimization",
        source_agent="agent_g",
    )
    assert "Refactor per review" in patch


def test_incomplete_patch_error_is_a_value_error_subclass() -> None:
    """Callers that catch ValueError also catch this — easy migration path."""
    assert issubclass(IncompletePatchError, ValueError)


def test_error_message_includes_source_agent_for_debugging() -> None:
    with pytest.raises(IncompletePatchError, match="source_x"):
        _extract_patch(
            {},
            "fix_patch",
            kind="bug-fix",
            source_agent="source_x",
        )
