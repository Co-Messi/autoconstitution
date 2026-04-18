"""Tests for ui.plain.PlainRenderer and ui.json_stream.JSONRenderer."""

from __future__ import annotations

import io
import json
from datetime import datetime

import pytest

from autoconstitution.ui.events import (
    Critique,
    LoopError,
    RatchetDecision,
    Revision,
    RoleEnd,
    RoleStart,
    RoundEnd,
    RoundStart,
    Token,
)
from autoconstitution.ui.json_stream import JSONRenderer
from autoconstitution.ui.plain import PlainRenderer

TS = datetime(2026, 4, 17, 14, 0, 0)


# --- PlainRenderer ----------------------------------------------------------


def _plain_lines(events: list) -> list[str]:
    out = io.StringIO()
    r = PlainRenderer(stream=out)
    for e in events:
        r.on_event(e)
    return out.getvalue().splitlines()


def test_plain_round_start_includes_prompt_preview() -> None:
    lines = _plain_lines([RoundStart(round=1, prompt="Hello world", timestamp=TS)])
    assert "[round=1] START" in lines[0]
    assert "Hello world" in lines[0]


def test_plain_role_start_and_end() -> None:
    lines = _plain_lines(
        [
            RoleStart(role="student", round=1, timestamp=TS),
            RoleEnd(role="student", round=1, output="the answer", timestamp=TS),
        ]
    )
    assert lines[0] == "[student round=1] start"
    assert lines[1].startswith("[student round=1] end output=")
    assert "the answer" in lines[1]


def test_plain_token_events_are_suppressed() -> None:
    lines = _plain_lines(
        [
            RoleStart(role="student", round=1, timestamp=TS),
            Token(role="student", round=1, text="tok1", timestamp=TS),
            Token(role="student", round=1, text="tok2", timestamp=TS),
            RoleEnd(role="student", round=1, output="tok1tok2", timestamp=TS),
        ]
    )
    assert len(lines) == 2
    assert all("token" not in line.lower() for line in lines)


def test_plain_critique_emits_verdict_and_count() -> None:
    lines = _plain_lines(
        [Critique(round=2, verdict="needs_revision", critique_count=3, raw="{}", timestamp=TS)]
    )
    assert "verdict=needs_revision" in lines[0]
    assert "critiques=3" in lines[0]


def test_plain_revision_identical_is_flagged() -> None:
    lines = _plain_lines(
        [Revision(round=2, before="a", after="a", identical=True, timestamp=TS)]
    )
    assert "identical" in lines[0].lower()


def test_plain_ratchet_decision_includes_all_fields() -> None:
    lines = _plain_lines(
        [
            RatchetDecision(
                round=3,
                metric_name="accuracy",
                decision="keep",
                score=0.92,
                previous_best=0.88,
                improvement_delta=0.04,
                timestamp=TS,
            )
        ]
    )
    assert "KEEP" in lines[0]
    assert "accuracy" in lines[0]
    assert "0.9200" in lines[0]
    assert "+0.0400" in lines[0]


def test_plain_round_end_converged_vs_continues() -> None:
    lines = _plain_lines(
        [
            RoundEnd(round=1, converged=True, timestamp=TS),
            RoundEnd(round=2, converged=False, timestamp=TS),
        ]
    )
    assert "converged" in lines[0]
    assert "continues" in lines[1]


def test_plain_loop_error_with_and_without_role() -> None:
    lines = _plain_lines(
        [
            LoopError(round=1, role="judge", message="timeout", timestamp=TS),
            LoopError(round=2, role=None, message="bootstrap failed", timestamp=TS),
        ]
    )
    assert "judge round=1" in lines[0]
    assert "timeout" in lines[0]
    assert "round=2" in lines[1]
    assert "bootstrap failed" in lines[1]


@pytest.mark.asyncio
async def test_plain_aclose_flushes_and_is_idempotent() -> None:
    out = io.StringIO()
    r = PlainRenderer(stream=out)
    r.on_event(RoleStart(role="student", round=1, timestamp=TS))
    await r.aclose()
    await r.aclose()  # idempotent


def test_plain_supports_streaming_is_false() -> None:
    assert PlainRenderer.supports_streaming is False


# --- JSONRenderer -----------------------------------------------------------


def _json_records(events: list) -> list[dict]:
    out = io.StringIO()
    r = JSONRenderer(stream=out)
    for e in events:
        r.on_event(e)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_json_every_event_serializes_round_trip() -> None:
    events = [
        RoundStart(round=1, prompt="p", timestamp=TS),
        RoleStart(role="student", round=1, timestamp=TS),
        Token(role="student", round=1, text="t", timestamp=TS),
        RoleEnd(role="student", round=1, output="o", timestamp=TS),
        Critique(round=1, verdict="compliant", critique_count=0, raw="{}", timestamp=TS),
        Revision(round=1, before="a", after="b", identical=False, timestamp=TS),
        RatchetDecision(
            round=1,
            metric_name="m",
            decision="keep",
            score=1.0,
            previous_best=0.5,
            improvement_delta=0.5,
            timestamp=TS,
        ),
        RoundEnd(round=1, converged=True, timestamp=TS),
        LoopError(round=1, role="judge", message="x", timestamp=TS),
    ]
    records = _json_records(events)
    assert len(records) == len(events)
    types = [r["type"] for r in records]
    assert types == [
        "round_start",
        "role_start",
        "token",
        "role_end",
        "critique",
        "revision",
        "ratchet_decision",
        "round_end",
        "loop_error",
    ]


def test_json_timestamp_is_renamed_to_ts_and_iso_formatted() -> None:
    records = _json_records([RoleStart(role="student", round=1, timestamp=TS)])
    assert records[0]["ts"] == "2026-04-17T14:00:00"
    assert "timestamp" not in records[0]


def test_json_token_includes_text() -> None:
    records = _json_records([Token(role="student", round=2, text="chunk", timestamp=TS)])
    assert records[0]["text"] == "chunk"
    assert records[0]["role"] == "student"
    assert records[0]["round"] == 2


def test_json_ratchet_decision_carries_numeric_fields() -> None:
    records = _json_records(
        [
            RatchetDecision(
                round=3,
                metric_name="acc",
                decision="discard",
                score=0.1,
                previous_best=0.5,
                improvement_delta=-0.4,
                timestamp=TS,
            )
        ]
    )
    rec = records[0]
    assert rec["decision"] == "discard"
    assert rec["score"] == 0.1
    assert rec["previous_best"] == 0.5
    assert rec["improvement_delta"] == -0.4


def test_json_loop_error_role_none_serializes_to_null() -> None:
    records = _json_records([LoopError(round=1, role=None, message="boom", timestamp=TS)])
    assert records[0]["role"] is None


def test_json_supports_streaming_is_true() -> None:
    assert JSONRenderer.supports_streaming is True


@pytest.mark.asyncio
async def test_json_aclose_is_idempotent() -> None:
    out = io.StringIO()
    r = JSONRenderer(stream=out)
    r.on_event(RoleStart(role="student", round=1, timestamp=TS))
    await r.aclose()
    await r.aclose()


def test_json_each_line_is_compact_single_line() -> None:
    out = io.StringIO()
    r = JSONRenderer(stream=out)
    r.on_event(RoundStart(round=1, prompt="p", timestamp=TS))
    text = out.getvalue()
    # Exactly one newline, compact separators, no leading whitespace
    assert text.count("\n") == 1
    assert not text.startswith(" ")
    # Compact: no spaces after separators
    assert '", "' not in text
    assert '":' in text or '":"' in text
