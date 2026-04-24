"""Tests for autoconstitution.substrate.capability_self_map."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from autoconstitution.substrate.capability_self_map import (
    CapabilitySelfMap,
    Outcome,
    TaskSignature,
)
from autoconstitution.substrate.manifold import Manifold


def _sig(domain: str = "code", difficulty: str = "medium", kind: str = "code") -> TaskSignature:
    return TaskSignature(domain=domain, difficulty=difficulty, kind=kind)


def _outcome(sig: TaskSignature, *, success: bool, score: float, rounds: int = 1) -> Outcome:
    return Outcome(
        signature=sig,
        success=success,
        score=score,
        rounds_used=rounds,
        ts=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────
# TaskSignature
# ─────────────────────────────────────────────


def test_task_signature_hash_equality() -> None:
    a = TaskSignature("math", "easy", "math")
    b = TaskSignature("math", "easy", "math")
    assert a == b
    assert hash(a) == hash(b)


def test_task_signature_inequality() -> None:
    a = TaskSignature("math", "easy", "math")
    b = TaskSignature("code", "easy", "code")
    assert a != b


def test_task_signature_set_dedup() -> None:
    sigs = {
        TaskSignature("math", "easy", "math"),
        TaskSignature("math", "easy", "math"),
        TaskSignature("code", "hard", "code"),
    }
    assert len(sigs) == 2


def test_task_signature_dict_roundtrip() -> None:
    sig = TaskSignature("reasoning", "hard", "free")
    d = sig.to_dict()
    restored = TaskSignature.from_dict(d)
    assert restored == sig


# ─────────────────────────────────────────────
# record / track_record
# ─────────────────────────────────────────────


def test_record_and_track_record(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    sig = _sig()
    csm.record(_outcome(sig, success=True, score=0.9))
    csm.record(_outcome(sig, success=False, score=0.2))
    record = csm.track_record(sig)
    assert record is not None
    assert record.n == 2
    assert record.success_rate == pytest.approx(0.5)


def test_track_record_none_when_empty(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    assert csm.track_record(_sig()) is None


def test_ewma_score_updates(tmp_manifold: Manifold) -> None:
    """EWMA should be closer to recent scores."""
    csm = CapabilitySelfMap(tmp_manifold)
    sig = _sig()
    # Record 5 bad outcomes then 5 good ones
    for _ in range(5):
        csm.record(_outcome(sig, success=False, score=0.1))
    for _ in range(5):
        csm.record(_outcome(sig, success=True, score=0.9))
    record = csm.track_record(sig)
    assert record is not None
    # EWMA after last 5 good outcomes should be > 0.5
    assert record.ewma_score > 0.5


# ─────────────────────────────────────────────
# predict
# ─────────────────────────────────────────────


def test_predict_unseen_returns_half_confidence(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    comp, cal = csm.predict(_sig("unseen", "easy", "free"))
    assert comp == pytest.approx(0.5)
    assert cal == pytest.approx(0.0)


def test_predict_calibration_saturates(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    sig = _sig()
    for i in range(6):  # more than 5 → calibration = 1.0
        csm.record(_outcome(sig, success=True, score=0.8))
    _, cal = csm.predict(sig)
    assert cal == pytest.approx(1.0)


def test_predict_calibration_partial(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    sig = _sig()
    for _ in range(2):
        csm.record(_outcome(sig, success=True, score=0.7))
    _, cal = csm.predict(sig)
    assert cal == pytest.approx(0.4)


# ─────────────────────────────────────────────
# weak_spots
# ─────────────────────────────────────────────


def test_weak_spots_threshold(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    sig_weak = _sig("math", "hard", "math")
    sig_strong = _sig("code", "easy", "code")

    for _ in range(3):
        csm.record(_outcome(sig_weak, success=False, score=0.1))
    for _ in range(3):
        csm.record(_outcome(sig_strong, success=True, score=0.95))

    weak = csm.weak_spots(threshold=0.4, min_n=3)
    ids = [(s.domain, s.difficulty, s.kind) for s in weak]
    assert ("math", "hard", "math") in ids
    assert ("code", "easy", "code") not in ids


def test_weak_spots_min_n_filter(tmp_manifold: Manifold) -> None:
    """A weak score with fewer than min_n attempts should not appear."""
    csm = CapabilitySelfMap(tmp_manifold)
    sig = _sig()
    csm.record(_outcome(sig, success=False, score=0.1))
    csm.record(_outcome(sig, success=False, score=0.1))
    weak = csm.weak_spots(threshold=0.4, min_n=3)
    assert sig not in weak


# ─────────────────────────────────────────────
# blank_regions
# ─────────────────────────────────────────────


def test_blank_regions_all_blank(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    candidates = [_sig("math", "easy", "math"), _sig("code", "hard", "code")]
    blanks = csm.blank_regions(candidates)
    assert len(blanks) == 2


def test_blank_regions_excludes_tried(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    tried = _sig("code", "easy", "code")
    untried = _sig("math", "hard", "math")
    csm.record(_outcome(tried, success=True, score=0.9))
    blanks = csm.blank_regions([tried, untried])
    assert untried in blanks
    assert tried not in blanks


# ─────────────────────────────────────────────
# summary
# ─────────────────────────────────────────────


def test_summary_empty(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    s = csm.summary()
    assert s["total"] == 0
    assert s["signatures"] == []


def test_summary_populated(tmp_manifold: Manifold) -> None:
    csm = CapabilitySelfMap(tmp_manifold)
    sig = _sig("reasoning", "medium", "free")
    for _ in range(3):
        csm.record(_outcome(sig, success=True, score=0.8))
    s = csm.summary()
    assert s["total"] == 1
    row = s["signatures"][0]
    assert row["domain"] == "reasoning"
    assert row["n"] == 3
