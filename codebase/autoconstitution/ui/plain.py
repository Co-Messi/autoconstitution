"""Line-based ``Renderer`` for non-TTY environments.

When stdout is piped or the user passes ``--ui=plain``, we want readable,
grep-able output instead of the Rich ``Live`` dashboard. Each event emits
one prefixed line: ``[role round=N] text``. Tokens are accumulated and
flushed on RoleEnd so streaming runs don't produce per-chunk noise.
"""

from __future__ import annotations

import contextlib
import sys
from typing import IO

from autoconstitution.ui.events import (
    Critique,
    Event,
    LoopError,
    RatchetDecision,
    Revision,
    RoleEnd,
    RoleStart,
    RoundEnd,
    RoundStart,
    Token,
)


class PlainRenderer:
    """One-line-per-event renderer. Safe under piping.

    Declares ``supports_streaming = False`` so the loop doesn't bother
    emitting per-token :class:`Token` events it would just accumulate.
    Call :meth:`on_event` for every event; :meth:`aclose` is a no-op.
    """

    supports_streaming: bool = False

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._out: IO[str] = stream if stream is not None else sys.stdout

    def on_event(self, event: Event) -> None:
        line = self._format(event)
        if line is not None:
            print(line, file=self._out, flush=True)

    async def aclose(self) -> None:
        # Stream may already be closed / detached — suppress harmlessly.
        with contextlib.suppress(ValueError, OSError):
            self._out.flush()

    def _format(self, event: Event) -> str | None:
        match event:
            case RoundStart(round=r, prompt=prompt):
                preview = _truncate(prompt, 80)
                return f"[round={r}] START prompt={preview!r}"
            case RoleStart(role=role, round=r):
                return f"[{role} round={r}] start"
            case Token():
                return None  # never emit per-token lines
            case RoleEnd(role=role, round=r, output=output):
                preview = _truncate(output, 120)
                return f"[{role} round={r}] end output={preview!r}"
            case Critique(round=r, verdict=verdict, critique_count=count):
                return f"[judge round={r}] verdict={verdict} critiques={count}"
            case Revision(round=r, identical=identical):
                tag = "(identical — halting)" if identical else ""
                return f"[student round={r}] revision {tag}".rstrip()
            case RatchetDecision(
                round=r, metric_name=metric, decision=decision, score=score,
                previous_best=prev, improvement_delta=delta,
            ):
                prev_s = f"{prev:.4f}" if prev is not None else "—"
                return (
                    f"[ratchet round={r}] {decision.upper()} metric={metric} "
                    f"score={score:.4f} prev={prev_s} delta={delta:+.4f}"
                )
            case RoundEnd(round=r, converged=converged):
                tag = "converged" if converged else "continues"
                return f"[round={r}] END {tag}"
            case LoopError(round=r, role=role, message=msg):
                where = f"{role} round={r}" if role else f"round={r}"
                return f"[error {where}] {msg}"
        return None  # pragma: no cover — match is exhaustive over Event union


def _truncate(text: str, n: int) -> str:
    """Truncate ``text`` to at most ``n`` chars, collapsing internal newlines."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= n:
        return collapsed
    return collapsed[: n - 1] + "…"


__all__ = ["PlainRenderer"]
