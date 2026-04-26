"""The ``Renderer`` protocol — the one seam between the CAI loop and any UI.

The CAI loop emits :mod:`autoconstitution.ui.events` instances. Anything that
wants to display (or record, or stream elsewhere) those events implements
this protocol. That's the whole contract.

Three concrete renderers ship with autoconstitution:

* ``ui.live`` — a Rich ``Live``/``Layout`` dashboard with role panels, token
  streaming, ratchet scoreboard, and a timer.
* ``ui.plain`` — one ``[role] text`` line per event, for piped output and CI.
* ``ui.json`` — one JSON object per event on stdout, for programmatic consumers.

Tests use :class:`CapturingRenderer` to assert the loop emitted the expected
events without depending on any terminal.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from autoconstitution.ui.events import Event


@runtime_checkable
class Renderer(Protocol):
    """A sink for the tagged-union :data:`~autoconstitution.ui.events.Event`.

    Implementations must be safe to call from any async context — the CAI
    loop emits events inside ``asyncio`` tasks. ``on_event`` must not raise
    on any known event type; unknown events should be ignored gracefully so
    renderers and loops can evolve at different paces.
    """

    supports_streaming: bool
    """True iff this renderer can meaningfully consume per-:class:`Token` events.

    Plain/JSON renderers set this True. The live dashboard sets this True.
    A "summary only" renderer that only wants :class:`RoleEnd` / :class:`RoundEnd`
    sets this False, and the loop will skip :class:`Token` emission for efficiency.
    """

    def on_event(self, event: Event) -> None:
        """Handle one event. Must not raise on known event types."""
        ...

    async def aclose(self) -> None:
        """Flush, finalize, release resources. Safe to call multiple times."""
        ...


class CapturingRenderer:
    """Test helper. Records every event for later assertions."""

    supports_streaming: bool = True

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.closed: bool = False

    def on_event(self, event: Event) -> None:
        self.events.append(event)

    async def aclose(self) -> None:
        self.closed = True


# Backwards-compat alias for peer code that used the in-flight name ``EventSink``
# (see autoconstitution/cai/critique_revision.py). Prefer ``Renderer`` going forward.
EventSink = Renderer

__all__ = ["CapturingRenderer", "EventSink", "Renderer"]
