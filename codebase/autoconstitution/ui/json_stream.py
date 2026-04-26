"""JSON-lines ``Renderer``: one JSON object per event on stdout.

For programmatic consumers (``--ui=json``) and CI log scraping. Every line
is a self-describing event record with ``type``, ``ts``, and type-specific
fields. Parseable with ``jq``, ``pandas.read_json(lines=True)``, or any
JSONL consumer.

Tokens ARE emitted in this mode (``supports_streaming = True``) so a
consumer can reconstruct the full streaming timeline. If you don't want
them, filter ``type != "token"`` on the consumer side.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import sys
from datetime import datetime
from typing import IO, Any

from autoconstitution.ui.events import Event


class JSONRenderer:
    """Emits one JSON object per event, newline-delimited.

    Output format (one per line):

    .. code-block:: json

       {"type": "role_start", "ts": "2026-04-17T14:00:00", "role": "student", "round": 1}

    Consumer snippet:

    .. code-block:: python

       import json
       for line in sys.stdin:
           event = json.loads(line)
           ...
    """

    supports_streaming: bool = True

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._out: IO[str] = stream if stream is not None else sys.stdout

    def on_event(self, event: Event) -> None:
        record = _event_to_dict(event)
        self._out.write(json.dumps(record, separators=(",", ":"), default=_default))
        self._out.write("\n")
        self._out.flush()

    async def aclose(self) -> None:
        with contextlib.suppress(ValueError, OSError):
            self._out.flush()


_EVENT_TYPE_NAMES: dict[str, str] = {
    "RoundStart": "round_start",
    "RoleStart": "role_start",
    "Token": "token",
    "RoleEnd": "role_end",
    "Critique": "critique",
    "Revision": "revision",
    "RatchetDecision": "ratchet_decision",
    "RoundEnd": "round_end",
    "LoopError": "loop_error",
}


def _event_to_dict(event: Event) -> dict[str, Any]:
    """Render an :data:`Event` into a JSON-friendly dict with ``type`` + fields."""
    cls_name = type(event).__name__
    record: dict[str, Any] = {"type": _EVENT_TYPE_NAMES.get(cls_name, cls_name.lower())}
    for f in dataclasses.fields(event):
        value = getattr(event, f.name)
        if f.name == "timestamp":
            record["ts"] = value.isoformat() if isinstance(value, datetime) else value
        else:
            record[f.name] = value
    return record


def _default(obj: Any) -> Any:
    """Fall-back encoder for non-stdlib JSON types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


__all__ = ["JSONRenderer"]
