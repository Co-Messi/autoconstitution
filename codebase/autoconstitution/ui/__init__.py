"""Terminal UX primitives for autoconstitution.

This module hosts the rendering layer. Leaf widgets live in their own files
(``scoreboard.py``, ``live.py``, ``plain.py``, ``json.py``). The public
``Renderer`` protocol and ``Event`` dataclasses live in ``protocol`` and
``events`` respectively.
"""

from autoconstitution.ui.events import (
    Critique,
    Event,
    LoopError,
    RatchetDecision,
    Revision,
    Role,
    RoleEnd,
    RoleStart,
    RoundEnd,
    RoundStart,
    Token,
)
from autoconstitution.ui.protocol import CapturingRenderer, Renderer
from autoconstitution.ui.scoreboard import render_scoreboard, scoreboard_line

__all__ = [
    "CapturingRenderer",
    "Critique",
    "Event",
    "LoopError",
    "RatchetDecision",
    "Renderer",
    "Revision",
    "Role",
    "RoleEnd",
    "RoleStart",
    "RoundEnd",
    "RoundStart",
    "Token",
    "render_scoreboard",
    "scoreboard_line",
]
