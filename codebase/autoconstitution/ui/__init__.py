"""Terminal UX primitives for autoconstitution.

This module hosts the rendering layer. Leaf widgets live in their own files
(``scoreboard.py``, ``live.py``, ``plain.py``, ``json.py``). The public
``Renderer`` protocol and ``Event`` dataclasses are added here as slices 2-4
land.
"""

from autoconstitution.ui.scoreboard import render_scoreboard, scoreboard_line

__all__ = ["render_scoreboard", "scoreboard_line"]
