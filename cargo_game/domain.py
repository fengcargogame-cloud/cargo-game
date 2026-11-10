"""Plain data types: cargo pieces, placements, actions and tunnels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class ActionMode(str, Enum):
    WHOLE = "whole"
    CUT_AT_MAX = "cut_at_max"


MODES: Tuple[ActionMode, ActionMode] = (ActionMode.WHOLE, ActionMode.CUT_AT_MAX)


@dataclass(frozen=True)
class CargoPiece:
    piece_id: int
    original_id: int
    length: int
    color: int
    min_left: int
    max_right: int
    was_cut: bool = False


@dataclass(frozen=True)
class PlacedPiece:
    piece_id: int
    original_id: int
    tunnel_id: int
    left: int
    right: int
    length: int
    color: int
    max_right: int


@dataclass(frozen=True)
class Action:
    piece_id: int          # stable id, not a list index
    tunnel_id: int
    mode: ActionMode


class Tunnel:
    def __init__(self, tunnel_id: int, placed: Optional[List[PlacedPiece]] = None) -> None:
        self.tunnel_id = tunnel_id
        self.placed: List[PlacedPiece] = placed if placed is not None else []

    def clone(self) -> "Tunnel":
        # PlacedPiece is immutable, so a shallow list copy is a full copy.
        return Tunnel(self.tunnel_id, list(self.placed))

    @property
    def current_right(self) -> int:
        return self.placed[-1].right if self.placed else 0

    @property
    def last_color(self) -> Optional[int]:
        return self.placed[-1].color if self.placed else None

    def next_left_for(self, piece: CargoPiece) -> int:
        return max(self.current_right, piece.min_left)
