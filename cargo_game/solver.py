"""Exact memoised search, for checking the agent on small instances."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .domain import Action, ActionMode
from .env import CargoTunnelEnv


def brute_force_optimum(env: CargoTunnelEnv) -> Tuple[float, List[Action]]:
    """Exhaustive search over every raw (unmasked) action; only for tiny games like StaticGameConfig.

    Memoised on everything future rewards depend on, so placement orders that reach the same
    board are searched once instead of once per permutation. Each cargo has at most one
    unassigned piece at a time, so moves are keyed by original_id (piece ids of cut remainders
    depend on cut order) and mapped back to piece ids by replaying the best line at the end.
    """
    memo: Dict[tuple, Tuple[float, List[Tuple[int, int, ActionMode]]]] = {}

    def key(e: CargoTunnelEnv) -> tuple:
        open_ids = {p.original_id for p in e.unassigned}
        worst: Dict[int, int] = {}
        for p in e.placed:
            if p.original_id in open_ids:
                worst[p.original_id] = max(worst.get(p.original_id, 0), p.right)
        pieces = tuple(sorted((p.original_id, p.length) for p in e.unassigned))
        return (tuple((t.current_right, t.last_color) for t in e.tunnels),
                pieces, tuple(sorted(worst.items())), e.step_count)

    def search(e: CargoTunnelEnv) -> Tuple[float, List[Tuple[int, int, ActionMode]]]:
        if e.is_terminal():
            return 0.0, []
        k = key(e)
        if k in memo:
            return memo[k]
        best: Tuple[float, List[Tuple[int, int, ActionMode]]] = (-math.inf, [])
        for piece in list(e.unassigned):
            for t in range(e.config.num_tunnels):
                for mode in e.raw_modes(piece, t):
                    child = e.clone()
                    r, done, _ = child.step(Action(piece.piece_id, t, mode))
                    rest, rest_moves = (0.0, []) if done else search(child)
                    if r + rest > best[0]:
                        best = (r + rest, [(piece.original_id, t, mode)] + rest_moves)
        memo[k] = best
        return best

    value, moves = search(env)
    replay, actions = env.clone(), []
    for original_id, t, mode in moves:
        piece = next(p for p in replay.unassigned if p.original_id == original_id)
        actions.append(Action(piece.piece_id, t, mode))
        replay.step(actions[-1])
    return value, actions
