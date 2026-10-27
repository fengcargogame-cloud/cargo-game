"""
Cargo-Tunnel placement: PyTorch + AlphaZero-style RL + 3-stage hierarchical MCTS.

Every placement is decided as three successive tree levels:

    real state s  ──(stage 0: which piece?)──►  (s, piece)
                  ──(stage 1: which tunnel?)─►  (s, piece, tunnel)
                  ──(stage 2: whole or cut?)─►  real state s'   (env.step happens here)

Each level is a genuine MCTS node with its own visit counts, Q-values and PUCT
selection, so branching per level is M, K and <=2 instead of M*K*2 at one level.

The network mirrors the tree: a set-transformer encodes tunnels + pieces, then
  - piece head   : pointer over candidate pieces          + V(s)
  - tunnel head  : pointer over tunnels, given the piece   + Q(s, piece)
  - mode head    : WHOLE/CUT, given piece and tunnel       + Q(s, piece, tunnel)
Each head is trained on the visit distribution of its own tree level.

Also included: urgency (deadline) prior, action masking, top-K expansion,
Dirichlet root noise, min-max normalised Q, and subtree reuse between moves.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================


@dataclass
class GameConfig:
    """Settings shared by every game variant. Use DynamicGameConfig or StaticGameConfig."""

    num_tunnels: int = 5
    tunnel_length: int = 10
    max_cargos: int = 10
    num_colors: int = 3

    penalty_exceed_max: float = 8.0
    penalty_color_change: float = 1.0
    penalty_cut: float = 1.5
    penalty_gap: float = 1.5             # per unit of empty space left between two cargos in a tunnel
    penalty_unplaced: float = 5.0
    cargo_reward: float = 2.0            # paid each time a cargo is fully placed
    completion_bonus: float = 5.0        # paid once every cargo is placed

    # Game rules
    enforce_tunnel_end: bool = True      # a placed segment may not extend past tunnel_length
    max_game_steps: int = 300

    # Action masking / pruning
    max_piece_candidates: int = 20       # urgency-ranked pieces considered at stage 0
    # Off by default: a cut is only possible when WHOLE overflows, so this mask would leave
    # stage 2 with a single option every time and the mode head would learn nothing.
    mask_overflow_whole: bool = False    # drop WHOLE when it overflows and CUT is possible
    mask_color_change: bool = True       # avoid colour change into a tunnel when a same-colour piece fits there

    # MCTS
    mcts_simulations: int = 160
    c_puct: float = 1.5
    top_k_pieces: int = 15                # top-K expansion at stage 0
    top_k_tunnels: int = 10               # top-K expansion at stage 1
    urgency_prior_weight: float = 0.25   # blend of urgency prior into piece prior
    dirichlet_alpha: float = 0.5
    dirichlet_frac: float = 0.25
    temperature_moves: int = 8
    gamma: float = 1.0
    value_scale: float = 50.0            # rewards are divided by this for the value heads
                                         # (StaticGameConfig overrides it for its own reward range)

    # Model
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    train_steps_per_episode: int = 8
    num_episodes: int = 2000
    replay_capacity: int = 20000
    device: str = "cpu"

    def episode_pieces(self, rng: random.Random) -> List["CargoPiece"]:
        raise NotImplementedError


@dataclass
class DynamicGameConfig(GameConfig):
    """A fresh random set of pieces every episode."""

    # Data generation
    min_cargo_length: int = 3
    max_cargo_length: int = 12
    min_min_left: int = 0
    max_min_left: int = 20
    min_slack: int = 0
    max_slack: int = 8

    def episode_pieces(self, rng: random.Random) -> List["CargoPiece"]:
        return sample_episode_pieces(self, rng)


@dataclass
class StaticGameConfig(GameConfig):
 
    # num_colors: int = 4
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (8, 0, 0, 4),
    #     (11, 0, 4, 10),
    #     (5, 1, 0, 5),
    #     (5, 1, 5, 10),
    #     (4, 3, 0, 2), 
    #     (7, 3, 2, 6),
    #     (9, 3, 5, 10),
    # )

    # value_scale: float = 40.0            # returns are about -50..+12.5
    # num_episodes: int = 1400
    # mcts_simulations: int = 200

    # num_tunnels: int = 6
    # tunnel_length: int = 11
    # num_colors: int = 3
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (7, 0, 0, 4),
    #     (7, 0, 3, 7),
    #     (4, 1, 0, 2),
    #     (6, 1, 2, 5),
    #     (6, 1, 5, 11), 
    #     (8, 0, 7, 11),
    #     (6, 1, 5, 11),
    #     (8, 2, 0, 4),
    #     (11, 2, 4, 10),
    #     (2, 2, 9, 11),
    # )

    # num_tunnels: int = 6
    # tunnel_length: int = 12
    # num_colors: int = 3
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (6, 0, 0, 3),
    #     (7, 1, 0, 4),
    #     (8, 1, 4, 12),
    #     (5, 1, 3, 8),
    #     (8, 0, 3, 7), 
    #     (8, 2, 0, 4),
    #     (12, 2, 4, 10),
    #     (6, 1, 8, 12),
    #     (4, 0, 7, 9),
    #     (6, 0, 9, 12),
    # )
    # value_scale: float = 70.0            # returns are about -50..+12.5
    # num_episodes: int = 2000
    # mcts_simulations: int = 300

    value_scale: float = 90.0            # returns are about -50..+12.5
    num_episodes: int = 5000
    mcts_simulations: int = 300
    num_tunnels: int = 7
    tunnel_length: int = 16
    num_colors: int = 4
    # (length, color, min_left, max_right) per cargo
    pieces: Tuple[Tuple[int, int, int, int], ...] = (
        (4, 0, 3, 5),
        (9, 0, 0, 3),
        (4, 0, 3, 7),
        (10, 0, 5, 9),
        (5, 3, 0, 5), 
        (5, 1, 0, 5),
        (7, 3, 5, 12),
        (4, 3, 12, 16),
        (5, 1, 5, 10),
        (6, 1, 10, 16),
        (19, 0, 9, 16),
        (6, 2, 0, 3),
        (19, 2, 3, 13),
        (7, 2, 12, 16),
    )

    # value_scale: float = 90.0            # returns are about -50..+12.5
    # num_episodes: int = 2000
    # mcts_simulations: int = 300
    # num_tunnels: int = 18
    # tunnel_length: int = 18
    # num_colors: int = 11
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (51, 0, 0, 17),
    #     (15, 1, 0, 4),
    #     (140, 2, 0, 18),
    #     (64, 3, 0, 16),
    #     (1, 4, 4, 5), 
    #     (1, 5, 5, 6),
    #     (1, 6, 6, 7),
    #     (1, 7, 7, 8),
    #     (6, 8, 7, 12),
    #     (11, 9, 7, 17),
    #     (12, 10, 10, 17),
    # )

    # penalty_exceed_max: float = 15.0
    # penalty_color_change: float = 1.0
    # penalty_cut: float = 1.0
    # penalty_gap: float = 1.0
    # penalty_unplaced: float = 5.0
    # cargo_reward: float = 5.0            # paid each time a cargo is fully placed
    # completion_bonus: float = 5.0        # paid once every cargo is placed

    # value_scale: float = 150.0            # returns are about -50..+12.5
    # num_episodes: int = 2000
    # mcts_simulations: int = 300
    # num_tunnels: int = 9
    # tunnel_length: int = 17
    # num_colors: int = 10
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (51, 0, 0, 17),
    #     (15, 1, 0, 4),
    #     (50, 2, 0, 18),
    #     (1, 3, 4, 5), 
    #     (1, 4, 5, 6),
    #     (1, 5, 6, 7),
    #     (1, 6, 7, 8),
    #     (6, 7, 7, 12),
    #     (11, 8, 7, 17),
    #     (12, 9, 10, 17),
    # )

    # penalty_exceed_max: float = 10.0
    # penalty_color_change: float = 1.0
    # penalty_cut: float = 1.5
    # penalty_gap: float = 1.0
    # penalty_unplaced: float = 3.0
    # cargo_reward: float = 8.0            # paid each time a cargo is fully placed
    # completion_bonus: float = 20.0        # paid once every cargo is placed

    # value_scale: float = 100.0            # returns are about -50..+12.5
    # num_episodes: int = 2000
    # mcts_simulations: int = 300
    # num_tunnels: int = 6
    # tunnel_length: int = 17
    # num_colors: int = 10
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (15, 1, 0, 4),
    #     (50, 2, 0, 18),
    #     (1, 3, 4, 5), 
    #     (1, 4, 5, 6),
    #     (1, 5, 6, 7),
    #     (1, 6, 7, 8),
    #     (6, 7, 7, 12),
    #     (11, 8, 7, 17),
    #     (12, 9, 10, 17),
    # )

    def __post_init__(self) -> None:
        self.max_cargos = len(self.pieces)

    def episode_pieces(self, rng: random.Random) -> List["CargoPiece"]:
        return [CargoPiece(i + 1, i, length, color, min_left, max_right)
                for i, (length, color, min_left, max_right) in enumerate(self.pieces)]


# ============================================================
# Domain model
# ============================================================


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


# ============================================================
# Dataset generation
# ============================================================


def sample_episode_pieces(config: DynamicGameConfig, rng: random.Random) -> List[CargoPiece]:
    cfg = config
    pieces: List[CargoPiece] = []
    for original_id in range(cfg.max_cargos):
        length = rng.randint(cfg.min_cargo_length, cfg.max_cargo_length)
        color = rng.randint(0, cfg.num_colors - 1)
        min_left = rng.randint(cfg.min_min_left, cfg.max_min_left)
        slack = rng.randint(cfg.min_slack, cfg.max_slack)
        max_right = min(cfg.tunnel_length, min_left + length + slack)
        pieces.append(CargoPiece(original_id + 1, original_id, length, color, min_left, max_right))
    return pieces


# ============================================================
# Environment
# ============================================================


class CargoTunnelEnv:
    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.tunnels: List[Tunnel] = []
        self.unassigned: List[CargoPiece] = []
        self.placed: List[PlacedPiece] = []
        self.original_lengths: Dict[int, int] = {}
        self.original_max_rights: Dict[int, int] = {}
        self.next_piece_id = 1
        self.num_cuts = 0
        self.step_count = 0
        self._cand_cache: Optional[List[Tuple[CargoPiece, float]]] = None

    # ---------------- lifecycle ----------------

    def reset(self, seed: Optional[int] = None, pieces: Optional[List[CargoPiece]] = None) -> None:
        cfg = self.config
        self.tunnels = [Tunnel(i) for i in range(cfg.num_tunnels)]
        self.unassigned = list(pieces) if pieces is not None else cfg.episode_pieces(random.Random(seed))
        self.placed = []
        self.original_lengths = {p.original_id: p.length for p in self.unassigned}
        self.original_max_rights = {p.original_id: p.max_right for p in self.unassigned}
        self.next_piece_id = max((p.piece_id for p in self.unassigned), default=0) + 1
        self.num_cuts = 0
        self.step_count = 0
        self._cand_cache = None

    def clone(self) -> "CargoTunnelEnv":
        env = CargoTunnelEnv(self.config)
        env.tunnels = [t.clone() for t in self.tunnels]
        env.unassigned = list(self.unassigned)
        env.placed = list(self.placed)
        env.original_lengths = self.original_lengths          # read-only after reset
        env.original_max_rights = self.original_max_rights    # read-only after reset
        env.next_piece_id = self.next_piece_id
        env.num_cuts = self.num_cuts
        env.step_count = self.step_count
        return env

    # ---------------- placement geometry ----------------

    def _fits(self, left: int, length: int) -> bool:
        return (not self.config.enforce_tunnel_end) or (left + length <= self.config.tunnel_length)

    def cut_length(self, piece: CargoPiece, tunnel_id: int) -> int:
        """Length placed by CUT_AT_MAX (0 if a cut is not possible)."""
        left = self.tunnels[tunnel_id].next_left_for(piece)
        fit = piece.max_right - left
        if 0 < fit < piece.length and self._fits(left, fit):
            return fit
        return 0

    def raw_modes(self, piece: CargoPiece, tunnel_id: int) -> List[ActionMode]:
        left = self.tunnels[tunnel_id].next_left_for(piece)
        modes = []
        if self._fits(left, piece.length):
            modes.append(ActionMode.WHOLE)
        if self.cut_length(piece, tunnel_id) > 0:
            modes.append(ActionMode.CUT_AT_MAX)
        return modes

    # ---------------- stage-wise legality (with masking) ----------------

    def candidates(self) -> List[Tuple[CargoPiece, float]]:
        """Urgency-ranked placeable pieces: [(piece, urgency)], most urgent first."""
        if self._cand_cache is not None:
            return self._cand_cache
        scored: List[Tuple[float, int, CargoPiece]] = []
        for piece in self.unassigned:
            best_slack = None
            for t in range(self.config.num_tunnels):
                if not self.raw_modes(piece, t):
                    continue
                slack = piece.max_right - self.tunnels[t].next_left_for(piece) - piece.length
                best_slack = slack if best_slack is None else max(best_slack, slack)
            if best_slack is None:
                continue  # cannot be placed anywhere right now
            urgency = 1.0 / (1.0 + max(0.0, float(best_slack)))
            scored.append((urgency, piece.piece_id, piece))
        scored.sort(key=lambda x: (-x[0], x[1]))
        self._cand_cache = [(p, u) for u, _, p in scored[: self.config.max_piece_candidates]]
        return self._cand_cache

    def legal_tunnels(self, piece: CargoPiece) -> List[int]:
        raw = [t for t in range(self.config.num_tunnels) if self.raw_modes(piece, t)]
        if not self.config.mask_color_change:
            return raw
        cand_pieces = [p for p, _ in self.candidates()]
        kept = []
        for t in raw:
            last = self.tunnels[t].last_color
            if last is not None and last != piece.color:
                same_colour_fits = any(
                    p.color == last and p.piece_id != piece.piece_id and self.raw_modes(p, t)
                    for p in cand_pieces
                )
                if same_colour_fits:
                    continue
            kept.append(t)
        return kept or raw   # never mask everything away

    def legal_modes(self, piece: CargoPiece, tunnel_id: int) -> List[ActionMode]:
        modes = self.raw_modes(piece, tunnel_id)
        if self.config.mask_overflow_whole and len(modes) == 2:
            left = self.tunnels[tunnel_id].next_left_for(piece)
            if left + piece.length > piece.max_right:
                return [ActionMode.CUT_AT_MAX]
        return modes

    def piece_by_id(self, piece_id: int) -> CargoPiece:
        for p in self.unassigned:
            if p.piece_id == piece_id:
                return p
        raise KeyError(piece_id)

    def is_terminal(self) -> bool:
        return (
            not self.unassigned
            or self.step_count >= self.config.max_game_steps
            or not self.candidates()
        )

    # ---------------- transition ----------------

    def step(self, action: Action) -> Tuple[float, bool, Dict]:
        cfg = self.config
        piece = self.piece_by_id(action.piece_id)
        if action.mode not in self.raw_modes(piece, action.tunnel_id):
            raise ValueError(f"Illegal action {action}")

        self.step_count += 1
        overflow_before = self.compute_overflow_penalty()
        tunnel = self.tunnels[action.tunnel_id]
        left = tunnel.next_left_for(piece)

        penalty = 0.0
        if tunnel.last_color is not None and tunnel.last_color != piece.color:
            penalty += cfg.penalty_color_change
        # if tunnel.placed:
        #     penalty += (left - tunnel.current_right) * cfg.penalty_gap

        if action.mode == ActionMode.WHOLE:
            placed_len = piece.length
        else:
            placed_len = self.cut_length(piece, action.tunnel_id)
            penalty += cfg.penalty_cut
            self.num_cuts += 1
        remainder_len = piece.length - placed_len

        self.unassigned = [p for p in self.unassigned if p.piece_id != piece.piece_id]
        placed = PlacedPiece(piece.piece_id, piece.original_id, action.tunnel_id,
                             left, left + placed_len, placed_len, piece.color, piece.max_right)
        tunnel.placed.append(placed)
        self.placed.append(placed)

        if remainder_len > 0:
            self.unassigned.append(CargoPiece(self.next_piece_id, piece.original_id, remainder_len,
                                              piece.color, piece.min_left, piece.max_right, True))
            self.next_piece_id += 1

        self._cand_cache = None
        penalty += self.compute_overflow_penalty() - overflow_before

        bonus = cfg.cargo_reward if remainder_len == 0 else 0.0
        done = self.is_terminal()
        if done:
            bonus += cfg.completion_bonus if not self.unassigned else -self.compute_unplaced_penalty()
        return -penalty + bonus, done, {"placed_len": placed_len, "remainder": remainder_len}

    # ---------------- scoring ----------------

    def compute_overflow_penalty(self) -> float:
        worst: Dict[int, int] = {}
        for p in self.placed:
            worst[p.original_id] = max(worst.get(p.original_id, 0), p.right)
        return sum(max(0, r - self.original_max_rights[o]) for o, r in worst.items()) * self.config.penalty_exceed_max

    def compute_color_change_penalty(self) -> float:
        changes = sum(
            1 for t in self.tunnels for a, b in zip(t.placed, t.placed[1:]) if a.color != b.color
        )
        return changes * self.config.penalty_color_change

    def compute_gap_penalty(self) -> float:
        gaps = sum(b.left - a.right for t in self.tunnels for a, b in zip(t.placed, t.placed[1:]))
        return gaps * self.config.penalty_gap

    def compute_unplaced_penalty(self) -> float:
        placed_len: Dict[int, int] = {}
        for p in self.placed:
            placed_len[p.original_id] = placed_len.get(p.original_id, 0) + p.length
        unfinished = sum(1 for o, L in self.original_lengths.items() if placed_len.get(o, 0) < L)
        return unfinished * self.config.penalty_unplaced

    def total_penalty(self) -> float:
        return (
            self.compute_overflow_penalty()
            + self.compute_color_change_penalty()
            # + self.compute_gap_penalty()
            + self.num_cuts * self.config.penalty_cut
            + (self.compute_unplaced_penalty() if self.is_terminal() else 0.0)
        )

    # ---------------- features for the network ----------------

    def piece_feature_dim(self) -> int:
        return 6 + self.config.num_colors

    def tunnel_feature_dim(self) -> int:
        return 2 + self.config.num_colors + 1

    PAIR_DIM = 8
    MODE_DIM = 5
    GLOBAL_DIM = 3

    def piece_features(self) -> np.ndarray:
        cfg, L = self.config, float(self.config.tunnel_length)
        rows = []
        for p, urg in self.candidates():
            color = np.zeros(cfg.num_colors)
            color[p.color] = 1.0
            rows.append(np.concatenate([[
                p.length / L, p.min_left / L, p.max_right / L,
                max(0, p.max_right - p.min_left - p.length) / L,
                1.0 if p.was_cut else 0.0, urg,
            ], color]))
        return np.asarray(rows, dtype=np.float32).reshape(-1, self.piece_feature_dim())

    def tunnel_features(self) -> np.ndarray:
        cfg, L = self.config, float(self.config.tunnel_length)
        rows = []
        for t in self.tunnels:
            color = np.zeros(cfg.num_colors + 1)
            color[0 if t.last_color is None else t.last_color + 1] = 1.0
            rows.append(np.concatenate([[t.current_right / L, len(t.placed) / max(1, cfg.max_cargos)], color]))
        return np.asarray(rows, dtype=np.float32)

    def global_features(self) -> np.ndarray:
        cfg = self.config
        return np.asarray([
            len(self.unassigned) / max(1, cfg.max_cargos),
            self.num_cuts / max(1, cfg.max_cargos),
            self.step_count / max(1, cfg.max_game_steps),
        ], dtype=np.float32)

    def pair_features(self, piece: CargoPiece) -> np.ndarray:
        """(K, PAIR_DIM): how the given piece would land in each tunnel."""
        cfg, L = self.config, float(self.config.tunnel_length)
        rows = []
        for t in self.tunnels:
            left = t.next_left_for(piece)
            last = t.last_color
            match = 0.5 if last is None else (1.0 if last == piece.color else 0.0)
            rows.append([
                left / L,
                (left - t.current_right) / L,
                match,
                max(0, left + piece.length - piece.max_right) / L,
                1.0 if self._fits(left, piece.length) else 0.0,
                1.0 if self.cut_length(piece, t.tunnel_id) > 0 else 0.0,
                t.current_right / L,
                len(t.placed) / max(1, cfg.max_cargos),
            ])
        return np.asarray(rows, dtype=np.float32)

    def mode_features(self, piece: CargoPiece, tunnel_id: int) -> np.ndarray:
        L = float(self.config.tunnel_length)
        left = self.tunnels[tunnel_id].next_left_for(piece)
        cut = self.cut_length(piece, tunnel_id)
        return np.asarray([
            max(0, left + piece.length - piece.max_right) / L,
            cut / piece.length,
            (piece.length - cut) / piece.length if cut else 0.0,
            1.0 if cut else 0.0,
            1.0 if piece.was_cut else 0.0,
        ], dtype=np.float32)

    # ---------------- display ----------------

    def render_text(self) -> str:
        lines = ["=" * 72,
                 f"Steps={self.step_count} | Cuts={self.num_cuts} | Unassigned={len(self.unassigned)}",
                 f"Penalty: overflow={self.compute_overflow_penalty():.1f} "
                 f"color={self.compute_color_change_penalty():.1f} "
                 # f"gap={self.compute_gap_penalty():.1f} "
                 f"cut={self.num_cuts * self.config.penalty_cut:.1f} "
                 f"total={self.total_penalty():.1f}"]
        for t in self.tunnels:
            desc = " ".join(f"[o{p.original_id} c{p.color} {p.left}->{p.right}]" for p in t.placed)
            lines.append(f"Tunnel {t.tunnel_id}: {desc or '<empty>'}")
        for p in self.unassigned:
            lines.append(f"  unplaced: o{p.original_id} len={p.length} c={p.color} "
                         f"[{p.min_left},{p.max_right}]")
        return "\n".join(lines)

    def render_grid(self) -> str:
        """One row per tunnel, one column per cell, each occupied cell showing its cargo id."""
        length = max([self.config.tunnel_length] + [p.right for p in self.placed])
        w = max(2, len(str(self.config.max_cargos - 1)) + 1)
        label = f"T{len(self.tunnels) - 1}"
        pad = " " * (len(label) + 2)
        lines = [pad + "".join(f"{x:>{w}}" if x % 5 == 0 else " " * w for x in range(length))]
        for t in self.tunnels:
            row = ["."] * length
            for p in t.placed:
                for x in range(p.left, p.right):
                    row[x] = str(p.original_id)
            lines.append(f"{f'T{t.tunnel_id}':>{len(label)}} |" + "".join(f"{c:>{w}}" for c in row) + " |")
        if self.unassigned:
            lines.append("unplaced: " + " ".join(f"o{p.original_id}(len={p.length})" for p in self.unassigned))
        return "\n".join(lines)


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


# ============================================================
# Policy-value network (structured 3-stage heads + pointer networks)
# ============================================================


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


class Pointer(nn.Module):
    """Additive (Bahdanau) attention pointer: score_i = v^T tanh(W_k k_i + W_q q)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.wk = nn.Linear(d, d, bias=False)
        self.wq = nn.Linear(d, d)
        self.v = nn.Linear(d, 1, bias=False)

    def forward(self, keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # keys (B, N, d), query (B, d) -> (B, N)
        return self.v(torch.tanh(self.wk(keys) + self.wq(query).unsqueeze(1))).squeeze(-1)


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class HierarchicalPolicyValueNet(nn.Module):
    def __init__(self, config: GameConfig, piece_dim: int, tunnel_dim: int) -> None:
        super().__init__()
        d = config.d_model
        self.piece_in = mlp(piece_dim, d, d)
        self.tunnel_in = mlp(tunnel_dim, d, d)
        self.type_emb = nn.Embedding(2, d)  # 0 = tunnel token, 1 = piece token
        layer = nn.TransformerEncoderLayer(d, config.n_heads, 2 * d, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.global_mlp = mlp(2 * d + CargoTunnelEnv.GLOBAL_DIM, d, d)

        # Stage 0: piece pointer + V(s)
        self.piece_ptr = Pointer(d)
        self.v0 = mlp(d, d, 1)
        # Stage 1: tunnel pointer conditioned on chosen piece + Q(s, piece)
        self.piece_ctx = mlp(2 * d, d, d)
        self.pair_in = mlp(CargoTunnelEnv.PAIR_DIM, d, d)
        self.tunnel_ptr = Pointer(d)
        self.v1 = mlp(d, d, 1)
        # Stage 2: mode head conditioned on piece and tunnel + Q(s, piece, tunnel)
        self.mode_ctx = mlp(2 * d + CargoTunnelEnv.MODE_DIM, d, d)
        self.mode_out = nn.Linear(d, 2)
        self.v2 = mlp(d, d, 1)

    # ---- shared encoder ----
    def encode(self, piece_x, piece_mask, tunnel_x, global_x):
        """piece_x (B,M,Fp) piece_mask (B,M) tunnel_x (B,K,Ft) global_x (B,G)."""
        B, K = tunnel_x.shape[:2]
        t_tok = self.tunnel_in(tunnel_x) + self.type_emb.weight[0]
        p_tok = self.piece_in(piece_x) + self.type_emb.weight[1]
        tokens = torch.cat([t_tok, p_tok], dim=1)
        pad = torch.cat([torch.zeros(B, K, dtype=torch.bool, device=tokens.device), ~piece_mask], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=pad)
        t_emb, p_emb = h[:, :K], h[:, K:]
        pm = piece_mask.unsqueeze(-1).float()
        p_pool = (p_emb * pm).sum(1) / pm.sum(1).clamp(min=1.0)
        g = self.global_mlp(torch.cat([t_emb.mean(1), p_pool, global_x], dim=-1))
        return g, p_emb, t_emb

    # ---- stage heads ----
    def stage0(self, g, p_emb, piece_mask):
        logits = masked_logits(self.piece_ptr(p_emb, g), piece_mask)
        return logits, torch.tanh(self.v0(g)).squeeze(-1)

    def stage1(self, g, e_piece, t_emb, pair_x, tunnel_mask):
        ctx = self.piece_ctx(torch.cat([g, e_piece], dim=-1))
        keys = t_emb + self.pair_in(pair_x)
        logits = masked_logits(self.tunnel_ptr(keys, ctx), tunnel_mask)
        return logits, torch.tanh(self.v1(ctx)).squeeze(-1), ctx, keys

    def stage2(self, ctx1, key_tunnel, mode_x, mode_mask):
        ctx = self.mode_ctx(torch.cat([ctx1, key_tunnel, mode_x], dim=-1))
        logits = masked_logits(self.mode_out(ctx), mode_mask)
        return logits, torch.tanh(self.v2(ctx)).squeeze(-1)


def _t(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(x, device=device).unsqueeze(0)


class NetEvaluator:
    """Adapter the MCTS uses; one encoder pass per real state, cached for stages 1-2."""

    def __init__(self, net: HierarchicalPolicyValueNet, device: str = "cpu") -> None:
        self.net = net
        self.device = device

    @torch.no_grad()
    def eval_state(self, env: CargoTunnelEnv) -> Tuple[np.ndarray, float, Dict]:
        self.net.eval()
        pf = env.piece_features()
        mask = np.ones(len(pf), dtype=bool)
        g, p_emb, t_emb = self.net.encode(_t(pf, self.device), _t(mask, self.device),
                                          _t(env.tunnel_features(), self.device),
                                          _t(env.global_features(), self.device))
        logits, v = self.net.stage0(g, p_emb, _t(mask, self.device))
        cache = {"g": g, "p_emb": p_emb, "t_emb": t_emb}
        return F.softmax(logits[0], -1).cpu().numpy(), float(v[0]), cache

    @torch.no_grad()
    def eval_piece(self, env, cache, slot: int, piece: CargoPiece, legal: List[int]):
        mask = np.zeros(env.config.num_tunnels, dtype=bool)
        mask[legal] = True
        logits, v, ctx, keys = self.net.stage1(cache["g"], cache["p_emb"][:, slot], cache["t_emb"],
                                               _t(env.pair_features(piece), self.device),
                                               _t(mask, self.device))
        cache[("piece", slot)] = (ctx, keys)
        return F.softmax(logits[0], -1).cpu().numpy(), float(v[0])

    @torch.no_grad()
    def eval_tunnel(self, env, cache, slot: int, piece: CargoPiece, tunnel_id: int, legal: List[ActionMode]):
        ctx, keys = cache[("piece", slot)]
        mask = np.array([m in legal for m in MODES])
        logits, v = self.net.stage2(ctx, keys[:, tunnel_id], _t(env.mode_features(piece, tunnel_id), self.device),
                                    _t(mask, self.device))
        return F.softmax(logits[0], -1).cpu().numpy(), float(v[0])


class UniformEvaluator:
    """Network-free evaluator (uniform priors, zero value) — used in tests."""

    def eval_state(self, env):
        n = len(env.candidates())
        return np.ones(n) / max(1, n), 0.0, {}

    def eval_piece(self, env, cache, slot, piece, legal):
        p = np.zeros(env.config.num_tunnels)
        p[legal] = 1.0 / len(legal)
        return p, 0.0

    def eval_tunnel(self, env, cache, slot, piece, tunnel_id, legal):
        return np.array([1.0 / len(legal) if m in legal else 0.0 for m in MODES]), 0.0


# ============================================================
# 3-stage hierarchical MCTS
# ============================================================


class Node:
    """
    stage 0: real env state, children keyed by candidate slot (piece choice)
    stage 1: piece chosen,     children keyed by tunnel id
    stage 2: tunnel chosen,    children keyed by mode -> stage-0 node of the next state
    `reward` is the reward on the edge INTO this node (non-zero only for stage-0 children of stage 2).
    """

    __slots__ = ("stage", "env", "parent", "prior", "reward", "terminal", "children",
                 "visits", "value_sum", "cache", "slot", "piece", "tunnel_id", "expanded")

    def __init__(self, stage: int, env: CargoTunnelEnv, parent: Optional["Node"], prior: float,
                 reward: float = 0.0, terminal: bool = False) -> None:
        self.stage = stage
        self.env = env
        self.parent = parent
        self.prior = prior
        self.reward = reward
        self.terminal = terminal
        self.children: Dict[object, Node] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.cache: Optional[Dict] = None
        self.slot: Optional[int] = None
        self.piece: Optional[CargoPiece] = None
        self.tunnel_id: Optional[int] = None
        self.expanded = False

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MinMaxStats:
    def __init__(self) -> None:
        self.lo, self.hi = float("inf"), -float("inf")

    def update(self, x: float) -> None:
        self.lo, self.hi = min(self.lo, x), max(self.hi, x)

    def norm(self, x: float) -> float:
        return (x - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0.5


class HierarchicalMCTS:
    def __init__(self, evaluator, config: GameConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.ev = evaluator
        self.cfg = config
        self.rng = rng or np.random.default_rng(0)

    # ---------- public ----------

    def new_root(self, env: CargoTunnelEnv) -> Node:
        return Node(0, env.clone(), None, 1.0, terminal=env.is_terminal())

    def run(self, root: Node, add_noise: bool = True) -> Node:
        stats = MinMaxStats()
        if not root.expanded and not root.terminal:
            self._expand(root)
        if add_noise:
            self._add_dirichlet(root)
        for _ in range(self.cfg.mcts_simulations):
            node, path = root, [root]
            while node.expanded and node.children and not node.terminal:
                node = self._select(node, stats)
                path.append(node)
            leaf_value = 0.0 if node.terminal else self._expand(node)
            self._backup(path, leaf_value, stats)
        return root

    # ---------- core ----------

    def _child_q(self, parent: Node, child: Node) -> float:
        if child.visits == 0:
            return parent.value()  # first-play urgency = parent's estimate
        return child.reward + self.cfg.gamma * child.value()

    def _select(self, node: Node, stats: MinMaxStats) -> Node:
        sqrt_n = math.sqrt(max(1, node.visits))
        best, best_score = None, -float("inf")
        for child in node.children.values():
            u = self.cfg.c_puct * child.prior * sqrt_n / (1 + child.visits)
            score = stats.norm(self._child_q(node, child)) + u
            if score > best_score:
                best, best_score = child, score
        return best

    def _backup(self, path: List[Node], value: float, stats: MinMaxStats) -> None:
        g = value
        for node in reversed(path):
            node.visits += 1
            node.value_sum += g
            g = node.reward + self.cfg.gamma * g
            stats.update(g)

    @staticmethod
    def _top_k(keys: List, priors: np.ndarray, k: int) -> Tuple[List, np.ndarray]:
        order = np.argsort(-priors)[:k]
        sel = priors[order]
        sel = sel / sel.sum() if sel.sum() > 0 else np.ones(len(order)) / len(order)
        return [keys[i] for i in order], sel

    def _expand(self, node: Node) -> float:
        cfg, env = self.cfg, node.env
        node.expanded = True

        if node.stage == 0:
            cands = env.candidates()
            priors, value, cache = self.ev.eval_state(env)
            node.cache = cache
            priors = np.asarray(priors[: len(cands)], dtype=np.float64)
            # Deadline-urgency prior blended into the learned piece prior
            urg = np.array([u for _, u in cands])
            urg = np.exp(urg * 4.0)
            urg /= urg.sum()
            w = cfg.urgency_prior_weight
            priors = (1 - w) * priors / max(priors.sum(), 1e-12) + w * urg
            slots, sel = self._top_k(list(range(len(cands))), priors, cfg.top_k_pieces)
            for slot, p in zip(slots, sel):
                child = Node(1, env, node, float(p))   # same env: nothing has changed yet
                child.cache, child.slot, child.piece = cache, slot, cands[slot][0]
                node.children[slot] = child
            return value

        if node.stage == 1:
            legal = env.legal_tunnels(node.piece)
            priors, value = self.ev.eval_piece(env, node.cache, node.slot, node.piece, legal)
            tunnels, sel = self._top_k(legal, np.asarray(priors)[legal], cfg.top_k_tunnels)
            for t, p in zip(tunnels, sel):
                child = Node(2, env, node, float(p))
                child.cache, child.slot, child.piece, child.tunnel_id = node.cache, node.slot, node.piece, t
                node.children[t] = child
            return value

        # stage 2: choose mode -> real transition
        legal = env.legal_modes(node.piece, node.tunnel_id)
        priors, value = self.ev.eval_tunnel(env, node.cache, node.slot, node.piece, node.tunnel_id, legal)
        for m in legal:
            nxt = env.clone()
            r, done, _ = nxt.step(Action(node.piece.piece_id, node.tunnel_id, m))
            node.children[m] = Node(0, nxt, node, float(priors[MODES.index(m)]),
                                    reward=r / cfg.value_scale, terminal=done)
        return value

    def _add_dirichlet(self, node: Node) -> None:
        if len(node.children) < 2:
            return
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(node.children))
        f = self.cfg.dirichlet_frac
        for child, n in zip(node.children.values(), noise):
            child.prior = (1 - f) * child.prior + f * n


def visit_policy(node: Node) -> Tuple[List, np.ndarray]:
    keys = list(node.children.keys())
    visits = np.array([node.children[k].visits for k in keys], dtype=np.float64)
    if visits.sum() == 0:  # unvisited level: fall back to priors
        visits = np.array([node.children[k].prior for k in keys])
    return keys, visits / visits.sum()


def pick(node: Node, explore: bool, rng: np.random.Generator) -> object:
    keys, probs = visit_policy(node)
    idx = rng.choice(len(keys), p=probs) if explore else int(np.argmax(probs))
    return keys[idx]


# ============================================================
# Replay buffer
# ============================================================


@dataclass
class Sample:
    piece_x: np.ndarray        # (M, Fp)
    tunnel_x: np.ndarray       # (K, Ft)
    global_x: np.ndarray       # (G,)
    pair_x: np.ndarray         # (K, PAIR_DIM) for the chosen piece
    mode_x: np.ndarray         # (MODE_DIM,)  for the chosen piece+tunnel
    slot: int                  # chosen piece slot
    tunnel_id: int             # chosen tunnel
    pi_piece: np.ndarray       # (M,)  stage-0 visit distribution
    pi_tunnel: np.ndarray      # (K,)  stage-1 visit distribution (under chosen piece)
    pi_mode: np.ndarray        # (2,)  stage-2 visit distribution (under chosen tunnel)
    tunnel_mask: np.ndarray    # (K,)  legal tunnels for chosen piece
    mode_mask: np.ndarray      # (2,)  legal modes
    value: float = 0.0         # normalised return-to-go


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: List[Sample] = []
        self.pos = 0

    def extend(self, samples: Sequence[Sample]) -> None:
        for s in samples:
            if len(self.data) < self.capacity:
                self.data.append(s)
            else:
                self.data[self.pos] = s
                self.pos = (self.pos + 1) % self.capacity

    def sample(self, n: int) -> List[Sample]:
        return random.sample(self.data, min(n, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)


def collate(batch: List[Sample], device: str) -> Dict[str, torch.Tensor]:
    B = len(batch)
    M = max(len(s.piece_x) for s in batch)
    Fp = batch[0].piece_x.shape[1]
    piece_x = np.zeros((B, M, Fp), np.float32)
    piece_mask = np.zeros((B, M), bool)
    pi_piece = np.zeros((B, M), np.float32)
    for i, s in enumerate(batch):
        n = len(s.piece_x)
        piece_x[i, :n] = s.piece_x
        piece_mask[i, :n] = True
        pi_piece[i, :n] = s.pi_piece
    out = {
        "piece_x": piece_x, "piece_mask": piece_mask, "pi_piece": pi_piece,
        "tunnel_x": np.stack([s.tunnel_x for s in batch]),
        "global_x": np.stack([s.global_x for s in batch]),
        "pair_x": np.stack([s.pair_x for s in batch]),
        "mode_x": np.stack([s.mode_x for s in batch]),
        "slot": np.array([s.slot for s in batch], np.int64),
        "tunnel_id": np.array([s.tunnel_id for s in batch], np.int64),
        "pi_tunnel": np.stack([s.pi_tunnel for s in batch]),
        "pi_mode": np.stack([s.pi_mode for s in batch]),
        "tunnel_mask": np.stack([s.tunnel_mask for s in batch]),
        "mode_mask": np.stack([s.mode_mask for s in batch]),
        "value": np.array([s.value for s in batch], np.float32),
    }
    return {k: torch.as_tensor(v, device=device) for k, v in out.items()}


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(target * logp).sum(-1).mean()


# ============================================================
# Trainer
# ============================================================


class Trainer:
    def __init__(self, config: GameConfig, seed: int = 42) -> None:
        self.cfg = config
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.episode_rng = random.Random(seed + 999)

        probe = CargoTunnelEnv(config)
        self.net = HierarchicalPolicyValueNet(config, probe.piece_feature_dim(), probe.tunnel_feature_dim()).to(config.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.mcts = HierarchicalMCTS(NetEvaluator(self.net, config.device), config, self.rng)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.history: Dict[str, List[float]] = {k: [] for k in
                                                ("reward", "penalty", "steps", "loss_piece", "loss_tunnel",
                                                 "loss_mode", "loss_value")}

    # ---------- self-play ----------

    def self_play_episode(self, explore: bool = True) -> Tuple[List[Sample], float, CargoTunnelEnv]:
        cfg = self.cfg
        env = CargoTunnelEnv(cfg)
        env.reset(seed=self.episode_rng.randint(0, 2 ** 31 - 1))
        records: List[Sample] = []
        rewards: List[float] = []
        root = self.mcts.new_root(env)
        move = 0

        while not env.is_terminal():
            self.mcts.run(root, add_noise=explore)
            temp = explore and move < cfg.temperature_moves

            # Stage 0: piece
            slots, pi_piece_vals = visit_policy(root)
            slot = pick(root, temp, self.rng)
            n1 = root.children[slot]
            if not n1.expanded:
                self.mcts._expand(n1)
            # Stage 1: tunnel (from the chosen piece's own subtree)
            tunnels, pi_t_vals = visit_policy(n1)
            tunnel_id = pick(n1, temp, self.rng)
            n2 = n1.children[tunnel_id]
            if not n2.expanded:
                self.mcts._expand(n2)
            # Stage 2: mode
            modes, pi_m_vals = visit_policy(n2)
            mode = pick(n2, temp, self.rng)

            piece = n1.piece
            K = cfg.num_tunnels
            pi_piece = np.zeros(len(env.candidates()), np.float32)
            pi_piece[slots] = pi_piece_vals
            pi_tunnel = np.zeros(K, np.float32)
            pi_tunnel[tunnels] = pi_t_vals
            pi_mode = np.zeros(2, np.float32)
            pi_mode[[MODES.index(m) for m in modes]] = pi_m_vals
            tmask = np.zeros(K, bool)
            tmask[env.legal_tunnels(piece)] = True
            mmask = np.array([m in env.legal_modes(piece, tunnel_id) for m in MODES])

            records.append(Sample(
                env.piece_features(), env.tunnel_features(), env.global_features(),
                env.pair_features(piece), env.mode_features(piece, tunnel_id),
                slot, tunnel_id, pi_piece, pi_tunnel, pi_mode, tmask, mmask,
            ))

            r, done, _ = env.step(Action(piece.piece_id, tunnel_id, mode))
            rewards.append(r)
            move += 1

            # Subtree reuse: the chosen stage-2 child already holds this exact next state.
            root = n2.children[mode]
            root.parent, root.reward = None, 0.0
            if done:
                break

        # Return-to-go targets for all three value heads
        g = 0.0
        for rec, r in zip(reversed(records), reversed(rewards)):
            g = r + cfg.gamma * g
            rec.value = float(np.clip(g / cfg.value_scale, -1.0, 1.0))
        return records, float(sum(rewards)), env

    # ---------- learning ----------

    def train_step(self, batch: List[Sample]) -> Tuple[float, float, float, float]:
        b = collate(batch, self.cfg.device)
        self.net.train()
        g, p_emb, t_emb = self.net.encode(b["piece_x"], b["piece_mask"], b["tunnel_x"], b["global_x"])
        logits0, v0 = self.net.stage0(g, p_emb, b["piece_mask"])

        idx = torch.arange(len(batch), device=g.device)
        e_piece = p_emb[idx, b["slot"]]
        logits1, v1, ctx1, keys = self.net.stage1(g, e_piece, t_emb, b["pair_x"], b["tunnel_mask"])
        logits2, v2 = self.net.stage2(ctx1, keys[idx, b["tunnel_id"]], b["mode_x"], b["mode_mask"])

        l_piece = soft_ce(logits0, b["pi_piece"])
        l_tunnel = soft_ce(logits1, b["pi_tunnel"])
        l_mode = soft_ce(logits2, b["pi_mode"])
        z = b["value"]
        l_value = F.mse_loss(v0, z) + F.mse_loss(v1, z) + F.mse_loss(v2, z)

        loss = l_piece + l_tunnel + l_mode + l_value
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return l_piece.item(), l_tunnel.item(), l_mode.item(), l_value.item()

    def train(self, verbose_every: int = 10) -> None:
        cfg = self.cfg
        for ep in range(1, cfg.num_episodes + 1):
            samples, ep_reward, env = self.self_play_episode(explore=True)
            self.replay.extend(samples)
            losses = np.zeros(4)
            if len(self.replay) >= cfg.batch_size:
                for _ in range(cfg.train_steps_per_episode):
                    losses += np.array(self.train_step(self.replay.sample(cfg.batch_size)))
                losses /= cfg.train_steps_per_episode

            h = self.history
            h["reward"].append(ep_reward)
            h["penalty"].append(env.total_penalty())
            h["steps"].append(env.step_count)
            for k, v in zip(("loss_piece", "loss_tunnel", "loss_mode", "loss_value"), losses):
                h[k].append(float(v))

            print(f"Ep {ep:04d} | reward={ep_reward:8.2f} | penalty={env.total_penalty():7.2f} | "
                  f"steps={env.step_count:3d} | unplaced={len(env.unassigned)} | "
                  f"L(piece/tunnel/mode/value)={losses[0]:.3f}/{losses[1]:.3f}/{losses[2]:.3f}/{losses[3]:.3f}")
            if verbose_every and ep % verbose_every == 0:
                # Greedy (no exploration) game, so the board shows what the agent actually does
                _, greedy_reward, greedy_env = self.self_play_episode(explore=False)
                print(f"  greedy reward={greedy_reward:8.2f} | penalty={greedy_env.total_penalty():7.2f}")
                print(greedy_env.render_grid())

    def save(self, path: str = "cargo_3stage.pt") -> None:
        torch.save({"model": self.net.state_dict(), "config": self.cfg.__dict__}, path)

    def plot(self, path: str = "training_curves.png") -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = self.history
        ep = np.arange(1, len(h["reward"]) + 1)

        def smooth(x, w=10):
            x = np.asarray(x, float)
            return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x

        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        ax[0, 0].plot(ep, h["reward"], alpha=0.3)
        ax[0, 0].plot(ep[len(ep) - len(smooth(h["reward"])):], smooth(h["reward"]))
        ax[0, 0].set_title("Episode reward")
        ax[0, 1].plot(ep, h["penalty"], alpha=0.3)
        ax[0, 1].plot(ep[len(ep) - len(smooth(h["penalty"])):], smooth(h["penalty"]))
        ax[0, 1].set_title("Episode penalty")
        for k in ("loss_piece", "loss_tunnel", "loss_mode", "loss_value"):
            ax[1, 0].plot(ep, h[k], label=k.replace("loss_", ""))
        ax[1, 0].legend()
        ax[1, 0].set_title("Losses per stage")
        ax[1, 1].plot(ep, h["steps"])
        ax[1, 1].set_title("Steps per episode")
        for a in ax.flat:
            a.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=110)
        print(f"Saved {path}")


# ============================================================
# Tests
# ============================================================


def _manual_env(**overrides) -> CargoTunnelEnv:
    cfg = DynamicGameConfig(num_tunnels=2, tunnel_length=20, max_cargos=2, **overrides)
    env = CargoTunnelEnv(cfg)
    env.reset(pieces=[
        CargoPiece(1, 0, 8, 0, 0, 5),    # needs a cut to avoid overflow
        CargoPiece(2, 1, 4, 1, 3, 10),
    ])
    return env


def test_env_basics() -> None:
    env = _manual_env()
    r, done, info = env.step(Action(1, 0, ActionMode.CUT_AT_MAX))
    assert info["placed_len"] == 5 and info["remainder"] == 3 and not done
    assert env.num_cuts == 1 and r < 0
    env2 = _manual_env(mask_overflow_whole=False)
    r2, _, _ = env2.step(Action(1, 0, ActionMode.WHOLE))
    assert env2.compute_overflow_penalty() == 3 * env2.config.penalty_exceed_max and r2 < 0


def test_masking_and_urgency() -> None:
    env = _manual_env(mask_overflow_whole=True)
    cands = env.candidates()
    assert cands[0][0].piece_id == 1, "tightest-deadline piece should rank first"
    assert env.legal_modes(cands[0][0], 0) == [ActionMode.CUT_AT_MAX], "overflowing WHOLE should be masked"


def test_clone_isolation() -> None:
    env = _manual_env()
    c = env.clone()
    c.step(Action(2, 1, ActionMode.WHOLE))
    assert len(env.unassigned) == 2 and not env.tunnels[1].placed


def test_tree_has_three_stages() -> None:
    cfg = DynamicGameConfig(num_tunnels=3, tunnel_length=30, max_cargos=5, mcts_simulations=60)
    env = CargoTunnelEnv(cfg)
    env.reset(seed=3)
    mcts = HierarchicalMCTS(UniformEvaluator(), cfg)
    root = mcts.run(mcts.new_root(env), add_noise=False)
    assert root.stage == 0 and len(root.children) <= cfg.top_k_pieces
    n1 = max(root.children.values(), key=lambda n: n.visits)
    assert n1.stage == 1 and n1.env is root.env, "piece choice must not change the env"
    assert all(k in range(cfg.num_tunnels) for k in n1.children)
    n2 = max(n1.children.values(), key=lambda n: n.visits)
    assert n2.stage == 2 and set(n2.children) <= set(MODES)
    n3 = max(n2.children.values(), key=lambda n: n.visits)
    assert n3.stage == 0 and n3.env.step_count == 1, "mode choice performs the real step"
    assert root.visits == cfg.mcts_simulations


def test_mcts_prefers_cut_over_overflow() -> None:
    # Two tunnels: cutting lets the remainder go to the empty tunnel without overflow
    # (+2 cargo +5 bonus -1.5 cut), whereas WHOLE overflows by 3 (-24 +2 +5). Search must find the cut.
    cfg = DynamicGameConfig(num_tunnels=2, tunnel_length=20, max_cargos=1, mask_overflow_whole=False,
                     mcts_simulations=200)
    env = CargoTunnelEnv(cfg)
    env.reset(pieces=[CargoPiece(1, 0, 8, 0, 0, 5)])
    mcts = HierarchicalMCTS(UniformEvaluator(), cfg)
    root = mcts.run(mcts.new_root(env), add_noise=False)
    n2 = root.children[0].children[0]
    assert pick(n2, False, np.random.default_rng(0)) == ActionMode.CUT_AT_MAX


def test_network_and_train_step() -> None:
    cfg = DynamicGameConfig(num_tunnels=3, tunnel_length=30, max_cargos=5, mcts_simulations=20,
                            batch_size=4, d_model=32)
    tr = Trainer(cfg, seed=1)
    samples, _, env = tr.self_play_episode()
    assert samples and env.is_terminal()
    tr.replay.extend(samples)
    losses = tr.train_step(tr.replay.sample(4))
    assert all(np.isfinite(losses))


def test_static_game() -> None:
    cfg = StaticGameConfig()
    a, b = CargoTunnelEnv(cfg), CargoTunnelEnv(cfg)
    a.reset(seed=1)
    b.reset(seed=2)
    assert a.unassigned == b.unassigned and cfg.max_cargos == len(cfg.pieces), "seed must not matter"

    # The solver must find a known zero-penalty optimum on a tiny instance, and its line must
    # replay to the value it reports. (The default instance takes too long to solve here.)
    tiny = StaticGameConfig(pieces=((6, 0, 0, 6), (4, 0, 0, 10), (5, 1, 0, 5), (5, 1, 0, 10)))
    env = CargoTunnelEnv(tiny)
    env.reset()
    best, actions = brute_force_optimum(env)
    assert best == tiny.completion_bonus + len(tiny.pieces) * tiny.cargo_reward
    assert len(actions) == len(tiny.pieces)
    assert math.isclose(sum(env.step(x)[0] for x in actions), best) and env.is_terminal()


def run_tests() -> None:
    for t in (test_env_basics, test_masking_and_urgency, test_clone_isolation,
              test_tree_has_three_stages, test_mcts_prefers_cut_over_overflow,
              test_network_and_train_step, test_static_game):
        t()
        print(f"  {t.__name__}: passed")
    print("All tests passed.\n")


# ============================================================
# Entry point
# ============================================================


def main() -> None:
    run_tests()
    config = StaticGameConfig()
    # config = DynamicGameConfig(num_episodes=200, mcts_simulations=160)
    trainer = Trainer(config, seed=42)
    trainer.train(verbose_every=20)
    trainer.save()
    trainer.plot()

    if isinstance(config, StaticGameConfig):
        env = CargoTunnelEnv(config)
        env.reset()
        best, actions = brute_force_optimum(env)
        _, agent_reward, agent_env = trainer.self_play_episode(explore=False)
        print(f"Optimal reward={best:.2f} | agent reward={agent_reward:.2f} | "
              f"{'OPTIMAL' if math.isclose(agent_reward, best) else 'SUBOPTIMAL'}")
        print("Optimal actions:", [(a.piece_id, a.tunnel_id, a.mode.value) for a in actions])
        print(agent_env.render_text())
        print(agent_env.render_grid())


if __name__ == "__main__":
    main()
