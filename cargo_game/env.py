"""The cargo-tunnel environment: legality, masking, transitions, scoring and features."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GameConfig
from .domain import Action, ActionMode, CargoPiece, PlacedPiece, Tunnel


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
