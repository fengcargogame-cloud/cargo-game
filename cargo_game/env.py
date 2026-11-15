"""The cargo-tunnel environment: legality, masking, transitions, scoring and features."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GameConfig
from .domain import Action, ActionMode, CargoPiece, PlacedPiece, Placement, Tunnel


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
        self._cut_cache: Optional[List[int]] = None
        # Per-state memo of placements() and legal_tunnels(); callers must not mutate the lists.
        self._place_cache: Dict[Tuple[int, int], List[Placement]] = {}
        self._tunnel_cache: Dict[int, List[int]] = {}

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
        self._cut_cache = None
        self._place_cache = {}
        self._tunnel_cache = {}

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

    def cut_points(self) -> List[int]:
        """Where an early cut may end: the min_left of every unplaced piece."""
        if self._cut_cache is None:
            self._cut_cache = sorted({p.min_left for p in self.unassigned})
        return self._cut_cache

    def placements(self, piece: CargoPiece, tunnel_id: int) -> List[Placement]:
        """Every raw (unmasked) way to put the piece in the tunnel: whole, cut at max_right, or
        cut early at a cut point strictly inside the segment and short of max_right."""
        left = self.tunnels[tunnel_id].next_left_for(piece)
        key = (piece.piece_id, left)       # the options depend only on the piece and where it starts
        cached = self._place_cache.get(key)
        if cached is not None:
            return cached
        options = self._place_cache[key] = []
        if self._fits(left, piece.length):
            options.append(Placement(ActionMode.WHOLE, left + piece.length))
        if self.cut_length(piece, tunnel_id) > 0:
            options.append(Placement(ActionMode.CUT_AT_MAX, piece.max_right))
        if self.config.allow_early_cut:
            limit = min(left + piece.length, piece.max_right)
            options.extend(Placement(ActionMode.CUT_EARLY, end) for end in self.cut_points()
                           if left < end < limit and self._fits(left, end - left))
        return options

    # ---------------- stage-wise legality (with masking) ----------------

    def segments_needed(self, piece: CargoPiece) -> int:
        """How many tunnels the piece needs to fit inside its [min_left, max_right] window."""
        window = min(piece.max_right, self.config.tunnel_length) - piece.min_left
        return 1 if window <= 0 else -(-piece.length // window)

    def candidates(self) -> List[Tuple[CargoPiece, float]]:
        """Urgency-ranked placeable pieces: [(piece, urgency)], most urgent first.

        Urgency is the pressure from waiting: segments the piece still needs divided by the
        tunnels where it can still start at its min_left. A piece that can wait (e.g. a short
        cargo near the tunnel end while many tunnels are still open) ranks low, even if its own
        window is tight. Once no tunnel can take it on time, urgency falls back to its slack.
        """
        if self._cand_cache is not None:
            return self._cand_cache
        scored: List[Tuple[float, int, int, CargoPiece]] = []
        for piece in self.unassigned:
            best_slack = None
            on_time = 0
            for t in range(self.config.num_tunnels):
                if not self.placements(piece, t):
                    continue
                left = self.tunnels[t].next_left_for(piece)
                if left == piece.min_left:
                    on_time += 1
                slack = piece.max_right - left - piece.length
                best_slack = slack if best_slack is None else max(best_slack, slack)
            if best_slack is None:
                continue  # cannot be placed anywhere right now
            if on_time:
                urgency = min(1.0, self.segments_needed(piece) / on_time)
            else:
                urgency = 1.0 / (1.0 + max(0.0, float(best_slack)))
            scored.append((urgency, best_slack, piece.piece_id, piece))
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        self._cand_cache = [(p, u) for u, _, _, p in scored[: self.config.max_piece_candidates]]
        return self._cand_cache

    def legal_tunnels(self, piece: CargoPiece) -> List[int]:
        cached = self._tunnel_cache.get(piece.piece_id)
        if cached is None:
            cached = self._tunnel_cache[piece.piece_id] = self._legal_tunnels(piece)
        return cached

    def _legal_tunnels(self, piece: CargoPiece) -> List[int]:
        raw = [t for t in range(self.config.num_tunnels) if self.placements(piece, t)]
        if not self.config.mask_color_change:
            return self._drop_symmetric(raw)
        cand_pieces = [p for p, _ in self.candidates()]
        on_time = [t for t in raw if self.tunnels[t].next_left_for(piece) == piece.min_left]
        need = self.segments_needed(piece)
        kept = []
        for t in raw:
            last = self.tunnels[t].last_color
            if last is not None and last != piece.color:
                # (a) a piece of the tunnel's own colour could take this slot instead
                same_colour_fits = any(
                    p.color == last and p.piece_id != piece.piece_id and self.placements(p, t)
                    for p in cand_pieces
                )
                if same_colour_fits:
                    continue
                # (b) the piece fits here whole but can afford to wait: enough other tunnels can
                #     still take it on time, so a same-colour tail may yet appear for it. A piece
                #     that would be cut here is already paying for the split; where its fragment
                #     goes is a packing choice left to the search.
                left = self.tunnels[t].next_left_for(piece)
                fits_whole = left + piece.length <= piece.max_right
                if (self.config.mask_wait_for_color and fits_whole
                        and sum(1 for u in on_time if u != t) >= need):
                    continue
            kept.append(t)
        return self._drop_symmetric(kept or raw)   # never mask everything away

    def _drop_symmetric(self, tunnel_ids: List[int]) -> List[int]:
        if not self.config.mask_symmetric_tunnels:
            return tunnel_ids
        seen, out = set(), []
        for t in tunnel_ids:
            state = (self.tunnels[t].current_right, self.tunnels[t].last_color)
            if state not in seen:
                seen.add(state)
                out.append(t)
        return out

    def legal_placements(self, piece: CargoPiece, tunnel_id: int) -> List[Placement]:
        options = self.placements(piece, tunnel_id)
        if self.config.mask_overflow_whole and any(o.mode == ActionMode.CUT_AT_MAX for o in options):
            left = self.tunnels[tunnel_id].next_left_for(piece)
            if left + piece.length > piece.max_right:
                return [o for o in options if o.mode != ActionMode.WHOLE]
        return options

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
        choice = next((o for o in self.placements(piece, action.tunnel_id) if o.mode == action.mode
                       and (o.mode != ActionMode.CUT_EARLY or o.end == action.cut_end)), None)
        if choice is None:
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

        placed_len = choice.end - left
        if action.mode != ActionMode.WHOLE:
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
        self._cut_cache = None
        self._place_cache = {}
        self._tunnel_cache = {}
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
    OPTION_DIM = 9
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

    def option_features(self, piece: CargoPiece, tunnel_id: int, options: List[Placement]) -> np.ndarray:
        """(len(options), OPTION_DIM): what each stage-2 option would do."""
        cfg, L = self.config, float(self.config.tunnel_length)
        left = self.tunnels[tunnel_id].next_left_for(piece)
        rows = []
        for o in options:
            placed = o.end - left
            # pieces that could start exactly on time right behind this segment
            followers = sum(1 for p in self.unassigned if p.min_left == o.end and p.piece_id != piece.piece_id)
            rows.append([
                1.0 if o.mode == ActionMode.WHOLE else 0.0,
                1.0 if o.mode == ActionMode.CUT_AT_MAX else 0.0,
                1.0 if o.mode == ActionMode.CUT_EARLY else 0.0,
                placed / piece.length,
                (piece.length - placed) / piece.length,
                o.end / L,
                max(0, o.end - piece.max_right) / L,
                followers / max(1, cfg.max_cargos),
                1.0 if piece.was_cut else 0.0,
            ])
        return np.asarray(rows, dtype=np.float32).reshape(-1, self.OPTION_DIM)

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
