"""Game settings, the static hand-made instance and random instance generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from .domain import CargoPiece


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
