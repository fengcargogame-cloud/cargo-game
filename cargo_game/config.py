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
    max_cargos: int = 14
    num_colors: int = 5

    penalty_exceed_max: float = 8.0
    penalty_color_change: float = 1.0
    penalty_cut: float = 1.5
    penalty_gap: float = 1.5             # per unit of empty space left between two cargos in a tunnel
    penalty_unplaced: float = 5.0
    cargo_reward: float = 3.0            # paid each time a cargo is fully placed
    completion_bonus: float = 8.0        # paid once every cargo is placed

    # Game rules
    enforce_tunnel_end: bool = True      # a placed segment may not extend past tunnel_length
    # A cut may also stop short of max_right, at the min_left of any unplaced piece, so that piece
    # can start on time behind it. Without this a cut always runs to max_right.
    allow_early_cut: bool = True
    max_game_steps: int = 300

    # Action masking / pruning
    max_piece_candidates: int = 20       # urgency-ranked pieces considered at stage 0
    # An overflow costs far more than a cut, and its huge negative reward stretches the search's
    # Q range until the difference between good layouts disappears.
    mask_overflow_whole: bool = True     # drop WHOLE when it overflows and CUT is possible
    mask_color_change: bool = True       # avoid colour change into a tunnel when a same-colour piece fits there
    # Off by default: it also hides a tunnel from a piece that "can wait" for a same-colour tail,
    # which forbids chaining different-coloured cargos in one tunnel even when the optimum needs it.
    mask_wait_for_color: bool = False
    # Tunnels with the same fill level and last colour are interchangeable; offer only the first.
    mask_symmetric_tunnels: bool = True

    # MCTS
    mcts_simulations: int = 160
    c_puct: float = 1.5
    top_k_pieces: int = 15                # top-K expansion at stage 0
    top_k_tunnels: int = 10               # top-K expansion at stage 1
    urgency_prior_weight: float = 0.25   # blend of urgency prior into piece prior
    dirichlet_alpha: float = 0.5
    dirichlet_frac: float = 0.25
    noise_all_stages: bool = True        # root noise on the tunnel and placement levels too, not only pieces
    # The game is deterministic and single-player, so the best finished line through a node is a
    # real achievable return. Q = (1 - w) * mean + w * best; 0 gives the plain AlphaZero mean.
    max_backup_weight: float = 0.5
    # After expanding a real state, play it out to the end with a cheap random policy. The finished
    # layout is a real return for the max backup, and the best one found is handed to the trainer
    # as a training episode. The policy prefers urgent pieces and placements that waste no cells:
    # each empty cell a placement would leave in front of the piece divides its odds by
    # e^rollout_gap_weight (0 = ignore gaps).
    rollouts: bool = True
    rollout_gap_weight: float = 3.0
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
    # Share of every training batch drawn from the best episode found so far, with its own moves
    # as policy targets. Only meaningful when every episode is the same instance, so it is 0 here
    # and switched on in StaticGameConfig.
    elite_fraction: float = 0.0
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

    elite_fraction: float = 0.25
 
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

    # value_scale: float = 90.0            # returns are about -50..+12.5
    # num_episodes: int = 5000
    # mcts_simulations: int = 300
    # num_tunnels: int = 7
    # tunnel_length: int = 16
    # num_colors: int = 4
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (4, 0, 3, 5),
    #     (9, 0, 0, 3),
    #     (4, 0, 3, 7),
    #     (10, 0, 5, 9),
    #     (5, 3, 0, 5), 
    #     (5, 1, 0, 5),
    #     (7, 3, 5, 12),
    #     (4, 3, 12, 16),
    #     (5, 1, 5, 10),
    #     (6, 1, 10, 16),
    #     (19, 0, 9, 16),
    #     (6, 2, 0, 3),
    #     (19, 2, 3, 13),
    #     (7, 2, 12, 16),
    # )

    # value_scale: float = 90.0            # returns are about -50..+12.5
    # num_episodes: int = 5000
    # mcts_simulations: int = 300
    # num_tunnels: int = 8
    # tunnel_length: int = 16
    # num_colors: int = 4
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (4, 0, 3, 5),
    #     (9, 0, 0, 3),
    #     (4, 0, 3, 7),
    #     (10, 0, 5, 9),
    #     (5, 3, 0, 5), 
    #     (10, 1, 0, 5),
    #     (7, 3, 5, 12),
    #     (4, 3, 12, 16),
    #     (10, 1, 5, 10),
    #     (11, 1, 10, 16),
    #     (19, 0, 9, 16),
    #     (6, 2, 0, 3),
    #     (19, 2, 3, 13),
    #     (7, 2, 12, 16),
    # )

    # value_scale: float = 90.0            # returns are about -50..+12.5
    # num_episodes: int = 5990
    # mcts_simulations: int = 300
    # num_tunnels: int = 9
    # tunnel_length: int = 16
    # num_colors: int = 5
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (4, 0, 3, 5),
    #     (9, 0, 0, 3),
    #     (4, 0, 3, 7),
    #     (10, 0, 5, 9),
    #     (5, 3, 0, 5), 
    #     (10, 1, 0, 5),
    #     (7, 3, 5, 12),
    #     (4, 3, 12, 16),
    #     (10, 1, 5, 10),
    #     (11, 1, 10, 16),
    #     (19, 0, 9, 16),
    #     (6, 2, 0, 3),
    #     (19, 2, 3, 13),
    #     (7, 2, 12, 16), 
    #     (2, 0, 14, 16),
    #     (14, 4, 0, 14),
    # )

    ### WORKED CONFIGURATION
    # value_scale: float = 40.0            # best return is about +10; anything below -40 is clipped
    # num_episodes: int = 5990
    # mcts_simulations: int = 300
    # num_tunnels: int = 4
    # tunnel_length: int = 17
    # num_colors: int = 9
    # # (length, color, min_left, max_right) per cargo
    # pieces: Tuple[Tuple[int, int, int, int], ...] = (
    #     (15, 0, 0, 4),
    #     (20, 1, 3, 17),
    #     (6, 2, 7, 12),
    #     (11, 3, 7, 17),
    #     (12, 4, 10, 17), 
    #     (1, 5, 4, 5),
    #     (1, 6, 5, 6),
    #     (1, 7, 6, 7),
    #     (1, 8, 7, 8),
    # )

    ### WORKED CONFIGURATION
    value_scale: float = 40.0            # best return is about +10; anything below -40 is clipped
    num_episodes: int = 5990
    mcts_simulations: int = 300
    num_tunnels: int = 6
    tunnel_length: int = 17
    num_colors: int = 9
    # (length, color, min_left, max_right) per cargo
    pieces: Tuple[Tuple[int, int, int, int], ...] = (
        (15, 0, 0, 4),
        (54, 1, 0, 17),
        (6, 2, 7, 12),
        (11, 3, 7, 17),
        (12, 4, 10, 17), 
        (1, 5, 4, 5),
        (1, 6, 5, 6),
        (1, 7, 6, 7),
        (1, 8, 7, 8),
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
        for i, piece in enumerate(self.pieces):
            if not isinstance(piece, tuple) or len(piece) != 4:
                raise ValueError(f"pieces[{i}] must be a 4-tuple (length, color, min_left, max_right), got {piece!r}")
            if not 0 <= piece[1] < self.num_colors:
                raise ValueError(f"pieces[{i}] has color {piece[1]}, but num_colors is {self.num_colors}")
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
