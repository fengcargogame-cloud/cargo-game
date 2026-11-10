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

Modules: config (settings and instances), domain (data types), env (the game), solver (exact
search), network, evaluators, mcts, replay and trainer.
"""

from .config import DynamicGameConfig, GameConfig, StaticGameConfig
from .domain import Action, ActionMode, CargoPiece
from .env import CargoTunnelEnv
from .solver import brute_force_optimum
from .trainer import Trainer

__all__ = [
    "Action", "ActionMode", "CargoPiece", "CargoTunnelEnv", "DynamicGameConfig", "GameConfig",
    "StaticGameConfig", "Trainer", "brute_force_optimum",
]
