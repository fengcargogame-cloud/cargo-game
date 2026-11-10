"""3-stage hierarchical MCTS: piece -> tunnel -> mode, one tree level each."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GameConfig
from .domain import Action, CargoPiece, MODES
from .env import CargoTunnelEnv


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
