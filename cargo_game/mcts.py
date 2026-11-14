"""3-stage hierarchical MCTS: piece -> tunnel -> mode, one tree level each."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GameConfig
from .domain import Action, CargoPiece
from .env import CargoTunnelEnv


class Node:
    """
    stage 0: real env state, children keyed by candidate slot (piece choice)
    stage 1: piece chosen,     children keyed by tunnel id
    stage 2: tunnel chosen,    children keyed by Placement -> stage-0 node of the next state
    `reward` is the reward on the edge INTO this node (non-zero only for stage-0 children of stage 2).
    """

    __slots__ = ("stage", "env", "parent", "prior", "reward", "terminal", "children",
                 "visits", "value_sum", "best", "cache", "slot", "piece", "tunnel_id", "expanded",
                 "action")

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
        self.best: Optional[float] = None   # best return of a finished line through this node
        self.cache: Optional[Dict] = None
        self.slot: Optional[int] = None
        self.piece: Optional[CargoPiece] = None
        self.tunnel_id: Optional[int] = None
        self.expanded = False
        self.action: Optional[Action] = None   # the move into this node (stage-0 children only)

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def q(self, max_weight: float) -> float:
        """Mean return, pulled towards the best finished line once one is known."""
        if self.best is None:
            return self.value()
        return (1.0 - max_weight) * self.value() + max_weight * self.best


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
        self._noise_root: Optional[Node] = None
        # Best finished line seen by the last run(): (return from the root, in reward units; its actions)
        self.found: Optional[Tuple[float, List[Action]]] = None

    # ---------- public ----------

    def new_root(self, env: CargoTunnelEnv) -> Node:
        return Node(0, env.clone(), None, 1.0, terminal=env.is_terminal())

    def run(self, root: Node, add_noise: bool = True) -> Node:
        stats = MinMaxStats()
        if not root.expanded and not root.terminal:
            self._expand(root)
        self._noise_root = root if add_noise and self.cfg.noise_all_stages else None
        if add_noise:
            self._add_dirichlet(root)
            if self.cfg.noise_all_stages:
                # Levels already expanded (subtree reuse); the rest get noise as they expand.
                for n1 in root.children.values():
                    if n1.expanded:
                        self._add_dirichlet(n1)
                        for n2 in n1.children.values():
                            if n2.expanded:
                                self._add_dirichlet(n2)
        self.found = None
        for _ in range(self.cfg.mcts_simulations):
            node, path = root, [root]
            while node.expanded and node.children and not node.terminal:
                node = self._select(node, stats)
                path.append(node)
            # `real` is the true return from the leaf to the end of the game, when one is known.
            real: Optional[float] = None
            tail: List[Action] = []
            if node.terminal or (node.expanded and not node.children):
                leaf_value, real = 0.0, 0.0
            else:
                leaf_value = self._expand(node)
                if self.cfg.rollouts and node.stage == 0:
                    real, tail = self._rollout(node)
            line_return = self._backup(path, leaf_value, stats, real)
            if line_return is not None and (self.found is None or line_return > self.found[0]):
                self.found = (line_return, [n.action for n in path[1:] if n.action is not None] + tail)
        self._noise_root = None
        return root

    def _draw(self, logits: List[float]) -> int:
        """Sample an index with probability proportional to exp(logit)."""
        w = np.exp(np.asarray(logits) - max(logits))
        return min(int(np.searchsorted(np.cumsum(w), self.rng.random() * w.sum())), len(logits) - 1)

    def _waste(self, env: CargoTunnelEnv, piece: CargoPiece, tunnel_id: int) -> float:
        """Rollout cost of a placement: empty cells left in front of the piece, plus the overflow
        penalty it cannot avoid there (a piece that cannot be cut and would end past max_right)."""
        cfg, tunnel = self.cfg, env.tunnels[tunnel_id]
        left = tunnel.next_left_for(piece)
        late = 0 if left < piece.max_right else left + piece.length - piece.max_right
        return cfg.rollout_gap_weight * (left - tunnel.current_right) + cfg.penalty_exceed_max * late

    def _rollout(self, node: Node) -> Tuple[float, List[Action]]:
        """Play a real state out to the end; returns (return / value_scale, actions played)."""
        cfg, env = self.cfg, node.env.clone()
        total, discount, actions = 0.0, 1.0, []
        while not env.is_terminal():
            cands = env.candidates()
            legal = [env.legal_tunnels(p) for p, _ in cands]
            # cost[i][j]: what putting candidate i in its j-th legal tunnel wastes
            cost = [[self._waste(env, p, t) for t in ts] for (p, _), ts in zip(cands, legal)]
            i = self._draw([4.0 * u - min(c) for (_, u), c in zip(cands, cost)])
            piece = cands[i][0]
            tunnel_id = legal[i][self._draw([-c for c in cost[i]])]
            options = env.legal_placements(piece, tunnel_id)
            option = options[int(self.rng.integers(len(options)))]
            action = Action(piece.piece_id, tunnel_id, option.mode, option.end)
            r, _, _ = env.step(action)
            total += discount * r
            discount *= cfg.gamma
            actions.append(action)
        return total / cfg.value_scale, actions

    # ---------- core ----------

    def _child_q(self, parent: Node, child: Node) -> float:
        if child.visits == 0:
            return parent.value()  # first-play urgency = parent's estimate
        return child.reward + self.cfg.gamma * child.q(self.cfg.max_backup_weight)

    def _select(self, node: Node, stats: MinMaxStats) -> Node:
        sqrt_n = math.sqrt(max(1, node.visits))
        best, best_score = None, -float("inf")
        for child in node.children.values():
            u = self.cfg.c_puct * child.prior * sqrt_n / (1 + child.visits)
            score = stats.norm(self._child_q(node, child)) + u
            if score > best_score:
                best, best_score = child, score
        return best

    def _backup(self, path: List[Node], value: float, stats: MinMaxStats,
                real: Optional[float] = None) -> Optional[float]:
        """`value` is the leaf estimate for the means. `real`, when given, is the true return from
        the leaf to the end of the game; it feeds each node's best finished line. Returns the true
        return of the whole line from the root in reward units, or None without `real`."""
        g, b = value, real
        for node in reversed(path):
            node.visits += 1
            node.value_sum += g
            g = node.reward + self.cfg.gamma * g
            stats.update(g)
            if b is not None:
                if node.best is None or b > node.best:
                    node.best = b
                b = node.reward + self.cfg.gamma * b
        return None if b is None else b * self.cfg.value_scale

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
            if self._noise_root is not None and node.parent is self._noise_root:
                self._add_dirichlet(node)
            return value

        # stage 2: choose placement (whole / cut at max / early cut) -> real transition
        legal = env.legal_placements(node.piece, node.tunnel_id)
        priors, value = self.ev.eval_tunnel(env, node.cache, node.slot, node.piece, node.tunnel_id, legal)
        for option, prior in zip(legal, priors):
            nxt = env.clone()
            action = Action(node.piece.piece_id, node.tunnel_id, option.mode, option.end)
            r, done, _ = nxt.step(action)
            child = Node(0, nxt, node, float(prior), reward=r / cfg.value_scale, terminal=done)
            child.action = action
            node.children[option] = child
        if self._noise_root is not None and node.parent.parent is self._noise_root:
            self._add_dirichlet(node)
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
