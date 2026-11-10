"""Evaluators the MCTS queries for priors and values at each stage."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .domain import ActionMode, CargoPiece, MODES
from .env import CargoTunnelEnv
from .network import HierarchicalPolicyValueNet


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
