"""Policy-value network: shared set encoder plus one head per tree stage."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import GameConfig
from .env import CargoTunnelEnv


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
