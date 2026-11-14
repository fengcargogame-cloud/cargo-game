"""Self-play samples, the replay buffer and batching."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch


@dataclass
class Sample:
    piece_x: np.ndarray        # (M, Fp)
    tunnel_x: np.ndarray       # (K, Ft)
    global_x: np.ndarray       # (G,)
    pair_x: np.ndarray         # (K, PAIR_DIM) for the chosen piece
    option_x: np.ndarray       # (O, OPTION_DIM) stage-2 options for the chosen piece+tunnel
    slot: int                  # chosen piece slot
    tunnel_id: int             # chosen tunnel
    option_id: int             # chosen stage-2 option
    pi_piece: np.ndarray       # (M,)  stage-0 visit distribution
    pi_tunnel: np.ndarray      # (K,)  stage-1 visit distribution (under chosen piece)
    pi_mode: np.ndarray        # (O,)  stage-2 visit distribution (under chosen tunnel)
    tunnel_mask: np.ndarray    # (K,)  legal tunnels for chosen piece
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
    O = max(len(s.option_x) for s in batch)
    option_x = np.zeros((B, O, batch[0].option_x.shape[1]), np.float32)
    option_mask = np.zeros((B, O), bool)
    pi_mode = np.zeros((B, O), np.float32)
    for i, s in enumerate(batch):
        n = len(s.option_x)
        option_x[i, :n] = s.option_x
        option_mask[i, :n] = True
        pi_mode[i, :n] = s.pi_mode
    out = {
        "piece_x": piece_x, "piece_mask": piece_mask, "pi_piece": pi_piece,
        "tunnel_x": np.stack([s.tunnel_x for s in batch]),
        "global_x": np.stack([s.global_x for s in batch]),
        "pair_x": np.stack([s.pair_x for s in batch]),
        "option_x": option_x, "option_mask": option_mask, "pi_mode": pi_mode,
        "slot": np.array([s.slot for s in batch], np.int64),
        "tunnel_id": np.array([s.tunnel_id for s in batch], np.int64),
        "pi_tunnel": np.stack([s.pi_tunnel for s in batch]),
        "tunnel_mask": np.stack([s.tunnel_mask for s in batch]),
        "value": np.array([s.value for s in batch], np.float32),
    }
    return {k: torch.as_tensor(v, device=device) for k, v in out.items()}
