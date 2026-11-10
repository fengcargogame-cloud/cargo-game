"""Self-play, training loop, checkpointing and plotting."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GameConfig
from .domain import Action, MODES
from .env import CargoTunnelEnv
from .evaluators import NetEvaluator
from .mcts import HierarchicalMCTS, pick, visit_policy
from .network import HierarchicalPolicyValueNet
from .replay import ReplayBuffer, Sample, collate


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(target * logp).sum(-1).mean()


# ============================================================
# Trainer
# ============================================================


class Trainer:
    def __init__(self, config: GameConfig, seed: int = 42) -> None:
        self.cfg = config
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.episode_rng = random.Random(seed + 999)

        probe = CargoTunnelEnv(config)
        self.net = HierarchicalPolicyValueNet(config, probe.piece_feature_dim(), probe.tunnel_feature_dim()).to(config.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.mcts = HierarchicalMCTS(NetEvaluator(self.net, config.device), config, self.rng)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.history: Dict[str, List[float]] = {k: [] for k in
                                                ("reward", "penalty", "steps", "loss_piece", "loss_tunnel",
                                                 "loss_mode", "loss_value")}

    # ---------- self-play ----------

    def self_play_episode(self, explore: bool = True) -> Tuple[List[Sample], float, CargoTunnelEnv]:
        cfg = self.cfg
        env = CargoTunnelEnv(cfg)
        env.reset(seed=self.episode_rng.randint(0, 2 ** 31 - 1))
        records: List[Sample] = []
        rewards: List[float] = []
        root = self.mcts.new_root(env)
        move = 0

        while not env.is_terminal():
            self.mcts.run(root, add_noise=explore)
            temp = explore and move < cfg.temperature_moves

            # Stage 0: piece
            slots, pi_piece_vals = visit_policy(root)
            slot = pick(root, temp, self.rng)
            n1 = root.children[slot]
            if not n1.expanded:
                self.mcts._expand(n1)
            # Stage 1: tunnel (from the chosen piece's own subtree)
            tunnels, pi_t_vals = visit_policy(n1)
            tunnel_id = pick(n1, temp, self.rng)
            n2 = n1.children[tunnel_id]
            if not n2.expanded:
                self.mcts._expand(n2)
            # Stage 2: mode
            modes, pi_m_vals = visit_policy(n2)
            mode = pick(n2, temp, self.rng)

            piece = n1.piece
            K = cfg.num_tunnels
            pi_piece = np.zeros(len(env.candidates()), np.float32)
            pi_piece[slots] = pi_piece_vals
            pi_tunnel = np.zeros(K, np.float32)
            pi_tunnel[tunnels] = pi_t_vals
            pi_mode = np.zeros(2, np.float32)
            pi_mode[[MODES.index(m) for m in modes]] = pi_m_vals
            tmask = np.zeros(K, bool)
            tmask[env.legal_tunnels(piece)] = True
            mmask = np.array([m in env.legal_modes(piece, tunnel_id) for m in MODES])

            records.append(Sample(
                env.piece_features(), env.tunnel_features(), env.global_features(),
                env.pair_features(piece), env.mode_features(piece, tunnel_id),
                slot, tunnel_id, pi_piece, pi_tunnel, pi_mode, tmask, mmask,
            ))

            r, done, _ = env.step(Action(piece.piece_id, tunnel_id, mode))
            rewards.append(r)
            move += 1

            # Subtree reuse: the chosen stage-2 child already holds this exact next state.
            root = n2.children[mode]
            root.parent, root.reward = None, 0.0
            if done:
                break

        # Return-to-go targets for all three value heads
        g = 0.0
        for rec, r in zip(reversed(records), reversed(rewards)):
            g = r + cfg.gamma * g
            rec.value = float(np.clip(g / cfg.value_scale, -1.0, 1.0))
        return records, float(sum(rewards)), env

    # ---------- learning ----------

    def train_step(self, batch: List[Sample]) -> Tuple[float, float, float, float]:
        b = collate(batch, self.cfg.device)
        self.net.train()
        g, p_emb, t_emb = self.net.encode(b["piece_x"], b["piece_mask"], b["tunnel_x"], b["global_x"])
        logits0, v0 = self.net.stage0(g, p_emb, b["piece_mask"])

        idx = torch.arange(len(batch), device=g.device)
        e_piece = p_emb[idx, b["slot"]]
        logits1, v1, ctx1, keys = self.net.stage1(g, e_piece, t_emb, b["pair_x"], b["tunnel_mask"])
        logits2, v2 = self.net.stage2(ctx1, keys[idx, b["tunnel_id"]], b["mode_x"], b["mode_mask"])

        l_piece = soft_ce(logits0, b["pi_piece"])
        l_tunnel = soft_ce(logits1, b["pi_tunnel"])
        l_mode = soft_ce(logits2, b["pi_mode"])
        z = b["value"]
        l_value = F.mse_loss(v0, z) + F.mse_loss(v1, z) + F.mse_loss(v2, z)

        loss = l_piece + l_tunnel + l_mode + l_value
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return l_piece.item(), l_tunnel.item(), l_mode.item(), l_value.item()

    def train(self, verbose_every: int = 10) -> None:
        cfg = self.cfg
        for ep in range(1, cfg.num_episodes + 1):
            samples, ep_reward, env = self.self_play_episode(explore=True)
            self.replay.extend(samples)
            losses = np.zeros(4)
            if len(self.replay) >= cfg.batch_size:
                for _ in range(cfg.train_steps_per_episode):
                    losses += np.array(self.train_step(self.replay.sample(cfg.batch_size)))
                losses /= cfg.train_steps_per_episode

            h = self.history
            h["reward"].append(ep_reward)
            h["penalty"].append(env.total_penalty())
            h["steps"].append(env.step_count)
            for k, v in zip(("loss_piece", "loss_tunnel", "loss_mode", "loss_value"), losses):
                h[k].append(float(v))

            print(f"Ep {ep:04d} | reward={ep_reward:8.2f} | penalty={env.total_penalty():7.2f} | "
                  f"steps={env.step_count:3d} | unplaced={len(env.unassigned)} | "
                  f"L(piece/tunnel/mode/value)={losses[0]:.3f}/{losses[1]:.3f}/{losses[2]:.3f}/{losses[3]:.3f}")
            if verbose_every and ep % verbose_every == 0:
                # Greedy (no exploration) game, so the board shows what the agent actually does
                _, greedy_reward, greedy_env = self.self_play_episode(explore=False)
                print(f"  greedy reward={greedy_reward:8.2f} | penalty={greedy_env.total_penalty():7.2f}")
                print(greedy_env.render_grid())

    def save(self, path: str = "cargo_3stage.pt") -> None:
        torch.save({"model": self.net.state_dict(), "config": self.cfg.__dict__}, path)

    def plot(self, path: str = "training_curves.png") -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = self.history
        ep = np.arange(1, len(h["reward"]) + 1)

        def smooth(x, w=10):
            x = np.asarray(x, float)
            return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x

        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        ax[0, 0].plot(ep, h["reward"], alpha=0.3)
        ax[0, 0].plot(ep[len(ep) - len(smooth(h["reward"])):], smooth(h["reward"]))
        ax[0, 0].set_title("Episode reward")
        ax[0, 1].plot(ep, h["penalty"], alpha=0.3)
        ax[0, 1].plot(ep[len(ep) - len(smooth(h["penalty"])):], smooth(h["penalty"]))
        ax[0, 1].set_title("Episode penalty")
        for k in ("loss_piece", "loss_tunnel", "loss_mode", "loss_value"):
            ax[1, 0].plot(ep, h[k], label=k.replace("loss_", ""))
        ax[1, 0].legend()
        ax[1, 0].set_title("Losses per stage")
        ax[1, 1].plot(ep, h["steps"])
        ax[1, 1].set_title("Steps per episode")
        for a in ax.flat:
            a.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=110)
        print(f"Saved {path}")
