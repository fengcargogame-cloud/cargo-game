"""Self-play, training loop, checkpointing and plotting."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GameConfig
from .domain import Action, ActionMode
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
        # Best line found so far (fixed-instance games), whether played in self-play or reached by
        # a search rollout: its positions, with its own moves as policy targets, are mixed into
        # every training batch.
        self.best_reward = -float("inf")
        self.best_env: Optional[CargoTunnelEnv] = None
        self.best_samples: List[Sample] = []
        self.best_line: List[Action] = []
        self.history: Dict[str, List[float]] = {k: [] for k in
                                                ("reward", "penalty", "steps", "loss_piece", "loss_tunnel",
                                                 "loss_mode", "loss_value", "best")}
        self.greedy_history: List[Tuple[int, float]] = []   # (episode, greedy reward)

    # ---------- self-play ----------

    def self_play_episode(self, explore: bool = True) -> Tuple[List[Sample], float, CargoTunnelEnv]:
        cfg = self.cfg
        env = CargoTunnelEnv(cfg)
        env.reset(seed=self.episode_rng.randint(0, 2 ** 31 - 1))
        records: List[Sample] = []
        rewards: List[float] = []
        played: List[Action] = []
        seed_state = env.clone()
        # `plan` is the best finished continuation known from the current state and `plan_value`
        # its return-to-go. The game is deterministic, so a plan is a guaranteed result: the greedy
        # agent follows it, and switches only when the search finds a better finished line.
        plan: List[Action] = []
        plan_value = -float("inf")
        if self.best_line:
            if not explore:
                plan, plan_value = list(self.best_line), self.best_reward
            elif self.rng.random() < cfg.elite_prefix_prob:
                for action in self.best_line[: int(self.rng.integers(len(self.best_line)))]:
                    played.append(action)
                    rewards.append(env.step(action)[0])
        root = self.mcts.new_root(env)
        move = 0

        while not env.is_terminal():
            self.mcts.run(root, add_noise=explore)
            if self.mcts.found is not None:
                # A simulation finished the game from here.
                rest, line = self.mcts.found
                if rest > plan_value + 1e-9:
                    plan, plan_value = list(line), rest
                if cfg.elite_fraction > 0:
                    self._adopt_line(seed_state, played + line, sum(rewards) + rest)
            temp = explore and move < cfg.temperature_moves
            follow = self._plan_nodes(root, plan[0]) if plan and not explore else None

            # Stage 0: piece
            slots, pi_piece_vals = visit_policy(root)
            slot = follow[0] if follow else pick(root, temp, self.rng)
            n1 = root.children[slot]
            if not n1.expanded:
                self.mcts._expand(n1)
            # Stage 1: tunnel (from the chosen piece's own subtree)
            tunnels, pi_t_vals = visit_policy(n1)
            tunnel_id = follow[1] if follow else pick(n1, temp, self.rng)
            n2 = n1.children[tunnel_id]
            if not n2.expanded:
                self.mcts._expand(n2)
            # Stage 2: placement option (whole / cut at max / early cut)
            chosen_options, pi_m_vals = visit_policy(n2)
            option = follow[2] if follow else pick(n2, temp, self.rng)

            piece = n1.piece
            K = cfg.num_tunnels
            pi_piece = np.zeros(len(env.candidates()), np.float32)
            pi_piece[slots] = pi_piece_vals
            pi_tunnel = np.zeros(K, np.float32)
            pi_tunnel[tunnels] = pi_t_vals
            options = env.legal_placements(piece, tunnel_id)
            pi_mode = np.zeros(len(options), np.float32)
            pi_mode[[options.index(o) for o in chosen_options]] = pi_m_vals
            tmask = np.zeros(K, bool)
            tmask[env.legal_tunnels(piece)] = True

            records.append(Sample(
                env.piece_features(), env.tunnel_features(), env.global_features(),
                env.pair_features(piece), env.option_features(piece, tunnel_id, options),
                slot, tunnel_id, options.index(option), pi_piece, pi_tunnel, pi_mode, tmask,
            ))

            played.append(Action(piece.piece_id, tunnel_id, option.mode, option.end))
            r, done, _ = env.step(played[-1])
            rewards.append(r)
            if plan and played[-1] == plan[0]:
                plan, plan_value = plan[1:], (plan_value - r) / cfg.gamma
            else:
                plan, plan_value = [], -float("inf")
            move += 1

            # Subtree reuse: the chosen stage-2 child already holds this exact next state.
            root = n2.children[option]
            root.parent, root.reward = None, 0.0
            if done:
                break

        # Return-to-go targets for all three value heads
        g = 0.0
        for rec, r in zip(reversed(records), reversed(rewards)):
            g = r + cfg.gamma * g
            rec.value = float(np.clip(g / cfg.value_scale, -1.0, 1.0))
        total = float(sum(rewards))
        if cfg.elite_fraction > 0:
            self._adopt_line(seed_state, played, total)
        return records, total, env

    def _plan_nodes(self, root, action: Action):
        """(slot, tunnel, placement) tree keys for `action`, or None if the tree does not offer it."""
        slot = next((s for s, n in root.children.items() if n.piece.piece_id == action.piece_id), None)
        if slot is None:
            return None
        n1 = root.children[slot]
        if not n1.expanded:
            self.mcts._expand(n1)
        n2 = n1.children.get(action.tunnel_id)
        if n2 is None:
            return None
        if not n2.expanded:
            self.mcts._expand(n2)
        option = next((o for o in n2.children if o.mode == action.mode
                       and (o.mode != ActionMode.CUT_EARLY or o.end == action.cut_end)), None)
        return None if option is None else (slot, action.tunnel_id, option)

    def _adopt_line(self, start: CargoTunnelEnv, actions: List[Action], total: float) -> None:
        """Make `actions` (a full game from `start`) the elite episode if it beats the best so far."""
        if total <= self.best_reward + 1e-9:
            return
        cfg, env = self.cfg, start.clone()
        samples: List[Sample] = []
        rewards: List[float] = []
        for action in actions:
            piece = env.piece_by_id(action.piece_id)
            slots = [p.piece_id for p, _ in env.candidates()]
            tunnels = env.legal_tunnels(piece)
            options = env.legal_placements(piece, action.tunnel_id)
            option = next((o for o in options if o.mode == action.mode
                           and (o.mode != ActionMode.CUT_EARLY or o.end == action.cut_end)), None)
            if piece.piece_id not in slots or action.tunnel_id not in tunnels or option is None:
                return   # the line uses a move the masks hide from the agent; it cannot imitate it
            pi_piece = np.zeros(len(slots), np.float32)
            pi_piece[slots.index(piece.piece_id)] = 1.0
            pi_tunnel = np.zeros(cfg.num_tunnels, np.float32)
            pi_tunnel[action.tunnel_id] = 1.0
            pi_mode = np.zeros(len(options), np.float32)
            pi_mode[options.index(option)] = 1.0
            tmask = np.zeros(cfg.num_tunnels, bool)
            tmask[tunnels] = True
            samples.append(Sample(
                env.piece_features(), env.tunnel_features(), env.global_features(),
                env.pair_features(piece), env.option_features(piece, action.tunnel_id, options),
                slots.index(piece.piece_id), action.tunnel_id, options.index(option),
                pi_piece, pi_tunnel, pi_mode, tmask,
            ))
            rewards.append(env.step(action)[0])
        g = 0.0
        for rec, r in zip(reversed(samples), reversed(rewards)):
            g = r + cfg.gamma * g
            rec.value = float(np.clip(g / cfg.value_scale, -1.0, 1.0))
        self.best_reward, self.best_env, self.best_samples = float(sum(rewards)), env, samples
        self.best_line = list(actions)

    def sample_batch(self) -> List[Sample]:
        cfg = self.cfg
        n_elite = round(cfg.elite_fraction * cfg.batch_size) if self.best_samples else 0
        batch = self.replay.sample(cfg.batch_size - n_elite)
        return batch + random.choices(self.best_samples, k=n_elite) if n_elite else batch

    # ---------- learning ----------

    def train_step(self, batch: List[Sample]) -> Tuple[float, float, float, float]:
        b = collate(batch, self.cfg.device)
        self.net.train()
        g, p_emb, t_emb = self.net.encode(b["piece_x"], b["piece_mask"], b["tunnel_x"], b["global_x"])
        logits0, v0 = self.net.stage0(g, p_emb, b["piece_mask"])

        idx = torch.arange(len(batch), device=g.device)
        e_piece = p_emb[idx, b["slot"]]
        logits1, v1, ctx1, keys = self.net.stage1(g, e_piece, t_emb, b["pair_x"], b["tunnel_mask"])
        logits2, v2 = self.net.stage2(ctx1, keys[idx, b["tunnel_id"]], b["option_x"], b["option_mask"])

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
                    losses += np.array(self.train_step(self.sample_batch()))
                losses /= cfg.train_steps_per_episode

            h = self.history
            h["reward"].append(ep_reward)
            h["penalty"].append(env.total_penalty())
            h["steps"].append(env.step_count)
            h["best"].append(self.best_reward if self.best_env is not None else float("nan"))
            for k, v in zip(("loss_piece", "loss_tunnel", "loss_mode", "loss_value"), losses):
                h[k].append(float(v))

            print(f"Ep {ep:04d} | reward={ep_reward:8.2f} | penalty={env.total_penalty():7.2f} | "
                  f"steps={env.step_count:3d} | unplaced={len(env.unassigned)} | "
                  f"L(piece/tunnel/mode/value)={losses[0]:.3f}/{losses[1]:.3f}/{losses[2]:.3f}/{losses[3]:.3f}")
            if verbose_every and ep % verbose_every == 0:
                # Greedy (no exploration) game, so the board shows what the agent actually does
                _, greedy_reward, greedy_env = self.self_play_episode(explore=False)
                self.greedy_history.append((ep, greedy_reward))
                best = f" | best so far={self.best_reward:8.2f}" if self.best_env is not None else ""
                print(f"  greedy reward={greedy_reward:8.2f} | penalty={greedy_env.total_penalty():7.2f}{best}")
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
        if not np.all(np.isnan(h["best"])):
            ax[0, 0].plot(ep, h["best"], color="tab:green", label="best layout so far")
        if self.greedy_history:
            ax[0, 0].plot(*zip(*self.greedy_history), "k.", markersize=4, label="greedy agent")
        lo = np.nanmin([np.nanmin(h["best"]) if not np.all(np.isnan(h["best"])) else np.inf,
                        np.percentile(h["reward"], 10)])
        ax[0, 0].set_ylim(lo - 5, max(h["reward"]) + 5)   # exploration crashes would flatten the rest
        ax[0, 0].legend(loc="lower right")
        ax[0, 0].set_title("Episode reward (exploring self-play, faint and smoothed)")
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
