"""Tests for the environment, search, network and solver."""

from __future__ import annotations

import math

import numpy as np

from cargo_game.config import DynamicGameConfig, StaticGameConfig
from cargo_game.domain import Action, ActionMode, CargoPiece, MODES
from cargo_game.env import CargoTunnelEnv
from cargo_game.evaluators import UniformEvaluator
from cargo_game.mcts import HierarchicalMCTS, pick
from cargo_game.solver import brute_force_optimum
from cargo_game.trainer import Trainer


def _manual_env(**overrides) -> CargoTunnelEnv:
    cfg = DynamicGameConfig(num_tunnels=2, tunnel_length=20, max_cargos=2, **overrides)
    env = CargoTunnelEnv(cfg)
    env.reset(pieces=[
        CargoPiece(1, 0, 8, 0, 0, 5),    # needs a cut to avoid overflow
        CargoPiece(2, 1, 4, 1, 3, 10),
    ])
    return env


def test_env_basics() -> None:
    env = _manual_env()
    r, done, info = env.step(Action(1, 0, ActionMode.CUT_AT_MAX))
    assert info["placed_len"] == 5 and info["remainder"] == 3 and not done
    assert env.num_cuts == 1 and r < 0
    env2 = _manual_env(mask_overflow_whole=False)
    r2, _, _ = env2.step(Action(1, 0, ActionMode.WHOLE))
    assert env2.compute_overflow_penalty() == 3 * env2.config.penalty_exceed_max and r2 < 0


def test_masking_and_urgency() -> None:
    env = _manual_env(mask_overflow_whole=True)
    cands = env.candidates()
    assert cands[0][0].piece_id == 1, "tightest-deadline piece should rank first"
    assert env.legal_modes(cands[0][0], 0) == [ActionMode.CUT_AT_MAX], "overflowing WHOLE should be masked"

    # A short cargo at the tunnel end has zero slack but can wait while tunnels are open;
    # a long cargo that needs several tunnels is the urgent one.
    env = CargoTunnelEnv(DynamicGameConfig(num_tunnels=3, tunnel_length=16, max_cargos=2))
    env.reset(pieces=[CargoPiece(1, 0, 2, 0, 14, 16), CargoPiece(2, 1, 19, 0, 9, 16)])
    assert env.candidates()[0][0].piece_id == 2, "piece needing 3 tunnels should outrank one that can wait"


def test_clone_isolation() -> None:
    env = _manual_env()
    c = env.clone()
    c.step(Action(2, 1, ActionMode.WHOLE))
    assert len(env.unassigned) == 2 and not env.tunnels[1].placed


def test_tree_has_three_stages() -> None:
    cfg = DynamicGameConfig(num_tunnels=3, tunnel_length=30, max_cargos=5, mcts_simulations=60)
    env = CargoTunnelEnv(cfg)
    env.reset(seed=3)
    mcts = HierarchicalMCTS(UniformEvaluator(), cfg)
    root = mcts.run(mcts.new_root(env), add_noise=False)
    assert root.stage == 0 and len(root.children) <= cfg.top_k_pieces
    n1 = max(root.children.values(), key=lambda n: n.visits)
    assert n1.stage == 1 and n1.env is root.env, "piece choice must not change the env"
    assert all(k in range(cfg.num_tunnels) for k in n1.children)
    n2 = max(n1.children.values(), key=lambda n: n.visits)
    assert n2.stage == 2 and set(n2.children) <= set(MODES)
    n3 = max(n2.children.values(), key=lambda n: n.visits)
    assert n3.stage == 0 and n3.env.step_count == 1, "mode choice performs the real step"
    assert root.visits == cfg.mcts_simulations


def test_mcts_prefers_cut_over_overflow() -> None:
    # Two tunnels: cutting lets the remainder go to the empty tunnel without overflow
    # (+2 cargo +5 bonus -1.5 cut), whereas WHOLE overflows by 3 (-24 +2 +5). Search must find the cut.
    cfg = DynamicGameConfig(num_tunnels=2, tunnel_length=20, max_cargos=1, mask_overflow_whole=False,
                     mcts_simulations=200)
    env = CargoTunnelEnv(cfg)
    env.reset(pieces=[CargoPiece(1, 0, 8, 0, 0, 5)])
    mcts = HierarchicalMCTS(UniformEvaluator(), cfg)
    root = mcts.run(mcts.new_root(env), add_noise=False)
    n2 = root.children[0].children[0]
    assert pick(n2, False, np.random.default_rng(0)) == ActionMode.CUT_AT_MAX


def test_network_and_train_step() -> None:
    cfg = DynamicGameConfig(num_tunnels=3, tunnel_length=30, max_cargos=5, mcts_simulations=20,
                            batch_size=4, d_model=32)
    tr = Trainer(cfg, seed=1)
    samples, _, env = tr.self_play_episode()
    assert samples and env.is_terminal()
    tr.replay.extend(samples)
    losses = tr.train_step(tr.replay.sample(4))
    assert all(np.isfinite(losses))


def test_static_game() -> None:
    cfg = StaticGameConfig()
    a, b = CargoTunnelEnv(cfg), CargoTunnelEnv(cfg)
    a.reset(seed=1)
    b.reset(seed=2)
    assert a.unassigned == b.unassigned and cfg.max_cargos == len(cfg.pieces), "seed must not matter"

    # The solver must find a known zero-penalty optimum on a tiny instance, and its line must
    # replay to the value it reports. (The default instance takes too long to solve here.)
    tiny = StaticGameConfig(pieces=((6, 0, 0, 6), (4, 0, 0, 10), (5, 1, 0, 5), (5, 1, 0, 10)))
    env = CargoTunnelEnv(tiny)
    env.reset()
    best, actions = brute_force_optimum(env)
    assert best == tiny.completion_bonus + len(tiny.pieces) * tiny.cargo_reward
    assert len(actions) == len(tiny.pieces)
    assert math.isclose(sum(env.step(x)[0] for x in actions), best) and env.is_terminal()


def run_tests() -> None:
    for t in (test_env_basics, test_masking_and_urgency, test_clone_isolation,
              test_tree_has_three_stages, test_mcts_prefers_cut_over_overflow,
              test_network_and_train_step, test_static_game):
        t()
        print(f"  {t.__name__}: passed")
    print("All tests passed.\n")


if __name__ == "__main__":
    run_tests()
