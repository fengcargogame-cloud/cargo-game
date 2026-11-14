"""Train the agent on the configured game and compare it with the exact optimum."""

from __future__ import annotations

import math

from cargo_game.config import DynamicGameConfig, StaticGameConfig  # noqa: F401 (DynamicGameConfig: see main)
from cargo_game.env import CargoTunnelEnv
from cargo_game.solver import brute_force_optimum
from cargo_game.trainer import Trainer
from tests.test_cargo_game import run_tests


def main() -> None:
    run_tests()
    config = StaticGameConfig()
    # config = DynamicGameConfig(num_episodes=200, mcts_simulations=160)
    trainer = Trainer(config, seed=42)
    trainer.train(verbose_every=20)
    trainer.save()
    trainer.plot()

    if isinstance(config, StaticGameConfig):
        env = CargoTunnelEnv(config)
        env.reset()
        best, actions = brute_force_optimum(env)
        _, agent_reward, agent_env = trainer.self_play_episode(explore=False)
        print(f"Optimal reward={best:.2f} | agent reward={agent_reward:.2f} | "
              f"{'OPTIMAL' if math.isclose(agent_reward, best) else 'SUBOPTIMAL'}")
        print("Optimal actions:", [(a.piece_id, a.tunnel_id, a.mode.value, a.cut_end) for a in actions])
        print(agent_env.render_text())
        print(agent_env.render_grid())


if __name__ == "__main__":
    main()
