# cargo-game

Cargo-tunnel placement with PyTorch, AlphaZero-style self-play and a 3-stage hierarchical MCTS
(which cargo, then which tunnel, then place it whole or cut it).

## Layout

```
train.py                  entry point: run tests, train, save, plot, compare with the optimum
cargo_game/
    config.py             GameConfig, DynamicGameConfig, StaticGameConfig (the instance to solve)
    domain.py             CargoPiece, PlacedPiece, Action, ActionMode, Tunnel
    env.py                CargoTunnelEnv: legal moves, rewards and penalties, network features
    solver.py             brute_force_optimum: exact search for small instances
    network.py            HierarchicalPolicyValueNet
    evaluators.py         NetEvaluator (network) and UniformEvaluator (no network)
    mcts.py               HierarchicalMCTS
    replay.py             Sample, ReplayBuffer, collate
    trainer.py            Trainer: self-play, training loop, save and plot
tests/test_cargo_game.py  tests
data/                     input data
```

## Usage

Run from the repository root:

```bash
python train.py                  # tests, then training
python -m tests.test_cargo_game  # tests only
```

The instance and training settings are in `StaticGameConfig` in `cargo_game/config.py`.
Training writes `cargo_3stage.pt` and `training_curves.png` to the current directory.
