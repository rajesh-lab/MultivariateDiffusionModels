import argparse
import click
import pytorch_lightning as pl
from mdm.utils.utils import get_config, print_config_nicely
from mdm.bin.train import train


def hyperparam_grid():
    hyperparam_grid = []
    for sde_type in ["learned_2", "malda", "cld", "vpsde"]:
        for lr in [1e-4, 1e-3, 5e-3]:
            for batch_size in [128]:
                D = dict(
                    lr=lr,
                    batch_size=batch_size,
                    sde_type=sde_type,
                )
                hyperparam_grid.append(D)

    return hyperparam_grid


def get_grid_config(config_path: argparse.ArgumentParser, index: int):
    config = get_config(config_path=config_path)
    hyperparam_grid = hyperparam_grid()

    print(f"Selecting Index {index} out {len(hyperparam_grid)} options")
    hp = hyperparam_grid[index]
    for key in hp:
        setattr(config, key, hp[key])

    return config


@click.command()
@click.option("--config_path", required=True, help="Config file path.")
@click.option("--index", type=int, required=True, help="Which grid index to run.")
@click.option(
    "--offline",
    type=bool,
    default=False,
    required=False,
    help="run wandb in offline mode.",
)
@click.option(
    "--debug_mode",
    type=bool,
    default=False,
    required=False,
    help="run code in debug mode.",
)
def main(config_path, index, offline, debug_mode):
    config = get_grid_config(config_path, index)
    pl.seed_everything(config.seed)
    print_config_nicely(config)

    train(config=config, offline=offline, debug_mode=debug_mode)


if __name__ == "__main__":
    main()
