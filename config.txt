"""CLI-аргументы и гиперпараметры."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCGAN on MNIST")
    parser.add_argument("--epoch", type=int, default=10, help="Число эпох")
    parser.add_argument(
        "--debug_samples_epoch",
        type=int,
        default=1,
        help="Как часто сохранять сэмплы (в эпохах)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    return {
        "noise_dim": 100,
        "lr": 0.0002,
        "b1": 0.5,
        "b2": 0.999,
        "batch_size": 24,
        "val_size": 0.1,
        "num_epochs": args.epoch,
        "log_every_n_steps": 50,
        "num_samples_to_log": 8,
        "debug_samples_epoch": args.debug_samples_epoch,
    }