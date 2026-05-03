import argparse
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple


TRAINER_PATHS: Dict[str, str] = {
    "mlp": "src/train_mlp.py",
    "cnn": "src/train_cnn.py",
}

DEFAULT_RUN_SPECS: Sequence[Tuple[str, str]] = (
    ("mlp", "data"),
    ("cnn", "outline_data"),
    ("mlp", "outline_data"),
)
RUN_SPEC_CHOICES = sorted(
    {
        f"{model}:{dataset}"
        for model in sorted(TRAINER_PATHS.keys())
        for dataset in ("data", "outline_data")
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain each selected model once and save ggplot-style loss curves."
    )
    parser.add_argument(
        "--run-spec",
        action="append",
        choices=RUN_SPEC_CHOICES,
        help=(
            "Specific model/dataset combo to run. "
            "Repeat to override the default queue of mlp:data, cnn:outline_data, mlp:outline_data."
        ),
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_command(model_name: str, dataset_name: str, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        TRAINER_PATHS[model_name],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--artifact-label",
        "ggplot",
        "--datasets",
        dataset_name,
    ]


def main() -> None:
    args = parse_args()

    if args.run_spec:
        run_specs = [tuple(run_spec.split(":", maxsplit=1)) for run_spec in args.run_spec]
    else:
        run_specs = list(DEFAULT_RUN_SPECS)

    for model_name, dataset_name in run_specs:
        cmd = build_command(model_name, dataset_name, args)
        print(
            f"[replot_loss_curves] Retraining {model_name} on {dataset_name} with ggplot loss curves..."
        )
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
