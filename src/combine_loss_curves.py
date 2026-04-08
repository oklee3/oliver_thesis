import argparse
import os
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw


RUN_ORDER = ["no_overlap", "no_overlap_bw", "overlap", "overlap_bw"]


def default_curve_paths(model_name: str) -> Dict[str, str]:
    return {
        run_name: os.path.join(
            "outline_images",
            model_name,
            "loss_curves",
            f"{model_name}_{run_name}_outline_loss_curve.png",
        )
        for run_name in RUN_ORDER
    }


def load_images(curve_paths: Dict[str, str]) -> Dict[str, Image.Image]:
    images: Dict[str, Image.Image] = {}
    missing: List[str] = []

    for run_name, path in curve_paths.items():
        if not os.path.isfile(path):
            missing.append(path)
            continue
        images[run_name] = Image.open(path).convert("RGB")

    if missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(f"Missing loss curve images:\n{missing_list}")

    return images


def combine_images(model_name: str, out_path: str, run_order: Sequence[str]) -> str:
    images = load_images(default_curve_paths(model_name))
    widths = [images[run_name].width for run_name in run_order]
    heights = [images[run_name].height for run_name in run_order]
    tile_width = max(widths)
    tile_height = max(heights)
    padding = 30
    title_height = 50

    canvas = Image.new(
        "RGB",
        (tile_width * 2 + padding * 3, tile_height * 2 + padding * 3 + title_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title = f"{model_name.upper()} Outline Loss Curves"
    draw.text((padding, 15), title, fill="black")

    for idx, run_name in enumerate(run_order):
        row, col = divmod(idx, 2)
        x = padding + col * (tile_width + padding)
        y = title_height + padding + row * (tile_height + padding)
        image = images[run_name]
        offset_x = x + (tile_width - image.width) // 2
        offset_y = y + (tile_height - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn", "mlp"],
        choices=["cnn", "mlp"],
        help="Model families to combine.",
    )
    args = parser.parse_args()

    for model_name in args.models:
        out_path = os.path.join("outline_images", model_name, f"{model_name}_outline_loss_curves_grid.png")
        saved_path = combine_images(model_name, out_path, RUN_ORDER)
        print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()