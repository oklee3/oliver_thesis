from PIL import Image, ImageDraw, ImageChops
import random
import math
import os
import sys

# basic parameters
IMAGE_SIZE = (128, 128)
BACKGROUND_COLOR = (0, 0, 0)
PRIMARY_SHAPE_COLOR = (255, 255, 255)
N_IMAGES = 100000

# output directories
NO_OVERLAP_CIRCLE_DIR = "data/no_overlap_circle"
NO_OVERLAP_CIRCLE_BW_DIR = "data/no_overlap_circle_bw"
NO_OVERLAP_TRIANGLE_DIR = "data/no_overlap_triangle"
NO_OVERLAP_TRIANGLE_BW_DIR = "data/no_overlap_triangle_bw"
OVERLAP_CIRCLE_DIR = "data/overlap_circle"
OVERLAP_CIRCLE_BW_DIR = "data/overlap_circle_bw"
OVERLAP_TRIANGLE_DIR = "data/overlap_triangle"
OVERLAP_TRIANGLE_BW_DIR = "data/overlap_triangle_bw"


def random_circle(radius_min=8, radius_max=18):
    radius = random.randint(radius_min, radius_max)
    x = random.randint(radius + 2, IMAGE_SIZE[0] - radius - 2)
    y = random.randint(radius + 2, IMAGE_SIZE[1] - radius - 2)
    return {"center": (x, y), "radius": radius, "color": PRIMARY_SHAPE_COLOR}


def triangle_vertices(cx, cy, size):
    return [
        (cx, cy - size),
        (cx - size, cy + size),
        (cx + size, cy + size),
    ]


def triangle_bounds(vertices):
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def draw_circle(draw, center, radius, color):
    x, y = center
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def draw_triangle(draw, vertices, color):
    draw.polygon(vertices, fill=color)


def in_bounds_circle(center, radius):
    x, y = center
    return (
        x - radius >= 0
        and y - radius >= 0
        and x + radius < IMAGE_SIZE[0]
        and y + radius < IMAGE_SIZE[1]
    )


def in_bounds_triangle(vertices):
    min_x, min_y, max_x, max_y = triangle_bounds(vertices)
    return min_x >= 0 and min_y >= 0 and max_x < IMAGE_SIZE[0] and max_y < IMAGE_SIZE[1]


def mask_circle(center, radius):
    mask = Image.new("1", IMAGE_SIZE, 0)
    draw = ImageDraw.Draw(mask)
    x, y = center
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=1)
    return mask


def mask_triangle(vertices):
    mask = Image.new("1", IMAGE_SIZE, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(vertices, fill=1)
    return mask


def mask_count(mask):
    # For mode "1", set pixels are 255 in getdata().
    return sum(1 for px in mask.getdata() if px)


def has_required_overlap_visibility(circle, tri, min_overlap_px=20, min_exclusive_px=20):
    c_mask = mask_circle(circle["center"], circle["radius"])
    t_mask = mask_triangle(tri)
    inter = ImageChops.logical_and(c_mask, t_mask)

    c_px = mask_count(c_mask)
    t_px = mask_count(t_mask)
    i_px = mask_count(inter)
    c_only = c_px - i_px
    t_only = t_px - i_px

    return i_px >= min_overlap_px and c_only >= min_exclusive_px and t_only >= min_exclusive_px


def placement_non_overlap(above="circle"):
    # Build shapes around same x-range while controlling vertical gap.
    for _ in range(500):
        radius = random.randint(8, 16)
        size = random.randint(8, 16)

        cx = random.randint(radius + 2, IMAGE_SIZE[0] - radius - 2)
        tx = random.randint(size + 2, IMAGE_SIZE[0] - size - 2)

        # Re-center horizontally a bit so they are both visible and near each other.
        mid_x = (cx + tx) // 2
        cx = max(radius + 2, min(IMAGE_SIZE[0] - radius - 2, mid_x + random.randint(-6, 6)))
        tx = max(size + 2, min(IMAGE_SIZE[0] - size - 2, mid_x + random.randint(-6, 6)))

        if above == "circle":
            # Circle strictly above triangle with guaranteed vertical gap.
            circle_top = random.randint(2, 32)
            cy = circle_top + radius

            tri_top_min = cy + radius + 8
            tri_top_max = IMAGE_SIZE[1] - (2 * size) - 2
            if tri_top_min > tri_top_max:
                continue
            tri_top = random.randint(tri_top_min, tri_top_max)
            ty = tri_top + size
        else:
            # Triangle strictly above circle with guaranteed vertical gap.
            tri_top = random.randint(2, 26)
            ty = tri_top + size

            circle_top_min = ty + size + 8
            circle_top_max = IMAGE_SIZE[1] - (2 * radius) - 2
            if circle_top_min > circle_top_max:
                continue
            circle_top = random.randint(circle_top_min, circle_top_max)
            cy = circle_top + radius

        circle = {"center": (cx, cy), "radius": radius}
        tri = triangle_vertices(tx, ty, size)

        if not in_bounds_circle(circle["center"], circle["radius"]):
            continue
        if not in_bounds_triangle(tri):
            continue

        return circle, tri

    raise RuntimeError("Could not place non-overlapping shapes")


def placement_overlap(above="circle"):
    # Force overlap by anchoring both shapes around a shared point.
    for _ in range(5000):
        radius = random.randint(10, 16)
        size = random.randint(10, 16)

        margin = max(radius, size) + 10
        anchor_x = random.randint(margin, IMAGE_SIZE[0] - margin)
        anchor_y = random.randint(margin, IMAGE_SIZE[1] - margin)

        x_jitter = random.randint(-3, 3)
        if above == "circle":
            # Circle centroid above triangle centroid.
            cx = anchor_x + x_jitter
            cy = anchor_y - random.randint(1, 4)
            tx = anchor_x - x_jitter
            ty = anchor_y + random.randint(1, 4)
        else:
            # Triangle centroid above circle centroid.
            tx = anchor_x + x_jitter
            ty = anchor_y - random.randint(1, 4)
            cx = anchor_x - x_jitter
            cy = anchor_y + random.randint(1, 4)

        tri = triangle_vertices(tx, ty, size)
        circle = {"center": (cx, cy), "radius": radius}

        if not in_bounds_circle(circle["center"], circle["radius"]):
            continue
        if not in_bounds_triangle(tri):
            continue

        # Ensure at least partial vertical ordering ("above") via centroid.
        if above == "circle" and not (cy <= ty):
            continue
        if above == "triangle" and not (ty <= cy):
            continue

        if not has_required_overlap_visibility(circle, tri):
            continue

        return circle, tri

    raise RuntimeError("Could not place overlapping shapes")


def save_image(path, filename, circle, triangle, circle_color, tri_color, top_shape):
    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    if top_shape == "circle":
        draw_triangle(draw, triangle, tri_color)
        draw_circle(draw, circle["center"], circle["radius"], circle_color)
    else:
        draw_circle(draw, circle["center"], circle["radius"], circle_color)
        draw_triangle(draw, triangle, tri_color)

    img.save(os.path.join(path, filename))


def generate_no_overlap_circle(idx, bw=False):
    circle, tri = placement_non_overlap(above="circle")
    c = PRIMARY_SHAPE_COLOR
    t = PRIMARY_SHAPE_COLOR if bw else (255, 0, 0)
    filename = f"no_overlap_{idx:04d}.png"
    outdir = NO_OVERLAP_CIRCLE_BW_DIR if bw else NO_OVERLAP_CIRCLE_DIR
    save_image(outdir, filename, circle, tri, c, t, top_shape="circle")


def generate_no_overlap_triangle(idx, bw=False):
    circle, tri = placement_non_overlap(above="triangle")
    c = PRIMARY_SHAPE_COLOR
    t = PRIMARY_SHAPE_COLOR if bw else (255, 0, 0)
    filename = f"no_overlap_{idx:04d}.png"
    outdir = NO_OVERLAP_TRIANGLE_BW_DIR if bw else NO_OVERLAP_TRIANGLE_DIR
    save_image(outdir, filename, circle, tri, c, t, top_shape="triangle")


def generate_overlap_circle(idx, bw=False):
    circle, tri = placement_overlap(above="circle")
    c = PRIMARY_SHAPE_COLOR
    t = PRIMARY_SHAPE_COLOR if bw else (255, 0, 0)
    filename = f"overlap_{idx:04d}.png"
    outdir = OVERLAP_CIRCLE_BW_DIR if bw else OVERLAP_CIRCLE_DIR
    save_image(outdir, filename, circle, tri, c, t, top_shape="circle")


def generate_overlap_triangle(idx, bw=False):
    circle, tri = placement_overlap(above="triangle")
    c = PRIMARY_SHAPE_COLOR
    t = PRIMARY_SHAPE_COLOR if bw else (255, 0, 0)
    filename = f"overlap_{idx:04d}.png"
    outdir = OVERLAP_TRIANGLE_BW_DIR if bw else OVERLAP_TRIANGLE_DIR
    save_image(outdir, filename, circle, tri, c, t, top_shape="triangle")


def generate_dataset(mode):
    mapping = {
        "no_overlap_circle": (NO_OVERLAP_CIRCLE_DIR, lambda i: generate_no_overlap_circle(i, bw=False)),
        "no_overlap_circle_bw": (NO_OVERLAP_CIRCLE_BW_DIR, lambda i: generate_no_overlap_circle(i, bw=True)),
        "no_overlap_triangle": (NO_OVERLAP_TRIANGLE_DIR, lambda i: generate_no_overlap_triangle(i, bw=False)),
        "no_overlap_triangle_bw": (NO_OVERLAP_TRIANGLE_BW_DIR, lambda i: generate_no_overlap_triangle(i, bw=True)),
        "overlap_circle": (OVERLAP_CIRCLE_DIR, lambda i: generate_overlap_circle(i, bw=False)),
        "overlap_circle_bw": (OVERLAP_CIRCLE_BW_DIR, lambda i: generate_overlap_circle(i, bw=True)),
        "overlap_triangle": (OVERLAP_TRIANGLE_DIR, lambda i: generate_overlap_triangle(i, bw=False)),
        "overlap_triangle_bw": (OVERLAP_TRIANGLE_BW_DIR, lambda i: generate_overlap_triangle(i, bw=True)),
    }

    if mode not in mapping:
        print(
            "Usage: python3 test_script.py [no_overlap_circle|no_overlap_circle_bw|"
            "no_overlap_triangle|no_overlap_triangle_bw|overlap_circle|"
            "overlap_circle_bw|overlap_triangle|overlap_triangle_bw]"
        )
        sys.exit(1)

    outdir, fn = mapping[mode]
    os.makedirs(outdir, exist_ok=True)
    for i in range(N_IMAGES):
        fn(i)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python3 test_script.py [no_overlap_circle|no_overlap_circle_bw|"
            "no_overlap_triangle|no_overlap_triangle_bw|overlap_circle|"
            "overlap_circle_bw|overlap_triangle|overlap_triangle_bw]"
        )
        sys.exit(1)

    generate_dataset(sys.argv[1])
