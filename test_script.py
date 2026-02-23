from PIL import Image, ImageDraw
import random, math
import os, sys

# basic parameters
IMAGE_SIZE = (128, 128)
BACKGROUND_COLOR = (255, 255, 255)

OVERLAP_DIR = "data/overlap_triange"
OVERLAP_BW_DIR = "data/overlap_triangle_bw"
CIRCLE_DIR = "data/overlap_circle"
CIRCLE_BW_DIR = "data/overlap_circle_bw"
NO_OVERLAP_DIR = "data/no_overlap"
NO_OVERLAP_BW_DIR = "data/no_overlap_bw"

N_IMAGES = 1000

# generate a random shape
def random_circle():
    radius = random.randint(8, 20)
    x = random.randint(radius, IMAGE_SIZE[0] - radius)
    y = random.randint(radius, IMAGE_SIZE[1] - radius)

    return {
        "center": (x, y),
        "radius": radius,
        "color": (0, 0, 0)
    }

def random_triangle():
    cx = random.randint(30, 100)
    cy = random.randint(30, 100)
    size = random.randint(12, 25)

    vertices = [
        (cx, cy - size),
        (cx - size, cy + size),
        (cx + size, cy + size)
    ]

    return {
        "vertices": vertices,
        "color": (255, 0, 0)
    }


# drawing functions
def draw_circle(draw, center, radius, color):
    x, y = center
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color
    )

def draw_triangle(draw, vertices, color):
    draw.polygon(vertices, fill=color)


# IMAGE GENERATING FUNCTIONS

# shapes overlapping (triangle over circle)
def generate_overlapping_triangle(idx):
    """
    shapes always overlap, varying size and position
    """
    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    circle = random_circle()

    cx, cy = circle["center"]
    size = circle["radius"] 

    max_offset = size // 2
    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)
    tx = cx + dx
    ty = cy + dy

    vertices = [
        (tx, ty - size),
        (tx - size, ty + size),
        (tx + size, ty + size)
    ]

    triangle = {
        "vertices": vertices,
        "color": (255, 0, 0)
    }

    draw_circle(draw, circle["center"], circle["radius"], circle["color"])
    draw_triangle(draw, triangle["vertices"], triangle["color"])

    filename = f"overlap_{idx:04d}.png"
    img.save(os.path.join(OVERLAP_DIR, filename))

# triangle over circle, black and white
def generate_overlapping_triangle_bw(idx):
    """
    same as above, no color
    """
    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    circle = random_circle()

    cx, cy = circle["center"]
    size = circle["radius"] 

    max_offset = size // 2
    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)
    tx = cx + dx
    ty = cy + dy

    vertices = [
        (tx, ty - size),
        (tx - size, ty + size),
        (tx + size, ty + size)
    ]

    triangle = {
        "vertices": vertices,
        "color": (0, 0, 0)
    }

    draw_circle(draw, circle["center"], circle["radius"], circle["color"])
    draw_triangle(draw, triangle["vertices"], triangle["color"])

    filename = f"overlap_{idx:04d}.png"
    img.save(os.path.join(OVERLAP_BW_DIR, filename))

# circle over triangle
def generate_overlapping_circle(idx):
    """
    circle drawn over triangle, varying size and position
    """
    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    circle = random_circle()

    cx, cy = circle["center"]
    size = circle["radius"]

    max_offset = size // 2
    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)
    tx = cx + dx
    ty = cy + dy

    vertices = [
        (tx, ty - size),
        (tx - size, ty + size),
        (tx + size, ty + size)
    ]

    triangle = {
        "vertices": vertices,
        "color": (255, 0, 0)
    }

    # draw triangle first
    draw_triangle(draw, triangle["vertices"], triangle["color"])
    # circle on top
    draw_circle(draw, circle["center"], circle["radius"], circle["color"])

    filename = f"overlap_{idx:04d}.png"
    img.save(os.path.join(CIRCLE_DIR, filename))

# circle over triangle, black and white
def generate_overlapping_circle_bw(idx):
    """
    circle over triangle, no color
    """
    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    circle = random_circle()
    circle["color"] = (0, 0, 0)

    cx, cy = circle["center"]
    size = circle["radius"]

    max_offset = size // 2
    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)
    tx = cx + dx
    ty = cy + dy

    vertices = [
        (tx, ty - size),
        (tx - size, ty + size),
        (tx + size, ty + size)
    ]

    triangle = {
        "vertices": vertices,
        "color": (0, 0, 0)
    }

    draw_triangle(draw, triangle["vertices"], triangle["color"])
    draw_circle(draw, circle["center"], circle["radius"], circle["color"])

    filename = f"overlap_{idx:04d}.png"
    img.save(os.path.join(CIRCLE_BW_DIR, filename))

# not overlapping
def generate_not_overlapping(idx):
    """
    shapes do not overlap
    """
    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    circle = random_circle()

    cx, cy = circle["center"]
    size = circle["radius"]

    # push triangle far enough away
    min_distance = size * 3

    angle = random.uniform(0, 2 * 3.14159)
    tx = int(cx + min_distance * math.cos(angle))
    ty = int(cy + min_distance * math.sin(angle))

    vertices = [
        (tx, ty - size),
        (tx - size, ty + size),
        (tx + size, ty + size)
    ]

    triangle = {
        "vertices": vertices,
        "color": (255, 0, 0)
    }

    draw_circle(draw, circle["center"], circle["radius"], circle["color"])
    draw_triangle(draw, triangle["vertices"], triangle["color"])

    filename = f"no_overlap_{idx:04d}.png"
    img.save(os.path.join(NO_OVERLAP_DIR, filename))

# not overlapping, black and white
def generate_not_overlapping_bw(idx):
    """
    shapes do not overlap, no color
    """
    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    circle = random_circle()
    circle["color"] = (0, 0, 0)

    cx, cy = circle["center"]
    size = circle["radius"]

    min_distance = size * 3

    angle = random.uniform(0, 2 * 3.14159)
    tx = int(cx + min_distance * math.cos(angle))
    ty = int(cy + min_distance * math.sin(angle))

    vertices = [
        (tx, ty - size),
        (tx - size, ty + size),
        (tx + size, ty + size)
    ]

    triangle = {
        "vertices": vertices,
        "color": (0, 0, 0)
    }

    draw_circle(draw, circle["center"], circle["radius"], circle["color"])
    draw_triangle(draw, triangle["vertices"], triangle["color"])

    filename = f"no_overlap_{idx:04d}.png"
    img.save(os.path.join(NO_OVERLAP_BW_DIR, filename))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_script.py [overlap]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "overlap_triangle":
        os.makedirs(OVERLAP_DIR, exist_ok=True)
        for i in range(N_IMAGES):
            generate_overlapping_triangle(i)

    if mode == "overlap_triangle_bw":
        os.makedirs(OVERLAP_BW_DIR, exist_ok=True)
        for i in range(N_IMAGES):
            generate_overlapping_triangle_bw(i)

    if mode == "overlap_circle":
        os.makedirs(CIRCLE_DIR, exist_ok=True)
        for i in range(N_IMAGES):
            generate_overlapping_circle(i)

    if mode == "overlap_circle_bw":
        os.makedirs(CIRCLE_BW_DIR, exist_ok=True)
        for i in range(N_IMAGES):
            generate_overlapping_circle_bw(i)

    if mode == "no_overlap":
        os.makedirs(NO_OVERLAP_DIR, exist_ok=True)
        for i in range(N_IMAGES):
            generate_not_overlapping(i)

    if mode == "no_overlap_bw":
        os.makedirs(NO_OVERLAP_BW_DIR, exist_ok=True)
        for i in range(N_IMAGES):
            generate_not_overlapping_bw(i)
    
