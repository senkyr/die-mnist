"""
Synthetic dataset generator that creates digit-like shapes using simple geometry.
Produces 32x32 grayscale BMP images that can be fed into DigitLearner.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# Kořen repozitáře do sys.path kvůli sdílenému balíčku die_common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from die_common.data import allocate_counts  # noqa: E402
from die_common.paths import SYNTHETIC_DIR  # noqa: E402

SIZE = 32  # Output image size.


@dataclass(frozen=True)
class ShapeSpec:
    """How to render a particular pseudo-digit."""

    slug: str
    drawer: Callable[[ImageDraw.ImageDraw, int, int, int], None]


def draw_circle(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    offset_x = random.randint(-3, 3)
    offset_y = random.randint(-3, 3)
    left = margin + offset_x
    top = margin + offset_y
    right = SIZE - margin + offset_x
    bottom = SIZE - margin + offset_y
    for t in range(thickness):
        draw.ellipse((left - t, top - t, right + t, bottom + t), outline=intensity)


def draw_vertical(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    x = SIZE // 2 + random.randint(-4, 4)
    draw.line((x, margin, x, SIZE - margin), fill=intensity, width=thickness)


def draw_horizontal(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    y = SIZE // 2 + random.randint(-4, 4)
    draw.line((margin, y, SIZE - margin, y), fill=intensity, width=thickness)


def draw_diag_tl_br(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    draw.line(
        (
            margin + random.randint(-2, 2),
            margin + random.randint(-2, 2),
            SIZE - margin + random.randint(-2, 2),
            SIZE - margin + random.randint(-2, 2),
        ),
        fill=intensity,
        width=thickness,
    )


def draw_diag_tr_bl(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    draw.line(
        (
            SIZE - margin + random.randint(-2, 2),
            margin + random.randint(-2, 2),
            margin + random.randint(-2, 2),
            SIZE - margin + random.randint(-2, 2),
        ),
        fill=intensity,
        width=thickness,
    )


def draw_square(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    left = margin + random.randint(-3, 3)
    top = margin + random.randint(-3, 3)
    right = SIZE - margin + random.randint(-3, 3)
    bottom = SIZE - margin + random.randint(-3, 3)
    for t in range(thickness):
        draw.rectangle((left - t, top - t, right + t, bottom + t), outline=intensity)


def draw_triangle_up(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    top_x = SIZE // 2 + random.randint(-3, 3)
    top_y = margin + random.randint(-3, 3)
    base_left = margin + random.randint(-3, 3)
    base_right = SIZE - margin + random.randint(-3, 3)
    base_y = SIZE - margin + random.randint(-3, 3)
    points = [(top_x, top_y), (base_left, base_y), (base_right, base_y)]
    for t in range(thickness):
        draw.polygon(
            [
                (points[0][0], points[0][1] - t),
                (points[1][0] - t, points[1][1] + t),
                (points[2][0] + t, points[2][1] + t),
            ],
            outline=intensity,
        )


def draw_triangle_down(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    bottom_x = SIZE // 2 + random.randint(-3, 3)
    bottom_y = SIZE - margin + random.randint(-3, 3)
    top_left = margin + random.randint(-3, 3)
    top_right = SIZE - margin + random.randint(-3, 3)
    top_y = margin + random.randint(-3, 3)
    points = [(bottom_x, bottom_y), (top_left, top_y), (top_right, top_y)]
    for t in range(thickness):
        draw.polygon(
            [
                (points[0][0], points[0][1] + t),
                (points[1][0] - t, points[1][1] - t),
                (points[2][0] + t, points[2][1] - t),
            ],
            outline=intensity,
        )


def draw_diamond(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    mid_x = SIZE // 2 + random.randint(-3, 3)
    mid_y = SIZE // 2 + random.randint(-3, 3)
    width = SIZE - 2 * margin + random.randint(-4, 4)
    height = SIZE - 2 * margin + random.randint(-4, 4)
    points = [
        (mid_x, mid_y - height // 2),
        (mid_x + width // 2, mid_y),
        (mid_x, mid_y + height // 2),
        (mid_x - width // 2, mid_y),
    ]
    for t in range(thickness):
        draw.polygon(
            [
                (points[0][0], points[0][1] - t),
                (points[1][0] + t, points[1][1]),
                (points[2][0], points[2][1] + t),
                (points[3][0] - t, points[3][1]),
            ],
            outline=intensity,
        )


def draw_cross(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    draw_vertical(draw, intensity, margin, thickness)
    draw_horizontal(draw, intensity, margin, thickness)


def draw_zigzag(draw: ImageDraw.ImageDraw, intensity: int, margin: int, thickness: int) -> None:
    steps = random.randint(3, 5)
    step_height = (SIZE - 2 * margin) // steps
    points = []
    x_left = margin + random.randint(-2, 2)
    x_right = SIZE - margin + random.randint(-2, 2)
    for step in range(steps + 1):
        y = margin + step * step_height + random.randint(-1, 1)
        x = x_left if step % 2 == 0 else x_right
        points.append((x, y))
    draw.line(points, fill=intensity, width=thickness)


SHAPES: Dict[int, ShapeSpec] = {
    0: ShapeSpec("circle", draw_circle),
    1: ShapeSpec("vertical_line", draw_vertical),
    2: ShapeSpec("horizontal_line", draw_horizontal),
    3: ShapeSpec("diagonal_tl_br", draw_diag_tl_br),
    4: ShapeSpec("diagonal_tr_bl", draw_diag_tr_bl),
    5: ShapeSpec("square", draw_square),
    6: ShapeSpec("triangle_up", draw_triangle_up),
    7: ShapeSpec("triangle_down", draw_triangle_down),
    8: ShapeSpec("diamond", draw_diamond),
    9: ShapeSpec("cross", draw_cross),
}


MORE_SHAPES = {
    10: ShapeSpec("zigzag", draw_zigzag),
}


def add_noise(image: Image.Image, noise_sigma: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    noise = np.random.normal(0.0, noise_sigma, size=array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    background_lift = random.uniform(0, 18)
    noisy = np.clip(noisy + background_lift, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)  # uint8 2D -> režim "L"; parametr mode je v Pillow 11.3+ deprecated


def draw_shape(shape: ShapeSpec, noise_sigma: float) -> Image.Image:
    image = Image.new("L", (SIZE, SIZE), color=0)
    draw = ImageDraw.Draw(image)

    intensity = random.randint(150, 255)
    margin = random.randint(3, 12)
    thickness = random.randint(2, 5)

    shape.drawer(draw, intensity=intensity, margin=margin, thickness=thickness)

    angle = random.uniform(-12, 12)
    if abs(angle) > 1e-2:
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    return add_noise(image, noise_sigma=noise_sigma)


def ensure_dirs(base: Path, labels: Iterable[int]) -> None:
    for label in labels:
        (base / str(label)).mkdir(parents=True, exist_ok=True)


def next_index_for(target_dir: Path) -> int:
    """Find the next available index for BMP files in target_dir.

    Returns 1 if no files exist, otherwise max_number + 1.
    Expects files named like 0001.bmp, 0042.bmp, etc.
    """
    max_index = 0
    for path in target_dir.glob("*.bmp"):
        try:
            file_number = int(path.stem)
            max_index = max(max_index, file_number)
        except ValueError:
            # Ignore non-numeric filenames
            continue
    return max_index + 1


OWN_PATTERNS = ("[0-9]*/*.bmp", "train/[0-9]*/*.bmp", "val/[0-9]*/*.bmp", "test/[0-9]*/*.bmp")


def own_images(base_dir: Path) -> list[Path]:
    """BMP ve struktuře, kterou Dreamer sám vytváří (<třída>/ nebo <split>/<třída>/)."""
    if not base_dir.exists():
        return []
    return [
        item
        for pattern in OWN_PATTERNS
        for item in base_dir.glob(pattern)
        if item.is_file() and item.parent.name.isdigit()
    ]


def pixel_digest(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


def existing_digests(base_dir: Path) -> set[str]:
    """Hashe pixelů už uložených obrázků, aby je doplňování nevyrobilo znovu."""
    digests = set()
    for path in own_images(base_dir):
        try:
            with Image.open(path) as img:
                digests.add(pixel_digest(img.convert("L")))
        except OSError:
            continue  # nečitelný soubor nijak nevadí, jen se s ním neporovnává
    return digests


def run_seed(seed: int, existing_count: int) -> int:
    """Seed pro tento běh: bez dat přesně --seed, při doplňování odvozený i z počtu souborů.

    Doplňování se stejným seedem vyrobilo tutéž sekvenci obrázků znovu pod
    novými čísly a Composer pak kopie rozhodil do train i test. Odvození
    zůstává reprodukovatelné: stejný seed a stejný stav složky dají stejná data.
    """
    if existing_count == 0:
        return seed
    digest = hashlib.sha256(f"{seed}:{existing_count}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big")


def seed_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_dataset(
    base_dir: Path,
    samples_per_class: int,
    noise_sigma: float,
    split_mode: bool,
    val_split: float,
    test_split: float,
    blur_max: float,
    invert_prob: float,
    max_retry: int,
    use_extra_shapes: bool,
) -> Tuple[int, Counter[str]]:
    shapes = dict(SHAPES)
    if use_extra_shapes:
        shapes.update(MORE_SHAPES)

    labels = sorted(shapes)

    # Validate samples_per_class limit (4-digit filenames: 0001-9999)
    if samples_per_class > 9999:
        raise ValueError(
            f"samples_per_class ({samples_per_class}) exceeds maximum of 9999 "
            "(limited by 4-digit filename format for DigitComposer compatibility)"
        )

    if split_mode:
        val_ratio = max(0.0, min(1.0, val_split))
        test_ratio = max(0.0, min(1.0, test_split))
        if val_ratio + test_ratio >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")
        train_ratio = 1.0 - val_ratio - test_ratio
        splits = [
            ("train", train_ratio),
            ("val", val_ratio),
            ("test", test_ratio),
        ]
        for split_name, _ in splits:
            ensure_dirs(base_dir / split_name, labels)
    else:
        ensure_dirs(base_dir, labels)
        splits = [(None, 1.0)]

    # Při doplňování se porovnává i s tím, co už ve složce leží.
    seen_hashes: set[str] = existing_digests(base_dir)
    created_total = 0
    created_by_tag: Counter[str] = Counter()

    def maybe_augment(image: Image.Image) -> Image.Image:
        if blur_max > 1e-3:
            radius = random.uniform(0.0, blur_max)
            if radius > 1e-2:
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        if random.random() < invert_prob:
            image = ImageOps.invert(image)
        return image

    def generate_unique(spec: ShapeSpec) -> Image.Image:
        last_digest = None
        last_image = None
        for _ in range(max(1, max_retry)):
            img = draw_shape(spec, noise_sigma)
            img = maybe_augment(img)
            digest = pixel_digest(img)
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                return img
            last_digest = digest
            last_image = img
        seen_hashes.add(last_digest)  # accept duplicate as a fallback
        return last_image

    for label in labels:
        spec = shapes[label]
        if split_mode:
            # Největší zbytek: round() u train i val uměl přestřelit (3 vzorky, val 0,5 → 2+2).
            counts = allocate_counts(samples_per_class, [ratio for _, ratio in splits])
            allocations = {name: count for (name, _), count in zip(splits, counts)}
        else:
            allocations = {None: samples_per_class}

        for split_name, _ in splits:
            count = allocations.get(split_name if split_mode else None, 0)
            target_root = base_dir / split_name if split_name else base_dir
            ensure_dirs(target_root, [label])
            target_dir = target_root / str(label)
            tag = split_name or "all"
            start_index = next_index_for(target_dir)

            # Check if adding new samples would exceed the 9999 limit
            final_index = start_index + count - 1
            if final_index > 9999:
                raise ValueError(
                    f"Cannot generate {count} samples for class {label} in {target_dir}: "
                    f"would exceed 9999 limit (existing files go up to {start_index - 1}, "
                    f"new samples would reach {final_index})"
                )

            for idx in range(count):
                image = generate_unique(spec)
                filename = f"{start_index + idx:04d}.bmp"
                image.save(target_dir / filename)
                created_total += 1
                created_by_tag[tag] += 1

    return created_total, created_by_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic pseudo-digit images for DigitLearner.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SYNTHETIC_DIR,
        help="Directory where class subfolders will be created (default: ../shared/data/synthetic/).",
    )
    parser.add_argument("--samples", type=int, default=600,
                        help="Number of images per class (with --split divided into train/val/test).")
    parser.add_argument("--noise", type=float, default=18.0,
                        help="Gaussian noise sigma applied to each image.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed for reproducibility.")
    parser.add_argument("--clean", action="store_true",
                        help="If set, existing files in output_dir will be removed before generation.")
    parser.add_argument("--split", action="store_true",
                        help="Create train/val/test subdirectories using split ratios.")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation proportion when --split is enabled (default 0.2).")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Test proportion when --split is enabled (default 0.1).")
    parser.add_argument("--blur", type=float, default=1.0,
                        help="Maximum Gaussian blur radius applied randomly (0 disables).")
    parser.add_argument("--invert_prob", type=float, default=0.15,
                        help="Probability of inverting foreground/background.")
    parser.add_argument("--max_retry", type=int, default=5,
                        help="How many attempts to keep a generated image unique before accepting duplicates.")
    parser.add_argument("--extra_shapes", action="store_true",
                        help="Add a few additional shape classes beyond the base 0-9.")
    return parser.parse_args()


def prepare_output_dir(base_dir: Path, clean: bool) -> None:
    if base_dir.exists() and clean:
        # Mažu jen obrázky ve struktuře, kterou Dreamer sám vytváří (<třída>/ nebo
        # <split>/<třída>/). --output_dir může omylem ukazovat výš (např. na
        # shared/data) a data jiných nástrojů ani README.md nesmí zmizet.
        for item in own_images(base_dir):
            item.unlink()
        for sub in sorted((p for p in base_dir.rglob("*") if p.is_dir()), reverse=True):
            if not any(sub.iterdir()):
                sub.rmdir()
    base_dir.mkdir(parents=True, exist_ok=True)


def has_existing_data(base_dir: Path) -> bool:
    if not base_dir.exists():
        return False
    return next(base_dir.rglob("*.bmp"), None) is not None


def resolve_existing_output(base_dir: Path, clean_requested: bool) -> str:
    """
    Decide how to handle existing data.

    Returns:
        "replace" to wipe and regenerate,
        "append" to keep and add new samples,
        "skip" to abort generation.
    """
    if clean_requested:
        return "replace"
    if not has_existing_data(base_dir):
        return "append"

    prompt = (
        f"Ve slozce '{base_dir}' jiz existuji vygenerovana data.\n"
        "Zvolte akci: [P]repsat / [Z]achovat / [D]oplnit: "
    )
    while True:
        choice = input(prompt).strip().lower()
        if choice in ("p", "prepsat", "overwrite", "replace", "y"):
            return "replace"
        if choice in ("z", "zachovat", "skip", "n", "ne"):
            return "skip"
        if choice in ("d", "doplnit", "append", "a"):
            return "append"
        print("Neplatna volba. Zadejte P, Z nebo D.")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    action = resolve_existing_output(output_dir, args.clean)
    if action == "skip":
        print(f"Data ve slozce {output_dir} zustala zachovana. Generovani neprobehlo.")
        return

    clean_flag = action == "replace"

    prepare_output_dir(output_dir, clean=clean_flag)

    existing_count = len(own_images(output_dir))
    seed = run_seed(args.seed, existing_count)
    if seed != args.seed:
        print(f"Doplnuji k {existing_count} existujicim obrazkum: seed {seed} "
              f"(odvozeny z --seed {args.seed} a poctu souboru, aby se obrazky neopakovaly).")
    seed_generators(seed)

    generated_total, generated_breakdown = generate_dataset(
        base_dir=output_dir,
        samples_per_class=args.samples,
        noise_sigma=args.noise,
        split_mode=args.split,
        val_split=args.val_split,
        test_split=args.test_split,
        blur_max=max(0.0, args.blur),
        invert_prob=max(0.0, min(1.0, args.invert_prob)),
        max_retry=max(1, args.max_retry),
        use_extra_shapes=args.extra_shapes,
    )

    if generated_total:
        print(f"Hotovo: vygenerovano {generated_total} obrazku v {output_dir}")
        if generated_breakdown:
            print("Souhrn podle slozek:")
            for tag, count in sorted(generated_breakdown.items()):
                label = "celkem" if tag == "all" else tag
                print(f"  {label}: {count}")
    else:
        print(f"Nebyly vytvoreny zadne nove soubory v {output_dir}")


if __name__ == "__main__":
    main()

