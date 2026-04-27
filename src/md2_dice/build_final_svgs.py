from __future__ import annotations

import argparse
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "assets" / "svg"
FINAL_DEBUG = FINAL / "_debug"
SOURCE = ROOT / "source"
REGENERATED_SOURCE_DEBUG = FINAL_DEBUG / "regenerated_from_source"

# Fusion 360 measurement from the usable flat circular face:
# area = 160.221 mm^2, circumference = 44.871 mm, diameter ~= 14.28 mm.
USABLE_CIRCLE_AREA_MM2 = 160.221
USABLE_CIRCLE_RADIUS_MM = math.sqrt(USABLE_CIRCLE_AREA_MM2 / math.pi)
SVG_CENTER = (8.0, 8.0)


@dataclass(frozen=True)
class FinalSvg:
    path: Path
    description: str


@dataclass(frozen=True)
class SourceSvgRecipe:
    path: Path
    source: Path
    title: str
    description: str
    foreground: str
    threshold: int
    target_bounds: tuple[float, float, float, float]
    simplify_pixels: float = 0.8


@dataclass(frozen=True)
class ReferenceSource:
    path: Path
    source: Path
    note: str


FINAL_SVGS = [
    FinalSvg(Path("defense_1_shield.svg"), "one shield"),
    FinalSvg(Path("defense_2_shields.svg"), "two shields"),
    FinalSvg(Path("defense_3_shields.svg"), "three shields"),
    FinalSvg(Path("monster_3_claws.svg"), "three red claws"),
    FinalSvg(Path("monster_paw.svg"), "red paw"),
    FinalSvg(Path("monster_paw_and_claws.svg"), "red paw and claws"),
    FinalSvg(Path("special_face.svg"), "face/mask symbol"),
    FinalSvg(Path("attack_1_sword_1_damage.svg"), "one sword and one damage"),
    FinalSvg(Path("attack_2_swords_1_damage.svg"), "two swords and one damage"),
    FinalSvg(Path("attack_1_sword.svg"), "one centered sword"),
    FinalSvg(Path("attack_2_swords.svg"), "two opposed swords"),
    FinalSvg(Path("attack_3_swords.svg"), "three swords"),
    FinalSvg(Path("attack_4_swords.svg"), "four swords"),
    FinalSvg(Path("magic_1_damage.svg"), "one damage"),
    FinalSvg(Path("magic_2_damage.svg"), "two damages"),
    FinalSvg(Path("elements/element_sword.svg"), "canonical sword"),
    FinalSvg(Path("elements/element_damage.svg"), "canonical damage"),
    FinalSvg(Path("elements/element_shield.svg"), "canonical shield"),
    FinalSvg(Path("elements/element_paw.svg"), "canonical paw"),
    FinalSvg(Path("elements/element_claws.svg"), "canonical claws"),
    FinalSvg(Path("elements/element_face.svg"), "canonical face"),
]

TRACEABLE_SOURCE_SVGS = [
    SourceSvgRecipe(
        Path("special_face.svg"),
        Path("source_special_face.png"),
        "visage_source_trace",
        "face/mask symbol traced from the clean source crop",
        "dark",
        128,
        (4.62, 3.61, 11.34, 12.35),
    ),
    SourceSvgRecipe(
        Path("elements/element_face.svg"),
        Path("source_special_face.png"),
        "visage_source_trace",
        "canonical face traced from the clean source crop",
        "dark",
        128,
        (4.62, 3.61, 11.34, 12.35),
    ),
    SourceSvgRecipe(
        Path("magic_2_damage.svg"),
        Path("source_magic_2_damage.png"),
        "magic_2_damage_source_trace",
        "two damage symbols traced from the clean source crop",
        "dark",
        128,
        (1.8, 1.8, 14.2, 14.2),
    ),
]

REFERENCE_SOURCES = [
    ReferenceSource(Path("attack_1_sword_1_damage.svg"), Path("source_attack_1_sword_1_damage.png"), "curated from noisy reference crop"),
    ReferenceSource(Path("attack_2_swords_1_damage.svg"), Path("source_attack_2_swords_1_damage.png"), "curated from noisy reference crop"),
    ReferenceSource(Path("defense_1_shield.svg"), Path("source_defense_1_shield.png"), "curated from noisy reference crop"),
    ReferenceSource(Path("defense_2_shields.svg"), Path("source_defense_2_shields.png"), "curated from noisy reference crop"),
    ReferenceSource(Path("defense_3_shields.svg"), Path("source_defense_3_shields.png"), "curated from noisy reference crop"),
    ReferenceSource(Path("monster_3_claws.svg"), Path("source_monster_3_claws.png"), "curated from noisy red-on-black reference crop"),
    ReferenceSource(Path("monster_paw.svg"), Path("source_monster_paw.png"), "curated from noisy red-on-black reference crop"),
    ReferenceSource(Path("monster_paw_and_claws.svg"), Path("source_monster_paw_and_claws.png"), "curated from noisy red-on-black reference crop"),
]


def path_points(path_data: str) -> list[tuple[float, float]]:
    values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path_data)]
    return list(zip(values[0::2], values[1::2]))


def transform_points(points: list[tuple[float, float]], transform: str | None) -> list[tuple[float, float]]:
    if not transform:
        return points

    match = re.search(
        r"translate\(([-\d.]+) ([-\d.]+)\) rotate\(([-\d.]+)\) scale\(([-\d.]+)\) translate\(-8 -8\)",
        transform,
    )
    if not match:
        return points

    tx, ty, angle_degrees, scale = (float(value) for value in match.groups())
    angle = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    transformed: list[tuple[float, float]] = []
    for x, y in points:
        x = (x - 8.0) * scale
        y = (y - 8.0) * scale
        transformed.append((x * cos_a - y * sin_a + tx, x * sin_a + y * cos_a + ty))
    return transformed


def svg_points(svg_path: Path) -> list[tuple[float, float]]:
    text = svg_path.read_text(encoding="utf-8")
    points: list[tuple[float, float]] = []
    grouped_pattern = re.compile(r'<g transform="([^"]+)">\s*<path d="([^"]+)"', re.S)
    grouped_spans: list[tuple[int, int]] = []
    for match in grouped_pattern.finditer(text):
        grouped_spans.append(match.span())
        points.extend(transform_points(path_points(match.group(2)), match.group(1)))

    for match in re.finditer(r'<path d="([^"]+)"', text):
        if any(start <= match.start() <= end for start, end in grouped_spans):
            continue
        points.extend(path_points(match.group(1)))
    return points


def fit_metrics(svg_path: Path) -> dict[str, float | bool]:
    points = svg_points(svg_path)
    if not points:
        return {
            "width": 0.0,
            "height": 0.0,
            "max_radius": 0.0,
            "fits_16mm_square": True,
            "fits_usable_circle": True,
        }

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    max_radius = max(math.hypot(x - SVG_CENTER[0], y - SVG_CENTER[1]) for x, y in points)
    return {
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
        "max_radius": max_radius,
        "fits_16mm_square": min(xs) >= 0 and max(xs) <= 16 and min(ys) >= 0 and max(ys) <= 16,
        "fits_usable_circle": max_radius <= USABLE_CIRCLE_RADIUS_MM,
    }


def validate_svg_header(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    if 'width="16mm"' not in text or 'height="16mm"' not in text or 'viewBox="0 0 16 16"' not in text:
        raise RuntimeError(f"{svg_path.relative_to(ROOT)} is not a 16mm square SVG")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def source_path(path: Path) -> Path:
    return SOURCE / path


def ensure_sources_exist() -> None:
    missing = [
        relative(source_path(recipe.source))
        for recipe in TRACEABLE_SOURCE_SVGS
        if not source_path(recipe.source).exists()
    ]
    missing.extend(
        relative(source_path(reference.source))
        for reference in REFERENCE_SOURCES
        if not source_path(reference.source).exists()
    )
    if missing:
        raise RuntimeError("Missing source images:\n" + "\n".join(f"- {path}" for path in sorted(set(missing))))


def source_mask(recipe: SourceSvgRecipe) -> np.ndarray:
    image = Image.open(source_path(recipe.source)).convert("RGBA")
    pixels = np.asarray(image)
    rgb = pixels[:, :, :3].astype(np.float32)
    alpha = pixels[:, :, 3] > 0
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    if recipe.foreground == "dark":
        mask = (luminance < recipe.threshold) & alpha
    elif recipe.foreground == "bright":
        mask = (luminance > recipe.threshold) & alpha
    else:
        raise RuntimeError(f"Unsupported foreground mode: {recipe.foreground}")
    return mask


def mask_geometry(mask: np.ndarray, simplify_pixels: float) -> Polygon | MultiPolygon:
    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0:
        raise RuntimeError("Source image produced an empty mask")

    pixels = [box(float(x), float(y), float(x + 1), float(y + 1)) for x, y in zip(x_indices, y_indices)]
    geometry = unary_union(pixels)
    if simplify_pixels:
        geometry = geometry.simplify(simplify_pixels, preserve_topology=True)
    if geometry.is_empty:
        raise RuntimeError("Source image produced empty vector geometry")
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return geometry
    polygons = [part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon)]
    if not polygons:
        raise RuntimeError("Source image did not produce polygon geometry")
    return MultiPolygon(polygons)


def fit_geometry_to_bounds(
    geometry: Polygon | MultiPolygon,
    target_bounds: tuple[float, float, float, float],
) -> Polygon | MultiPolygon:
    min_x, min_y, max_x, max_y = geometry.bounds
    target_min_x, target_min_y, target_max_x, target_max_y = target_bounds
    source_width = max_x - min_x
    source_height = max_y - min_y
    target_width = target_max_x - target_min_x
    target_height = target_max_y - target_min_y
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("Source geometry has no area")

    scale = min(target_width / source_width, target_height / source_height)
    geometry = affinity.translate(geometry, xoff=-min_x, yoff=-min_y)
    geometry = affinity.scale(geometry, xfact=scale, yfact=scale, origin=(0, 0))
    fitted_width = source_width * scale
    fitted_height = source_height * scale
    return affinity.translate(
        geometry,
        xoff=target_min_x + (target_width - fitted_width) / 2,
        yoff=target_min_y + (target_height - fitted_height) / 2,
    )


def format_number(value: float) -> str:
    rounded = round(value + 0.0, 3)
    text = f"{rounded:.3f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def ring_path(coords) -> str:
    points = [(float(x), float(y)) for x, y in coords]
    if points and points[0] == points[-1]:
        points = points[:-1]
    if not points:
        return ""
    first_x, first_y = points[0]
    parts = [f"M {format_number(first_x)} {format_number(first_y)}"]
    parts.extend(f"L {format_number(x)} {format_number(y)}" for x, y in points[1:])
    parts.append("Z")
    return " ".join(parts)


def polygon_path(polygon: Polygon) -> str:
    parts = [ring_path(polygon.exterior.coords)]
    parts.extend(ring_path(interior.coords) for interior in polygon.interiors)
    return " ".join(part for part in parts if part)


def geometry_path_data(geometry: Polygon | MultiPolygon) -> str:
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    polygons = sorted(polygons, key=lambda polygon: (-polygon.area, polygon.bounds))
    return " ".join(polygon_path(polygon) for polygon in polygons if polygon.area > 0.001)


def traced_svg_text(recipe: SourceSvgRecipe) -> str:
    geometry = mask_geometry(source_mask(recipe), recipe.simplify_pixels)
    geometry = fit_geometry_to_bounds(geometry, recipe.target_bounds)
    path_data = geometry_path_data(geometry)
    if not path_data:
        raise RuntimeError(f"{relative(source_path(recipe.source))} produced no path data")
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="16mm" height="16mm" viewBox="0 0 16 16">',
            f"  <title>{recipe.title}</title>",
            f"  <desc>Regenerated from {relative(source_path(recipe.source))}: {recipe.description}.</desc>",
            '  <g fill="#000000" stroke="none" fill-rule="evenodd">',
            f'    <path d="{path_data}"/>',
            "  </g>",
            "</svg>",
            "",
        ]
    )


def write_regeneration_report(replaced: bool) -> None:
    report = [
        "# SVG Regeneration",
        "",
        "This report documents which source images can be traced automatically and which final SVGs remain curated.",
        "",
        "| output | source | action |",
        "| --- | --- | --- |",
    ]
    action = "traced into assets/svg" if replaced else "traced into debug candidate; curated asset preserved"
    for recipe in TRACEABLE_SOURCE_SVGS:
        report.append(f"| `{recipe.path.as_posix()}` | `{(Path('source') / recipe.source).as_posix()}` | {action} |")
    for reference in REFERENCE_SOURCES:
        report.append(
            f"| `{reference.path.as_posix()}` | `{(Path('source') / reference.source).as_posix()}` | "
            f"{reference.note}; curated asset preserved |"
        )
    report.extend(
        [
            "",
            "Composed symbols such as sword counts, single damage, and canonical element SVGs without clean source crops remain curated SVG assets.",
            "",
        ]
    )
    (FINAL_DEBUG / "regeneration.md").write_text("\n".join(report), encoding="utf-8")


def regenerate_from_sources(*, replace_from_source: bool = False) -> None:
    ensure_sources_exist()
    if REGENERATED_SOURCE_DEBUG.exists():
        shutil.rmtree(REGENERATED_SOURCE_DEBUG)
    REGENERATED_SOURCE_DEBUG.mkdir(parents=True, exist_ok=True)

    for recipe in TRACEABLE_SOURCE_SVGS:
        text = traced_svg_text(recipe)
        debug_path = REGENERATED_SOURCE_DEBUG / recipe.path
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(text, encoding="utf-8")
        validate_svg_header(debug_path)

        if replace_from_source:
            final_path = FINAL / recipe.path
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text(text, encoding="utf-8")

    write_regeneration_report(replace_from_source)


def write_reports() -> None:
    FINAL_DEBUG.mkdir(parents=True, exist_ok=True)
    manifest = [
        "# Final SVG Export",
        "",
        f"Usable Fusion 360 face circle radius: `{USABLE_CIRCLE_RADIUS_MM:.3f} mm`.",
        "All SVGs are `16mm x 16mm` with `viewBox=\"0 0 16 16\"`.",
        "",
        "| file | description | size mm | max radius mm | fits circle |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    fit_report = [
        "# Fit Check",
        "",
        f"- useful circle area: `{USABLE_CIRCLE_AREA_MM2} mm^2`",
        f"- useful circle radius: `{USABLE_CIRCLE_RADIUS_MM:.3f} mm`",
        f"- useful circle diameter: `{USABLE_CIRCLE_RADIUS_MM * 2:.3f} mm`",
        "",
    ]

    failures: list[str] = []
    for svg in FINAL_SVGS:
        svg_path = FINAL / svg.path
        if not svg_path.exists():
            raise RuntimeError(f"Missing final SVG: {svg_path.relative_to(ROOT)}")
        validate_svg_header(svg_path)
        metrics = fit_metrics(svg_path)
        fits = "yes" if metrics["fits_usable_circle"] else "NO"
        manifest.append(
            f"| `{svg.path}` | {svg.description} | "
            f"{metrics['width']:.2f} x {metrics['height']:.2f} | {metrics['max_radius']:.2f} | {fits} |"
        )
        if not metrics["fits_usable_circle"]:
            failures.append(
                f"- `{svg.path}` exceeds useful circle: radius `{metrics['max_radius']:.3f} mm`, "
                f"bbox `{metrics['width']:.3f} x {metrics['height']:.3f} mm`"
            )

    fit_report.extend(failures or ["- all exported SVGs fit the useful circle"])
    (FINAL / "MANIFEST.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (FINAL_DEBUG / "fit_check.md").write_text("\n".join(fit_report) + "\n", encoding="utf-8")

    if failures:
        raise RuntimeError("Some final SVGs do not fit the useful circle")


def validate() -> None:
    write_reports()


def regenerate(*, replace_from_source: bool = False) -> None:
    regenerate_from_sources(replace_from_source=replace_from_source)
    write_reports()


def main() -> None:
    validate()


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate or regenerate SVG assets.")
    parser.add_argument(
        "--replace-from-source",
        action="store_true",
        help="Overwrite automatically traceable final SVGs with source-image traces.",
    )
    args = parser.parse_args(argv)
    regenerate(replace_from_source=args.replace_from_source)


if __name__ == "__main__":
    cli_main()
