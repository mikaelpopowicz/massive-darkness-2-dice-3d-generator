from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "build" / "3mf" / "individual"
OUTPUT_DIR = ROOT / "docs" / "previews"

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
MAT_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"

DIE_LABELS = {
    "black": "Enemy die x6",
    "blue": "Blue Defense die x5",
    "green": "Green Defense die x5",
    "orange": "Orange Attack die x3",
    "yellow": "Yellow Attack die x3",
    "red": "Red Attack die x3",
    "purple": "Shadow die x1",
}

PREVIEW_FILENAMES = {
    "black": "enemy_die_x6",
    "blue": "blue_defense_die_x5",
    "green": "green_defense_die_x5",
    "orange": "orange_attack_die_x3",
    "yellow": "yellow_attack_die_x3",
    "red": "red_attack_die_x3",
    "purple": "shadow_die_x1",
}


def normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(left * right for left, right in zip(a, b))


def color_from_hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def shade(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * amount))) for channel in color)  # type: ignore[return-value]


def load_objects(path: Path) -> list[tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], tuple[int, int, int]]]:
    ns = {"c": CORE_NS, "m": MAT_NS}
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))

    colors = [color_from_hex(color.attrib["color"]) for color in root.findall(".//m:color", ns)]
    objects = []
    for obj in root.findall(".//c:object", ns):
        vertices = [
            (float(vertex.attrib["x"]), float(vertex.attrib["y"]), float(vertex.attrib["z"]))
            for vertex in obj.findall(".//c:vertex", ns)
        ]
        triangles = [
            (int(triangle.attrib["v1"]), int(triangle.attrib["v2"]), int(triangle.attrib["v3"]))
            for triangle in obj.findall(".//c:triangle", ns)
        ]
        material_index = int(obj.attrib.get("pindex", "0"))
        objects.append((vertices, triangles, colors[material_index]))
    return objects


def project(
    point: tuple[float, float, float],
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    view: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (dot(point, right), dot(point, up), dot(point, view))


def symbol_cap_axis(vertices: list[tuple[float, float, float]]) -> int:
    extents = []
    for axis in range(3):
        values = [vertex[axis] for vertex in vertices]
        extents.append(max(values) - min(values))
    return min(range(3), key=lambda axis: extents[axis])


def visible_symbol_cap_value(
    vertices: list[tuple[float, float, float]],
    cap_axis: int,
    view: tuple[float, float, float],
) -> float:
    values = [vertex[cap_axis] for vertex in vertices]
    return max(values) if view[cap_axis] >= 0 else min(values)


PREVIEW_VIEWS = {
    "view_a": {"vector": (1.4, -1.9, 1.2), "flip_up": False, "light": (0.4, -0.6, 1.0)},
    # Opposite yaw plus underside view. The screen-up vector is flipped so the
    # bottom face appears at the top of the preview, mirroring the top face in
    # view A instead of looking visually upside down.
    "view_b": {"vector": (-1.4, 1.9, -1.2), "flip_up": True, "light": (-0.4, 0.6, -1.0)},
}


def render_die(
    path: Path,
    output: Path,
    *,
    view_vector: tuple[float, float, float],
    light_vector: tuple[float, float, float],
    flip_up: bool = False,
    size: int = 2000,
) -> None:
    objects = load_objects(path)
    view = normalize(view_vector)
    right = normalize(cross((0.0, 0.0, 1.0), view))
    up = normalize(cross(view, right))
    if flip_up:
        up = (-up[0], -up[1], -up[2])
    light = normalize(light_vector)

    projected_points = []
    triangles_to_draw = []
    for object_index, (vertices, triangles, color) in enumerate(objects):
        projected = [project(vertex, right, up, view) for vertex in vertices]
        projected_points.extend(projected)
        cap_axis = symbol_cap_axis(vertices) if object_index > 0 else None
        cap_value = visible_symbol_cap_value(vertices, cap_axis, view) if cap_axis is not None else None
        for v1, v2, v3 in triangles:
            a, b, c = vertices[v1], vertices[v2], vertices[v3]
            normal = normalize(cross((b[0] - a[0], b[1] - a[1], b[2] - a[2]), (c[0] - a[0], c[1] - a[1], c[2] - a[2])))
            if cap_axis is not None and abs(normal[cap_axis]) < 0.85:
                continue
            if cap_value is not None:
                centroid = (a[cap_axis] + b[cap_axis] + c[cap_axis]) / 3.0
                if abs(centroid - cap_value) > 0.05:
                    continue
            elif dot(normal, view) <= -0.05:
                continue
            # Symbols are rendered flat to keep README previews readable.
            # Draw only the camera-facing symbol cap; extrusion side walls make
            # the software preview look jagged even though the 3MF is fine.
            lighting = 1.0 if object_index > 0 else 0.58 + 0.42 * max(0.0, dot(normal, light))
            depth = (projected[v1][2] + projected[v2][2] + projected[v3][2]) / 3.0
            symbol_bias = 0.02 if object_index > 0 else 0.0
            triangles_to_draw.append((depth + symbol_bias, [projected[v1], projected[v2], projected[v3]], shade(color, lighting)))

    min_x = min(point[0] for point in projected_points)
    max_x = max(point[0] for point in projected_points)
    min_y = min(point[1] for point in projected_points)
    max_y = max(point[1] for point in projected_points)
    scale = (size * 0.72) / max(max_x - min_x, max_y - min_y)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    def to_screen(point: tuple[float, float, float]) -> tuple[float, float]:
        return (size / 2 + (point[0] - center_x) * scale, size / 2 - (point[1] - center_y) * scale)

    color_buffer = np.full((size, size, 3), (244, 246, 248), dtype=np.uint8)
    depth_buffer = np.full((size, size), -np.inf, dtype=np.float32)

    for _depth, triangle, color in triangles_to_draw:
        screen = [to_screen(point) for point in triangle]
        xs = [point[0] for point in screen]
        ys = [point[1] for point in screen]
        min_px = max(0, int(math.floor(min(xs))))
        max_px = min(size - 1, int(math.ceil(max(xs))))
        min_py = max(0, int(math.floor(min(ys))))
        max_py = min(size - 1, int(math.ceil(max(ys))))
        if min_px > max_px or min_py > max_py:
            continue

        x1, y1 = screen[0]
        x2, y2 = screen[1]
        x3, y3 = screen[2]
        z1, z2, z3 = triangle[0][2], triangle[1][2], triangle[2][2]
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-9:
            continue

        yy, xx = np.mgrid[min_py : max_py + 1, min_px : max_px + 1]
        px = xx + 0.5
        py = yy + 0.5
        w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
        w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
        w3 = 1.0 - w1 - w2
        inside = (w1 >= -1e-6) & (w2 >= -1e-6) & (w3 >= -1e-6)
        if not inside.any():
            continue

        z = w1 * z1 + w2 * z2 + w3 * z3
        current = depth_buffer[min_py : max_py + 1, min_px : max_px + 1]
        visible = inside & (z > current)
        if not visible.any():
            continue
        current[visible] = z[visible]
        color_buffer[min_py : max_py + 1, min_px : max_px + 1][visible] = color

    image = Image.fromarray(color_buffer, mode="RGB")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def create_die_sheet(die_name: str, view_paths: list[Path]) -> Path:
    images = [Image.open(path).resize((760, 760), Image.Resampling.LANCZOS) for path in view_paths]
    sheet = Image.new("RGB", (1560, 880), "#f4f6f8")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        x = 20 + index * 780
        sheet.paste(image, (x, 20))
        draw.text((x + 12, 808), f"{DIE_LABELS[die_name]} - view {index + 1}", fill="#222222")
    output = OUTPUT_DIR / f"{PREVIEW_FILENAMES[die_name]}_preview.png"
    sheet.save(output)
    return output


def create_contact_sheet(previews: list[Path]) -> None:
    images = [Image.open(path).resize((720, 406), Image.Resampling.LANCZOS) for path in previews]
    labels = []
    for path in previews:
        slug = path.stem.replace("_preview", "")
        die_name = next(name for name, filename in PREVIEW_FILENAMES.items() if filename == slug)
        labels.append(DIE_LABELS[die_name])
    width, height = 2 * 800, 4 * 480
    sheet = Image.new("RGB", (width, height), "#f4f6f8")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % 2) * 800 + 40
        y = (index // 2) * 480 + 30
        sheet.paste(image, (x, y))
        draw.text((x + 12, y + 420), label, fill="#222222")
    sheet.save(OUTPUT_DIR / "all_dice_preview.png")


def main() -> None:
    preview_paths = []
    for model in sorted(INPUT_DIR.glob("*_die.3mf")):
        die_name = model.stem.replace("_die", "")
        view_paths = []
        for view_name, view_config in PREVIEW_VIEWS.items():
            output = OUTPUT_DIR / f"{PREVIEW_FILENAMES[die_name]}_{view_name}.png"
            render_die(
                model,
                output,
                view_vector=view_config["vector"],
                light_vector=view_config["light"],
                flip_up=view_config["flip_up"],
            )
            view_paths.append(output)
        preview_paths.append(create_die_sheet(die_name, view_paths))
    create_contact_sheet(preview_paths)


if __name__ == "__main__":
    main()
