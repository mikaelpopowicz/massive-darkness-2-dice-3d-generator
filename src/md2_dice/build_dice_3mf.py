from __future__ import annotations

import math
import re
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import mapbox_earcut as earcut
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
import trimesh


ROOT = Path(__file__).resolve().parents[2]
BASE_3MF = ROOT / "source" / "base_die.3mf"
SVG_DIR = ROOT / "assets" / "svg"
BUILD_3MF_OUT = ROOT / "build" / "3mf"

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
MAT_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
PROD_NS = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"

ET.register_namespace("", CORE_NS)
ET.register_namespace("m", MAT_NS)
ET.register_namespace("p", PROD_NS)

SYMBOL_DEPTH_MM = 0.5
RAISED_DEPTH_MM = 0.5
SLIGHT_RELIEF_MM = 0.08
PRINT_OPTIMIZED_DEPTH_MM = 0.6
PRINT_OPTIMIZED_RELIEF_MM = 0.2
PRINT_OPTIMIZED_BUFFER_MM = 0.04
NOZZLE_04_BLACK_SYMBOL_BUFFER_MM = -0.03
INSERT_DEPTH_MM = 0.45
INSERT_PLANAR_SCALE = 0.985


@dataclass(frozen=True)
class Face:
    name: str
    normal: tuple[float, float, float]
    axis_u: tuple[float, float, float]
    axis_v: tuple[float, float, float]


@dataclass(frozen=True)
class DieSpec:
    name: str
    base_color: str
    symbol_color: str
    faces: list[str | None]


FACES = [
    Face("top", (0, 0, 1), (1, 0, 0), (0, 1, 0)),
    Face("front", (0, -1, 0), (1, 0, 0), (0, 0, 1)),
    Face("right", (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    Face("back", (0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    Face("left", (-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    Face("bottom", (0, 0, -1), (1, 0, 0), (0, -1, 0)),
]


DICE = [
    DieSpec(
        "yellow",
        "#F7C900FF",
        "#F6F6F3FF",
        [None, "attack_1_sword.svg", "attack_1_sword.svg", "attack_1_sword_1_damage.svg", "attack_1_sword_1_damage.svg", "attack_2_swords.svg"],
    ),
    DieSpec(
        "orange",
        "#F36C16FF",
        "#F6F6F3FF",
        [None, "attack_1_sword.svg", "attack_1_sword_1_damage.svg", "attack_2_swords.svg", "attack_2_swords.svg", "attack_3_swords.svg"],
    ),
    DieSpec(
        "red",
        "#B70F2BFF",
        "#F6F6F3FF",
        ["attack_1_sword.svg", "attack_1_sword.svg", "attack_1_sword_1_damage.svg", "attack_2_swords_1_damage.svg", "attack_3_swords.svg", "attack_4_swords.svg"],
    ),
    DieSpec(
        "blue",
        "#1266C9FF",
        "#F6F6F3FF",
        [None, None, "defense_1_shield.svg", "defense_1_shield.svg", "defense_1_shield.svg", "defense_2_shields.svg"],
    ),
    DieSpec(
        "green",
        "#109A20FF",
        "#F6F6F3FF",
        [None, "defense_1_shield.svg", "defense_1_shield.svg", "defense_1_shield.svg", "defense_2_shields.svg", "defense_3_shields.svg"],
    ),
    DieSpec(
        "purple",
        "#5B168EFF",
        "#F6F6F3FF",
        ["magic_1_damage.svg", "magic_2_damage.svg", "attack_1_sword.svg", "attack_2_swords.svg", "attack_3_swords.svg", "special_face.svg"],
    ),
    DieSpec(
        "black",
        "#1C1C1CFF",
        "#D93A34FF",
        [None, None, "monster_3_claws.svg", "monster_paw.svg", "monster_paw.svg", "monster_paw_and_claws.svg"],
    ),
]


def vector_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vector_scale(a: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def read_base_mesh() -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    with ZipFile(BASE_3MF) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
    ns = {"c": CORE_NS}
    vertices = [
        (float(vertex.attrib["x"]), float(vertex.attrib["y"]), float(vertex.attrib["z"]))
        for vertex in root.findall(".//c:vertex", ns)
    ]
    triangles = [
        (int(triangle.attrib["v1"]), int(triangle.attrib["v2"]), int(triangle.attrib["v3"]))
        for triangle in root.findall(".//c:triangle", ns)
    ]
    return vertices, triangles


def tokenize_path(path_data: str) -> list[str]:
    return re.findall(r"[MLQZmlqz]|-?\d+(?:\.\d+)?", path_data)


def parse_path(path_data: str) -> list[list[tuple[float, float]]]:
    tokens = tokenize_path(path_data)
    contours: list[list[tuple[float, float]]] = []
    current: tuple[float, float] | None = None
    contour: list[tuple[float, float]] = []
    index = 0
    command = ""

    def next_number() -> float:
        nonlocal index
        value = float(tokens[index])
        index += 1
        return value

    while index < len(tokens):
        if re.match(r"[MLQZmlqz]", tokens[index]):
            command = tokens[index].upper()
            index += 1

        if command == "M":
            if contour:
                contours.append(contour)
            current = (next_number(), next_number())
            contour = [current]
            command = "L"
        elif command == "L":
            current = (next_number(), next_number())
            contour.append(current)
        elif command == "Q":
            if current is None:
                raise RuntimeError("Q command before current point")
            control = (next_number(), next_number())
            end = (next_number(), next_number())
            start = current
            for step in range(1, 11):
                t = step / 10.0
                x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
                y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]
                contour.append((x, y))
            current = end
        elif command == "Z":
            if contour:
                contours.append(contour)
                contour = []
            current = None
            command = ""
        else:
            raise RuntimeError(f"Unsupported SVG command: {command}")

    if contour:
        contours.append(contour)
    return contours


def transform_2d(points: list[tuple[float, float]], transform: str | None) -> list[tuple[float, float]]:
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


def svg_contours(svg_path: Path) -> list[list[tuple[float, float]]]:
    text = svg_path.read_text(encoding="utf-8")
    contours: list[list[tuple[float, float]]] = []
    grouped_pattern = re.compile(r'<g transform="([^"]+)">\s*<path d="([^"]+)"', re.S)
    grouped_spans: list[tuple[int, int]] = []
    for match in grouped_pattern.finditer(text):
        grouped_spans.append(match.span())
        contours.extend(transform_2d(contour, match.group(1)) for contour in parse_path(match.group(2)))
    for match in re.finditer(r'<path d="([^"]+)"', text):
        if any(start <= match.start() <= end for start, end in grouped_spans):
            continue
        contours.extend(parse_path(match.group(1)))
    return [contour for contour in contours if len(contour) >= 3]


def polygon_from_contours(contours: list[list[tuple[float, float]]]) -> Polygon | MultiPolygon:
    geometry = None
    for contour in contours:
        polygon = Polygon(contour)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.001:
            continue
        geometry = polygon if geometry is None else geometry.symmetric_difference(polygon)
    if geometry is None:
        raise RuntimeError("SVG produced no polygon geometry")
    return geometry


def scale_contours_about_center(
    contours: list[list[tuple[float, float]]],
    scale: float,
) -> list[list[tuple[float, float]]]:
    return [[(8.0 + (x - 8.0) * scale, 8.0 + (y - 8.0) * scale) for x, y in contour] for contour in contours]


def polygon_parts(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if polygon.area > 0.001]
    return [geometry]


def ring_without_duplicate_close(ring) -> list[tuple[float, float]]:
    coords = [(float(x), float(y)) for x, y in ring.coords]
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


def triangulate_polygon_2d(polygon: Polygon) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    polygon = orient(polygon, sign=1.0)
    rings = [ring_without_duplicate_close(polygon.exterior)]
    rings.extend(ring_without_duplicate_close(interior) for interior in polygon.interiors)
    vertices_2d = [point for ring in rings for point in ring]
    ring_ends = []
    total = 0
    for ring in rings:
        total += len(ring)
        ring_ends.append(total)

    if len(vertices_2d) < 3:
        return []

    indices = earcut.triangulate_float64(
        np.array(vertices_2d, dtype=np.float64),
        np.array(ring_ends, dtype=np.uint32),
    )
    return [
        (vertices_2d[int(indices[i])], vertices_2d[int(indices[i + 1])], vertices_2d[int(indices[i + 2])])
        for i in range(0, len(indices), 3)
    ]


def point_on_face(face: Face, x: float, y: float, depth: float) -> tuple[float, float, float]:
    # SVG coordinates are 0..16 with y downward. Face-local coordinates are
    # centered and y-up, so y is inverted here.
    u = x - 8.0
    v = 8.0 - y
    base = vector_scale(face.normal, 8.0 - depth)
    return vector_add(vector_add(base, vector_scale(face.axis_u, u)), vector_scale(face.axis_v, v))


def point_outside_face(face: Face, x: float, y: float, depth: float) -> tuple[float, float, float]:
    u = x - 8.0
    v = 8.0 - y
    base = vector_scale(face.normal, 8.0 + depth)
    return vector_add(vector_add(base, vector_scale(face.axis_u, u)), vector_scale(face.axis_v, v))


def point_at_face_offset(face: Face, x: float, y: float, outward_offset: float) -> tuple[float, float, float]:
    u = x - 8.0
    v = 8.0 - y
    base = vector_scale(face.normal, 8.0 + outward_offset)
    return vector_add(vector_add(base, vector_scale(face.axis_u, u)), vector_scale(face.axis_v, v))


def add_vertex(vertices: list[tuple[float, float, float]], vertex: tuple[float, float, float]) -> int:
    vertices.append(vertex)
    return len(vertices) - 1


def extrude_polygon_to_face(
    geometry: Polygon | MultiPolygon,
    face: Face,
    depth_mm: float,
    *,
    raised: bool = False,
    relief_mm: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    triangles_3d: list[tuple[int, int, int]] = []

    for polygon in polygon_parts(geometry):
        top_indices: dict[tuple[float, float], int] = {}
        bottom_indices: dict[tuple[float, float], int] = {}

        def top_index(point: tuple[float, float]) -> int:
            key = (round(point[0], 6), round(point[1], 6))
            if key not in top_indices:
                if relief_mm > 0:
                    top_indices[key] = add_vertex(vertices, point_at_face_offset(face, point[0], point[1], relief_mm))
                elif raised:
                    top_indices[key] = add_vertex(vertices, point_outside_face(face, point[0], point[1], depth_mm))
                else:
                    top_indices[key] = add_vertex(vertices, point_on_face(face, point[0], point[1], 0.0))
            return top_indices[key]

        def bottom_index(point: tuple[float, float]) -> int:
            key = (round(point[0], 6), round(point[1], 6))
            if key not in bottom_indices:
                if relief_mm > 0:
                    bottom_indices[key] = add_vertex(vertices, point_on_face(face, point[0], point[1], max(depth_mm - relief_mm, 0.0)))
                elif raised:
                    bottom_indices[key] = add_vertex(vertices, point_on_face(face, point[0], point[1], 0.0))
                else:
                    bottom_indices[key] = add_vertex(vertices, point_on_face(face, point[0], point[1], depth_mm))
            return bottom_indices[key]

        for coords in triangulate_polygon_2d(polygon):
            top = [top_index((x, y)) for x, y in coords]
            bottom = [bottom_index((x, y)) for x, y in coords]
            triangles_3d.append((top[0], top[1], top[2]))
            triangles_3d.append((bottom[2], bottom[1], bottom[0]))

        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coords = list(ring.coords)
            for point_a, point_b in zip(coords, coords[1:]):
                a_top = top_index((point_a[0], point_a[1]))
                b_top = top_index((point_b[0], point_b[1]))
                a_bottom = bottom_index((point_a[0], point_a[1]))
                b_bottom = bottom_index((point_b[0], point_b[1]))
                triangles_3d.append((a_top, a_bottom, b_bottom))
                triangles_3d.append((a_top, b_bottom, b_top))

    return vertices, triangles_3d


def add_mesh_object(
    resources: ET.Element,
    object_id: int,
    name: str,
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    material_index: int,
) -> None:
    vertices, triangles = repair_mesh(vertices, triangles)
    obj = ET.SubElement(
        resources,
        f"{{{CORE_NS}}}object",
        {"id": str(object_id), "name": name, "type": "model", "pid": "1", "pindex": str(material_index)},
    )
    mesh = ET.SubElement(obj, f"{{{CORE_NS}}}mesh")
    verts = ET.SubElement(mesh, f"{{{CORE_NS}}}vertices")
    for x, y, z in vertices:
        ET.SubElement(verts, f"{{{CORE_NS}}}vertex", {"x": f"{x:.6f}", "y": f"{y:.6f}", "z": f"{z:.6f}"})
    tris = ET.SubElement(mesh, f"{{{CORE_NS}}}triangles")
    for v1, v2, v3 in triangles:
        ET.SubElement(tris, f"{{{CORE_NS}}}triangle", {"v1": str(v1), "v2": str(v2), "v3": str(v3)})


def create_model_xml(
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]],
    *,
    colors: list[str] | None = None,
) -> bytes:
    model = ET.Element(
        f"{{{CORE_NS}}}model",
        {"unit": "millimeter", "{http://www.w3.org/XML/1998/namespace}lang": "en-US"},
    )
    ET.SubElement(model, f"{{{CORE_NS}}}metadata", {"name": "Title"}).text = "Massive Darkness Dice"
    resources = ET.SubElement(model, f"{{{CORE_NS}}}resources")
    color_group = ET.SubElement(resources, f"{{{MAT_NS}}}colorgroup", {"id": "1"})
    for color in colors or [
        "#F7C900FF",
        "#F36C16FF",
        "#B70F2BFF",
        "#1266C9FF",
        "#109A20FF",
        "#5B168EFF",
        "#1C1C1CFF",
        "#F6F6F3FF",
        "#D93A34FF",
    ]:
        ET.SubElement(color_group, f"{{{MAT_NS}}}color", {"color": color})

    build = ET.SubElement(model, f"{{{CORE_NS}}}build", {f"{{{PROD_NS}}}UUID": str(uuid.uuid4())})
    for object_id, (name, vertices, triangles, material_index) in enumerate(objects, start=1):
        add_mesh_object(resources, object_id, name, vertices, triangles, material_index)
        ET.SubElement(build, f"{{{CORE_NS}}}item", {"objectid": str(object_id), f"{{{PROD_NS}}}UUID": str(uuid.uuid4())})
    return ET.tostring(model, encoding="utf-8", xml_declaration=True)


def write_3mf(
    path: Path,
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]],
    *,
    colors: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_xml = create_model_xml(objects, colors=colors)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
""",
        )
        archive.writestr("3D/3dmodel.model", model_xml)


def repair_mesh(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    return [tuple(float(value) for value in vertex) for vertex in mesh.vertices], [
        tuple(int(value) for value in face) for face in mesh.faces
    ]


def triangle_normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (nx / length, ny / length, nz / length)


def write_stl(path: Path, vertices: list[tuple[float, float, float]], triangles: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, triangles = repair_mesh(vertices, triangles)
    with path.open("wb") as handle:
        header = f"Generated from {path.stem}".encode("ascii", errors="ignore")[:80]
        handle.write(header.ljust(80, b"\0"))
        handle.write(struct.pack("<I", len(triangles)))
        for v1, v2, v3 in triangles:
            a, b, c = vertices[v1], vertices[v2], vertices[v3]
            normal = triangle_normal(a, b, c)
            handle.write(struct.pack("<12fH", *normal, *a, *b, *c, 0))


def write_stl_assembly(path: Path, objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, vertices, triangles, _material_index in objects:
        write_stl(path / f"{name}.stl", vertices, triangles)


def combine_mesh_objects(
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    combined_vertices: list[tuple[float, float, float]] = []
    combined_triangles: list[tuple[int, int, int]] = []
    for _name, vertices, triangles, _material_index in objects:
        vertex_offset = len(combined_vertices)
        combined_vertices.extend(vertices)
        combined_triangles.extend(
            (v1 + vertex_offset, v2 + vertex_offset, v3 + vertex_offset) for v1, v2, v3 in triangles
        )
    return combined_vertices, combined_triangles


def write_combined_stl(
    path: Path,
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]],
) -> None:
    vertices, triangles = combine_mesh_objects(objects)
    write_stl(path, vertices, triangles)


def material_index_for_base(die: DieSpec) -> int:
    return {"yellow": 0, "orange": 1, "red": 2, "blue": 3, "green": 4, "purple": 5, "black": 6}[die.name]


def two_color_objects(
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]],
) -> list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]]:
    return [(name, vertices, triangles, 0 if index == 0 else 1) for index, (name, vertices, triangles, _material) in enumerate(objects)]


def translated(vertices: list[tuple[float, float, float]], offset: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    return [(x + offset[0], y + offset[1], z + offset[2]) for x, y, z in vertices]


def symbol_mesh(
    svg_name: str,
    face: Face,
    *,
    depth_mm: float = SYMBOL_DEPTH_MM,
    planar_scale: float = 1.0,
    raised: bool = False,
    relief_mm: float = 0.0,
    buffer_mm: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    svg_path = SVG_DIR / svg_name
    contours = svg_contours(svg_path)
    if planar_scale != 1.0:
        contours = scale_contours_about_center(contours, planar_scale)
    geometry = polygon_from_contours(contours)
    if buffer_mm:
        geometry = geometry.buffer(buffer_mm, join_style="round")
    return extrude_polygon_to_face(geometry, face, depth_mm, raised=raised, relief_mm=relief_mm)


def die_objects(
    die: DieSpec,
    base_vertices: list[tuple[float, float, float]],
    base_triangles: list[tuple[int, int, int]],
    offset: tuple[float, float, float] = (0, 0, 0),
) -> list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]]:
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]] = [
        (f"{die.name}_body", translated(base_vertices, offset), base_triangles, material_index_for_base(die))
    ]
    symbol_material = 8 if die.name == "black" else 7
    for face, svg_name in zip(FACES, die.faces):
        if svg_name is None:
            continue
        vertices, triangles = symbol_mesh(svg_name, face)
        objects.append((f"{die.name}_{face.name}_{Path(svg_name).stem}", translated(vertices, offset), triangles, symbol_material))
    return objects


def raised_die_objects(
    die: DieSpec,
    base_vertices: list[tuple[float, float, float]],
    base_triangles: list[tuple[int, int, int]],
    offset: tuple[float, float, float] = (0, 0, 0),
) -> list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]]:
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]] = [
        (f"{die.name}_body", translated(base_vertices, offset), base_triangles, material_index_for_base(die))
    ]
    symbol_material = 8 if die.name == "black" else 7
    for face, svg_name in zip(FACES, die.faces):
        if svg_name is None:
            continue
        vertices, triangles = symbol_mesh(svg_name, face, depth_mm=RAISED_DEPTH_MM, raised=True)
        objects.append((f"{die.name}_{face.name}_{Path(svg_name).stem}_raised", translated(vertices, offset), triangles, symbol_material))
    return objects


def slight_relief_die_objects(
    die: DieSpec,
    base_vertices: list[tuple[float, float, float]],
    base_triangles: list[tuple[int, int, int]],
    offset: tuple[float, float, float] = (0, 0, 0),
) -> list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]]:
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]] = [
        (f"{die.name}_body", translated(base_vertices, offset), base_triangles, material_index_for_base(die))
    ]
    symbol_material = 8 if die.name == "black" else 7
    for face, svg_name in zip(FACES, die.faces):
        if svg_name is None:
            continue
        vertices, triangles = symbol_mesh(svg_name, face, depth_mm=SYMBOL_DEPTH_MM, relief_mm=SLIGHT_RELIEF_MM)
        objects.append((f"{die.name}_{face.name}_{Path(svg_name).stem}_slight_relief", translated(vertices, offset), triangles, symbol_material))
    return objects


def print_optimized_die_objects(
    die: DieSpec,
    base_vertices: list[tuple[float, float, float]],
    base_triangles: list[tuple[int, int, int]],
    offset: tuple[float, float, float] = (0, 0, 0),
) -> list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]]:
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]] = [
        (f"{die.name}_body", translated(base_vertices, offset), base_triangles, material_index_for_base(die))
    ]
    symbol_material = 8 if die.name == "black" else 7
    for face, svg_name in zip(FACES, die.faces):
        if svg_name is None:
            continue
        vertices, triangles = symbol_mesh(
            svg_name,
            face,
            depth_mm=PRINT_OPTIMIZED_DEPTH_MM,
            relief_mm=PRINT_OPTIMIZED_RELIEF_MM,
            buffer_mm=PRINT_OPTIMIZED_BUFFER_MM,
        )
        objects.append(
            (f"{die.name}_{face.name}_{Path(svg_name).stem}_print_optimized", translated(vertices, offset), triangles, symbol_material)
        )
    return objects


def print_optimized_04_nozzle_die_objects(
    die: DieSpec,
    base_vertices: list[tuple[float, float, float]],
    base_triangles: list[tuple[int, int, int]],
    offset: tuple[float, float, float] = (0, 0, 0),
) -> list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]]:
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]] = [
        (f"{die.name}_body", translated(base_vertices, offset), base_triangles, material_index_for_base(die))
    ]
    symbol_material = 8 if die.name == "black" else 7
    symbol_buffer = NOZZLE_04_BLACK_SYMBOL_BUFFER_MM if die.name == "black" else PRINT_OPTIMIZED_BUFFER_MM
    for face, svg_name in zip(FACES, die.faces):
        if svg_name is None:
            continue
        vertices, triangles = symbol_mesh(
            svg_name,
            face,
            depth_mm=PRINT_OPTIMIZED_DEPTH_MM,
            relief_mm=PRINT_OPTIMIZED_RELIEF_MM,
            buffer_mm=symbol_buffer,
        )
        objects.append(
            (f"{die.name}_{face.name}_{Path(svg_name).stem}_print_04_nozzle", translated(vertices, offset), triangles, symbol_material)
        )
    return objects


def insert_objects(
    die: DieSpec,
    offset: tuple[float, float, float] = (0, 0, 0),
) -> list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]]:
    objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]] = []
    symbol_material = 8 if die.name == "black" else 7
    for face, svg_name in zip(FACES, die.faces):
        if svg_name is None:
            continue
        vertices, triangles = symbol_mesh(
            svg_name,
            face,
            depth_mm=INSERT_DEPTH_MM,
            planar_scale=INSERT_PLANAR_SCALE,
        )
        objects.append((f"{die.name}_{face.name}_{Path(svg_name).stem}_insert", translated(vertices, offset), triangles, symbol_material))
    return objects


def main() -> None:
    if not BASE_3MF.exists():
        raise RuntimeError(f"Missing base die 3MF: {BASE_3MF}")
    base_vertices, base_triangles = read_base_mesh()
    individual_dir = BUILD_3MF_OUT / "individual"
    global_final_objects: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]], int]] = []

    for index, die in enumerate(DICE):
        final_objects = print_optimized_04_nozzle_die_objects(die, base_vertices, base_triangles)
        write_3mf(
            individual_dir / f"{die.name}_die.3mf",
            two_color_objects(final_objects),
            colors=[die.base_color, die.symbol_color],
        )
        global_final_objects.extend(print_optimized_04_nozzle_die_objects(die, base_vertices, base_triangles, offset=(index * 24.0, 0.0, 0.0)))

    write_3mf(BUILD_3MF_OUT / "all_dice.3mf", global_final_objects)
    (BUILD_3MF_OUT / "README.md").write_text(
        "# Final 3MF Files\n\n"
        "Use these files in Bambu Studio.\n\n"
        "- `all_dice.3mf`: all dice in one file.\n"
        "- `individual/`: one file per die color.\n\n"
        "These are tuned for a Bambu P2S with a 0.4 mm nozzle and 0.12 mm High Quality profile.\n"
        f"Symbols have {PRINT_OPTIMIZED_RELIEF_MM} mm visible relief and {PRINT_OPTIMIZED_DEPTH_MM} mm total thickness.\n"
        f"The black die red symbols use a {NOZZLE_04_BLACK_SYMBOL_BUFFER_MM} mm buffer to preserve black gaps between claw marks.\n"
        "Each individual die file contains only two material colors: die body and symbols.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
