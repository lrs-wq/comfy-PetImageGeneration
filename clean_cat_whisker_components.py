from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_GLB = (
    ROOT
    / "outputs"
    / "cat_3d_hunyuan2mini_turbo_clean_alpha_final"
    / "pose_02_clean_alpha"
    / "pose_02_clean_alpha_hunyuan_texgen_uv_fur_clean.glb"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "cat_3d_hunyuan2mini_turbo_clean_alpha_whisker_clean"
    / "pose_02_clean_alpha"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove small mouth/whisker artifact components from a textured cat GLB."
    )
    parser.add_argument("--input-glb", default=str(DEFAULT_INPUT_GLB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default="")
    parser.add_argument("--max-faces", type=int, default=80)
    parser.add_argument("--front-limit", type=float, default=0.14)
    parser.add_argument("--y-min", type=float, default=0.56)
    parser.add_argument("--y-max", type=float, default=0.92)
    parser.add_argument("--z-min", type=float, default=0.50)
    parser.add_argument("--z-max", type=float, default=0.70)
    parser.add_argument("--max-span-x", type=float, default=0.06)
    parser.add_argument("--max-span-z", type=float, default=0.12)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unable to load mesh from {path}")
    return mesh


def clean_whisker_components(mesh: trimesh.Trimesh, args: argparse.Namespace) -> tuple[trimesh.Trimesh, dict]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    dims = np.maximum(bounds_max - bounds_min, 1e-9)
    norm_vertices = (vertices - bounds_min) / dims
    face_centers = norm_vertices[faces].mean(axis=1)

    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(faces)),
        min_len=1,
    )

    keep_faces = np.ones(len(faces), dtype=bool)
    removed_components = []
    for component in components:
        component = np.asarray(list(component), dtype=np.int64)
        if len(component) > args.max_faces:
            continue

        centers = face_centers[component]
        center = centers.mean(axis=0)
        span = centers.max(axis=0) - centers.min(axis=0)
        in_muzzle_band = (
            center[0] < args.front_limit
            and args.y_min < center[1] < args.y_max
            and args.z_min < center[2] < args.z_max
        )
        compact_fragment = span[0] < args.max_span_x and span[2] < args.max_span_z
        if not (in_muzzle_band and compact_fragment):
            continue

        keep_faces[component] = False
        removed_components.append(
            {
                "faces": int(len(component)),
                "center": [round(float(value), 4) for value in center],
                "span": [round(float(value), 4) for value in span],
            }
        )

    cleaned = mesh.copy()
    cleaned.update_faces(keep_faces)
    cleaned.remove_unreferenced_vertices()
    cleaned.fix_normals()
    report = {
        "input_faces": int(len(mesh.faces)),
        "output_faces": int(len(cleaned.faces)),
        "removed_faces": int((~keep_faces).sum()),
        "removed_components": removed_components,
        "component_count": int(len(components)),
        "bounds_min": [float(value) for value in bounds_min],
        "bounds_max": [float(value) for value in bounds_max],
    }
    return cleaned, report


def main() -> None:
    args = parse_args()
    input_glb = Path(args.input_glb)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_glb.stem}_whisker_clean"

    mesh = load_mesh(input_glb)
    cleaned, report = clean_whisker_components(mesh, args)

    output_glb = output_dir / f"{name}.glb"
    cleaned.export(output_glb)
    report["output_glb"] = str(output_glb)

    report_path = output_dir / f"{name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
