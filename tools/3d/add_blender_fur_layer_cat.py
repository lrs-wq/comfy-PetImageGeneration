import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT_GLB = (
    Path(__file__).resolve().parent
    / "outputs"
    / "cat_3d_hunyuan2mini_turbo_clean_alpha_final"
    / "pose_02_clean_alpha"
    / "pose_02_clean_alpha_hunyuan_texgen_uv_fur_clean.glb"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "cat_3d_blender_fur"
    / "pose_02_clean_alpha"
)
DEFAULT_BLENDER_PATHS = [
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender\blender.exe",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Add a Blender particle-hair fur layer to a textured cat GLB."
    )
    parser.add_argument("--input-glb", default=str(DEFAULT_INPUT_GLB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXE", ""))
    parser.add_argument("--name", default="")
    parser.add_argument("--hair-count", type=int, default=36000)
    parser.add_argument("--child-hairs", type=int, default=12)
    parser.add_argument("--hair-length-ratio", type=float, default=0.0075)
    parser.add_argument("--root-radius-ratio", type=float, default=0.00020)
    parser.add_argument("--tip-radius-ratio", type=float, default=0.000018)
    parser.add_argument("--guard-hair-count", type=int, default=7000)
    parser.add_argument("--guard-child-hairs", type=int, default=3)
    parser.add_argument("--guard-length-ratio", type=float, default=0.014)
    parser.add_argument("--longhair-count", type=int, default=9000)
    parser.add_argument("--longhair-child-hairs", type=int, default=5)
    parser.add_argument("--longhair-length-ratio", type=float, default=0.018)
    parser.add_argument("--density-dark-cutoff", type=float, default=0.11)
    parser.add_argument("--density-bright-full", type=float, default=0.36)
    parser.add_argument("--min-density", type=float, default=0.0)
    parser.add_argument("--fur-color", default="0.88,0.82,0.72")
    parser.add_argument("--no-texture-fur-color", action="store_true")
    parser.add_argument("--clean-whisker-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--whisker-clean-strength", type=float, default=1.0)
    parser.add_argument("--add-muzzle-repair", action="store_true")
    parser.add_argument("--add-natural-whiskers", action="store_true")
    parser.add_argument("--render-size", type=int, default=1100)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--camera-axis",
        default="-X",
        choices=["-Y", "+Y", "-X", "+X"],
        help="Primary preview direction in Blender coordinates.",
    )
    parser.add_argument(
        "--engine",
        default="CYCLES",
        choices=["CYCLES", "BLENDER_EEVEE_NEXT"],
    )
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--inside-blender", action="store_true")
    return parser.parse_args(argv)


def find_blender(explicit):
    if explicit:
        path = Path(explicit)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Blender executable not found: {path}")

    found = shutil.which("blender")
    if found:
        return found

    for candidate in DEFAULT_BLENDER_PATHS:
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "Blender was not found. Install Blender or pass --blender C:\\path\\to\\blender.exe"
    )


def external_main(args):
    blender = find_blender(args.blender)
    script = str(Path(__file__).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        blender,
        "--background",
        "--python",
        script,
        "--",
        "--inside-blender",
        "--input-glb",
        str(Path(args.input_glb).resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--hair-count",
        str(args.hair_count),
        "--child-hairs",
        str(args.child_hairs),
        "--hair-length-ratio",
        str(args.hair_length_ratio),
        "--root-radius-ratio",
        str(args.root_radius_ratio),
        "--tip-radius-ratio",
        str(args.tip_radius_ratio),
        "--guard-hair-count",
        str(args.guard_hair_count),
        "--guard-child-hairs",
        str(args.guard_child_hairs),
        "--guard-length-ratio",
        str(args.guard_length_ratio),
        "--longhair-count",
        str(args.longhair_count),
        "--longhair-child-hairs",
        str(args.longhair_child_hairs),
        "--longhair-length-ratio",
        str(args.longhair_length_ratio),
        "--density-dark-cutoff",
        str(args.density_dark_cutoff),
        "--density-bright-full",
        str(args.density_bright_full),
        "--min-density",
        str(args.min_density),
        "--fur-color",
        args.fur_color,
        "--render-size",
        str(args.render_size),
        "--samples",
        str(args.samples),
        f"--camera-axis={args.camera_axis}",
        "--engine",
        args.engine,
    ]
    if args.name:
        cmd.extend(["--name", args.name])
    if args.skip_render:
        cmd.append("--skip-render")
    if args.no_texture_fur_color:
        cmd.append("--no-texture-fur-color")
    if not args.clean_whisker_artifacts:
        cmd.append("--no-clean-whisker-artifacts")
    cmd.extend(["--whisker-clean-strength", str(args.whisker_clean_strength)])
    if args.add_muzzle_repair:
        cmd.append("--add-muzzle-repair")
    if args.add_natural_whiskers:
        cmd.append("--add-natural-whiskers")

    print("Running Blender:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def blender_args():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def inside_blender_main(args):
    import bpy
    from mathutils import Vector

    input_glb = Path(args.input_glb)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or input_glb.stem

    fur_color = tuple(float(part.strip()) for part in args.fur_color.split(","))
    if len(fur_color) != 3:
        raise ValueError("--fur-color must be R,G,B")

    clear_scene(bpy)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {input_glb}")

    target = max(mesh_objects, key=lambda obj: len(obj.data.polygons))
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    if args.clean_whisker_artifacts:
        clean_whisker_artifacts(target, strength=args.whisker_clean_strength)
    smooth_mesh(target)

    bounds_min, bounds_max, dims, diag = mesh_bounds(mesh_objects, Vector)
    hair_length = diag * args.hair_length_ratio
    root_radius = diag * args.root_radius_ratio
    tip_radius = diag * args.tip_radius_ratio

    texture_image = find_first_texture_image(bpy, target)
    density_group = build_fur_density_group(
        target,
        texture_image,
        dark_cutoff=args.density_dark_cutoff,
        bright_full=args.density_bright_full,
        min_density=args.min_density,
    )

    fur_material = create_fur_material(
        bpy,
        fur_color,
        texture_image=None if args.no_texture_fur_color else texture_image,
    )
    target.data.materials.append(fur_material)
    add_particle_fur(
        bpy,
        target,
        name="cat_dense_short_undercoat",
        density_group_name=density_group.name,
        material_slot=len(target.data.materials),
        hair_count=args.hair_count,
        child_hairs=args.child_hairs,
        hair_length=hair_length,
        root_radius=root_radius,
        tip_radius=tip_radius,
        roughness_scale=0.34,
        child_radius_scale=2.2,
        clump_factor=0.025,
    )
    if args.guard_hair_count > 0:
        add_particle_fur(
            bpy,
            target,
            name="cat_sparse_guard_hairs",
            density_group_name=density_group.name,
            material_slot=len(target.data.materials),
            hair_count=args.guard_hair_count,
            child_hairs=args.guard_child_hairs,
            hair_length=diag * args.guard_length_ratio,
            root_radius=root_radius * 0.72,
            tip_radius=tip_radius * 0.70,
            roughness_scale=0.48,
            child_radius_scale=1.4,
            clump_factor=0.01,
        )
    if args.longhair_count > 0:
        longhair_group = build_longhair_tuft_group(target, density_group)
        add_particle_fur(
            bpy,
            target,
            name="cat_ragdoll_collar_and_ear_tufts",
            density_group_name=longhair_group.name,
            material_slot=len(target.data.materials),
            hair_count=args.longhair_count,
            child_hairs=args.longhair_child_hairs,
            hair_length=diag * args.longhair_length_ratio,
            root_radius=root_radius * 0.62,
            tip_radius=tip_radius * 0.55,
            roughness_scale=0.72,
            child_radius_scale=1.1,
            clump_factor=0.055,
        )
    if args.add_muzzle_repair:
        add_muzzle_repair_pads(bpy, Vector, bounds_min, dims, diag)
    if args.add_natural_whiskers:
        add_natural_whiskers(bpy, Vector, bounds_min, dims, diag)

    setup_scene(
        bpy,
        Vector,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        dims=dims,
        camera_axis=args.camera_axis,
        render_size=args.render_size,
        samples=args.samples,
        engine=args.engine,
    )

    blend_path = output_dir / f"{name}_blender_fur.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    preview_path = output_dir / f"{name}_blender_fur_preview.png"
    if not args.skip_render:
        bpy.context.scene.render.filepath = str(preview_path)
        bpy.ops.render.render(write_still=True)

    report = {
        "input_glb": str(input_glb),
        "blend": str(blend_path),
        "preview": str(preview_path) if not args.skip_render else "",
        "target_mesh": target.name,
        "texture_used": texture_image.name if texture_image else "",
        "hair_count": args.hair_count,
        "child_hairs": args.child_hairs,
        "guard_hair_count": args.guard_hair_count,
        "guard_child_hairs": args.guard_child_hairs,
        "longhair_count": args.longhair_count,
        "longhair_child_hairs": args.longhair_child_hairs,
        "hair_length": hair_length,
        "guard_hair_length": diag * args.guard_length_ratio,
        "longhair_length": diag * args.longhair_length_ratio,
        "root_radius": root_radius,
        "tip_radius": tip_radius,
        "texture_fur_color": not args.no_texture_fur_color,
        "clean_whisker_artifacts": args.clean_whisker_artifacts,
        "whisker_clean_strength": args.whisker_clean_strength,
        "add_muzzle_repair": args.add_muzzle_repair,
        "add_natural_whiskers": args.add_natural_whiskers,
        "density_vertices": len(target.data.vertices),
        "mesh_faces_after_whisker_clean": len(target.data.polygons),
        "engine": args.engine,
    }
    report_path = output_dir / f"{name}_blender_fur_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def clear_scene(bpy):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def smooth_mesh(obj):
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def clean_whisker_artifacts(obj, strength=1.0):
    mesh = obj.data
    vertices = mesh.vertices
    if not vertices or not mesh.polygons:
        return

    xs = [vertex.co.x for vertex in vertices]
    ys = [vertex.co.y for vertex in vertices]
    zs = [vertex.co.z for vertex in vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    dx = max(max_x - min_x, 1e-6)
    dy = max(max_y - min_y, 1e-6)
    dz = max(max_z - min_z, 1e-6)

    def norm(co):
        return (
            (co.x - min_x) / dx,
            (co.y - min_y) / dy,
            (co.z - min_z) / dz,
        )

    face_neighbors = build_face_neighbors(mesh)
    visited = set()
    selected = []
    max_faces = max(8, int(80 * max(strength, 0.25)))
    front_limit = 0.14 + 0.03 * max(strength - 1.0, 0.0)
    for polygon in mesh.polygons:
        if polygon.index in visited:
            continue
        stack = [polygon.index]
        component = []
        visited.add(polygon.index)
        while stack:
            face_index = stack.pop()
            component.append(face_index)
            for neighbor in face_neighbors[face_index]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if len(component) > max_faces:
            continue

        centers = [norm(mesh.polygons[index].center) for index in component]
        center = (
            sum(c[0] for c in centers) / len(centers),
            sum(c[1] for c in centers) / len(centers),
            sum(c[2] for c in centers) / len(centers),
        )
        span = (
            max(c[0] for c in centers) - min(c[0] for c in centers),
            max(c[1] for c in centers) - min(c[1] for c in centers),
            max(c[2] for c in centers) - min(c[2] for c in centers),
        )

        in_muzzle_band = (
            center[0] < front_limit
            and 0.50 < center[2] < 0.70
            and 0.56 < center[1] < 0.92
        )
        compact_fragment = span[0] < 0.06 and span[2] < 0.12
        if in_muzzle_band and compact_fragment:
            selected.extend(component)

    if not selected:
        print("Whisker artifact clean: no faces selected")
        return

    import bmesh

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    selected_set = set(selected)
    faces_to_delete = [face for face in bm.faces if face.index in selected_set]
    bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    print(f"Whisker artifact clean: removed {len(selected)} faces")


def build_face_neighbors(mesh):
    edge_to_faces = {}
    for polygon in mesh.polygons:
        indices = list(polygon.vertices)
        edges = [
            tuple(sorted((indices[0], indices[1]))),
            tuple(sorted((indices[1], indices[2]))),
            tuple(sorted((indices[2], indices[0]))),
        ]
        for edge in edges:
            edge_to_faces.setdefault(edge, []).append(polygon.index)

    neighbors = {polygon.index: set() for polygon in mesh.polygons}
    for faces in edge_to_faces.values():
        if len(faces) < 2:
            continue
        for face in faces:
            neighbors[face].update(other for other in faces if other != face)
    return neighbors


def mesh_bounds(objects, Vector):
    world_points = []
    for obj in objects:
        world_points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    min_v = Vector((min(p.x for p in world_points), min(p.y for p in world_points), min(p.z for p in world_points)))
    max_v = Vector((max(p.x for p in world_points), max(p.y for p in world_points), max(p.z for p in world_points)))
    dims = max_v - min_v
    diag = max(dims.length, 1e-6)
    return min_v, max_v, dims, diag


def find_first_texture_image(bpy, obj):
    for material in obj.data.materials:
        if not material or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image and node.image.size[0] > 1:
                return node.image
    for image in bpy.data.images:
        if image.size[0] > 1 and image.size[1] > 1:
            return image
    return None


def build_fur_density_group(obj, image, dark_cutoff, bright_full, min_density):
    mesh = obj.data
    group = obj.vertex_groups.new(name="cat_fur_density")

    uv_by_vertex = collect_vertex_uvs(mesh)
    sampler = make_image_sampler(image)
    weights = []
    for vertex in mesh.vertices:
        color = sampler(uv_by_vertex.get(vertex.index))
        weight = density_weight(color, dark_cutoff, bright_full, min_density)
        weights.append(weight)

    for vertex_index, weight in enumerate(weights):
        group.add([vertex_index], weight, "REPLACE")

    average = sum(weights) / max(len(weights), 1)
    print(
        f"Fur density weights: average={average:.3f}, "
        f"min={min(weights):.3f}, max={max(weights):.3f}"
    )
    return group


def build_longhair_tuft_group(obj, base_group):
    mesh = obj.data
    group = obj.vertex_groups.new(name="cat_longhair_tuft_density")
    coords = [vertex.co for vertex in mesh.vertices]
    min_x = min(v.x for v in coords)
    max_x = max(v.x for v in coords)
    min_y = min(v.y for v in coords)
    max_y = max(v.y for v in coords)
    min_z = min(v.z for v in coords)
    max_z = max(v.z for v in coords)
    dx = max(max_x - min_x, 1e-6)
    dy = max(max_y - min_y, 1e-6)
    dz = max(max_z - min_z, 1e-6)
    center_y = (min_y + max_y) * 0.5

    base_weights = {}
    for vertex in mesh.vertices:
        try:
            base_weights[vertex.index] = base_group.weight(vertex.index)
        except RuntimeError:
            base_weights[vertex.index] = 0.0

    weights = []
    for vertex in mesh.vertices:
        x = (vertex.co.x - min_x) / dx
        y_abs = abs(vertex.co.y - center_y) / dy
        z = (vertex.co.z - min_z) / dz

        front = 1.0 - smoothstep(0.22, 0.62, x)
        centerline = 1.0 - smoothstep(0.10, 0.34, y_abs)
        side = smoothstep(0.18, 0.45, y_abs)

        chest_height = smoothstep(0.18, 0.38, z) * (1.0 - smoothstep(0.62, 0.82, z))
        cheek_height = smoothstep(0.50, 0.66, z) * (1.0 - smoothstep(0.76, 0.92, z))
        ear_height = smoothstep(0.74, 0.96, z)

        chest = front * centerline * chest_height
        cheek = front * smoothstep(0.08, 0.30, y_abs) * cheek_height
        ear = front * side * ear_height
        weight = max(chest, cheek * 0.72, ear * 0.55) * base_weights[vertex.index]
        weights.append(clamp(weight, 0.0, 1.0))

    for vertex_index, weight in enumerate(weights):
        group.add([vertex_index], weight, "REPLACE")

    average = sum(weights) / max(len(weights), 1)
    print(
        f"Longhair tuft weights: average={average:.3f}, "
        f"min={min(weights):.3f}, max={max(weights):.3f}"
    )
    return group


def collect_vertex_uvs(mesh):
    if not mesh.uv_layers.active:
        return {}
    uv_layer = mesh.uv_layers.active.data
    sums = {}
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv = uv_layer[loop_index].uv
            if vertex_index not in sums:
                sums[vertex_index] = [0.0, 0.0, 0]
            sums[vertex_index][0] += uv.x
            sums[vertex_index][1] += uv.y
            sums[vertex_index][2] += 1

    return {
        vertex_index: (values[0] / values[2], values[1] / values[2])
        for vertex_index, values in sums.items()
        if values[2]
    }


def make_image_sampler(image):
    if image is None:
        return lambda uv: (0.86, 0.80, 0.70)

    width, height = int(image.size[0]), int(image.size[1])
    pixels = [0.0] * (width * height * 4)
    image.pixels.foreach_get(pixels)

    def sample(uv):
        if uv is None:
            return 0.86, 0.80, 0.70
        u = uv[0] % 1.0
        v = uv[1] % 1.0
        x = min(width - 1, max(0, int(u * (width - 1))))
        y = min(height - 1, max(0, int((1.0 - v) * (height - 1))))
        idx = (y * width + x) * 4
        return pixels[idx], pixels[idx + 1], pixels[idx + 2]

    return sample


def density_weight(color, dark_cutoff, bright_full, min_density):
    r, g, b = color
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    bright_weight = smoothstep(dark_cutoff, bright_full, luma)
    return clamp(min_density + (1.0 - min_density) * bright_weight, 0.0, 1.0)


def smoothstep(edge0, edge1, value):
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    x = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def clamp(value, low, high):
    return max(low, min(high, value))


def create_fur_material(bpy, color, texture_image=None):
    material = bpy.data.materials.new("texture_following_cat_fur")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")

    try:
        shader = nodes.new(type="ShaderNodeBsdfHairPrincipled")
        for input_name in ("Color", "Base Color"):
            if input_name in shader.inputs:
                shader.inputs[input_name].default_value = (*color, 1.0)
                if texture_image is not None:
                    texture = nodes.new(type="ShaderNodeTexImage")
                    texture.image = texture_image
                    material.node_tree.links.new(texture.outputs["Color"], shader.inputs[input_name])
                break
        if "Roughness" in shader.inputs:
            shader.inputs["Roughness"].default_value = 0.58
        material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    except Exception:
        shader = nodes.new(type="ShaderNodeBsdfPrincipled")
        if "Base Color" in shader.inputs:
            shader.inputs["Base Color"].default_value = (*color, 1.0)
            if texture_image is not None:
                texture = nodes.new(type="ShaderNodeTexImage")
                texture.image = texture_image
                material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        if "Roughness" in shader.inputs:
            shader.inputs["Roughness"].default_value = 0.82
        material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    return material


def add_particle_fur(
    bpy,
    obj,
    name,
    density_group_name,
    material_slot,
    hair_count,
    child_hairs,
    hair_length,
    root_radius,
    tip_radius,
    roughness_scale,
    child_radius_scale,
    clump_factor,
):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_add(type="PARTICLE_SYSTEM")
    modifier = obj.modifiers[-1]
    modifier.name = name
    particle_system = obj.particle_systems[-1]
    settings = particle_system.settings

    settings.name = name
    settings.type = "HAIR"
    settings.count = hair_count
    settings.hair_length = hair_length
    settings.use_advanced_hair = True
    settings.render_type = "PATH"
    settings.use_modifier_stack = True
    settings.material = material_slot
    settings.emit_from = "FACE"
    settings.distribution = "RAND"
    settings.use_even_distribution = True
    settings.length_random = 0.30

    settings.hair_step = 5
    settings.display_step = 4
    settings.render_step = 5
    settings.use_rotations = True
    settings.phase_factor_random = 0.35
    settings.tangent_factor = hair_length * 0.09

    settings.child_type = "INTERPOLATED"
    settings.child_percent = min(child_hairs, 12)
    settings.rendered_child_count = child_hairs
    settings.child_radius = hair_length * child_radius_scale
    settings.child_length = 0.90
    settings.child_size = 0.45
    settings.child_size_random = 0.28
    settings.roughness_1_size = hair_length * 0.85
    settings.roughness_1 = hair_length * 0.10 * roughness_scale
    settings.roughness_2_size = hair_length * 0.45
    settings.roughness_2 = hair_length * 0.035 * roughness_scale
    settings.roughness_endpoint = hair_length * 0.055 * roughness_scale
    settings.clump_factor = clump_factor
    settings.clump_shape = -0.18
    settings.root_radius = root_radius
    settings.tip_radius = tip_radius
    settings.radius_scale = 1.0

    particle_system.vertex_group_density = density_group_name
    particle_system.vertex_group_length = density_group_name


def add_muzzle_repair_pads(bpy, Vector, bounds_min, dims, diag):
    material = create_surface_material(bpy, "soft_clean_muzzle_patch", (0.84, 0.80, 0.72, 1.0))
    front_x = bounds_min.x - dims.x * 0.014
    z = bounds_min.z + dims.z * 0.635
    patch_specs = [
        (bounds_min.y + dims.y * 0.805, z, (dims.x * 0.012, dims.y * 0.026, dims.z * 0.020)),
        (bounds_min.y + dims.y * 0.915, z + dims.z * 0.004, (dims.x * 0.011, dims.y * 0.024, dims.z * 0.018)),
    ]
    for index, (y, pad_z, scale) in enumerate(patch_specs, start=1):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32,
            ring_count=16,
            radius=1.0,
            location=(front_x, y, pad_z),
        )
        pad = bpy.context.object
        pad.name = f"clean_muzzle_pad_{index}"
        pad.scale = scale
        pad.data.materials.append(material)
        for polygon in pad.data.polygons:
            polygon.use_smooth = True


def add_natural_whiskers(bpy, Vector, bounds_min, dims, diag):
    material = create_curve_material(bpy, "fine_cat_whiskers", (0.94, 0.92, 0.86, 1.0))
    front_x = bounds_min.x - dims.x * 0.030
    starts = [
        (bounds_min.y + dims.y * 0.800, -1.0),
        (bounds_min.y + dims.y * 0.910, 1.0),
    ]
    z_values = [0.612, 0.636, 0.660]
    for side_index, (start_y, direction) in enumerate(starts):
        for row, z_norm in enumerate(z_values):
            start = Vector((front_x, start_y, bounds_min.z + dims.z * z_norm))
            mid = Vector((
                front_x - dims.x * 0.010,
                start_y + direction * dims.y * (0.060 + row * 0.010),
                start.z + dims.z * (row - 1) * 0.006,
            ))
            end = Vector((
                front_x - dims.x * 0.014,
                start_y + direction * dims.y * (0.145 + row * 0.018),
                start.z + dims.z * (row - 1) * 0.014,
            ))
            create_bezier_curve(
                bpy,
                f"fine_cat_whisker_{side_index}_{row}",
                [start, mid, end],
                material,
                bevel_depth=diag * 0.00028,
            )


def create_surface_material(bpy, name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader:
        if "Base Color" in shader.inputs:
            shader.inputs["Base Color"].default_value = color
        if "Roughness" in shader.inputs:
            shader.inputs["Roughness"].default_value = 0.86
    return material


def create_curve_material(bpy, name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader:
        if "Base Color" in shader.inputs:
            shader.inputs["Base Color"].default_value = color
        if "Roughness" in shader.inputs:
            shader.inputs["Roughness"].default_value = 0.55
    return material


def create_bezier_curve(bpy, name, points, material, bevel_depth):
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 16
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 2
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, co in zip(spline.bezier_points, points):
        point.co = co
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def setup_scene(
    bpy,
    Vector,
    bounds_min,
    bounds_max,
    dims,
    camera_axis,
    render_size,
    samples,
    engine,
):
    scene = bpy.context.scene
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        try_enable_cycles_gpu(bpy)
    else:
        scene.eevee.taa_render_samples = samples

    scene.render.resolution_x = render_size
    scene.render.resolution_y = render_size
    scene.render.film_transparent = False
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (1.0, 1.0, 1.0)
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    center = (bounds_min + bounds_max) * 0.5
    axis = axis_vector(camera_axis, Vector)
    max_dim = max(dims.x, dims.y, dims.z, 1e-6)
    distance = max_dim * 3.0

    camera_data = bpy.data.cameras.new("fur_preview_camera")
    camera = bpy.data.objects.new("fur_preview_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + axis * distance + Vector((0.0, 0.0, dims.z * 0.08))
    look_at(camera, center, Vector)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(dims.x, dims.z) * 1.24
    scene.camera = camera

    light_data = bpy.data.lights.new("softbox_key", type="AREA")
    light = bpy.data.objects.new("softbox_key", light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + Vector((-max_dim * 1.2, -max_dim * 1.8, max_dim * 2.5))
    light.data.energy = 520
    light.data.size = max_dim * 2.4

    fill_data = bpy.data.lights.new("softbox_fill", type="AREA")
    fill = bpy.data.objects.new("softbox_fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = center + Vector((max_dim * 1.5, max_dim * 1.2, max_dim * 1.6))
    fill.data.energy = 90
    fill.data.size = max_dim * 3.0


def try_enable_cycles_gpu(bpy):
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        for device_type in ("OPTIX", "CUDA", "HIP"):
            try:
                preferences.compute_device_type = device_type
                preferences.get_devices()
                usable = False
                for device in preferences.devices:
                    if device.type != "CPU":
                        device.use = True
                        usable = True
                if usable:
                    bpy.context.scene.cycles.device = "GPU"
                    print(f"Cycles GPU enabled: {device_type}")
                    return
            except Exception:
                continue
    except Exception:
        pass
    bpy.context.scene.cycles.device = "CPU"
    print("Cycles GPU not available; using CPU")


def axis_vector(axis, Vector):
    mapping = {
        "-Y": Vector((0.0, -1.0, 0.0)),
        "+Y": Vector((0.0, 1.0, 0.0)),
        "-X": Vector((-1.0, 0.0, 0.0)),
        "+X": Vector((1.0, 0.0, 0.0)),
    }
    return mapping[axis]


def look_at(obj, target, Vector):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main():
    raw_args = blender_args() if "--" in sys.argv else sys.argv[1:]
    args = parse_args(raw_args)
    if args.inside_blender:
        inside_blender_main(args)
    else:
        external_main(args)


if __name__ == "__main__":
    main()
