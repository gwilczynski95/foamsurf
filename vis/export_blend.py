"""Export a Poly Haven style .blend (meshes + a Principled BSDF with image textures) for vis/render_clip.py.

    blender -b MODEL.blend --python vis/export_blend.py -- OUT_DIR [--size 2048]
    blender -b --python vis/export_blend.py -- OUT_DIR --import MODEL.gltf   (glTF / glb: imported first)

Writes OUT_DIR/mesh.npz (vertices [V,3] y-up, faces [F,3], corner_uv [F,3,2], part [F] = material slot),
OUT_DIR/<material>_{albedo,metal,rough}.png (textures resized to --size, metal/rough as greyscale) and
OUT_DIR/manifest.json ({"format": "npz", "parts": [{"id", "material": {"mode": "pbr", "maps": {...}}}]}),
which vis.mesh.load_manifest reads like FALCON's manifests.
"""
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
out = Path(argv[0])
size = int(argv[argv.index("--size") + 1]) if "--size" in argv else 2048
out.mkdir(parents=True, exist_ok=True)
if "--import" in argv:  # a glTF / glb into an empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=argv[argv.index("--import") + 1])

materials = []  # names, in part order
vertices, faces, uvs, parts = [], [], [], []
offset = 0
depsgraph = bpy.context.evaluated_depsgraph_get()
for obj in bpy.data.objects:
    if obj.type != "MESH" or obj.hide_render:
        continue
    mesh = obj.evaluated_get(depsgraph).to_mesh()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.transform(obj.matrix_world)
    uv_layer = bm.loops.layers.uv.active
    slot_names = [s.material.name if s.material else "default" for s in obj.material_slots] or ["default"]
    for name in slot_names:
        if name not in materials:
            materials.append(name)
    v = np.array([vert.co[:] for vert in bm.verts], dtype=np.float32)
    f = np.array([[vert.index for vert in face.verts] for face in bm.faces], dtype=np.int64)
    uv = np.array([[loop[uv_layer].uv[:] for loop in face.loops] for face in bm.faces], dtype=np.float32)
    p = np.array([materials.index(slot_names[min(face.material_index, len(slot_names) - 1)]) for face in bm.faces])
    vertices.append(v)
    faces.append(f + offset)
    uvs.append(uv)
    parts.append(p)
    offset += len(v)
    bm.free()
    obj.evaluated_get(depsgraph).to_mesh_clear()

v = np.concatenate(vertices)
v = np.stack((v[:, 0], v[:, 2], -v[:, 1]), -1)  # Blender z-up -> y-up
np.savez_compressed(out / "mesh.npz", vertices=v, faces=np.concatenate(faces), corner_uv=np.concatenate(uvs),
                    part=np.concatenate(parts))


CHANNELS = {"Red": 0, "R": 0, "Green": 1, "G": 1, "Blue": 2, "B": 2}


def socket_image(node, socket_name):
    """Image texture feeding a BSDF input (through normal-map / reroute / separate-colour nodes) and the channel a
    separate-colour node picks from it (glTF imports pack metal in blue and roughness in green), or None."""
    socket = node.inputs.get(socket_name)
    channel = None
    while socket is not None and socket.is_linked:
        link = socket.links[0]
        src = link.from_node
        if src.type == "TEX_IMAGE":
            return src.image, channel
        if src.type in ("SEPARATE_COLOR", "SEPRGB"):
            channel = CHANNELS.get(link.from_socket.name, channel)
        socket = next((s for s in src.inputs if s.is_linked), None)
    return None, None


def save(image, path, grey, channel=None):
    image = image.copy()
    image.scale(size, size)
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(size, size, 4)[::-1]  # Blender: bottom row first
    if image.colorspace_settings.name not in ("sRGB",) and not grey:
        pixels[..., :3] = np.clip(pixels[..., :3], 0, 1) ** (1 / 2.2)
    c = channel or 0
    data = np.clip(pixels[..., c:c + 1] if grey else pixels[..., :3], 0, 1)
    from PIL import Image  # noqa: E402  (Blender's Python may lack PIL: fall back to Blender's writer)
    Image.fromarray((data[..., 0] if grey else data).__mul__(255).round().astype(np.uint8)).save(path)


def save_blender(image, path, grey, channel=None):
    image = image.copy()
    image.scale(size, size)
    if grey:
        px = np.array(image.pixels[:], dtype=np.float32).reshape(-1, 4)
        px[:, 0] = px[:, channel or 0]
        px[:, 1] = px[:, 2] = px[:, 0]
        px[:, 3] = 1
        image.pixels[:] = px.reshape(-1)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()


parts_json = []
for name in materials:
    mat = bpy.data.materials.get(name)
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mat and mat.use_nodes else None
    maps = {}
    for key, socket, grey in (("albedo", "Base Color", False), ("metal", "Metallic", True), ("rough", "Roughness", True)):
        image, channel = socket_image(bsdf, socket) if bsdf else (None, None)
        if image is None:
            continue
        path = out / f"{name}_{key}.png"
        try:
            save(image, path, grey, channel)
        except ImportError:
            save_blender(image, path, grey, channel)
        maps[key] = path.name
    parts_json.append({"id": name, "material": {"mode": "pbr", "rgb": [1, 1, 1], "maps": maps}})
(out / "manifest.json").write_text(json.dumps({"format": "npz", "mesh": "mesh.npz", "parts": parts_json}, indent=1))
print("exported", out, len(v), "vertices", sum(len(f) for f in faces), "faces", materials)
