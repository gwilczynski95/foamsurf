"""Download Poly Haven models (CC0) and export them for vis/render_clip.py.

    python vis/fetch_polyhaven.py ASSET [ASSET ...] [--res 1k] [--blender BLENDER]

Each asset's glTF and its textures (Poly Haven's file API) go to assets/models/raw/ASSET/, then
vis/export_blend.py writes assets/models/ASSET/{mesh.npz, manifest.json, textures}.
"""
import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = {"User-Agent": "foamsurf-vis"}


def fetch(url: str, path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT)) as response:
        path.write_bytes(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("assets", nargs="+")
    parser.add_argument("--res", default="1k")
    parser.add_argument("--size", type=int, default=1024, help="texture size of the export")
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    for asset in args.assets:
        with urllib.request.urlopen(urllib.request.Request(f"https://api.polyhaven.com/files/{asset}",
                                                           headers=AGENT)) as response:
            files = json.load(response)
        source = files["gltf"][args.res]["gltf"]  # the .blend files need Blender 5
        raw = ROOT / "assets/models/raw" / asset
        main_file = raw / Path(source["url"]).name
        fetch(source["url"], main_file)
        for relative, info in source.get("include", {}).items():
            fetch(info["url"], raw / relative)
        out = ROOT / "assets/models" / asset
        subprocess.run([args.blender, "-b", "--disable-autoexec", "--python", str(ROOT / "vis/export_blend.py"),
                        "--", str(out), "--size", str(args.size), "--import", str(main_file)], check=True)


if __name__ == "__main__":
    main()
