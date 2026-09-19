"""Executed only by the GPU test in a fresh interpreter, never collected by pytest."""
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[3] / "radfoam"
    extension_dirs = sorted(
        directory for directory in (root / "build").glob("lib.*")
        if (directory / "radfoam" / "__init__.py").is_file()
        and any((directory / "radfoam").glob("torch_bindings*.so"))
    )
    if not extension_dirs:
        raise RuntimeError(f"No compiled RadFoam reference package under {root / 'build'}")
    sys.path[:0] = [str(extension_dirs[-1]), str(root)]
    print("Importing RadFoam before CUDA rendering:", extension_dirs[-1], flush=True)
    import radfoam
    import torch
    from radfoam_model.render import TraceRays

    # Load FoamSurf only AFTER RadFoam has initialized its CUDA heap.
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project))
    import runpy
    test_helpers = runpy.run_path(str(Path(__file__).with_name("test_quantiles_gpu.py")))
    fixture = test_helpers["fixture"]
    render_quantiles = test_helpers["render_quantiles"]
    renderer, scene, rays = fixture(sigma=(.4, .8), far=1000.)
    q = torch.tensor([[.9, .6]], device="cuda")
    depth, _ = render_quantiles(renderer, scene, rays, q)
    depth.sum().backward()
    torch.cuda.synchronize()
    density = scene.densities.detach().clone().requires_grad_()
    attributes = torch.cat(((scene.colors.detach()-.5)/.28209479177387814, density), -1)
    topology = renderer._voronoi_topology
    output = TraceRays.apply(
        radfoam.create_pipeline(0, torch.float32), scene.sites.detach(), attributes,
        topology.adjacency, topology.offsets, rays,
        torch.zeros(1, dtype=torch.uint32, device="cuda"), q, False,
    )
    reference_depth = output[1]
    torch.testing.assert_close(depth, reference_depth, atol=2e-4, rtol=2e-4)
    reference_depth.sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(scene.densities.grad, density.grad, atol=2e-3, rtol=2e-3)
    print("RADFOAM_REFERENCE_PASSED", flush=True)


if __name__ == "__main__":
    main()
