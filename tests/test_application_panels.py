"""CPU checks of paper illumination, placement, and the figure reproduction workflow."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from apps.raytrace.sun import Sun, WATER_IOR, splat, gaussian_blur_periodic
from apps.raytrace.tracer import SceneTracer
from apps.raytrace.shading import Sphere, TraceSettings

ROOT = Path(__file__).resolve().parents[1]


def test_sun_refraction_obeys_snell_and_surface_intersections():
    sun = Sun(up=(0., 0., 1.), elevation_deg=30., surface=2.)
    air, water = sun.air_direction('cpu'), sun.water_direction('cpu')
    assert torch.linalg.vector_norm(water).item() == pytest.approx(1.)
    assert water[2] < 0
    assert torch.linalg.vector_norm(air[:2]).item() == pytest.approx(WATER_IOR * torch.linalg.vector_norm(water[:2]).item())
    rays = torch.tensor([[0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., -1.], [0., 0., 3., 0., 0., 1.]])
    hits = sun.surface_hit(rays)
    assert hits[0] == 2. and torch.isinf(hits[1:]).all()
    flat = Sun(up=(0., 0., 1.), slope=0.)
    normals = flat.surface_normal(torch.zeros(3, 3), torch.ones(3))
    torch.testing.assert_close(normals, torch.tensor([[0., 0., 1.]]).expand(3, 3))


def test_caustic_splat_and_blur_conserve_energy():
    field = torch.zeros(16)
    splat(field, torch.tensor([[-.2, 3.8], [2.3, 1.2]]), torch.tensor([2., 3.]), 4)
    assert field.sum().item() == pytest.approx(5.)
    blurred = gaussian_blur_periodic(field.reshape(1, 4, 4), torch.tensor([.8]))
    assert blurred.sum().item() == pytest.approx(5., abs=1e-5)


def test_environment_restores_sun_when_tracing_fails():
    tracer = object.__new__(SceneTracer)
    sun = tracer.sun = object()
    tracer._environment = {}
    tracer.nearest_sites = lambda x: torch.zeros(len(x), dtype=torch.long)
    def trace(*args, **kwargs):
        assert tracer.sun is None
        raise ValueError('test interruption')
    tracer.trace = trace
    with pytest.raises(ValueError, match='interruption'):
        tracer.environment(torch.zeros(3), 8, TraceSettings())
    assert tracer.sun is sun


def test_world_size_placement_preserves_requested_size_and_tilt(monkeypatch):
    import insert
    tracer = SimpleNamespace(trainer=SimpleNamespace(device='cpu'))
    anchor = SimpleNamespace(origin=torch.zeros(3))
    monkeypatch.setattr(insert, 'seabed_point', lambda *args: torch.tensor([0., 0., 20.]))
    monkeypatch.setattr(insert, 'load_manifest', lambda *args, **kwargs:
                        (np.zeros((3, 3)), np.array([[0, 1, 2]]), np.array([0]), [SimpleNamespace()], ['hull'], None))
    captured = {}
    def place(*args):
        captured['fraction'], captured['tilt'] = args[4], args[-1]
        return torch.zeros(3, 3), {}
    monkeypatch.setattr(insert, 'place_mesh_on_seabed', place)
    monkeypatch.setattr(insert, 'Mesh', lambda *args: object())
    spec = {'type': 'mesh', 'manifest': 'fixture', 'u': .5, 'v': .5, 'size': 1.2, 'max_tilt_deg': 12.}
    insert.build_objects(tracer, anchor, [spec])
    assert captured == {'fraction': .06, 'tilt': 12.}
    assert spec['size'] == 1.2 and 'span_fraction' not in spec


def test_figure_plan_uses_derived_models_and_supports_scene_selection(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import reproduce_applications as apps
    monkeypatch.setenv('FS_EDIT_PACKAGE', str(tmp_path / 'editing'))
    monkeypatch.setenv('FS_FIGURE_PACKAGE', str(tmp_path / 'figure'))
    args = SimpleNamespace(applications='figure', scenes='', out=tmp_path)
    commands = apps.plan(args)
    assert len(commands) == 4
    expected = ['models/style/iui3_both', 'models/edits/bulge', 'models/style/curasao_both', 'models/jg_haze/model']
    for command, model in zip(commands, expected):
        assert command[command.index('--run') + 1].endswith(model)
        shot = json.loads((ROOT / command[command.index('--shot') + 1]).read_text())
        TraceSettings(**shot['trace'])
        assert 'anchor_view' in shot and '--still' in command
        for obj in shot['objects']:
            if 'manifest' in obj:
                assert obj['manifest'].startswith('models/') and '$' not in obj['manifest']
    args.scenes = 'IUI3'
    assert len(apps.plan(args)) == 2


def test_reference_comparison_rejects_wrong_shape_and_wrong_pixels(monkeypatch, tmp_path):
    from PIL import Image
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from compare_application_panels import compare
    a, b = tmp_path / 'a.png', tmp_path / 'b.png'
    Image.new('RGB', (8, 8), (128, 64, 32)).save(a)
    Image.new('RGB', (8, 8), (128, 64, 32)).save(b)
    assert compare(a, b)['exact']
    Image.new('RGB', (8, 8), (0, 0, 0)).save(b)
    assert not compare(a, b)['passed']
    Image.new('RGB', (9, 8)).save(b)
    assert 'shapes differ' in compare(a, b)['error']


def test_lamp_reach_attenuates_spherical_lights_and_dispatches_sun():
    tracer = object.__new__(SceneTracer)
    tracer.water_coefficients = lambda: (torch.full((3,), 0.2), None)
    points = torch.zeros(1, 3)
    lamp = Sphere(center=(0., 0., 2.), radius=0.1, power=4.)
    direction, irradiance = tracer.lamp_reach(points, lamp, [], visibility=False)
    torch.testing.assert_close(direction, torch.tensor([[0., 0., 1.]]))
    torch.testing.assert_close(irradiance, torch.full((1, 3), math.exp(-0.4)))
    calls = []
    def reach(scene, positions, objects):
        calls.append((scene, positions, objects))
        return direction, irradiance
    sun = SimpleNamespace(reach=reach)
    actual = tracer.lamp_reach(points, sun, [])
    assert len(calls) == 1 and calls[0][0] is tracer
    assert calls[0][1] is points
    torch.testing.assert_close(actual, (direction, irradiance))
