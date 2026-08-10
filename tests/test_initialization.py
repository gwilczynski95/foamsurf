import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import radfoam
from radfoam_model.scene import RadFoamScene


class FakeTriangulation:
    """Minimal CPU stand-in for tests that do not exercise Delaunay itself."""

    def __init__(self, points):
        self.points = points

    def permutation(self):
        # A non-trivial permutation catches point/attribute alignment bugs.
        return torch.arange(
            self.points.shape[0] - 1,
            -1,
            -1,
            dtype=torch.long,
            device=self.points.device,
        )


def make_args(init_method="auto", **overrides):
    values = {
        "sh_degree": 1,
        "init_points": 16,
        "final_points": 64,
        "activation_scale": 1.0,
        "init_method": init_method,
        "synthetic_bbox_min": [-1.5, -1.0, -0.5],
        "synthetic_bbox_max": [1.5, 1.0, 0.5],
        "synthetic_outer_points": 4,
        "synthetic_outer_scale": 10.0,
        "synthetic_inner_density": 0.25,
        "synthetic_outer_density": -0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class InitializationTest(unittest.TestCase):
    def build_scene(self, args, points=None, colors=None):
        with (
            patch.object(radfoam, "Triangulation", FakeTriangulation),
            patch.object(radfoam, "create_pipeline", return_value=object()),
            patch.object(RadFoamScene, "update_triangulation"),
        ):
            return RadFoamScene(
                args,
                points=points,
                points_colors=colors,
                device=torch.device("cpu"),
            )

    def test_synthetic_bbox_builds_inner_and_outer_populations(self):
        torch.manual_seed(7)
        model = self.build_scene(make_args("synthetic_bbox"))

        self.assertEqual(tuple(model.primal_points.shape), (16, 3))
        self.assertEqual(tuple(model.density.shape), (16, 1))
        self.assertEqual(tuple(model.att_dc.shape), (16, 3))
        self.assertEqual(tuple(model.att_sh.shape), (16, 9))
        self.assertIsNone(model.faces)

        density = model.density.detach().squeeze(-1)
        inner_mask = density == 0.25
        outer_mask = density == -0.5
        self.assertEqual(int(inner_mask.sum()), 12)
        self.assertEqual(int(outer_mask.sum()), 4)

        inner = model.primal_points.detach()[inner_mask]
        bbox_min = torch.tensor([-1.5, -1.0, -0.5])
        bbox_max = torch.tensor([1.5, 1.0, 0.5])
        self.assertTrue(torch.all(inner >= bbox_min))
        self.assertTrue(torch.all(inner <= bbox_max))

    def test_explicit_random_ignores_supplied_point_cloud(self):
        points = torch.zeros(3, 3)
        colors = torch.zeros(3, 3)
        model = self.build_scene(make_args("random"), points, colors)

        self.assertEqual(model.primal_points.shape[0], 16)
        self.assertTrue(
            torch.equal(model.density, torch.zeros_like(model.density))
        )

    def test_auto_dispatches_to_point_cloud_when_available(self):
        points = torch.zeros(3, 3)
        colors = torch.zeros(3, 3)
        args = make_args("auto")

        with (
            patch.object(RadFoamScene, "initialize_from_pcd") as pcd_init,
            patch.object(RadFoamScene, "random_initialize") as random_init,
            patch.object(radfoam, "create_pipeline", return_value=object()),
        ):
            RadFoamScene(args, points, colors, device=torch.device("cpu"))

        pcd_init.assert_called_once_with(points, colors)
        random_init.assert_not_called()

    def test_auto_dispatches_to_random_without_point_cloud(self):
        args = make_args("auto")
        with (
            patch.object(RadFoamScene, "initialize_from_pcd") as pcd_init,
            patch.object(RadFoamScene, "random_initialize") as random_init,
            patch.object(radfoam, "create_pipeline", return_value=object()),
        ):
            RadFoamScene(args, device=torch.device("cpu"))

        random_init.assert_called_once_with()
        pcd_init.assert_not_called()

    def test_pcd_requires_a_point_cloud(self):
        with self.assertRaisesRegex(ValueError, "requires a dataset"):
            self.build_scene(make_args("pcd"))

    def test_unknown_initialization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown init_method"):
            self.build_scene(make_args("not-a-method"))

    def test_synthetic_bbox_validates_configuration(self):
        invalid_overrides = (
            {"synthetic_bbox_min": [0.0, 0.0]},
            {
                "synthetic_bbox_min": [0.0, 0.0, 0.0],
                "synthetic_bbox_max": [1.0, 0.0, 1.0],
            },
            {"synthetic_outer_points": -1},
            {"synthetic_outer_points": 16},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.build_scene(make_args("synthetic_bbox", **overrides))


if __name__ == "__main__":
    unittest.main()
