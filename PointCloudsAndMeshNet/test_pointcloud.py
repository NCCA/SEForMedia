import unittest
import inspect
from tempfile import TemporaryDirectory

import numpy as np
import torch

try:
    from PointCloudForML import pointcloud as pc
except ImportError:
    pc = None


class PointCloudTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            pc, "The point-cloud lesson helpers have not been implemented"
        )

    def test_cloud_view_is_interactive_and_preserves_part_names_and_points(self):
        self.assertIn("labels", inspect.signature(pc.cloud_figure).parameters)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        figure = pc.cloud_figure(
            points, "Parts", np.array([0, 1, 0]), labels=["seat", "back"]
        )
        self.assertEqual(
            [trace.type for trace in figure.data], ["scatter3d", "scatter3d"]
        )
        self.assertEqual([trace.name for trace in figure.data], ["seat", "back"])
        np.testing.assert_array_equal(figure.data[0].x, [0, 0])
        self.assertEqual(figure.layout.scene.dragmode, "orbit")
        self.assertEqual(figure.layout.scene.camera.up.y, 1)

    def test_normalisation_preserves_shape_and_handles_degenerate_clouds(self):
        points = np.array([[1, 2, 3], [3, 2, 3], [2, 5, 3]], dtype=float)
        result = pc.normalise(points)
        np.testing.assert_allclose(result.mean(0), 0, atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(result, axis=1).max()), 1)
        np.testing.assert_array_equal(pc.normalise(np.ones((4, 3))), np.zeros((4, 3)))
        with self.assertRaises(ValueError):
            pc.normalise(np.empty((0, 3)))

    def test_surface_sampling_weights_faces_by_area(self):
        triangles = np.array(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 2], [2, 0, 2], [0, 2, 2]]],
            dtype=float,
        )
        points, faces = pc.sample_triangles(triangles, 10000, seed=7)
        self.assertLess(abs((faces == 1).mean() - 0.8), 0.02)
        self.assertTrue(np.all(points[:, :2] >= 0))
        self.assertTrue(
            np.all(points[:, :2].sum(1) <= np.where(faces == 0, 1, 2) + 1e-6)
        )

    def test_classifier_is_invariant_and_accepts_different_point_counts(self):
        torch.manual_seed(3)
        model = pc.PointNet(3).eval()
        points = torch.randn(2, 19, 3)
        order = torch.randperm(19)
        with torch.no_grad():
            torch.testing.assert_close(model(points), model(points[:, order]))
            self.assertEqual(model(points[:, :7]).shape, (2, 3))

    def test_segmentation_follows_the_input_permutation(self):
        torch.manual_seed(3)
        model = pc.SegmentNet(4).eval()
        points = torch.randn(2, 19, 3)
        order = torch.randperm(19)
        with torch.no_grad():
            torch.testing.assert_close(
                model(points)[:, :, order], model(points[:, order])
            )
        self.assertEqual(model(points).shape, (2, 4, 19))

    def test_rotation_preserves_distances(self):
        points = torch.randn(5, 21, 3)
        rotated = pc.rotate(points, torch.Generator().manual_seed(2), full=True)
        torch.testing.assert_close(
            torch.cdist(points, points),
            torch.cdist(rotated, rotated),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_generated_data_and_part_labels_have_matching_dimensions(self):
        x, y, names = pc.make_shapes(6, 32, 2)
        self.assertEqual(x.shape, (18, 32, 3))
        self.assertEqual(y.bincount().tolist(), [6, 6, 6])
        self.assertEqual(len(names), 3)
        x, y, names = pc.make_parts(5, 48, 2)
        self.assertEqual(x.shape, (5, 48, 3))
        self.assertEqual(y.shape, (5, 48))
        self.assertEqual(set(y.unique().tolist()), {0, 1, 2})

    def test_iou_is_computed_per_shape_with_an_explicit_absent_part_rule(self):
        target = torch.tensor([[0, 0, 1, 1]])
        prediction = torch.tensor([[0, 1, 1, 1]])
        scores = pc.part_iou(prediction, target, 3)
        torch.testing.assert_close(scores, torch.tensor([[0.5, 2 / 3, 1.0]]))

    def test_incomplete_shapenet_cache_is_rejected_before_import_or_download(self):
        with TemporaryDirectory() as root:
            try:
                pc.load_shapenet(root, "train", 32, 4, 42)
            except Exception as error:
                self.assertIsInstance(
                    error,
                    ValueError,
                    "Reject an incomplete cache before importing the optional library",
                )
            else:
                self.fail("An incomplete cache should be rejected")

    def test_split_keeps_every_class_and_never_reuses_a_shape(self):
        labels = torch.tensor([0] * 10 + [1] * 10)
        train, val = pc.split_indices(labels, seed=4)
        self.assertFalse(set(train.tolist()) & set(val.tolist()))
        self.assertEqual(len(train) + len(val), 20)
        self.assertEqual(labels[val].bincount().tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()
