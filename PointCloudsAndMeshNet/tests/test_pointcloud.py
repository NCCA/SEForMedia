"""Checks the geometry and model helpers in ``pointcloud.py``."""

import numpy as np
import pointcloud as pc
import pytest
import torch


def test_normalise_centres_the_cloud_and_scales_it_into_the_unit_sphere():
    points = np.array([[1, 2, 3], [3, 2, 3], [2, 5, 3]], dtype=float)
    result = pc.normalise(points)
    np.testing.assert_allclose(result.mean(0), 0, atol=1e-7)
    assert np.linalg.norm(result, axis=1).max() == pytest.approx(1)


def test_normalise_maps_a_cloud_of_identical_points_to_the_origin():
    np.testing.assert_array_equal(pc.normalise(np.ones((4, 3))), np.zeros((4, 3)))


def test_normalise_rejects_an_empty_cloud():
    with pytest.raises(ValueError):
        pc.normalise(np.empty((0, 3)))


def test_cloud_figure_draws_one_orbiting_trace_per_named_part():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    figure = pc.cloud_figure(
        points, "Parts", np.array([0, 1, 0]), labels=["seat", "back"]
    )
    assert [trace.type for trace in figure.data] == ["scatter3d", "scatter3d"]
    assert [trace.name for trace in figure.data] == ["seat", "back"]
    np.testing.assert_array_equal(figure.data[0].x, [0, 0])
    assert figure.layout.scene.dragmode == "orbit"
    assert figure.layout.scene.camera.up.y == 1


def test_surface_sampling_picks_faces_in_proportion_to_their_area():
    # the second triangle is twice the size in each direction, so four times
    # the area, which means 4 / 5 of the samples should land on it
    triangles = np.array(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 2], [2, 0, 2], [0, 2, 2]]],
        dtype=float,
    )
    points, faces = pc.sample_triangles(triangles, 10000, seed=7)
    assert (faces == 1).mean() == pytest.approx(0.8, abs=0.02)
    assert np.all(points[:, :2] >= 0)
    assert np.all(points[:, :2].sum(1) <= np.where(faces == 0, 1, 2) + 1e-6)


def test_classifier_output_ignores_the_order_of_the_points():
    torch.manual_seed(3)
    model = pc.PointNet(3).eval()
    points = torch.randn(2, 19, 3)
    order = torch.randperm(19)
    with torch.no_grad():
        torch.testing.assert_close(model(points), model(points[:, order]))


def test_classifier_accepts_any_number_of_points():
    model = pc.PointNet(3).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 7, 3)).shape == (2, 3)


def test_segmentation_scores_follow_the_input_permutation():
    torch.manual_seed(3)
    model = pc.SegmentNet(4).eval()
    points = torch.randn(2, 19, 3)
    order = torch.randperm(19)
    with torch.no_grad():
        torch.testing.assert_close(model(points)[:, :, order], model(points[:, order]))
        assert model(points).shape == (2, 4, 19)


def test_rotation_preserves_the_distances_between_points():
    points = torch.randn(5, 21, 3)
    rotated = pc.rotate(points, torch.Generator().manual_seed(2), full=True)
    torch.testing.assert_close(
        torch.cdist(points, points), torch.cdist(rotated, rotated), atol=1e-5, rtol=1e-5
    )


def test_generated_shapes_are_balanced_across_the_three_classes():
    x, y, names = pc.make_shapes(6, 32, 2)
    assert x.shape == (18, 32, 3)
    assert y.bincount().tolist() == [6, 6, 6]
    assert len(names) == 3


def test_generated_parts_label_every_point_with_one_of_three_parts():
    x, y, _ = pc.make_parts(5, 48, 2)
    assert x.shape == (5, 48, 3)
    assert y.shape == (5, 48)
    assert set(y.unique().tolist()) == {0, 1, 2}


def test_part_iou_is_per_shape_and_scores_an_absent_part_as_one():
    target = torch.tensor([[0, 0, 1, 1]])
    prediction = torch.tensor([[0, 1, 1, 1]])
    scores = pc.part_iou(prediction, target, 3)
    torch.testing.assert_close(scores, torch.tensor([[0.5, 2 / 3, 1.0]]))


def test_incomplete_shapenet_cache_is_rejected_before_any_download(tmp_path):
    # a ValueError here (rather than ImportError) shows the cache was checked
    # before the optional torch_geometric import was attempted
    with pytest.raises(ValueError):
        pc.load_shapenet(tmp_path, "train", 32, 4, 42)


def test_split_keeps_every_class_and_never_reuses_a_shape():
    labels = torch.tensor([0] * 10 + [1] * 10)
    train, val = pc.split_indices(labels, seed=4)
    assert not set(train.tolist()) & set(val.tolist())
    assert len(train) + len(val) == 20
    assert labels[val].bincount().tolist() == [2, 2]
