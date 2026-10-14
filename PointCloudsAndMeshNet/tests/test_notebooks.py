"""
Runs the lesson notebooks headless, including the submitted training forms.

marimo lets me replace any cell's definitions when running an app, so a form
is "submitted" by passing a ``SimpleNamespace`` with the ``value`` the form
would have returned.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def submitted(**settings) -> SimpleNamespace:
    return SimpleNamespace(value=settings)


@pytest.mark.parametrize("part", range(1, 7))
def test_lesson_runs_by_default_without_training_or_optional_data(run_lesson, part):
    definitions = run_lesson(part)
    assert "trained_model" not in definitions
    assert "segmentation_model" not in definitions
    assert "comparison_models" not in definitions


def test_sampling_view_uses_matching_mesh_faces_and_cloud_coordinates(run_lesson):
    pytest.importorskip("plotly")
    pytest.importorskip("trimesh")
    definitions = run_lesson(1, {"mesh_form": submitted(path="", count=256)})
    mesh = definitions["mesh"]
    centre = definitions["mesh_points"].mean(axis=0)
    radius = np.linalg.norm(definitions["mesh_points"] - centre, axis=1).max()
    np.testing.assert_allclose(
        definitions["display_vertices"], (mesh.vertices - centre) / radius
    )
    np.testing.assert_allclose(
        definitions["display_points"], (definitions["mesh_points"] - centre) / radius
    )
    for figure in (definitions["mesh_comparison"], definitions["mesh_overlay"]):
        assert [trace.type for trace in figure.data] == ["mesh3d", "scatter3d"]
        np.testing.assert_array_equal(figure.data[0].i, mesh.faces[:, 0])
        np.testing.assert_allclose(
            figure.data[1].x, definitions["display_points"][:, 0]
        )
        assert figure.layout.scene.dragmode == "orbit"


def test_feature_channels_and_contributors_are_plotly_views(run_lesson):
    figure = run_lesson(3)["feature_plot"]
    assert [trace.type for trace in figure.data] == ["scatter3d"] * 3
    assert figure.data[2].marker.showscale


@pytest.mark.slow
def test_classifier_learns_and_reports_a_matching_confusion_matrix(run_lesson):
    definitions = run_lesson(
        4,
        {
            "training_form": submitted(
                root="", points=128, limit=24, epochs=12, lr=0.003, seed=42
            )
        },
    )
    history = definitions["history"]
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert definitions["test_logits"].shape == (36, 3)
    assert torch.isfinite(definitions["test_logits"]).all()
    expected = torch.bincount(
        definitions["test_y"] * 3 + definitions["predictions"], minlength=9
    ).reshape(3, 3)
    plot = definitions["confusion_plot"]
    np.testing.assert_array_equal(plot.data[0].z, expected.numpy())
    assert plot.layout.yaxis.autorange == "reversed"


@pytest.mark.slow
def test_robustness_shuffle_matches_clean_and_plot_shows_the_rows(run_lesson):
    definitions = run_lesson(
        5,
        {
            "training_form": submitted(
                root="", points=128, limit=24, epochs=30, lr=0.003, seed=42
            )
        },
    )
    rows = definitions["robustness_rows"]

    def accuracy(model: str, condition: str) -> float:
        return next(
            row["accuracy"]
            for row in rows
            if row["model"] == model and row["condition"] == condition
        )

    for model in definitions["comparison_models"]:
        assert accuracy(model, "Clean") == accuracy(model, "Shuffle")
    assert all(0 <= row["accuracy"] <= 1 for row in rows)
    for trace in definitions["robustness_plot"].data:
        condition = "Noise" if trace.xaxis == "x" else "Points"
        expected = [
            row
            for row in rows
            if row["model"] == trace.name and row["condition"] == condition
        ]
        np.testing.assert_allclose(
            trace.error_y.array, [row["std"] for row in expected]
        )
        np.testing.assert_allclose(trace.y, [row["accuracy"] for row in expected])


def test_segmentation_loss_demo_runs_before_training(run_lesson):
    definitions = run_lesson(6)
    torch.testing.assert_close(
        definitions["point_losses"], definitions["manual_point_losses"]
    )
    torch.testing.assert_close(
        definitions["point_losses"].mean(), definitions["flattened_point_loss"]
    )


@pytest.mark.slow
def test_segmentation_checkpoint_reloads_the_same_predictions(run_lesson, tmp_path):
    destination = tmp_path / "segmentation.pt"
    definitions = run_lesson(
        6,
        {
            "segmentation_form": submitted(
                root="", points=192, limit=24, epochs=2, lr=0.003, seed=42
            ),
            "seg_checkpoint_path": SimpleNamespace(value=str(destination)),
            "seg_save_button": SimpleNamespace(value=True),
        },
    )
    assert destination.is_file()
    checkpoint = torch.load(destination, weights_only=True)
    assert checkpoint["parts"] == definitions["seg_names"]
    test_x = definitions["seg_test_x"]
    with torch.no_grad():
        torch.testing.assert_close(
            definitions["reloaded_segmentation_model"](test_x),
            definitions["segmentation_model"](test_x),
        )


@pytest.mark.slow
def test_segmentation_learns_aligned_point_labels(run_lesson):
    definitions = run_lesson(
        6,
        {
            "segmentation_form": submitted(
                root="", points=192, limit=24, epochs=12, lr=0.003, seed=42
            )
        },
    )
    assert definitions["seg_predictions"].shape == definitions["seg_test_y"].shape
    assert definitions["seg_ious"].mean().item() > 0.7
    view = definitions["segmentation_plot"]
    for scene in (view.layout.scene, view.layout.scene2, view.layout.scene3):
        assert scene.dragmode == "orbit"
        assert scene.xaxis.range == (-1.1, 1.1)
    assert [trace.name for trace in view.data[:3]] == definitions["seg_names"]
    assert [trace.name for trace in view.data[-2:]] == ["Correct", "Incorrect"]
    assert all(trace.type == "scatter3d" for trace in view.data)
