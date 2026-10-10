#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 1: surface sampling

    We will turn a mesh into a collection of points, then centre and scale those
    points ready for a network. We will first sample two triangles where we can
    check the answer by hand, then try a mesh with [trimesh](https://trimesh.org/).

    By the end we should be able to explain area-weighted sampling, distinguish
    surface points from mesh vertices, and normalise a cloud without changing its proportions.
    We use NumPy operations from [NumPy for ML](../NumPyForML/README.md).
    The first examples need no downloaded assets. We use
    [Plotly](https://plotly.com/python/3d-scatter-plots/) to rotate the views:
    drag to orbit, use the wheel to zoom, and use the toolbar to reset the camera.
    These controls change the view, rather than the coordinates passed to our model.

    Run from the repository root with the optional plotting and mesh dependencies:

    ```bash
    uv run --with plotly --with 'trimesh>=4.6' marimo edit PointCloudForML/PointCloudForMLPart1Sampling.py
    ```
    """)
    return


@app.cell
def _():
    import inspect
    import sys
    from pathlib import Path
    import marimo as mo
    import numpy as np

    # This also works when marimo starts from the notebook's own folder.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pointcloud as pc

    return (inspect, mo, np, pc, Path)


@app.cell
def _(mo):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError:
        mo.stop(
            True,
            mo.md(
                "Plotly is missing. Run this notebook using the uv command above to enable the interactive views."
            ),
        )
    return go, make_subplots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unequal triangles

    Our second triangle has four times the area of the first. Uniform surface
    sampling should therefore put about four fifths of the points on it.
    Selecting a face uniformly would put about half on each, over-sampling the small triangle.

    After selecting a face we use two barycentric coordinates. If their sum is
    larger than one, reflecting them back gives a point inside the triangle.
    This avoids the bias introduced by normalising three independent uniform weights.
    """)
    return


@app.cell
def _(np, pc):
    triangles = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 2], [2, 0, 2], [0, 2, 2]],
        ],
        dtype=float,
    )
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    print("Face areas:", areas)
    print("Expected probabilities:", areas / areas.sum())
    print("Sampler implementation:")
    return (triangles,)


@app.cell
def _(inspect, mo, pc):
    mo.md("```python\n" + inspect.getsource(pc.sample_triangles) + "\n```")
    return


@app.cell
def _(mo):
    point_count = mo.ui.slider(
        32, 4096, value=512, step=32, label="Sampled points", show_value=True
    )
    point_count
    return (point_count,)


@app.cell
def _(pc, point_count, triangles, np):
    samples, face_ids = pc.sample_triangles(triangles, point_count.value, seed=7)
    print("Cloud shape and dtype:", samples.shape, samples.dtype)
    print(
        "Observed face proportions:", np.bincount(face_ids, minlength=2) / len(face_ids)
    )
    return samples, face_ids


@app.cell
def _(go, triangles, samples, face_ids, mo):
    _vertices = triangles.reshape(-1, 3)
    sampling_view = go.Figure(
        [
            go.Mesh3d(
                x=_vertices[:, 0],
                y=_vertices[:, 1],
                z=_vertices[:, 2],
                i=[0, 3],
                j=[1, 4],
                k=[2, 5],
                color="lightgrey",
                opacity=0.3,
                name="Original triangles",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=samples[:, 0],
                y=samples[:, 1],
                z=samples[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=face_ids,
                    colorscale=[
                        [0, "#1f77b4"],
                        [0.5, "#1f77b4"],
                        [0.5, "#ff7f0e"],
                        [1, "#ff7f0e"],
                    ],
                    cmin=0,
                    cmax=1,
                ),
                name="Surface samples",
                customdata=face_ids,
                hovertemplate="Face %{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            ),
        ]
    )
    sampling_view.update_layout(
        title="Unequal triangles and their surface samples",
        height=550,
        scene=dict(
            aspectmode="data", dragmode="orbit", camera=dict(up=dict(x=0, y=1, z=0))
        ),
    )
    mo.ui.plotly(sampling_view, config={"displaylogo": False, "scrollZoom": True})
    return (sampling_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Centre and scale

    Subtract the point centroid, then divide by the largest distance from it:

    $$c=\frac{1}{N}\sum_i x_i,\qquad r=\max_i\|x_i-c\|,\qquad \hat{x}_i=\frac{x_i-c}{r}.$$

    This places the cloud **inside** the unit sphere; it does not project every
    point onto the sphere. One scalar scale factor preserves proportions.
    Dividing each coordinate by its own range would distort the shape.
    The sampled centroid can move slightly when we resample the same mesh.
    """)
    return


@app.cell
def _(samples, np, pc):
    centred = samples - samples.mean(axis=0)
    radius = np.linalg.norm(centred, axis=1).max()
    normalised = centred / radius
    print("Centroid:", normalised.mean(axis=0).round(7))
    print("Largest radius:", np.linalg.norm(normalised, axis=1).max())
    print("Zero extent handled:", pc.normalise(np.ones((3, 3))))
    try:
        pc.normalise(np.empty((0, 3)))
    except ValueError as _error:
        print("Empty cloud:", _error)
    return (normalised,)


@app.cell
def _(normalised, face_ids, go, mo):
    normalised_view = go.Figure(
        go.Scatter3d(
            x=normalised[:, 0],
            y=normalised[:, 1],
            z=normalised[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=face_ids,
                colorscale=[
                    [0, "#1f77b4"],
                    [0.5, "#1f77b4"],
                    [0.5, "#ff7f0e"],
                    [1, "#ff7f0e"],
                ],
                cmin=0,
                cmax=1,
            ),
            name="Normalised samples",
        )
    )
    normalised_view.update_layout(
        title="Centred and scaled surface samples",
        height=550,
        scene=dict(
            aspectmode="cube",
            dragmode="orbit",
            xaxis=dict(range=[-1.1, 1.1]),
            yaxis=dict(range=[-1.1, 1.1]),
            zaxis=dict(range=[-1.1, 1.1]),
            camera=dict(up=dict(x=0, y=1, z=0)),
        ),
    )
    mo.ui.plotly(normalised_view, config={"displaylogo": False, "scrollZoom": True})
    return (normalised_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and sample a mesh

    Leave the path blank to use a generated box or enter an OBJ, STL or OFF
    mesh path. Run this notebook with the optional dependencies:

    ```bash
    uv run --with plotly --with 'trimesh>=4.6' marimo edit PointCloudForML/PointCloudForMLPart1Sampling.py
    ```

    [`sample_surface(mesh, count, seed=...)`](https://trimesh.org/trimesh.sample.html)
    weights faces by area and returns both points and face indices. Its samples
    are random, rather than evenly spaced. We keep the seed fixed whilst changing N.
    No file is read until we press the button. Submit once with the blank path,
    then change the point count and submit again. Does the sparse cloud still
    show the thin dimension of the box?

    We calculate centring and scale from the sampled points and apply that
    **same transform** to the original mesh vertices. Normalising the mesh and
    cloud separately could move them apart. Original vertices may extend
    slightly outside the sampled cloud's unit sphere, especially with few points.
    """)
    return


@app.cell
def _(mo):
    mesh_form = mo.ui.dictionary(
        {
            "path": mo.ui.text(value="", label="Mesh path (blank for a generated box)"),
            "count": mo.ui.number(32, 8192, value=1024, label="Points"),
        }
    ).form(submit_button_label="Sample mesh")
    mesh_form
    return (mesh_form,)


@app.cell
def _(mesh_form, mo, Path):
    mo.stop(mesh_form.value is None, mo.md("Submit the form to sample a mesh."))
    try:
        import trimesh
    except ModuleNotFoundError:
        mo.stop(
            True,
            mo.md(
                "trimesh is missing. Use the uv command above to enable this example."
            ),
        )
    _mesh_path = mesh_form.value["path"].strip()
    try:
        mesh = (
            trimesh.load_mesh(Path(_mesh_path).expanduser(), process=False)
            if _mesh_path
            else trimesh.creation.box(extents=[1, 2, 0.5])
        )
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("Select a single mesh with triangle faces")
        mesh_points, mesh_faces = trimesh.sample.sample_surface(
            mesh, int(mesh_form.value["count"]), seed=42
        )
    except (ValueError, OSError) as _error:
        mo.stop(True, mo.md(f"Could not sample this mesh: {_error}"))
    print("Mesh vertices:", len(mesh.vertices), "faces:", len(mesh.faces))
    print("Sampled surface points:", mesh_points.shape)
    return mesh, mesh_points, mesh_faces


@app.cell
def _(mesh, mesh_points, np):
    mesh_centre = mesh_points.mean(axis=0)
    mesh_radius = np.linalg.norm(mesh_points - mesh_centre, axis=1).max()
    if mesh_radius <= 0:
        raise ValueError("The sampled mesh has zero extent")
    display_points = (mesh_points - mesh_centre) / mesh_radius
    display_vertices = (np.asarray(mesh.vertices) - mesh_centre) / mesh_radius
    print("Cloud centroid:", display_points.mean(axis=0).round(7))
    print("Cloud maximum radius:", np.linalg.norm(display_points, axis=1).max())
    return display_points, display_vertices


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare the original surface with its samples

    The left panel draws the original indexed triangles; the right draws the
    sampled points. Both use the same axis ranges, equal axis scales and starting
    camera. Each panel can be rotated independently. Hover over a point to see
    its source face index.

    In the overlay below, mesh and points share one scene, so dragging rotates
    them together. The translucent mesh lets us inspect whether the points sit
    on the surface. Click either legend entry to hide or show that representation.
    """)
    return


@app.cell
def _(mesh, mesh_faces, display_vertices, display_points, go, make_subplots, np, mo):
    _mesh_trace = go.Mesh3d(
        x=display_vertices[:, 0],
        y=display_vertices[:, 1],
        z=display_vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color="lightsteelblue",
        name="Original mesh",
        flatshading=True,
        hoverinfo="skip",
        showlegend=True,
    )
    _point_trace = go.Scatter3d(
        x=display_points[:, 0],
        y=display_points[:, 1],
        z=display_points[:, 2],
        mode="markers",
        marker=dict(size=3, color="#ff7f0e"),
        name="Surface samples",
        customdata=mesh_faces,
        hovertemplate="Source face %{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    )
    _extent = float(
        max(np.abs(display_vertices).max(), np.abs(display_points).max()) * 1.1
    )
    mesh_scene = dict(
        aspectmode="cube",
        dragmode="orbit",
        xaxis=dict(title="x", range=[-_extent, _extent]),
        yaxis=dict(title="y", range=[-_extent, _extent]),
        zaxis=dict(title="z", range=[-_extent, _extent]),
        camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.5, y=1.2, z=1.5)),
    )
    mesh_comparison = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Original mesh", "Sampled point cloud"),
    )
    mesh_comparison.add_trace(_mesh_trace, row=1, col=1)
    mesh_comparison.add_trace(_point_trace, row=1, col=2)
    mesh_comparison.update_layout(
        scene=mesh_scene,
        scene2=mesh_scene,
        height=600,
        title="One surface, two representations",
        margin=dict(l=0, r=0, b=0, t=70),
    )
    mo.ui.plotly(mesh_comparison, config={"displaylogo": False, "scrollZoom": True})
    return mesh_comparison, mesh_scene


@app.cell
def _(mesh_comparison, mesh_scene, go, mo):
    mesh_overlay = go.Figure(mesh_comparison.data)
    mesh_overlay.update_traces(scene="scene")
    mesh_overlay.update_traces(opacity=0.25, selector=dict(type="mesh3d"))
    mesh_overlay.update_layout(
        scene=mesh_scene,
        title="Rotate mesh and samples together",
        height=600,
        margin=dict(l=0, r=0, b=0, t=50),
    )
    mo.ui.plotly(mesh_overlay, config={"displaylogo": False, "scrollZoom": True})
    return (mesh_overlay,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Predict the face proportions if the second triangle's side lengths are tripled.
    2. Compare 32 and 4096 samples. Which details can a network no longer see?
    3. Translate and uniformly scale a cloud before normalisation. Compare the result.
    4. Explain why normalisation does not remove orientation.
    5. Hide the mesh in the overlay and reduce N. Which surface details disappear?
    6. Explain why moving the camera does not test rotation robustness of a model.

    [Part 2](PointCloudForMLPart2Sets.py) investigates what happens when we shuffle the points.
    """)
    return


if __name__ == "__main__":
    app.run()
