#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 1: surface sampling

    A mesh describes a surface using vertices and faces. This is useful for modelling and rendering, but for this lesson we want to prepare data for a network that works with points. We will represent each object as a point cloud: a collection of positions stored in a NumPy array with shape `(N, 3)`, where each row contains an `x`, `y` and `z` coordinate.

    This gives us a common representation for meshes with different numbers of vertices and faces. We can sample the same number of points from each object, making it easier to group examples into training batches. Point clouds are also used for data captured by depth cameras and laser scanners, so working with them connects our mesh examples to other sources of 3D data. We lose the connections between vertices when we make this conversion; the network must learn about the shape from the spatial arrangement of the points.

    We could use the mesh vertices directly, but their distribution reflects how the mesh was built. One part of an object might contain many small triangles whilst another uses a few large ones. Sampling only the vertices would give the more densely modelled regions greater representation, even if they cover very little of the surface.

    Instead, we will sample points across the triangle surfaces. We choose triangles in proportion to their area, then sample uniformly within each chosen triangle. This gives equal areas of the surface the same chance of contributing points, regardless of how they have been divided into faces. Larger triangles will therefore contribute more points on average.

    We will then centre the cloud and scale it using one scale factor for all three axes. For this lesson, we want the network to work with shape without also having to account for differences in position or overall size. Using a single scale factor preserves the object's proportions. This preparation removes information about location and absolute size, so we would need to keep that information separately for a task where it matters.

    We will first sample two triangles where we can check the answer by hand, then try a mesh with [trimesh](https://trimesh.org/). By the end we should be able to explain area weighted sampling, distinguish surface points from mesh vertices, and normalise a cloud without changing its proportions.

    We use NumPy operations from [NumPy for ML](../NumPyForML/README.md). The first examples need no downloaded assets. We use [Plotly](https://plotly.com/python/3d-scatter-plots/) to rotate the views: drag to orbit, use the wheel to zoom, and use the toolbar to reset the camera. These controls change the view, rather than the coordinates passed to our model.
    """)
    return


@app.cell
def _():
    import inspect
    import sys
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # This also works when marimo starts from the notebook's own folder.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # this module is one I have written to handle point cloud operations.
    import pointcloud as pc

    return Path, go, inspect, make_subplots, mo, np, pc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unequal triangles

    The NumPy array below stores the vertex positions of two triangles. The second triangle has four times the area of the first, so it should receive about four fifths of the sampled points. The smaller triangle should receive the remaining fifth. These proportions will vary for a finite random sample, but should get closer to the expected values as we sample more points.

    Here, *uniform surface sampling* means that any two regions with the same surface area have the same probability of receiving a point. To achieve this we need to make two choices: which triangle to sample, and where to place the point within that triangle.

    We choose each triangle with a probability proportional to its area:

    \[
    P(\text{triangle } i) = \frac{A_i}{\sum_j A_j}.
    \]

    Selecting faces with equal probability would put about half the points on each triangle. As the smaller triangle has only a quarter of the area, it would then have four times the point density. Our cloud would reflect the way the surface was divided into triangles, rather than giving an even sample of the surface.

    Once we have selected a triangle, we use *barycentric coordinates* to place a point inside it. For vertices \(\mathbf{a}\), \(\mathbf{b}\) and \(\mathbf{c}\), we write the point as a weighted combination:

    \[
    \mathbf{p} = w_a\mathbf{a} + w_b\mathbf{b} + w_c\mathbf{c},
    \qquad
    w_a + w_b + w_c = 1.
    \]

    When all three weights are non-negative, the point lies inside the triangle or on its boundary. A weight of one selects the corresponding vertex, whilst three weights of \(1/3\) give the triangle's centroid. As the weights must sum to one, we only need to generate two values; the third follows from them.

    We start with two independent random values \(u\) and \(v\), each uniformly distributed between zero and one. Valid coordinates must satisfy \(u + v \leq 1\), which covers half of the unit square. If \(u + v > 1\), we reflect the pair into the valid half:

    \[
    u \leftarrow 1-u,
    \qquad
    v \leftarrow 1-v.
    \]

    We then calculate the point using:

    \[
    \mathbf{p}
    = (1-u-v)\mathbf{a} + u\mathbf{b} + v\mathbf{c}
    = \mathbf{a} + u(\mathbf{b}-\mathbf{a}) + v(\mathbf{c}-\mathbf{a}).
    \]

    The reflection folds the two halves of the square onto the same triangular region without changing their area. This gives a uniform distribution in barycentric coordinates. Mapping that region onto our mesh triangle scales every small region by the same area factor, so the resulting points are also uniformly distributed across the triangle's surface.

    Generating three independent uniform values and dividing each by their sum also produces valid barycentric weights, but it does not produce a uniform surface sample. That method favours points towards the centre of the triangle. The reflection method gives us both the required constraint on the weights and the distribution we want.


    <img src="public/USS.png" width="80%" />
    """)
    return


@app.cell
def _(np):
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
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        / 2
    )
    print("Face areas:", areas)
    print("Expected probabilities:", areas / areas.sum())
    print("Sampler implementation:")
    return (triangles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following function (from point_cloud.py) implements the processes outlined above to generate sample points for the triangles.
    """)
    return


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
def _(np, pc, point_count, triangles):
    samples, face_ids = pc.sample_triangles(triangles, point_count.value, seed=7)
    print("Cloud shape and dtype:", samples.shape, samples.dtype)
    print(
        "Observed face proportions:",
        np.bincount(face_ids, minlength=2) / len(face_ids),
    )
    return face_ids, samples


@app.cell
def _(face_ids, go, mo, samples, triangles):
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
            aspectmode="data",
            dragmode="orbit",
            camera=dict(up=dict(x=0, y=1, z=0)),
        ),
    )
    mo.ui.plotly(sampling_view, config={"displaylogo": False, "scrollZoom": True})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Centre and scale

    Our meshes may have different positions, sizes and modelling units. One object might be centred at the origin whilst another is offset, and the same shape could be stored in metres or millimetres. For this lesson we want to compare shapes, so we will move each cloud to a common centre and give it a common overall scale.

    We first calculate the *point centroid*, the mean position of all \(N\) sampled points:

    \[
    c=\frac{1}{N}\sum_{i=1}^{N}x_i.
    \]

    Here, each \(x_i\) is a three-dimensional position, so \(c\) contains the mean of the \(x\), \(y\) and \(z\) coordinates. Subtracting \(c\) from every point translates the whole cloud so that its centroid is at the origin:

    \[
    x_i'=x_i-c.
    \]

    This changes the cloud's position without changing the distances between its points. The centroid does not have to lie on the surface; for a hollow or concave object it may lie in empty space.

    Next, we find the largest [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) from the centroid to any sampled point:

    \[
    r=\max_i\|x_i-c\|.
    \]

    We use this distance as a single scale factor and divide every centred point by it:

    \[
    \hat{x}_i=\frac{x_i-c}{r}.
    \]

    The most distant sampled point now has distance one from the origin, and every other point has distance at most one. This places the cloud **inside the unit sphere**. We are scaling the whole object; we are not projecting each point onto the sphere's surface. Dividing each point by its own distance would force all points onto the sphere and remove the radial variation that helps describe the shape.

    Using the same scalar \(r\) for all three coordinates preserves proportions, angles and relative distances. A tall, narrow object remains tall and narrow. If we divided each coordinate by its own range, we would stretch the axes by different amounts. For example, a rectangular box could become a cube, changing the shape we want the network to learn.

    Both the centroid and the radius are calculated from the sampled cloud. Resampling the same mesh will therefore produce slightly different values. With uniform surface sampling, the point centroid estimates the surface's area-weighted centroid, which may differ from the mean of the mesh vertices or the centroid of its enclosed volume. The radius also depends on whether our sample includes points close to the most distant part of the surface. The unit-sphere guarantee applies to the sampled points.

    Finally, this normalisation removes absolute position and size. That is useful when those quantities are irrelevant to our task, but we should retain \(c\) and \(r\) if we need to recover the original coordinates:

    \[
    x_i=r\hat{x}_i+c.
    \]

    If all points occupy the same position, then \(r=0\) and we cannot divide by it. Our implementation must handle that case explicitly.
    """)
    return


@app.cell
def _(np, pc, samples):
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
def _(face_ids, go, mo, normalised):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and sample a mesh



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


@app.cell(hide_code=True)
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
def _(Path, mesh_form, mo):
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
    return mesh, mesh_faces, mesh_points


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
def _(
    display_points,
    display_vertices,
    go,
    make_subplots,
    mesh,
    mesh_faces,
    mo,
    np,
):
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
def _(go, mesh_comparison, mesh_scene, mo):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Predict the face proportions if the second triangle's side lengths are tripled.
    2. Compare 32 and 4096 samples. Which details can a network no longer see?
    3. Translate and uniformly scale a cloud before normalisation. Compare the result.
    4. Explain why normalisation does not remove orientation.
    5. Explain why moving the camera does not test rotation robustness of a model.

    [Part 2](PointCloudForMLPart2Sets.py) investigates what happens when we shuffle the points.
    """)
    return


if __name__ == "__main__":
    app.run()
