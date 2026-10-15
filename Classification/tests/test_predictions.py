import ast
from pathlib import Path

import pytest
import torch
from torch import nn


@pytest.mark.parametrize(
    ("notebook", "inputs", "expected"),
    [
        ("MultiClassificationMarimo.py", [[-2.0, 3.0], [4.0, -1.0]], [1, 0]),
        ("BinaryClassificationMarimo.py", [[-2.0], [4.0]], [0, 1]),
    ],
)
def test_prediction_display_matches_held_out_targets(notebook, inputs, expected):
    path = Path(__file__).parents[1] / notebook
    tree = ast.parse(path.read_text())
    cell = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, ast.Name) and child.id == "y_preds"
            for child in ast.walk(node)
        )
    )
    X = torch.tensor(inputs)
    model = nn.Linear(X.shape[1], X.shape[1], bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(X.shape[1]))
    targets = torch.tensor(expected)
    namespace = {"torch": torch, "model": model, "y_1": torch.tensor([3, 3])}
    for arg in cell.args.args:
        if arg.arg.startswith("X_"):
            namespace[arg.arg] = X
        elif arg.arg.startswith("y_") and arg.arg != "y_1":
            namespace[arg.arg] = targets
    statements = [node for node in cell.body if not isinstance(node, ast.Return)]
    statements[-1] = ast.Assign(
        targets=[ast.Name(id="displayed", ctx=ast.Store())],
        value=statements[-1].value,
    )
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))

    exec(compile(module, str(path), "exec"), namespace)
    predictions, displayed_targets = namespace["displayed"]

    assert predictions.tolist() == expected
    assert displayed_targets.tolist() == expected
