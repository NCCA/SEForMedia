import importlib.util
from pathlib import Path

import polars as pl


def test_cleaning_and_solutions(tmp_path: Path) -> None:
    """Run the notebook against missing values and ancient artwork dates."""
    path = Path(__file__).resolve().parents[1] / "IntroductionToPolarsMarimo.py"
    assert path.exists(), "Polars notebook has not been created"
    spec = importlib.util.spec_from_file_location("polars_notebook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = pl.DataFrame(
        {
            "Object Number": ["a", "b", "c", "d"],
            "Is Public Domain": ["True"] * 4,
            "Department": ["Medieval Art", "Medieval Art", "Other", "Other"],
            "AccessionYear": ["1900", "2000", "unknown", "2001"],
            "Object Name": ["Vase"] * 4,
            "Culture": ["Greek"] * 4,
            "Title": [None, "B", "C", "D"],
            "Object Begin Date": ["-500", "1800", "1900", "bad"],
            "Medium": ["Clay"] * 4,
            "Dimensions": ["10 cm"] * 4,
            "Tags": ["Birds|Trees", "Birds", "Trees", None],
            "Empty": [None] * 4,
        }
    )
    fixture.write_csv(tmp_path / "MetObjects.csv")
    _, definitions = module.app.run(defs={"data_dir": tmp_path, "Path": Path})
    cleaned = definitions["dataset_5"]
    assert cleaned["Object Number"].to_list() == ["a", "b"]
    assert cleaned["Title"].to_list() == ["Untitled", "B"]
    assert cleaned["Object Begin Date"].to_list() == [-500, 1800]
    assert "Empty" not in definitions["dataset_2"].columns
    assert definitions["g"]["Counts"].to_list() == [4]
    namespace = dict(definitions)
    for solution in definitions["solutions"]:
        exec(solution, namespace)
    assert namespace["oldest_record"]["Object Number"] == "a"
    assert namespace["departments"]["Counts"].sum() == 2
    assert namespace["tags_df"].row(0) == ("Birds", 2)
