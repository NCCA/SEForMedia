import importlib.util
from pathlib import Path
import tempfile
import unittest

import polars as pl


class PolarsNotebookTests(unittest.TestCase):
    def test_cleaning_and_solutions(self) -> None:
        """Run the notebook against missing values and ancient artwork dates."""
        path = Path(__file__).resolve().parents[1] / "IntroductionToPolarsMarimo.py"
        self.assertTrue(path.exists(), "Polars notebook has not been created")
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
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "MetObjects.csv"
            fixture.write_csv(csv)
            _, definitions = module.app.run(
                defs={"data_dir": Path(folder), "Path": Path}
            )
        cleaned = definitions["dataset_5"]
        self.assertEqual(cleaned["Object Number"].to_list(), ["a", "b"])
        self.assertEqual(cleaned["Title"].to_list(), ["Untitled", "B"])
        self.assertEqual(cleaned["Object Begin Date"].to_list(), [-500, 1800])
        self.assertNotIn("Empty", definitions["dataset_2"].columns)
        self.assertEqual(definitions["g"]["Counts"].to_list(), [4])
        namespace = dict(definitions)
        for solution in definitions["solutions"]:
            exec(solution, namespace)
        self.assertEqual(namespace["oldest_record"]["Object Number"], "a")
        self.assertEqual(namespace["departments"]["Counts"].sum(), 2)
        self.assertEqual(namespace["tags_df"].row(0), ("Birds", 2))


if __name__ == "__main__":
    unittest.main()
