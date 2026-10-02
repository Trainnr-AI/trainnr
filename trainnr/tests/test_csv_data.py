"""The CSV on-ramp: any robot's log becomes identification data."""

import tempfile
import unittest
from pathlib import Path

from trainnr.collect.csv_data import excitation_from_csv


def _write(directory: str, text: str) -> Path:
    path = Path(directory) / "run.csv"
    path.write_text(text, encoding="utf-8")
    return path


class FromCsv(unittest.TestCase):
    def test_named_columns_become_row_aligned_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                directory,
                "t,duty,angle,extra\n0.0,0,0.0,9\n0.02,25,0.01,9\n0.04,25,0.05,9\n",
            )
            data = excitation_from_csv(
                path, time="t", controls=["duty"], measurements=["angle"]
            )
        self.assertEqual(len(data.times), 3)
        self.assertEqual(data.controls.shape, (3, 1))
        self.assertEqual(data.measurements.shape, (3, 1))
        self.assertEqual(data.controls[1, 0], 25.0)

    def test_missing_column_named_in_the_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(directory, "t,duty\n0,0\n1,1\n")
            with self.assertRaises(ValueError) as caught:
                excitation_from_csv(
                    path, time="t", controls=["duty"], measurements=["angle"]
                )
        self.assertIn("angle", str(caught.exception))

    def test_non_increasing_time_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(directory, "t,u,y\n0,0,0\n0,1,1\n")
            with self.assertRaises(ValueError):
                excitation_from_csv(path, time="t", controls=["u"], measurements=["y"])

    def test_non_numeric_value_names_the_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(directory, "t,u,y\n0,0,0\n1,oops,1\n")
            with self.assertRaises(ValueError) as caught:
                excitation_from_csv(path, time="t", controls=["u"], measurements=["y"])
        self.assertIn("'u'", str(caught.exception))

    def test_single_row_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(directory, "t,u,y\n0,0,0\n")
            with self.assertRaises(ValueError):
                excitation_from_csv(path, time="t", controls=["u"], measurements=["y"])

    def test_headerless_file_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(directory, "")
            with self.assertRaises(ValueError):
                excitation_from_csv(path, time="t", controls=["u"], measurements=["y"])


if __name__ == "__main__":
    unittest.main()
