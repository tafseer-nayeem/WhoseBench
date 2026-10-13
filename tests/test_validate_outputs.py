import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import validate_outputs


class ReadRowsTests(unittest.TestCase):
    def test_released_outputs_pass(self):
        self.assertEqual(validate_outputs.main(), 0)

    def test_rejects_empty_duplicate_header_and_extra_fields(self):
        cases = {
            "empty": "",
            "duplicate header": "value,value\n1,2\n",
            "extra fields": "value\n1,2\n",
        }
        for name, contents in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "sample.csv").write_text(contents, encoding="utf-8")
                with patch.object(validate_outputs, "ROOT", root):
                    with self.assertRaises(AssertionError):
                        validate_outputs.read_rows("sample.csv")


if __name__ == "__main__":
    unittest.main()
