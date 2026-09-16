import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.curation import build_cell_line_map, curate_cancer_rows, normalize_name


class CurationTests(unittest.TestCase):
    def test_normalized_exact_matching_and_ambiguity(self):
        metadata = pd.DataFrame(
            {
                "model_id": ["M1", "M2", "M3"],
                "BROAD_ID": ["ACH-1", "ACH-2", "ACH-3"],
                "model_name": ["A-2058", "Shared", "Shared"],
                "synonyms": ["A2058;A 2058", "COLLISION", "COLLISION"],
                "model_type": ["Cell Line"] * 3,
                "species": ["Homo Sapiens"] * 3,
                "cancer_type": ["Melanoma", "Cancer", "Cancer"],
                "tissue": ["Skin", "Other", "Other"],
                "expression_data": [True] * 3,
            }
        )
        combinations = pd.DataFrame(
            {"drug_a": ["A", "A"], "drug_b": ["B", "B"], "cell_line": ["A2058", "COLLISION"], "synergy": [1.0, 2.0]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.csv"
            metadata.to_csv(path, index=False)
            mapping, _ = build_cell_line_map(path)
            curated, _, report = curate_cancer_rows(combinations, path)
        self.assertEqual(normalize_name("A-2058"), "A2058")
        self.assertEqual(mapping["A2058"], "ACH-1")
        self.assertNotIn("COLLISION", mapping)
        self.assertEqual(curated.model_id.tolist(), ["ACH-1"])
        self.assertEqual(report["matched_rows"], 1)


if __name__ == "__main__":
    unittest.main()
