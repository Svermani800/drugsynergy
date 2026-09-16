import io
import unittest

import pandas as pd

from app.server import app, model_bundle


@unittest.skipIf(model_bundle is None, "production artifact is not trained")
class AppIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_metadata_and_order_invariant_prediction(self):
        self.assertTrue(self.client.get("/health").get_json()["model_ready"])
        metadata = self.client.get("/api/metadata").get_json()
        drug_a, drug_b = metadata["drugs"][:2]
        cell = metadata["cell_lines"][0]
        first = self.client.post("/api/predict", json={"drug_a": drug_a, "drug_b": drug_b, "cell_line": cell})
        second = self.client.post("/api/predict", json={"drug_a": drug_b, "drug_b": drug_a, "cell_line": cell})
        self.assertEqual(first.status_code, 200)
        self.assertAlmostEqual(first.get_json()["predicted_bliss"], second.get_json()["predicted_bliss"], places=7)

    def test_same_structure_and_all_invalid_batch_are_handled(self):
        metadata = self.client.get("/api/metadata").get_json()
        drug = metadata["drugs"][0]
        cell = metadata["cell_lines"][0]
        response = self.client.post("/api/predict", json={"drug_a": drug, "drug_b": drug, "cell_line": cell})
        self.assertEqual(response.status_code, 400)
        frame = pd.DataFrame({"drug_a": ["UNKNOWN"], "drug_b": [drug], "cell_line": [cell]})
        upload = {"file": (io.BytesIO(frame.to_csv(index=False).encode()), "invalid.csv")}
        batch = self.client.post("/api/batch", data=upload, content_type="multipart/form-data")
        self.assertEqual(batch.status_code, 200)
        self.assertIn("No validated structure mapping", batch.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
