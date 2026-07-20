import unittest

import torch

from mib_solution.infer import normalize_fields
from mib_solution.model import PacketCNN


class ModelTests(unittest.TestCase):
    def test_packet_model_handles_variable_page_counts(self):
        model = PacketCNN({"adjudication": 3, "fee_status": 4})
        output = model(
            torch.rand(2, 6, 3, 96, 64),
            torch.tensor([[True] * 6, [True] * 3 + [False] * 3]),
        )
        self.assertEqual(tuple(output["adjudication"].shape), (2, 3))
        self.assertEqual(tuple(output["fee_status"].shape), (2, 4))


class InferenceNormalizationTests(unittest.TestCase):
    def test_schema_sensitive_fields_receive_safe_defaults(self):
        record = {"fee_status": "PAID - receipt", "sponsor_id": "missing", "arrival_date": "tomorrow"}
        normalize_fields(record)
        self.assertEqual(record["fee_status"], "paid")
        self.assertEqual(record["sponsor_id"], "SPN-0000")
        self.assertEqual(record["arrival_date"], "1900-01-01")


if __name__ == "__main__":
    unittest.main()
