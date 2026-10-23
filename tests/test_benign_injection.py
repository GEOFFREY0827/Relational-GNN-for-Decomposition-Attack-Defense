"""Pure preprocessing checks; no models, downloads, or training."""
import unittest

from prepare_benign_injection import inject_records


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "h", "label": "harmful", "Q": "Invented target", "S": ["original A", "original B"],
             "harmful_index": 1, "canary": "invented-canary"},
            {"id": "b1", "label": "benign", "Q": "Picnic", "S": [f"picnic task {i}" for i in range(30)]},
            {"id": "b2", "label": "benign", "Q": "Gardening", "S": [f"garden task {i}" for i in range(30)]},
        ]

    def test_count_order_labels_and_no_source_mutation(self):
        output = inject_records(self.rows)
        for old, new in zip(self.rows, output):
            self.assertEqual(len(new["S"]), len(old["S"]) + 25)
            self.assertEqual(new["Q"], old["Q"])
            self.assertEqual(new["label"], old["label"])
            metadata = new["benign_injection"]
            self.assertEqual([new["S"][i] for i in metadata["original_positions"]], old["S"])
            self.assertEqual(len(metadata["donors"]), 25)
            self.assertEqual(len({new["S"][d["position"]] for d in metadata["donors"]}), 25)
            self.assertTrue(all(d["donor_id"] in ("b1", "b2") and d["donor_id"] != old["id"]
                                for d in metadata["donors"]))
        self.assertEqual(len(self.rows[0]["S"]), 2)
        self.assertEqual(output[0]["canary"], "invented-canary")
        self.assertIsNone(output[0]["harmful_index"])
        self.assertEqual(output[0]["benign_injection"]["original_harmful_index"], 1)

    def test_seed_and_placement(self):
        a = inject_records(self.rows)
        self.assertEqual(a, inject_records(self.rows))
        self.assertNotEqual(a, inject_records(self.rows, seed=43))
        self.assertNotEqual(a, inject_records(self.rows, split="test"))
        self.assertEqual(inject_records(self.rows, placement="append")[0]["S"][:2], self.rows[0]["S"])
        self.assertEqual(inject_records(self.rows, placement="prepend")[0]["S"][-2:], self.rows[0]["S"])

    def test_insufficient_pool_fails_instead_of_repeating(self):
        with self.assertRaises(ValueError):
            inject_records(self.rows[:2])
