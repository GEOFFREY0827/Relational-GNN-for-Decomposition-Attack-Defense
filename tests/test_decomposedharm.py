"""Adapter tests use invented text; no gated dataset content is published."""
import json
import unittest

from prepare_decomposedharm import CANARY, convert_qa


class ConversionTests(unittest.TestCase):
    def test_csv_nested_list_and_filter(self):
        row = {"id": "qa_1", "task_type": "QA", "original_task": "Plan a picnic",
               "decomposition": json.dumps(str(["Check weather", "Choose a park"])),
               "label": "benign", "harm_index": float("nan")}
        result = convert_qa([dict(row, task_type="agent"), row])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["S"], ["Check weather", "Choose a park"])
        self.assertEqual(result[0]["Q"], "Plan a picnic")
        self.assertIsNone(result[0]["harmful_index"])
        self.assertEqual(result[0]["canary"], CANARY)
        json.dumps(result, allow_nan=False)

    def test_card_schema(self):
        result = convert_qa([{"task_type": "jailbreak", "original_task": "Plan",
                              "decomposed_subtasks": ["First", "Second"]}])
        self.assertEqual(result[0]["S"], ["First", "Second"])

    def test_invalid_and_empty(self):
        for subtasks in ([], [""], "__import__('os').getcwd()", {"a": "b"}):
            with self.subTest(subtasks=subtasks), self.assertRaises((ValueError, SyntaxError)):
                convert_qa([{"task_type": "QA", "original_task": "Plan", "decomposition": subtasks}])
        with self.assertRaises(ValueError):
            convert_qa([])
