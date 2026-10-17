"""Small regression checks for the supplied data and feature engineering."""

import unittest
from pathlib import Path

from src.data_preparation import _parse_currency, load_and_prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataPreparationTests(unittest.TestCase):
    def test_currency_parser(self) -> None:
        values = _parse_currency(__import__("pandas").Series(["$300,385.62", "$1.00"]))
        self.assertEqual(values.tolist(), [300385.62, 1.0])

    def test_feature_table_has_one_row_per_client(self) -> None:
        prepared = load_and_prepare_data(PROJECT_ROOT)
        features = prepared.client_features
        self.assertEqual(len(features), 2000)
        self.assertTrue(features["client_id"].is_unique)
        self.assertTrue((features["purchase_count"] >= 0).all())
        self.assertTrue({"age", "total_spend", "country"}.issubset(features.columns))


if __name__ == "__main__":
    unittest.main()
