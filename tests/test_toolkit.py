import unittest

from arabic_ai_toolkit.evaluation import evaluate_case, evaluate_cases
from arabic_ai_toolkit.normalization import NormalizationOptions, normalize_arabic


class NormalizationTests(unittest.TestCase):
    def test_removes_diacritics_and_tatweel(self):
        self.assertEqual(normalize_arabic("الذَّكــاء"), "الذكاء")

    def test_normalizes_common_alef_and_yeh_variants(self):
        self.assertEqual(normalize_arabic("إلى آفاق"), "الي افاق")

    def test_preserves_teh_marbuta_by_default(self):
        self.assertEqual(normalize_arabic("مدرسة"), "مدرسة")
        options = NormalizationOptions(normalize_teh_marbuta=True)
        self.assertEqual(normalize_arabic("مدرسة", options), "مدرسه")


class EvaluationTests(unittest.TestCase):
    def test_exact_match_after_normalization(self):
        score = evaluate_case("a", "الذكاء الاصطناعي", "الذَّكاء الاصطناعي")
        self.assertEqual(score.exact_match, 1.0)
        self.assertEqual(score.token_f1, 1.0)

    def test_token_metrics_count_duplicates(self):
        score = evaluate_case("b", "علم بيانات بيانات", "علم بيانات")
        self.assertEqual(score.token_precision, 1.0)
        self.assertAlmostEqual(score.token_recall, 2 / 3)
        self.assertAlmostEqual(score.token_f1, 0.8)

    def test_empty_collection_is_well_defined(self):
        summary = evaluate_cases([])
        self.assertEqual(summary.to_dict(), {
            "cases": 0,
            "exact_match": 0.0,
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
        })

    def test_missing_fields_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "reference and prediction"):
            evaluate_cases([{"id": "x", "reference": "اختبار"}])


if __name__ == "__main__":
    unittest.main()
