import json
from pathlib import Path
import unittest
import app as har_app

BASE_DIR = Path(__file__).resolve().parent
DUMMY_USER = {"id": 1, "username": "guest_researcher"}


class TestHARModelNamesAndBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_text = (BASE_DIR / "app.py").read_text(encoding="utf-8")
        cls.html_text = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")
        cls.metrics = json.loads((BASE_DIR / "results" / "metrics.json").read_text(encoding="utf-8"))

    def test_benchmark_source_of_truth_alignment(self):
        """Verify benchmark metrics in results/metrics.json match displayed UI and API figures."""
        self.assertEqual(self.metrics["task3_leakage"]["svm_tuned_test_acc"], 96.2)
        self.assertEqual(self.metrics["task1_loso_cnnlstm"]["svm_loso_mean"], 95.83)
        self.assertEqual(self.metrics["task1_loso_cnnlstm"]["svm_loso_std"], 4.84)

        self.assertEqual(self.metrics["task2_app_models"]["reg_ann_test_acc"], 96.06)
        self.assertEqual(self.metrics["task2_app_models"]["reg_ann_loso_mean"], 94.13)
        self.assertEqual(self.metrics["task2_app_models"]["reg_ann_loso_std"], 5.85)

        self.assertEqual(self.metrics["task2_app_models"]["cnn1d_test_acc"], 93.93)
        self.assertEqual(self.metrics["task2_app_models"]["cnn1d_loso_mean"], 93.51)
        self.assertEqual(self.metrics["task2_app_models"]["cnn1d_loso_std"], 7.78)

        self.assertEqual(self.metrics["task3_leakage"]["cnnlstm_clean_21sub_test_acc"], 90.57)
        self.assertEqual(self.metrics["task1_loso_cnnlstm"]["cnnlstm_loso_mean"], 90.14)
        self.assertEqual(self.metrics["task1_loso_cnnlstm"]["cnnlstm_loso_std"], 10.25)

    def test_no_stale_accuracies_in_app_or_html(self):
        """Ensure stale figures (94.6%, 95.7%, 96.3%, 95.8%, 96.8%, etc.) are completely absent."""
        stale_patterns = [
            "94.6%", "94.62",
            "95.7%", "95.72",
            "96.3%", "96.27",
            "95.8%", "95.84",
            "96.8%", "96.84",
            "94.03",
        ]
        for pat in stale_patterns:
            self.assertNotIn(pat, self.app_text, f"Stale figure {pat!r} found in app.py")
            self.assertNotIn(pat, self.html_text, f"Stale figure {pat!r} found in static/index.html")

    def test_html_labels_protocols_and_hybrid_distinction(self):
        """Verify static/index.html displays truthful predictor names, separate protocols, and unvalidated tags."""
        required_snippets = [
            "Linear SVM",
            "96.20%",
            "95.83% ± 4.84%",
            "Regularized ANN",
            "96.06%",
            "94.13% ± 5.85%",
            "HAR 1D-CNN (Standalone)",
            "93.93%",
            "93.51% ± 7.78%",
            "CNN-LSTM",
            "90.57%",
            "90.14% ± 10.25%",
            "1D-CNN + LOSO-SVM Hybrid",
            "CombinAI (Ensemble)",
            "Not independently validated",
            "live conf",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, self.html_text, f"Required snippet {snippet!r} missing from static/index.html")

    def test_api_status_and_live_comparison_metadata(self):
        """Verify /api/status and /api/demo/{activity} return truthful predictor names and protocol-separated scores."""
        st = har_app.api_status(user=DUMMY_USER)

        expected_models = [
            "CNN-LSTM",
            "Regularized ANN",
            "Linear SVM",
            "1D-CNN + LOSO-SVM Hybrid",
            "CombinAI (Ensemble)",
        ]
        self.assertEqual(st["models"], expected_models)
        self.assertEqual(st["live_predictors"]["base_predictors"], expected_models[:4])
        self.assertEqual(st["live_predictors"]["ensemble"], "CombinAI (Ensemble)")

        bm = st["benchmarks"]["benchmarked_models"]
        self.assertEqual(bm["linear_svm_leak_free"]["held_out_test_accuracy"], 96.20)
        self.assertEqual(bm["linear_svm_leak_free"]["loso_summary"], "95.83% ± 4.84%")
        self.assertEqual(bm["regularized_ann"]["held_out_test_accuracy"], 96.06)
        self.assertEqual(bm["regularized_ann"]["loso_summary"], "94.13% ± 5.85%")
        self.assertEqual(bm["har_1dcnn_standalone"]["held_out_test_accuracy"], 93.93)
        self.assertEqual(bm["har_1dcnn_standalone"]["loso_summary"], "93.51% ± 7.78%")
        self.assertEqual(bm["cnn_lstm_leak_free"]["held_out_test_accuracy"], 90.57)
        self.assertEqual(bm["cnn_lstm_leak_free"]["loso_summary"], "90.14% ± 10.25%")

        demo = har_app.predict_preset("WALKING", user=DUMMY_USER).model_dump()
        by_key = {m["model_key"]: m for m in demo["all_models"]}

        self.assertEqual(by_key["cnn_lstm"]["model_name"], "CNN-LSTM")
        self.assertEqual(by_key["cnn_lstm"]["held_out_test_accuracy"], "90.57%")
        self.assertEqual(by_key["cnn_lstm"]["loso_accuracy"], "90.14% ± 10.25%")

        self.assertEqual(by_key["ann"]["model_name"], "Regularized ANN")
        self.assertEqual(by_key["ann"]["held_out_test_accuracy"], "96.06%")
        self.assertEqual(by_key["ann"]["loso_accuracy"], "94.13% ± 5.85%")

        self.assertEqual(by_key["svm"]["model_name"], "Linear SVM")
        self.assertEqual(by_key["svm"]["held_out_test_accuracy"], "96.20%")
        self.assertEqual(by_key["svm"]["loso_accuracy"], "95.83% ± 4.84%")

        self.assertEqual(by_key["loso"]["model_name"], "1D-CNN + LOSO-SVM Hybrid")
        self.assertEqual(by_key["loso"]["role"], "Base Predictor (Hybrid)")
        self.assertEqual(by_key["loso"]["held_out_test_accuracy"], "Not independently validated")
        self.assertEqual(by_key["loso"]["loso_accuracy"], "Not independently validated")

        self.assertEqual(by_key["combinai"]["model_name"], "CombinAI (Ensemble)")
        self.assertEqual(by_key["combinai"]["role"], "Ensemble (4 Base Predictors)")
        self.assertEqual(by_key["combinai"]["held_out_test_accuracy"], "Not independently validated")
        self.assertEqual(by_key["combinai"]["loso_accuracy"], "Not independently validated")

    def test_chatbot_loso_and_hybrid_distinction(self):
        """Verify chatbot answers explain LOSO as a 30-fold evaluation protocol and distinguish the hybrid predictor."""
        ans_loso = har_app.chat_with_model_assistant(
            har_app.ChatRequestPayload(
                message="What does LOSO mean, what is the 1D-CNN + LOSO-SVM Hybrid predictor, and how does it differ from standalone HAR 1D-CNN?"
            ),
            user=DUMMY_USER,
        ).answer
        self.assertIn("Leave-One-Subject-Out", ans_loso)
        self.assertIn("93.93%", ans_loso)
        self.assertIn("93.51% ± 7.78%", ans_loso)
        self.assertIn("1D-CNN + LOSO-SVM Hybrid", ans_loso)
        self.assertIn("Not independently validated", ans_loso)

        ans_comb = har_app.chat_with_model_assistant(
            har_app.ChatRequestPayload(
                message="How does the CombinAI ensemble combine the 4 base predictors and why is it marked not independently validated?"
            ),
            user=DUMMY_USER,
        ).answer
        self.assertIn("CombinAI (Ensemble)", ans_comb)
        self.assertIn("Not independently validated", ans_comb)


if __name__ == "__main__":
    unittest.main(verbosity=2)
