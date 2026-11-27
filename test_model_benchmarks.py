from pathlib import Path
import unittest
import numpy as np
import app as har_app

BASE_DIR = Path(__file__).resolve().parent
DUMMY_USER = {"id": 1, "username": "guest_researcher"}


class TestHARModelsImagesAndSteps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_text = (BASE_DIR / "app.py").read_text(encoding="utf-8")
        cls.html_text = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")

    def test_two_models_and_exact_accuracies(self):
        """Verify only Linear SVM (96.27% test / 95.84% LOSO) and CNN-LSTM (91.75%) are exposed."""
        st = har_app.api_status(user=DUMMY_USER)
        self.assertEqual(st["models"], ["Linear SVM", "CNN-LSTM"])
        self.assertEqual(st["benchmarks"]["svm_test_accuracy"], 96.27)
        self.assertEqual(st["benchmarks"]["svm_macro_f1"], 0.963)
        self.assertEqual(st["benchmarks"]["svm_loso_mean_accuracy"], 95.84)
        self.assertEqual(st["benchmarks"]["cnn_lstm_test_accuracy"], 91.75)

        required_html_snippets = [
            "Linear SVM",
            "96.27%",
            "Linear SVM test accuracy (macro F1 0.963)",
            "95.84%",
            "Linear SVM mean LOSO accuracy (30 subjects)",
            "CNN-LSTM",
            "91.75%",
            "CNN-LSTM test accuracy (best epoch)",
            "Reset Step Count",
            "liveResetStepsBtn",
            "analyticsResetStepsBtn",
            "data-mimg",
        ]
        for snippet in required_html_snippets:
            self.assertIn(snippet, self.html_text, f"Missing {snippet!r} in static/index.html")

    def test_all_six_activities_and_side_by_side_comparison(self):
        """Verify all 6 activities return Linear SVM and CNN-LSTM side by side and map to their activity images."""
        expected_images = {
            "WALKING": "/static/images/walking.jpg",
            "WALKING_UPSTAIRS": "/static/images/upstairs.jpg",
            "WALKING_DOWNSTAIRS": "/static/images/downstairs.jpg",
            "SITTING": "/static/images/sitting.jpg",
            "STANDING": "/static/images/standing.jpg",
            "LAYING": "/static/images/laying.jpg",
        }
        for act, img_path in expected_images.items():
            self.assertIn(img_path, self.html_text)
            demo = har_app.predict_preset(act, user=DUMMY_USER).model_dump()
            self.assertEqual(demo["ensemble_activity"], act)
            self.assertEqual(len(demo["all_models"]), 2)
            by_key = {m["model_key"]: m for m in demo["all_models"]}
            self.assertEqual(list(by_key.keys()), ["svm", "cnn_lstm"])
            self.assertEqual(by_key["svm"]["model_name"], "Linear SVM")
            self.assertEqual(by_key["svm"]["accuracy"], "96.27%")
            self.assertEqual(by_key["svm"]["loso_accuracy"], "95.84% (30 subjects)")
            self.assertEqual(by_key["cnn_lstm"]["model_name"], "CNN-LSTM")
            self.assertEqual(by_key["cnn_lstm"]["accuracy"], "91.75%")

    def test_real_world_step_counting_accuracy(self):
        """Verify step detector counts walking steps reliably and returns 0 for stationary postures."""
        t = np.arange(128, dtype=np.float32) / 50.0
        # 2.56s walking window at 1.95 Hz (~5 steps in 2.56s)
        f = 1.95
        signals = np.zeros((9, 128), dtype=np.float32)
        ax = 0.18 * np.sin(2 * np.pi * f * t)
        ay = 0.08 * np.cos(2 * np.pi * f * t)
        az = 0.06 * np.sin(2 * np.pi * f * t)
        signals[0], signals[1], signals[2] = ax, ay, az
        signals[6], signals[7], signals[8] = ax, 0.96 + ay, 0.22 + az

        steps, spm, peaks = har_app.detect_steps_in_window(signals, 50.0, "WALKING")
        self.assertGreaterEqual(steps, 4)
        self.assertLessEqual(steps, 6)
        self.assertTrue(100.0 <= spm <= 135.0, f"Unexpected cadence: {spm}")

        # Stationary sitting window must report 0 steps
        steps_sit, spm_sit, _ = har_app.detect_steps_in_window(signals * 0.01, 50.0, "SITTING")
        self.assertEqual(steps_sit, 0)
        self.assertEqual(spm_sit, 0.0)

    def test_chatbot_two_models_answers(self):
        """Verify chatbot answers reference Linear SVM (96.27% / 95.84% LOSO) and CNN-LSTM (91.75%)."""
        reply = har_app.chat_with_model_assistant(
            har_app.ChatRequestPayload(
                message="Compare the two built-in models: Linear SVM (96.27% test / 95.84% LOSO) vs. CNN-LSTM (91.75%)"
            ),
            user=DUMMY_USER,
        )
        self.assertIn("96.27%", reply.answer)
        self.assertIn("95.84%", reply.answer)
        self.assertIn("91.75%", reply.answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
