"""Check evaluation semantics and that label metadata cannot enter the prompt."""
import importlib.util
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("baseline", Path(__file__).resolve().parents[1] / "scripts/baseline.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BaselineTests(unittest.TestCase):
    def test_test_protocol_follows_id_even_when_svt_video_has_audio(self):
        self.assertEqual(b.get_protocol({"sample_id": "gen:accept_a_svt:S1", "video": "videos/gen/accept_a_svt/video_audio/S1.mp4"}), "VT")
        self.assertEqual(b.get_protocol({"sample_id": "gen:accept_c_svt:S1", "video": "x.mp4"}), "VT")
        self.assertEqual(b.get_protocol({"sample_id": "gen:accept_a_va:S1", "video": "x.mp4"}), "VA")
        self.assertEqual(b.get_protocol({"sample_id": "ch_sims_v2:VA:x", "video": "x.mp4"}), "VA")
        item = b.model_input({"sample_id": "ch_sims_v2:VT:x", "video": "x.mp4", "text": "released transcript"})
        self.assertEqual(item["text"], "released transcript")

    def test_new_runs_are_separate_and_explicit_directory_resumes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(b, "ROOT", Path(temp)):
            first = b.output_directory("baseline_v1", "val")
            second = b.output_directory("baseline_v1", "val")
            test = b.output_directory("baseline_v1", "test")
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent.name, "val")
            self.assertEqual(test.parent.name, "test")
            self.assertEqual(b.output_directory("ignored", "val", first), first)

    def test_prompt_ignores_annotations_and_va_transcript(self):
        row = {"sample_id": "x", "video": "v.mp4", "protocol": "VA", "text_content": "SECRET_TRANSCRIPT", "sample_type": "GOLD_CONFLICT", "views": {"M12": {"label": "SECRET_LABEL"}}}
        messages = str(b.messages_for(b.model_input(row)))
        for secret in ("SECRET_TRANSCRIPT", "SECRET_LABEL", "GOLD_CONFLICT"):
            self.assertNotIn(secret, messages)

    def test_output_validation_preserves_failure(self):
        with self.assertRaises(ValueError):
            b.parse_prediction('{"pred_emotion":"positive","is_uncertain":"yes","clarifying_question":"none"}')
        with self.assertRaises(ValueError):
            b.parse_prediction("probably happy")
        p = b.parse_prediction('```json\n{"pred_emotion":"negative","is_uncertain":"no","clarifying_question":"none"}\n```')
        self.assertEqual(p["pred_emotion"], "negative")

    def test_f1_conventions_and_missing_predictions(self):
        rows = [{"sample_id": "a", "sample_type": "Conflict", "views": {"M12": {"label": "positive"}}}, {"sample_id": "b", "sample_type": "Aligned", "views": {"M12": {"label": "negative"}}}]
        pred = [{"sample_id": "a", "prediction": {"pred_emotion": "positive", "is_uncertain": "yes"}}, {"sample_id": "b", "prediction": {"pred_emotion": "negative", "is_uncertain": "no"}}]
        score = b.score_rows(rows, pred)
        self.assertAlmostEqual(score["macro_f1_fixed_3"], 2/3)
        self.assertEqual(score["score_sklearn_union"], 1)
        missing = b.score_rows(rows, pred[:1])
        self.assertEqual(missing["missing_or_invalid_predictions"], 1)
        self.assertEqual(missing["aligned_calibration_rate"], 0)

    def test_test_export_needs_no_labels_and_uses_template_order(self):
        rows = [{"sample_id": "ch_sims_v2:VT:a", "video": "a.mp4", "text": "hello"},
                {"sample_id": "gen:accept_a_va:b", "video": "b.mp4", "text": "bye"}]
        predictions = [
            {"sample_id": rows[0]["sample_id"], "prediction": {"pred_emotion": "positive", "is_uncertain": "no", "clarifying_question": "none"}},
            {"sample_id": rows[1]["sample_id"], "prediction": {"pred_emotion": "neutral", "is_uncertain": "yes", "clarifying_question": "Could you explain,\nin your own words?"}},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template = root / "sample_submission.csv"
            with template.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(b.CSV_FIELDS)
                for row in reversed(rows):
                    writer.writerow([row["sample_id"], "neutral", "no", "none"])
            result = b.finalize(argparse.Namespace(split="test", dataset_root=root), rows, rows, predictions, root)
            self.assertTrue(result["submission"]["valid"])
            self.assertFalse((root / "metrics.json").exists())
            with (root / "submission.csv").open(newline="") as f:
                exported = list(csv.DictReader(f))
            self.assertEqual([p["sample_id"] for p in exported], [r["sample_id"] for r in reversed(rows)])
            self.assertEqual(exported[0]["clarifying_question"], "Could you explain,\nin your own words?")
            self.assertTrue(json.loads((root / "status.json").read_text())["submission_ready"])

            # A missing prediction must not be replaced with the sample default.
            with self.assertRaises(ValueError):
                b.export_submission(rows, predictions[:1], template, root)
            self.assertFalse((root / "submission.csv").exists())
            self.assertFalse(json.loads((root / "submission_validation.json").read_text())["valid"])

            # Duplicate IDs must not silently overwrite a prediction.
            with self.assertRaises(ValueError):
                b.export_submission(rows, predictions + predictions[:1], template, root)

    def test_invalid_question_blocks_submission_export(self):
        row = {"sample_id": "ch_sims_v2:VT:a", "video": "a.mp4"}
        bad = {"sample_id": row["sample_id"], "prediction": {"pred_emotion": "positive", "is_uncertain": "yes", "clarifying_question": "none"}}
        valid, invalid = b.checked_predictions([row], [bad])
        self.assertFalse(valid)
        self.assertEqual(len(invalid), 1)


if __name__ == "__main__":
    unittest.main()
