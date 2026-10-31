import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from model_benchmark.mot import (
    _maximum_weight_assignment,
    discover_mot_sequences,
    evaluate_mot_sequences,
    read_mot_annotations,
)


class MotEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.sequence_dir = self.root / "sequences" / "sequence-1"
        self.sequence_dir.mkdir(parents=True)
        self.annotations_dir = self.root / "annotations"
        self.annotations_dir.mkdir()
        for frame_name in ("000001.jpg", "000002.jpg"):
            (self.sequence_dir / frame_name).touch()
        self.annotation_path = self.annotations_dir / "sequence-1.txt"
        self.annotation_path.write_text(
            "1,5,10,10,20,20,1,1,1,1\n"
            "2,5,10,10,20,20,1,1,1,1\n"
            "2,-1,0,0,10,10,1,1,1,1\n"
            "2,6,0,0,10,10,0,1,1,1\n",
            encoding="utf-8",
        )

    def test_assignment_optimizes_globally_instead_of_greedy_pairing(self):
        assignments = _maximum_weight_assignment([[10.0, 9.0], [9.0, 0.0]])

        self.assertEqual(set(assignments), {(0, 1), (1, 0)})

    def test_discovers_visdrone_sequences_annotations_and_frame_directory(self):
        sequences = discover_mot_sequences(self.root)

        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].name, "sequence-1")
        self.assertEqual(sequences[0].frames_dir, self.sequence_dir)
        self.assertEqual(sequences[0].annotations, self.annotation_path)

    def test_reads_visdrone_class_ids_and_skips_ignored_rows(self):
        annotations = read_mot_annotations(self.annotation_path)

        self.assertEqual(set(annotations), {1, 2})
        self.assertEqual(len(annotations[1]), 1)
        self.assertEqual(annotations[1][0].track_id, 5)
        self.assertEqual(annotations[1][0].class_id, 0)
        self.assertEqual(annotations[1][0].box, (10.0, 10.0, 30.0, 30.0))
        self.assertEqual(len(annotations[2]), 1)

    def test_evaluates_tracking_ids_across_consecutive_frames(self):
        class FakeModel:
            def track(self, **kwargs):
                self.calls = getattr(self, "calls", [])
                self.calls.append(kwargs)
                track_id = 11
                return [_result(Path(kwargs["source"]).name, track_id)]

        model = FakeModel()
        metrics = evaluate_mot_sequences(
            model,
            self.root,
            tracker="bytetrack.yaml",
            imgsz=640,
            device="cpu",
            conf=0.001,
            iou=0.7,
            quantize=None,
        )

        self.assertEqual(len(model.calls), 2)
        self.assertTrue(all(call["persist"] for call in model.calls))
        self.assertEqual(metrics["status"], "evaluated")
        self.assertEqual(metrics["sequences"], 1)
        self.assertEqual(metrics["frames"], 2)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["id_switches"], 0)
        self.assertEqual(metrics["mota"], 1.0)
        self.assertEqual(metrics["idf1"], 1.0)

    def test_reports_identity_switches(self):
        class FakeModel:
            def track(self, **kwargs):
                frame_name = Path(kwargs["source"]).name
                track_id = 11 if frame_name == "000001.jpg" else 12
                return [_result(frame_name, track_id)]

        metrics = evaluate_mot_sequences(
            FakeModel(),
            self.root,
            tracker="bytetrack.yaml",
            imgsz=640,
            device="cpu",
            conf=0.001,
            iou=0.7,
            quantize=None,
        )

        self.assertEqual(metrics["id_switches"], 1)
        self.assertEqual(metrics["mota"], 0.5)
        self.assertEqual(metrics["idf1"], 0.5)


def _result(frame_name, track_id):
    box = SimpleNamespace(
        xyxy=[[10.0, 10.0, 30.0, 30.0]],
        cls=[0],
        id=[track_id],
    )
    return SimpleNamespace(path=frame_name, boxes=[box])


if __name__ == "__main__":
    unittest.main()
