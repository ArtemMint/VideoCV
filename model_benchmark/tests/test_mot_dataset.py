import tempfile
import unittest
import zipfile
from pathlib import Path

from model_benchmark.mot_dataset import prepare_visdrone_mot


def create_mot_archive(path, *, unsafe_member=False):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "VisDrone2019-MOT-val/sequences/sequence-1/000001.jpg",
            b"frame-one",
        )
        archive.writestr(
            "VisDrone2019-MOT-val/annotations/sequence-1.txt",
            "1,5,10,10,20,20,1,1,1,1\n",
        )
        if unsafe_member:
            archive.writestr("../outside.txt", "unsafe")


class PrepareVisDroneMotTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.archive = self.root / "VisDrone2019-MOT-val.zip"
        self.dataset_root = self.root / "datasets" / "VisDrone-MOT"
        create_mot_archive(self.archive)

    def test_extracts_and_reuses_visdrone_mot_validation_archive(self):
        result = prepare_visdrone_mot(self.archive, self.dataset_root)

        self.assertTrue(result["prepared"])
        self.assertEqual(result["sequences"], 1)
        self.assertEqual(result["frames"], 1)
        self.assertTrue(
            (self.dataset_root / "sequences/sequence-1/000001.jpg").is_file()
        )
        self.assertTrue(
            (self.dataset_root / "annotations/sequence-1.txt").is_file()
        )

        cached = prepare_visdrone_mot(dataset_root=self.dataset_root)
        self.assertFalse(cached["prepared"])
        self.assertEqual(cached["sequences"], 1)

    def test_rejects_unsafe_zip_member_without_creating_dataset(self):
        unsafe_archive = self.root / "unsafe.zip"
        create_mot_archive(unsafe_archive, unsafe_member=True)

        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            prepare_visdrone_mot(unsafe_archive, self.dataset_root)

        self.assertFalse(self.dataset_root.exists())

    def test_does_not_replace_an_incomplete_dataset_folder(self):
        self.dataset_root.mkdir(parents=True)

        with self.assertRaisesRegex(FileExistsError, "incomplete"):
            prepare_visdrone_mot(self.archive, self.dataset_root)

    def test_rejects_archive_without_matching_annotations(self):
        invalid_archive = self.root / "invalid.zip"
        with zipfile.ZipFile(invalid_archive, "w") as archive:
            archive.writestr(
                "dataset/sequences/sequence-1/000001.jpg",
                b"frame-one",
            )

        with self.assertRaisesRegex(ValueError, "No VisDrone-MOT sequences"):
            prepare_visdrone_mot(invalid_archive, self.dataset_root)

        self.assertFalse(self.dataset_root.exists())


if __name__ == "__main__":
    unittest.main()
