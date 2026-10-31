"""MOT dataset discovery and tracking evaluation helpers."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class MotSequence:
    name: str
    frames_dir: Path
    annotations: Path


@dataclass(frozen=True)
class MotDetection:
    track_id: Optional[int]
    class_id: int
    box: Tuple[float, float, float, float]


def _image_directory(sequence_dir: Path) -> Optional[Path]:
    for name in ("img1", "images", "frames"):
        candidate = sequence_dir / name
        if candidate.is_dir():
            return candidate
    if any(path.suffix.lower() in IMAGE_SUFFIXES for path in sequence_dir.iterdir()):
        return sequence_dir
    return None


def _sequence_for_annotation(root: Path, annotation: Path) -> Optional[Path]:
    if annotation.name.lower() == "gt.txt":
        if _image_directory(annotation.parent.parent) is not None:
            return annotation.parent.parent
        if _image_directory(annotation.parent) is not None:
            return annotation.parent
        return annotation.parent.parent

    candidates = [
        annotation.parent.parent / "sequences" / annotation.stem,
        annotation.parent.parent / annotation.stem,
        root / "sequences" / annotation.stem,
        root / annotation.stem,
    ]
    for candidate in candidates:
        if candidate.is_dir() and _image_directory(candidate) is not None:
            return candidate

    for candidate in root.rglob(annotation.stem):
        if candidate.is_dir() and _image_directory(candidate) is not None:
            return candidate
    return None


def discover_mot_sequences(dataset_root: Path) -> List[MotSequence]:
    """Find VisDrone-MOT annotation/frame pairs in common MOT layouts."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("MOT dataset directory does not exist: {}".format(root))

    annotation_files = set(root.rglob("gt.txt"))
    annotation_dirs = [
        path for path in root.rglob("annotations") if path.is_dir()
    ]
    for annotation_dir in annotation_dirs:
        annotation_files.update(annotation_dir.glob("*.txt"))

    sequences: Dict[Path, MotSequence] = {}
    for annotation in sorted(annotation_files):
        sequence_dir = _sequence_for_annotation(root, annotation)
        if sequence_dir is None:
            continue
        frames_dir = _image_directory(sequence_dir)
        if frames_dir is None:
            continue
        if not any(path.suffix.lower() in IMAGE_SUFFIXES for path in frames_dir.iterdir()):
            continue
        sequences[sequence_dir] = MotSequence(
            name=sequence_dir.name,
            frames_dir=frames_dir,
            annotations=annotation,
        )

    if not sequences:
        raise ValueError(
            "No VisDrone-MOT sequences found. Expected sequences/<name>/<frame>.jpg "
            "with annotations/<name>.txt, or <sequence>/img1 and <sequence>/gt/gt.txt."
        )
    return [sequences[path] for path in sorted(sequences)]


def _frame_number(frame_path: Path, ordinal: int) -> int:
    try:
        return int(frame_path.stem)
    except ValueError:
        return ordinal


def _frame_sort_key(frame_path: Path) -> Tuple[int, int, str]:
    try:
        return (0, int(frame_path.stem), frame_path.name)
    except ValueError:
        return (1, 0, frame_path.name)


def read_mot_annotations(annotation_path: Path) -> Dict[int, List[MotDetection]]:
    """Read VisDrone MOT CSV labels and map its 1-10 classes to YOLO 0-9."""
    frames: Dict[int, List[MotDetection]] = {}
    with annotation_path.open("r", encoding="utf-8", newline="") as annotation_file:
        for line_number, row in enumerate(csv.reader(annotation_file), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 8:
                raise ValueError(
                    "Invalid VisDrone-MOT annotation at {}:{}; expected at least 8 columns"
                    .format(annotation_path, line_number)
                )
            frame_number, track_id = int(row[0]), int(row[1])
            x, y, width, height = map(float, row[2:6])
            confidence, category = float(row[6]), int(row[7])
            if confidence <= 0 or track_id < 0 or category < 1 or category > 10:
                continue
            if width <= 0 or height <= 0:
                continue
            frames.setdefault(frame_number, []).append(
                MotDetection(
                    track_id=track_id,
                    class_id=category - 1,
                    box=(x, y, x + width, y + height),
                )
            )
    return frames


def _prediction_detections(result: Any) -> List[MotDetection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    detections = []
    for box in boxes:
        coords = tuple(float(value) for value in box.xyxy[0])
        track_id = getattr(box, "id", None)
        detections.append(
            MotDetection(
                track_id=int(track_id[0]) if track_id is not None else None,
                class_id=int(box.cls[0]),
                box=(coords[0], coords[1], coords[2], coords[3]),
            )
        )
    return detections


def _iou(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _maximum_weight_assignment(
    weights: Sequence[Sequence[float]],
) -> List[Tuple[int, int]]:
    """Return a maximum-weight one-to-one assignment for a rectangular matrix."""
    row_count = len(weights)
    column_count = len(weights[0]) if row_count else 0
    size = max(row_count, column_count)
    if size == 0:
        return []

    costs = [
        [
            -weights[row][column]
            if row < row_count and column < column_count
            else 0.0
            for column in range(size)
        ]
        for row in range(size)
    ]
    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    assigned_row = [0] * (size + 1)
    previous_column = [0] * (size + 1)

    for row in range(1, size + 1):
        assigned_row[0] = row
        current_column = 0
        minimum_cost = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[current_column] = True
            current_row = assigned_row[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum_cost[column]:
                    minimum_cost[column] = reduced_cost
                    previous_column[column] = current_column
                if minimum_cost[column] < delta:
                    delta = minimum_cost[column]
                    next_column = column
            for column in range(size + 1):
                if used[column]:
                    row_potential[assigned_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum_cost[column] -= delta
            current_column = next_column
            if assigned_row[current_column] == 0:
                break
        while True:
            next_column = previous_column[current_column]
            assigned_row[current_column] = assigned_row[next_column]
            current_column = next_column
            if current_column == 0:
                break

    assignments = []
    for column in range(1, size + 1):
        row = assigned_row[column] - 1
        actual_column = column - 1
        if row < row_count and actual_column < column_count:
            assignments.append((row, actual_column))
    return assignments


def _match_detections(
    ground_truth: Sequence[MotDetection],
    predictions: Sequence[MotDetection],
    iou_threshold: float,
) -> List[Tuple[int, int]]:
    if not ground_truth or not predictions:
        return []
    gt_count = len(ground_truth)
    pred_count = len(predictions)
    match_reward = min(gt_count, pred_count) + 1.0
    weights = [[0.0] * pred_count for _ in range(gt_count)]
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            overlap = _iou(gt.box, prediction.box)
            if gt.class_id == prediction.class_id and overlap >= iou_threshold:
                weights[gt_index][pred_index] = match_reward + overlap
    return [
        (row, column)
        for row, column in _maximum_weight_assignment(weights)
        if weights[row][column] > 0
    ]


def _id_true_positives(
    identity_matches: Dict[Tuple[Tuple[str, int], Tuple[str, int]], int],
    gt_identities: Sequence[Tuple[str, int]],
    pred_identities: Sequence[Tuple[str, int]],
) -> int:
    if not gt_identities or not pred_identities:
        return 0
    weights = [
        [
            identity_matches.get((gt_identity, pred_identity), 0)
            for pred_identity in pred_identities
        ]
        for gt_identity in gt_identities
    ]
    return sum(
        weights[row][column]
        for row, column in _maximum_weight_assignment(weights)
    )


def evaluate_mot_sequences(
    model: Any,
    dataset_root: Path,
    *,
    tracker: str,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    quantize: Optional[int],
    match_iou: float = 0.5,
) -> Dict[str, Any]:
    """Run a persistent tracker over each sequence and calculate MOT metrics."""
    sequences = discover_mot_sequences(dataset_root)
    true_positives = false_positives = false_negatives = id_switches = 0
    total_gt = total_predictions = frame_count = 0
    last_prediction_by_gt: Dict[Tuple[str, int], int] = {}
    identity_matches: Dict[Tuple[Tuple[str, int], Tuple[str, int]], int] = {}
    gt_identities = set()
    pred_identities = set()

    for sequence in sequences:
        annotations = read_mot_annotations(sequence.annotations)
        image_paths = sorted(
            (
                path
                for path in sequence.frames_dir.iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=_frame_sort_key,
        )
        frames_by_name = {
            image_path.name: _frame_number(image_path, ordinal)
            for ordinal, image_path in enumerate(image_paths, start=1)
        }
        missing_frames = set(annotations) - set(frames_by_name.values())
        if missing_frames:
            raise ValueError(
                "Ground-truth annotations reference missing frames in sequence '{}': {}"
                .format(sequence.name, ", ".join(map(str, sorted(missing_frames)[:5])))
            )
        processed = 0
        predictor = getattr(model, "predictor", None)
        if predictor is not None:
            for sequence_tracker in getattr(predictor, "trackers", []):
                sequence_tracker.reset()

        for image_path in image_paths:
            results = model.track(
                source=str(image_path),
                persist=True,
                tracker=tracker,
                imgsz=imgsz,
                device=device,
                conf=conf,
                iou=iou,
                quantize=quantize,
                verbose=False,
                save=False,
            )
            if not results:
                raise ValueError(
                    "Tracker returned no result for frame '{}'".format(image_path)
                )
            for result in results:
                result_path = Path(str(result.path)).name
                if result_path not in frames_by_name:
                    raise ValueError(
                        "Tracker returned unexpected frame '{}' for sequence '{}'".format(
                            result_path, sequence.name
                        )
                    )
                frame_number = frames_by_name[result_path]
                gt = annotations.get(frame_number, [])
                predictions = _prediction_detections(result)
                matches = _match_detections(gt, predictions, match_iou)
                frame_count += 1
                processed += 1
                total_gt += len(gt)
                total_predictions += len(predictions)
                true_positives += len(matches)
                false_negatives += len(gt) - len(matches)
                false_positives += len(predictions) - len(matches)

                for gt_index, prediction_index in matches:
                    gt_track_id = gt[gt_index].track_id
                    prediction_id = predictions[prediction_index].track_id
                    if gt_track_id is None or prediction_id is None:
                        continue
                    gt_identity = (sequence.name, gt_track_id)
                    pred_identity = (sequence.name, prediction_id)
                    gt_identities.add(gt_identity)
                    pred_identities.add(pred_identity)
                    identity_matches[(gt_identity, pred_identity)] = (
                        identity_matches.get((gt_identity, pred_identity), 0) + 1
                    )
                    previous_prediction = last_prediction_by_gt.get(gt_identity)
                    if (
                        previous_prediction is not None
                        and previous_prediction != prediction_id
                    ):
                        id_switches += 1
                    last_prediction_by_gt[gt_identity] = prediction_id

        if processed != len(image_paths):
            raise ValueError(
                "Tracker processed {} of {} frames in sequence '{}'".format(
                    processed, len(image_paths), sequence.name
                )
            )

    id_true_positives = _id_true_positives(
        identity_matches, sorted(gt_identities), sorted(pred_identities)
    )
    id_false_positives = total_predictions - id_true_positives
    id_false_negatives = total_gt - id_true_positives
    id_denominator = 2 * id_true_positives + id_false_positives + id_false_negatives

    return {
        "status": "evaluated",
        "sequences": len(sequences),
        "frames": frame_count,
        "ground_truth_detections": total_gt,
        "predicted_detections": total_predictions,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "id_switches": id_switches,
        "mota": (
            1.0 - (false_negatives + false_positives + id_switches) / total_gt
            if total_gt
            else None
        ),
        "id_true_positives": id_true_positives,
        "id_false_positives": id_false_positives,
        "id_false_negatives": id_false_negatives,
        "idf1": (
            2.0 * id_true_positives / id_denominator if id_denominator else None
        ),
        "match_iou_threshold": match_iou,
    }
