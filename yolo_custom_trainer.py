from unittest import result

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

class WeightedDetectionLoss(v8DetectionLoss):
    """Detection loss with class weights applied to BCE classification loss."""

    def __init__(self, model, class_weights=None, tal_topk=10, tal_topk2=None):
        """Initialize loss with optional per-class weights for BCE."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if class_weights is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.to(self.device),
                reduction="none",
            )

class WeightedE2ELoss(E2ELoss):
    """E2E Loss with class weights for YOLO26."""

    def __init__(self, model, class_weights=None):
        """Initialize E2E loss with weighted detection loss."""

        def weighted_loss_fn(model, tal_topk=10, tal_topk2=None):
            return WeightedDetectionLoss(model, class_weights=class_weights, tal_topk=tal_topk, tal_topk2=tal_topk2)

        super().__init__(model, loss_fn=weighted_loss_fn)

class WeightedDetectionModel(DetectionModel):
    """Detection model that uses class-weighted loss."""

    def init_criterion(self):
        """Initialize weighted loss criterion with per-class weights."""
        class_weights = torch.ones(self.nc)
        class_weights[0] = 2.0  # upweight class 0
        class_weights[1] = 3.0  # upweight rare class 1
        return WeightedE2ELoss(self, class_weights=class_weights)

class WeightedTrainer(DetectionTrainer):
    """Trainer that returns a WeightedDetectionModel."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a WeightedDetectionModel."""
        model = WeightedDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

model = YOLO("vis_epoch50_openvino_model")

# model.train(data="coco8.yaml", epochs=10, trainer=WeightedTrainer)
results = model.track(
    source="video_example/12805671_3840_2160_60fps.mp4",
    task="detect",
    imgsz=640,
    save=True,
    conf=0.25,
    persist=True,
)

for r in results:
    pass