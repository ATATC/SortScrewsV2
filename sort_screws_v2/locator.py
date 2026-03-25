from typing import Sequence

import torch
from torch import nn
from mipcandy import HasDevice, Device
from transformers import GroundingDinoProcessor, AutoModelForZeroShotObjectDetection

from sort_screws_v2.data import CaseLabel


class Locator(HasDevice):
    def __init__(self, classes: Sequence[str], *, model_id: str = "IDEA-Research/grounding-dino-tiny",
                 box_threshold: float = .3, text_threshold: float = .25, device: Device = "cpu") -> None:
        super().__init__(device)
        self.classes: tuple[str, ...] = tuple(classes)
        self.box_threshold: float = box_threshold
        self.text_threshold: float = text_threshold
        self._processor: GroundingDinoProcessor = GroundingDinoProcessor.from_pretrained(model_id)
        self._model: nn.Module = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self._device)

    @torch.no_grad()
    def predict_image(self, x: torch.Tensor, *, batch: bool = False) -> CaseLabel | list[CaseLabel]:
        if not batch:
            x = x.unsqueeze(0)
        b, c, h, w = x.shape
        x = self._processor(x, ".".join(self.classes) + ".", return_tensors="pt").to(self._device)
        outputs = self._model(**x)
        outputs = self._processor.post_process_grounded_object_detection(
            outputs, x.input_ids, self.box_threshold, self.text_threshold, target_sizes=[(c, h, w)] * b
        )
        return list(map(CaseLabel.from_hf, outputs)) if batch else CaseLabel.from_hf(outputs[0])