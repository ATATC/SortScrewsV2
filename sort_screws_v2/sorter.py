from os import PathLike
from typing import override, Sequence

import cv2
import numpy as np
import torch
from mipcandy import HasDevice, Device
from sort_screws import Camera, ResNetPredictor, cv2pt
from torchvision.transforms import Resize

from sort_screws_v2.controller import Controller


class Sorter(Camera, HasDevice):
    def __init__(self, controller_port: str, gears: Sequence[int], experiment_folder: str | PathLike[str],
                 roi_size: int, *, resize: int = 224, device: Device = "cpu") -> None:
        Camera.__init__(self, roi_size)
        HasDevice.__init__(self, device)
        self.controller: Controller = Controller(controller_port)
        if len(gears) != ResNetPredictor.num_classes:
            raise ValueError(f"Expected {ResNetPredictor.num_classes} gears, got {len(gears)}")
        self.gears: tuple[int, ...] = tuple(gears)
        self.predictor: ResNetPredictor = ResNetPredictor(str(experiment_folder), (3, roi_size, roi_size),
                                                          device=device)
        self.paused: bool = False
        self.resize: Resize = Resize(resize)

    @override
    def job(self, frame: np.ndarray, roi: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        if not self.paused:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image = cv2pt(roi, device=self._device)
            image = self.resize(image)
            logits = self.predictor.predict_image(image)
            probs = logits.softmax(0)
            confidence, class_id = torch.max(probs, 0)
            confidence, class_id = confidence.item(), class_id.item()
            cv2.putText(frame, f"Class: {class_id} @ {confidence * 100:.2f}%", (40, 80), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Camera Preview", frame)
            if confidence > 0.5:
                self.controller.turn_to(self.gears[class_id])
        key = self.wait_key()
        if key == ord(" "):
            self.paused = not self.paused
        if key == ord("q"):
            return True
        return False
