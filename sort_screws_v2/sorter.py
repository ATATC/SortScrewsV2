from os import PathLike
from typing import override, Sequence
from collections import deque

import cv2
import numpy as np
import torch
from mipcandy import HasDevice, Device
from sort_screws import Camera, ResNetPredictor, cv2pt
from torchvision.transforms import Resize

from sort_screws_v2.controller import Controller


class Sorter(Camera, HasDevice):
    def __init__(self, controller_port: str, gears: Sequence[int], experiment_folder: str | PathLike[str],
                 roi_size: int, num_classes: int, *, resize: int = 224, window_size: int = 30,
                 device: Device = "cpu") -> None:
        Camera.__init__(self, roi_size)
        HasDevice.__init__(self, device)
        self.controller: Controller = Controller(controller_port)
        if len(gears) != num_classes:
            raise ValueError(f"Expected {num_classes} gears, got {len(gears)}")
        self.gears: tuple[int, ...] = tuple(gears)
        self.predictor: ResNetPredictor = ResNetPredictor(str(experiment_folder), (3, roi_size, roi_size),
                                                          device=device)
        self.predictor.num_classes = num_classes
        self.paused: bool = False
        self.resize: Resize = Resize(resize)
        self.class_id_window: deque[int] = deque(maxlen=window_size)
        self.confidence_window: deque[float] = deque(maxlen=window_size)

    def is_class_recognized(self, confidence: float, class_id: int) -> bool:
        self.confidence_window.append(confidence)
        self.class_id_window.append(class_id)
        class_id_match_ratio = sum(1 for cid in self.class_id_window if cid == class_id) / len(self.class_id_window)
        return np.percentile(self.confidence_window, 90) > 0.8 and class_id_match_ratio >= 0.9

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
            if self.is_class_recognized(confidence, class_id):
                self.controller.turn_to(self.gears[class_id])
            else:
                self.controller.reset()
        key = self.wait_key()
        if key == ord(" "):
            self.paused = not self.paused
        if key == ord("q"):
            return True
        return False
