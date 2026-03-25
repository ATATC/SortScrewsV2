from os import PathLike
from typing import override

import cv2
import numpy as np
import torch
from mipcandy import HasDevice, Device, auto_device
from sort_screws import Camera, ResNetPredictor, cv2pt
from torchvision.transforms import Resize

from sort_screws_v2 import Locator


GEARS: list[int] = [-39, -22, 0, 22, 39, 50]


class Predictor(Camera, HasDevice):
    def __init__(self, experiment_folder: str | PathLike[str], *, device: Device = "cpu") -> None:
        Camera.__init__(self, 512)
        HasDevice.__init__(self, device)
        self.predictor: ResNetPredictor = ResNetPredictor(str(experiment_folder), (3, 512, 512),
                                                          device=device)
        self.paused: bool = False
        self.resize: Resize = Resize(224)

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
            cv2.putText(frame, f"Class: {class_id} @ {confidence * 100:.2f}%", (40, 80), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (0, 255, 0), 2, cv2.LINE_AA)
            label = self.locator.predict_image(image)
            for bbox in label.bboxes:
                x3, y3, x4, y4 = bbox
                cv2.rectangle(frame, (x1 + x3, y1 + y3), (x1 + x4, y1 + y4), (0, 0, 255), 2)
            cv2.imshow("Camera Preview", frame)
        key = self.wait_key()
        if key == ord(" "):
            self.paused = not self.paused
        if key == ord("q"):
            return True
        return False


if __name__ == "__main__":
    device = auto_device()
    print(device)
    app = Predictor(f"trainer/{ResNetPredictor.__name__.replace("Predictor", "Trainer")}/final", device=device)
    app.run()
