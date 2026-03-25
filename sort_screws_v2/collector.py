from os import PathLike, makedirs, remove
from typing import override

import cv2
import numpy as np
from mipcandy import HasDevice, Device

from sort_screws_v2.camera import Camera
from sort_screws_v2.pre_labeler import PreLabeler
from sort_screws_v2.utils import cv2pt
from sort_screws_v2.dataset import CaseLabel, ClassManager, DEFAULT_CLASS_MANAGER, load_labels, save_labels


class Collector(Camera, HasDevice):
    def __init__(self, dataset_dir: str | PathLike[str], *, class_manager: ClassManager = DEFAULT_CLASS_MANAGER,
                 append: bool = False, device: Device = "cpu") -> None:
        Camera.__init__(self, 512)
        HasDevice.__init__(self, device)
        self.images_dir: str = f"{dataset_dir}/images"
        self.json_path: str = f"{dataset_dir}/labels.json"
        self.class_manager: ClassManager = class_manager
        self.labeler: PreLabeler = PreLabeler(class_manager.classes, device=device)
        # runtime
        self.labels: list[CaseLabel] = []
        if append:
            self.labels = load_labels(self.json_path)
        else:
            makedirs(self.images_dir)

    @property
    def num_cases(self) -> int:
        return len(self.labels)

    @override
    def job(self, frame: np.ndarray, roi: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Num cases: {self.num_cases}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                    2,
                    cv2.LINE_AA)
        pre_label = self.labeler.predict_image(cv2pt(roi, device=self._device))
        for i in range(pre_label.num_objects):
            x1, y1, x2, y2 = pre_label.bboxes[i]
            class_name = pre_label.text_labels[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.class_manager.color(class_name), 2)
        cv2.imshow("Camera Preview", frame)
        key = self.wait_key()
        if key == ord("c"):
            filename = f"case_{self.num_cases:04d}.png"
            cv2.imwrite(f"{self.images_dir}/{filename}", roi)
            self.labels.append(pre_label)
            save_labels(self.labels, self.json_path)
        if key == ord("s"):
            self.labels.pop(-1)
            remove(f"{self.images_dir}/case_{self.num_cases:04d}.png")
            save_labels(self.labels, self.json_path)
        elif key == ord("q"):
            return True
        return False