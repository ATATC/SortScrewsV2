import cv2
import numpy as np


class Camera(object):
    def __init__(self, roi_size: int, *, resize: int | None = None, device_id: int = 0) -> None:
        self._roi_size: int = roi_size
        self._resize: int | None = resize
        self._device_id: int = device_id

    @staticmethod
    def wait_key(*, delay: int = 1) -> int:
        return cv2.waitKey(delay) & 0xFF

    def job(self, frame: np.ndarray, roi: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        ...

    def run(self) -> None:
        cap = cv2.VideoCapture(self._device_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam {self._device_id}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            cx = w // 2
            cy = h // 2
            half = self._roi_size // 2
            x1 = cx - half
            y1 = cy - half
            x2 = cx + half
            y2 = cy + half
            roi = frame[y1:y2, x1:x2]
            if self._resize:
                roi = cv2.resize(roi, (self._resize, self._resize), interpolation=cv2.INTER_AREA)
            if self.job(frame, roi.copy(), (x1, y1, x2, y2)):
                break
        cap.release()
        cv2.destroyAllWindows()
