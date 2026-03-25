import cv2
import numpy as np
import torch
from mipcandy import Device


def cv2pt(x: np.ndarray, *, device: Device = "cpu") -> torch.Tensor:
    return torch.as_tensor(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), dtype=torch.float, device=device).permute(2, 0, 1)
