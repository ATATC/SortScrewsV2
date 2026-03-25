from mipcandy import auto_device
from sort_screws import ResNetPredictor

from sort_screws_v2 import Sorter

GEARS: list[int] = [180, 90 - 39, 90 - 22, 90, 90 + 22, 90 + 39, 90 + 50]

if __name__ == "__main__":
    device = auto_device()
    print(device)
    app = Sorter(
        "/dev/cu.usbserial-110", GEARS, f"trainer/{ResNetPredictor.__name__.replace("Predictor", "Trainer")}/final",
        512, 7, device=device
    )
    app.run()
