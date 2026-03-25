from mipcandy import auto_device
from sort_screws import ResNetPredictor

from sort_screws_v2 import Sorter

GEARS: list[int] = [180, -39, -22, 0, 22, 39, 50]

if __name__ == "__main__":
    device = auto_device()
    print(device)
    app = Sorter(
        "/dev/cu.usbserial-110", GEARS, f"trainer/{ResNetPredictor.__name__.replace("Predictor", "Trainer")}/final",
        512, device=device
    )
    app.run()
