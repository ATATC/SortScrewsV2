from mipcandy import auto_device
from sort_screws import ConvNeXtPredictor

from sort_screws_v2 import Sorter

GEARS: list[tuple[int | None, int | None]] = [
    (None, None),
    (11, None),
    (21, None),
    (30, None),
    (37, None),
    (45, None),
    (134, 11),
    (141, 21),
    (146, 30),
    (150, 37)
]

if __name__ == "__main__":
    device = auto_device()
    print(device)
    app = Sorter(
        "/dev/cu.usbserial-110", GEARS, f"trainer/{ConvNeXtPredictor.__name__.replace("Predictor", "Trainer")}/final",
        512, 10, device=device, offset_a=77, offset_b=70
    )
    app.run()
