from mipcandy import auto_device
from sort_screws import ConvNeXtPredictor

from sort_screws_v2 import Sorter

GEARS: list[int] = [180, 150, 148, 145, 140, 135, 129, 121, 112, 102]

if __name__ == "__main__":
    device = auto_device()
    print(device)
    app = Sorter(
        "/dev/cu.usbserial-1110", GEARS, f"trainer/{ConvNeXtPredictor.__name__.replace("Predictor", "Trainer")}/final",
        224, 10, device=device, servo_offset=-10, default_angle=180
    )
    app.run()
