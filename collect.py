from mipcandy import auto_device

from sort_screws_v2 import Collector

DATASET_DIR: str = "SortScrewsV2"

if __name__ == "__main__":
    app = Collector(DATASET_DIR, device=auto_device())
    app.run()
