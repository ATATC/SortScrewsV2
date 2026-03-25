from dataclasses import dataclass, asdict
from os import PathLike
from typing import Self, Sequence
from json import dump, load

import torch

type BBox = tuple[int, int, int, int]


@dataclass
class ClassManager(object):
    classes: list[str]
    assigned_colors: list[tuple[int, int, int]]

    def class_id(self, class_name: str) -> int:
        return self.classes.index(class_name)

    def class_name(self, class_id: int) -> str:
        return self.classes[class_id]

    def color(self, class_name) -> tuple[int, int, int]:
        return self.assigned_colors[self.class_id(class_name)]


DEFAULT_CLASS_MANAGER: ClassManager = ClassManager(["round", "flat", "truss"], [(255, 0, 0), (0, 255, 0), (0, 0, 255)])


@dataclass
class CaseLabel(object):
    num_objects: int
    bboxes: list[BBox]
    text_labels: list[str]
    labels: list[str]
    scores: list[float]

    @classmethod
    def from_hf(cls, outputs: dict[str, torch.Tensor | list[str]]) -> Self:
        scores = outputs["scores"].tolist()
        num_objects = len(scores)
        bboxes = outputs["boxes"].tolist()
        text_labels = outputs["text_labels"]
        labels = outputs["labels"]
        if not (num_objects == len(bboxes) == len(text_labels) == len(labels)):
            raise ValueError(f"Inconsistent number of objects: {num_objects} vs {len(bboxes)} vs {len(text_labels)} vs {len(labels)}")
        return cls(num_objects, list(map(tuple, bboxes)), text_labels, labels, scores)


def save_labels(labels: Sequence[CaseLabel], path: str | PathLike[str]) -> None:
    with open(path, "w") as f:
        dump([asdict(case) for case in labels], f)


def load_labels(path: str | PathLike[str]) -> list[CaseLabel]:
    with open(path) as f:
        return [CaseLabel(**case) for case in load(f)]
