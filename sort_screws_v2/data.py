from dataclasses import dataclass, asdict
from os import PathLike
from typing import Self, Sequence
from json import dump, load

import torch

type BBox = tuple[int, int, int, int]


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
        bboxes = outputs["boxes"].round().int().tolist()
        text_labels = outputs["text_labels"]
        if "" in text_labels:
            text_labels.remove("")
        labels = outputs["labels"]
        if "" in labels:
            labels.remove("")
        if not (num_objects == len(bboxes) == len(text_labels) == len(labels)):
            raise ValueError(f"Inconsistent number of objects: {scores}, {bboxes}, {text_labels}, {labels}")
        return cls(num_objects, list(map(tuple, bboxes)), text_labels, labels, scores)


def save_labels(labels: Sequence[CaseLabel], path: str | PathLike[str]) -> None:
    with open(path, "w") as f:
        dump([asdict(case) for case in labels], f)


def load_labels(path: str | PathLike[str]) -> list[CaseLabel]:
    with open(path) as f:
        return [CaseLabel(**case) for case in load(f)]
