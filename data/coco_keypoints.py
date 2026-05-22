from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import (
    apply_obfuscation,
    aspect_ratio_box,
    crop_and_resize,
    generate_keypoint_heatmaps,
    keypoints_to_crop,
    normalize_image,
    to_hw,
)


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class CocoKeypointsTopDownDataset(Dataset):
    """COCO person-instance crops with Gaussian keypoint heatmap targets."""

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
        input_size: tuple[int, int] = (256, 192),
        heatmap_size: tuple[int, int] = (64, 48),
        sigma: float = 2.0,
        bbox_padding: float = 1.25,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        skip_empty: bool = True,
        max_samples: int | None = None,
        obfuscation: dict[str, Any] | None = None,
        deterministic_obfuscation: bool = False,
        obfuscation_seed: int = 0,
    ) -> None:
        self.image_root = Path(image_root)
        self.annotation_file = Path(annotation_file)
        self.input_size = to_hw(input_size)
        self.heatmap_size = to_hw(heatmap_size)
        self.sigma = sigma
        self.bbox_padding = bbox_padding
        self.image_mean = image_mean
        self.image_std = image_std
        self.obfuscation = obfuscation or {"mode": "none"}
        self.deterministic_obfuscation = deterministic_obfuscation
        self.obfuscation_seed = obfuscation_seed

        with self.annotation_file.open("r", encoding="utf-8") as handle:
            coco: dict[str, Any] = json.load(handle)

        self.images = {int(image["id"]): image for image in coco["images"]}
        annotations = []
        for ann in coco["annotations"]:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            valid_bbox = len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0
            is_person = ann.get("category_id", 1) == 1
            has_keypoints = ann.get("num_keypoints", 0) > 0
            if is_person and valid_bbox and (has_keypoints or not skip_empty):
                annotations.append(ann)

        if max_samples is not None:
            annotations = annotations[:max_samples]
        self.annotations = annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ann = self.annotations[index]
        image_info = self.images[int(ann["image_id"])]
        image_path = self.image_root / image_info["file_name"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop_box = aspect_ratio_box(ann["bbox"], self.input_size, padding=self.bbox_padding)
            crop = crop_and_resize(image, crop_box, self.input_size)
            if self.deterministic_obfuscation:
                rng = random.Random(self.obfuscation_seed + index)
            else:
                rng = random
            crop = apply_obfuscation(crop, self.obfuscation, rng=rng)

        raw_keypoints = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        keypoints_xy = keypoints_to_crop(raw_keypoints[:, :2], crop_box, self.input_size)
        visibility = raw_keypoints[:, 2].astype(np.float32)
        target_heatmaps, target_weights = generate_keypoint_heatmaps(
            keypoints_xy=keypoints_xy,
            visibility=visibility,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
        )

        return {
            "pixel_values": normalize_image(crop, self.image_mean, self.image_std),
            "target_heatmaps": target_heatmaps,
            "target_weights": target_weights,
            "keypoints": torch.from_numpy(keypoints_xy),
            "visibility": torch.from_numpy(visibility),
            "bbox": torch.tensor(ann["bbox"], dtype=torch.float32),
            "crop_box": torch.from_numpy(crop_box),
            "image_id": torch.tensor(int(ann["image_id"]), dtype=torch.long),
            "annotation_id": torch.tensor(int(ann["id"]), dtype=torch.long),
        }
