from __future__ import annotations

import logging
import time
from typing import Hashable, Iterable, Mapping
import gc

import torch
from monai.config import KeysCollection
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    MapTransform,
    Transform,
)

from sw_fastedit.click_definitions import LABELS_KEY
from sw_fastedit.utils.helper import (
    describe_batch_data,
)
from sw_fastedit.utils.logger import get_logger, setup_loggers

logger = None

cast_labels_to_zero_and_one = lambda x: torch.where(x > 0, 1, 0)

def threshold_foreground(x):
    return (x > 0.005) & (x < 0.995)


class TrackTimed(Transform):
    """
    A transform that tracks and logs the execution time of another transform.

    Args:
        transform (Transform): The transform to be timed.
    """
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        global logger
        start_time = time.perf_counter()
        data = self.transform(data)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"-------- {self.transform.__class__.__qualname__:<20.20}() took {total_time:.3f} seconds")

        return data


class CheckTheAmountOfInformationLossByCropd(MapTransform):
    """
    Prints information about the amount of information lost due to cropping on labeled data.

    Args:
        keys (KeysCollection): Keys to apply the transform on.
        roi_size (Iterable): Size of the region of interest (ROI) after cropping.
        crop_foreground (bool): Whether to crop the foreground before the main crop. Default is True.
    """
    def __init__(self, keys: KeysCollection, roi_size: Iterable, crop_foreground=True):
        super().__init__(keys)
        self.roi_size = roi_size
        self.crop_foreground = crop_foreground

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        labels = data[LABELS_KEY]
        for key in self.key_iterator(data):
            if key == "label":
                t = []
                if self.crop_foreground:
                    t.append(
                        CropForegroundd(
                            keys=("image", "label"),
                            source_key="image",
                            select_fn=threshold_foreground,
                        )
                    )
                if self.roi_size is not None:
                    t.append(CenterSpatialCropd(keys="label", roi_size=self.roi_size))

                if len(t):
                    # copy the label and crop it to the desired size
                    label = data[key]
                    new_data = {"label": label.clone(), "image": data["image"].clone()}

                    cropped_label = Compose(t)(new_data)["label"]

                    for idx, (key_label, _) in enumerate(labels.items(), start=1):
                        # Only count non-background lost labels
                        if key_label != "background":
                            sum_label = torch.sum(label == idx).item()
                            sum_cropped_label = torch.sum(cropped_label == idx).item()
                            # then check how much of the labels is lost
                            lost_pixels = sum_label - sum_cropped_label
                            if sum_label != 0:
                                lost_pixels_ratio = lost_pixels / sum_label * 100
                                logger.info(
                                    f"{lost_pixels_ratio:.1f} % of labelled pixels of the type {key_label} have been lost when cropping"
                                )
                            else:
                                logger.info("No labeled pixels found for current image")
                                logger.debug(f"image {data['image_meta_dict']['filename_or_obj']}")
            else:
                raise UserWarning("This transform only applies to key 'label'")
        return data




class InitLoggerd(MapTransform):
    """
    Initializes the logger inside the dataloader thread.

    Args:
        loglevel (int): Logging level. Default is logging.INFO.
        no_log (bool): Whether to disable logging. Default is True.
        log_dir (str): Directory to save log files. Default is None.

    Note:
        If `no_log` is True, logging will be disabled, and `log_dir` will be ignored.
    """
    def __init__(self, loglevel=logging.INFO, no_log=True, log_dir=None):

        global logger
        super().__init__(None)

        self.loglevel = loglevel
        self.log_dir = log_dir
        self.no_log = no_log

        if self.no_log:
            self.log_dir = None

        setup_loggers(self.loglevel, self.log_dir)
        logger = get_logger()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        global logger
        if logger is None:
            setup_loggers(self.loglevel, self.log_dir)
        logger = get_logger()
        return data
