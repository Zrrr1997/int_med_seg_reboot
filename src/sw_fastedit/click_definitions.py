from __future__ import annotations

from enum import IntEnum

LABELS_KEY = "label_names"


class ClickGenerationStrategy(IntEnum):
    """
    Enumeration representing different strategies for generating clicks during interactive segmentation.

    Attributes:
        GLOBAL_NON_CORRECTIVE (int): Sample a click randomly based on the label, without correction based on the prediction.
        GLOBAL_CORRECTIVE (int): Sample a click based on the discrepancy between the label and prediction, generating corrective clicks where the network predicts incorrectly.
        PATCH_BASED_CORRECTIVE (int): Subdivide the volume into patches of size train_crop_size, calculate the dice score for each, then sample a click on the worst-performing one. Supported only during validation and testing.
        DEEPGROW_GLOBAL_CORRECTIVE (int): At each iteration, sample from the probability and don't add a click if it yields False.
    """
    GLOBAL_NON_CORRECTIVE = 1
    GLOBAL_CORRECTIVE = 2
    PATCH_BASED_CORRECTIVE = 3
    DEEPGROW_GLOBAL_CORRECTIVE = 4
    READ_FROM_JSON = 5


class StoppingCriterion(IntEnum):
    """
    Enumeration representing different stopping criteria for the interactive segmentation process.

    Attributes:
        MAX_ITER (int): Sample max_train_interactions amount of clicks (can be done in the first iteration if non-corrective).
        MAX_ITER_AND_PROBABILITY (int): Sample clicks iteratively. At each step, sample p~(0,1). If p > x, continue sampling.
        MAX_ITER_AND_DICE (int): Sample clicks iteratively. Stop when dice is good enough (e.g., 0.9) or when max_train_interactions amount of clicks is reached.
        MAX_ITER_PROBABILITY_AND_DICE (int): Sample clicks iteratively. At each step, stop if max_train_interactions is reached. Otherwise, sample p~(0,1).
            If p > dice, continue sampling, then check if dice is good enough. If so, no more clicks are required.
        DEEPGROW_PROBABILITY (int): Stopping criterion as previously implemented with Deepgrow.
    """
    MAX_ITER = 1
    MAX_ITER_AND_PROBABILITY = 2
    MAX_ITER_AND_DICE = 3
    MAX_ITER_PROBABILITY_AND_DICE = 4
    DEEPGROW_PROBABILITY = 5
