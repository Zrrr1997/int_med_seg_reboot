# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import os
import cc3d
import json
import logging
from scipy import ndimage
from skimage.morphology import skeletonize


from typing import Dict, Hashable, List, Mapping, Tuple

import torch
import numpy as np
import nibabel as nib
from monai.config import KeysCollection
from monai.data import MetaTensor, PatchIterd
from monai.losses import DiceLoss
from monai.networks.layers import GaussianFilter
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    MapTransform,
    Randomizable,
    LazyTransform,
    InvertibleTransform,
    Flip,
)
from collections.abc import Hashable, Mapping, Sequence


from monai.utils.enums import CommonKeys

from sw_fastedit.click_definitions import LABELS_KEY, ClickGenerationStrategy
from sw_fastedit.utils.distance_transform import get_random_choice_from_tensor
from monai.transforms.utils import distance_transform_edt
from sw_fastedit.utils.helper import get_global_coordinates_from_patch_coordinates, get_tensor_at_coordinates, timeit
from FastGeodis import generalised_geodesic3d


logger = logging.getLogger("sw_fastedit")


def get_guidance_tensor_for_key_label(data, key_label, device) -> torch.Tensor:
    """Makes sure the guidance is in a tensor format."""
    tmp_gui = data.get(key_label, torch.tensor([], dtype=torch.int32, device=device))
    if isinstance(tmp_gui, list):
        tmp_gui = torch.tensor(tmp_gui, dtype=torch.int32, device=device)
    assert type(tmp_gui) is torch.Tensor or type(tmp_gui) is MetaTensor
    return tmp_gui


# TODO Franzi - one transform class - AddExtremePoints - already included in MONAI

class AddEmptySignalChannels(MapTransform):
    """
        Adds empty channels to the signal which will be filled with the guidance signal later.
        E.g. for two labels: 1x192x192x256 -> 3x192x192x256
    """
    def __init__(self, device, keys: KeysCollection = None):
        super().__init__(keys)
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set up the initial batch data
        in_channels = 1 + len(data[LABELS_KEY]) 
        tmp_image = data[CommonKeys.IMAGE][0 : 0 + 1, ...]
        assert len(tmp_image.shape) == 4
        new_shape = list(tmp_image.shape)
        new_shape[0] = in_channels
        # Set the signal to 0 for all input images
        # image is on channel 0 of e.g. (1,128,128,128) and the signals get appended, so
        # e.g. (3,128,128,128) for two labels
        inputs = torch.zeros(new_shape) #, device=self.device)
        inputs[0] = data[CommonKeys.IMAGE][0]
        if isinstance(data[CommonKeys.IMAGE], MetaTensor):
            data[CommonKeys.IMAGE].array = inputs
        else:
            data[CommonKeys.IMAGE] = inputs

        return data


class NormalizeLabelsInDatasetd(MapTransform):
    """
    Normalize label values according to label names dictionary

    Args:
        keys: the ``keys`` parameter will be used to get and set the actual data item to transform
        labels: all label names
        allow_missing_keys: whether to ignore it if keys are missing.
        device: device this transform shall run on

    Returns: data and also the new labels will be stored in data with key LABELS_KEY
    """
    def __init__(
        self,
        keys: KeysCollection,
        labels=None,
        allow_missing_keys: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.labels = labels
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set the labels dict in case no labels were provided
        data[LABELS_KEY] = self.labels

        for key in self.key_iterator(data):
            if key == "label":
                try:
                    label = data[key]
                    if isinstance(label, str):
                        # Special case since label has been defined to be a string in MONAILabel
                        raise AttributeError
                except AttributeError:
                    # label does not exist - this might be a validation run
                    break

                # Dictionary containing new label numbers
                new_labels = {}
                label = torch.zeros(data[key].shape, device=self.device)
                # Making sure the range values and number of labels are the same
                for idx, (key_label, val_label) in enumerate(self.labels.items(), start=1):
                    if key_label != "background":
                        new_labels[key_label] = idx
                        label[data[key] == val_label] = idx
                    if key_label == "background":
                        new_labels["background"] = 0
                    else:
                        new_labels[key_label] = idx
                        label[data[key] == val_label] = idx

                data[LABELS_KEY] = new_labels
                if isinstance(data[key], MetaTensor):
                    data[key].array = label
                else:
                    data[key] = label
            else:
                raise UserWarning("Only the key label is allowed here!")
        return data


class AddGuidanceSignal(MapTransform):
    """
    Add Guidance signal for input image.

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
        disks: This paraemters fill spheres with a radius of sigma centered around each click.
        device: device this transform shall run on.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: int = 1,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
        disks: bool = False,
        gdt: bool = False,
        spacing: Tuple = None,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.disks = disks
        self.gdt = gdt
        self.spacing = spacing
        self.device = device

    def _get_corrective_signal(self, image, guidance, key_label):
        dimensions = 3 if len(image.shape) > 3 else 2
        assert (
            type(guidance) is torch.Tensor or type(guidance) is MetaTensor
        ), f"guidance is {type(guidance)}, value {guidance}"



        if guidance.size()[0]:
            first_point_size = guidance[0].numel()
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                # Assuming the guidance has either shape (1, x, y , z) or (x, y, z)
                assert (
                    first_point_size == 4 or first_point_size == 3
                ), f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                assert first_point_size == 3, f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)

            sshape = signal.shape

            for point in guidance:
                if torch.any(point < 0):
                    continue
                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2] = 1.0

            # Apply a Gaussian filter to the signal
            if torch.max(signal[0]) > 0:
                signal_tensor = signal[0]
                if self.sigma != 0:
                    pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                    signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                    signal_tensor = signal_tensor.squeeze(0).squeeze(0)

                signal[0] = signal_tensor
                signal[0] = (signal[0] - torch.min(signal[0])) / (torch.max(signal[0]) - torch.min(signal[0]))
                if self.disks:
                    signal[0] = (signal[0] > 0.1) * 1.0  # 0.1 with sigma=1 --> radius = 3, otherwise it is a cube

                if self.gdt:
                    geos = generalised_geodesic3d(image.unsqueeze(0).to(self.device),
                                                signal[0].unsqueeze(0).unsqueeze(0).to(self.device),
                                                self.spacing,
                                                10e10,
                                                1.0,
                                                2)




                    signal[0] = geos[0][0]

            if not (torch.min(signal[0]).item() >= 0 and torch.max(signal[0]).item() <= 1.0) and (not self.gdt):
                raise UserWarning(
                    "[WARNING] Bad signal values",
                    torch.min(signal[0]),
                    torch.max(signal[0]),
                )



            if signal is None:
                raise UserWarning("[ERROR] Signal is None")
            return signal
        else:
            if dimensions == 3:
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)
            if signal is None:
                logger.warning("Guidance Signal is None")
            return signal

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key == "image":
                image = data[key]
                assert image.is_cuda 
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]


                for _, (label_key, _) in enumerate(data[LABELS_KEY].items()):
                    label_guidance = get_guidance_tensor_for_key_label(data, label_key, self.device)
                    logger.debug(f"Converting guidance for label {label_key}:{label_guidance} into a guidance signal..")

                    if label_guidance is not None and label_guidance.numel():
                        signal = self._get_corrective_signal(
                            image,
                            label_guidance.to(device=self.device),
                            key_label=label_key,
                        )
                        if not self.gdt:
                            assert torch.sum(signal) > 0
                    else:
                        signal = self._get_corrective_signal(
                            image,
                            torch.Tensor([]).to(device=self.device),
                            key_label=label_key,
                        )

                    assert signal.is_cuda 
                    assert tmp_image.is_cuda 
                    tmp_image = torch.cat([tmp_image, signal], dim=0)
                    if isinstance(data[key], MetaTensor):
                        data[key].array = tmp_image
                    else:
                        data[key] = tmp_image
                return data
            else:
                raise UserWarning("This transform only applies to image key")
        raise UserWarning("image key has not been been found")


class FindDiscrepancyRegions(MapTransform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        pred_key: key to prediction source.
        discrepancy_key: key to store discrepancies found between label and prediction.
        device: device this transform shall run on.
    """

    def __init__(
        self,
        keys: KeysCollection,
        pred_key: str = "pred",
        discrepancy_key: str = "discrepancy",
        allow_missing_keys: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred_key = pred_key
        self.discrepancy_key = discrepancy_key
        self.device = device

    def disparity(self, label, pred):
        disparity = label - pred
        # +1 means predicted label is not part of the ground truth
        # -1 means predicted label missed that region of the ground truth
        pos_disparity = (disparity > 0).to(dtype=torch.float32, device=self.device)  # FN
        neg_disparity = (disparity < 0).to(dtype=torch.float32, device=self.device)  # FP
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        return self.disparity(label, pred)

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key == "label":
                assert (
                    (type(data[key]) is torch.Tensor)
                    or (type(data[key]) is MetaTensor)
                    and (type(data[self.pred_key]) is torch.Tensor or type(data[self.pred_key]) is MetaTensor)
                )
                all_discrepancies = {}
                assert data[key].is_cuda and data["pred"].is_cuda

                for _, (label_key, label_value) in enumerate(data[LABELS_KEY].items()):
                    if label_key != "background":
                        label = torch.clone(data[key].detach())
                        # Label should be represented in 1
                        label[label != label_value] = 0
                        label = (label > 0.5).to(dtype=torch.float32)

                        # Taking single prediction
                        pred = torch.clone(data[self.pred_key].detach())
                        pred[pred != label_value] = 0
                        # Prediction should be represented in one
                        pred = (pred > 0.5).to(dtype=torch.float32)
                    else:
                        # Taking single label
                        label = torch.clone(data[key].detach())
                        label[label != label_value] = 1
                        label = 1 - label
                        # Label should be represented in 1
                        label = (label > 0.5).to(dtype=torch.float32)
                        # Taking single prediction
                        pred = torch.clone(data[self.pred_key].detach())
                        pred[pred != label_value] = 1
                        pred = 1 - pred
                        # Prediction should be represented in one
                        pred = (pred > 0.5).to(dtype=torch.float32)
                    all_discrepancies[label_key] = self._apply(label, pred)
                data[self.discrepancy_key] = all_discrepancies
                return data
            else:
                logger.error("This transform only applies to 'label' key")
        raise UserWarning


class AddGuidance(Randomizable, MapTransform):
    """
    Add guidance based on different click generation strategies.

    Args:
        discrepancy_key: key to discrepancy map between label and prediction shape (2, C, H, W, D) or (2, C, H, W)
        probability_key: key to click/interaction probability, shape (1)
        device: device this transform shall run on.
        click_generation_strategy_key: sets the used ClickGenerationStrategy.
        patch_size: Only relevant for the patch-based click generation strategy. Sets the size of the cropped patches
        on which then further analysis is run.
    """

    def __init__(
        self,
        keys: KeysCollection,
        discrepancy_key: str = "discrepancy",
        probability_key: str = "probability",
        allow_missing_keys: bool = False,
        device=None,
        click_generation_strategy_key: str = "click_generation_strategy",
        patch_size: Tuple[int] = (128, 128, 128),
        load_from_json: bool = False,
        json_dir : str = None,
        center_click : bool = False, # Center of largest error,
        click_noise : bool = False,
        noise_level : int = 1,
        noise_probability : float = 0.25,
        systematic_error : bool = False,
        systematic_error_probability : float = 0.25,
        random_click : bool = False,
        intensity_based : bool = False,
        uncertainty_based : bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.discrepancy_key = discrepancy_key
        self.probability_key = probability_key
        self._will_interact = None
        self.is_other = None
        self.default_guidance = None
        self.device = device
        self.click_generation_strategy_key = click_generation_strategy_key
        self.patch_size = patch_size
        self.load_from_json = load_from_json
        self.json_dir = json_dir
        self.center_click = center_click
        self.click_noise = click_noise
        self.noise_level = noise_level
        self.noise_probability = noise_probability
        self.systematic_error = systematic_error
        self.systematic_error_probability = systematic_error_probability
        self.uncertainty_based = uncertainty_based
        self.intensity_based = intensity_based

        self.random_click = random_click

    def randomize(self, data: Mapping[Hashable, torch.Tensor]):
        probability = data[self.probability_key]
        self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy, image, label, key_label, sample_id) -> List[int | List[int]] | None:
        assert discrepancy.is_cuda

        if self.center_click:
            distance = distance_transform_edt(discrepancy).detach().cpu().clone().numpy() 
            
            comps = discrepancy[0].detach().cpu().clone().numpy()
            
            labels_out = cc3d.largest_k(
                comps, k=1, 
                connectivity=6, delta=0,
                )
            
            labels_out = skeletonize(labels_out).astype(np.float32) # Lee et al.
            labels_out *= distance[0]


            t_index = np.unravel_index(np.argmax(labels_out), labels_out.shape)
            t_index = np.insert(t_index, 0, 0)
        elif self.random_click: 
            # It seems that it does not differ at all from the EDT-based sampling
            # The probabilities are so small that it is equivalent to uniform sampling
            t_index, t_value = get_random_choice_from_tensor(label)  
        else:
            distance = distance_transform_edt(discrepancy) 
            t_index, t_value = get_random_choice_from_tensor(distance)  

        if self.systematic_error:
            assert self.uncertainty_based or self.intensity_based, 'Must be either uncertainty or intensity-based systematic error!'
            if np.random.rand() < self.systematic_error_probability and key_label != 'background': # Systematic Error 

                if self.uncertainty_based: # Discrepancy between tumor and background prediction channels
                    uncertainty = nib.load(os.path.join('outputs/disks_4_robot_user_center_with_nifti_noise_2_systematic_error_balanced_uncertainty/data/', sample_id, 'uncertainty_0.nii.gz'))
                    uncertainty = np.array(uncertainty.dataobj)
                    uncertainty = torch.Tensor(uncertainty).unsqueeze(0)
                    
                    uncertainty[:,:,:,:80] = 0 # de-bladder
                    uncertainty[:,:,:,-40:] = 0 # de-brain
                    uncertainty = (label.cpu() == 0) * (uncertainty > 0.01) # 0.01 is a fixed hyperparameter

                if self.intensity_based: # High-uptake region sampling
                    label_mean = torch.mean(image[:1][label == 1])
                    high_uptake_background_regions = (label == 0) * (image >= label_mean)
                    high_uptake_background_regions = high_uptake_background_regions[:1,:,:,:]
                    high_uptake_background_regions[:,:,:,:80] = 0 # de-bladder
                    high_uptake_background_regions[:,:,:,-40:] = 0 # de-brain

                    if self.uncertainty_based:
                        t_index, _ = get_random_choice_from_tensor(high_uptake_background_regions.cpu() * uncertainty)  
                    else:
                        t_index, _ = get_random_choice_from_tensor(high_uptake_background_regions.cpu())  
                else:
                    t_index, _ = get_random_choice_from_tensor(uncertainty)  


 


            
        if self.click_noise: # Random Error
            if np.random.rand() < self.noise_probability: # Conditional random noise --> add noise in 25% of the cases 
                noise = np.insert(np.random.randint(-self.noise_level, self.noise_level, 3), 0, 0)
                t_index += noise


        gc.collect()  
        return t_index

    def add_guidance_based_on_discrepancy(
        self,
        data: Dict,
        guidance: torch.Tensor,
        key_label: str,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert guidance.dtype == torch.int32
        # Positive clicks of the segment in the iteration
        discrepancy = data[self.discrepancy_key][key_label]
        # idx 0 is positive discrepancy and idx 1 is negative discrepancy
        pos_discr = discrepancy[0]

        sample_id = data['image_meta_dict']['filename_or_obj'].split('/')[-1].split('.')[0]

        if coordinates is None:
            # Add guidance to the current key label
            if torch.sum(pos_discr) > 0:
                tmp_gui = self.find_guidance(pos_discr, data['image'], data['label'], key_label, sample_id)
                self.check_guidance_length(data, tmp_gui)
                if tmp_gui is not None:
                    guidance = torch.cat(
                        (
                            guidance,
                            torch.tensor([tmp_gui], dtype=torch.int32, device=guidance.device),
                        ),
                        0,
                    )
            else:
                logger.info(f'Nothing to improve, not adding any click for the {key_label}...')
        else:
            pos_discr = get_tensor_at_coordinates(pos_discr, coordinates=coordinates)
            if torch.sum(pos_discr) > 0:
                # TODO Add suport for 2d
                tmp_gui = self.find_guidance(pos_discr, data['image'], data['label'], key_label, sample_id)
                if tmp_gui is not None:
                    tmp_gui = get_global_coordinates_from_patch_coordinates(tmp_gui, coordinates)
                    self.check_guidance_length(data, tmp_gui)
                    guidance = torch.cat(
                        (
                            guidance,
                            torch.tensor([tmp_gui], dtype=torch.int32, device=guidance.device),
                        ),
                        0,
                    )
        return guidance

    def add_guidance_based_on_label(self, data, guidance, label):
        assert guidance.dtype == torch.int32
        # Add guidance to the current key label
        if torch.sum(label) > 0:

            #if self.center_click:
            #    tmp_gui_index = self.find_guidance(data['label'], data['image'], data['label'], label)


            assert label.is_cuda
            # generate a random sample
            tmp_gui_index, tmp_gui_value = get_random_choice_from_tensor(label)
            if tmp_gui_index is not None:
                self.check_guidance_length(data, tmp_gui_index)
                guidance = torch.cat(
                    (
                        guidance,
                        torch.tensor([tmp_gui_index], dtype=torch.int32, device=guidance.device),
                    ),
                    0,
                )
        return guidance

    def check_guidance_length(self, data, new_guidance):
        dimensions = 3 if len(data[CommonKeys.IMAGE].shape) > 3 else 2
        if dimensions == 3:
            assert len(new_guidance) == 4, f"len(new_guidance) is {len(new_guidance)}, new_guidance is {new_guidance}"
        else:
            assert len(new_guidance) == 3, f"len(new_guidance) is {len(new_guidance)}, new_guidance is {new_guidance}"

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        click_generation_strategy = data[self.click_generation_strategy_key]



        if self.load_from_json:
            assert self.json_dir is not None

            for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                num_clicks = len(data[key_label]) + 1 if key_label in data.keys() else 1

                im_fn = data['image_meta_dict']['filename_or_obj'].split('/')[-1]
                json_fn = os.path.join(self.json_dir, im_fn.replace('.nii.gz', '_clicks.json'))
                with open(json_fn, 'r') as f:
                    json_data = json.load(f)
                    json_data[key_label] = [[0] + el for el in json_data[key_label]]
                    json_data[key_label] = json_data[key_label][:num_clicks]
                data[key_label] = json_data[key_label]

        elif click_generation_strategy == ClickGenerationStrategy.GLOBAL_NON_CORRECTIVE:
            # uniform random sampling on label
            for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                tmp_gui = get_guidance_tensor_for_key_label(data, key_label, self.device)
                data[key_label] = self.add_guidance_based_on_label(
                    data, tmp_gui, data["label"].eq(idx).to(dtype=torch.int32)
                )
        elif (
            click_generation_strategy == ClickGenerationStrategy.GLOBAL_CORRECTIVE
            or click_generation_strategy == ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE
        ):
            if click_generation_strategy == ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE:
                # sets self._will_interact
                self.randomize(data)
            else:
                self._will_interact = True

            if self._will_interact:
                for key_label in data[LABELS_KEY].keys():
                    tmp_gui = get_guidance_tensor_for_key_label(data, key_label, self.device)

                    # Add guidance based on discrepancy
                    data[key_label] = self.add_guidance_based_on_discrepancy(data, tmp_gui, key_label)
        elif click_generation_strategy == ClickGenerationStrategy.PATCH_BASED_CORRECTIVE:
            assert data[CommonKeys.LABEL].shape == data[CommonKeys.PRED].shape

            t = [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(
                    keys=("pred", "label"),
                    argmax=(True, False),
                    to_onehot=(len(data[LABELS_KEY]), len(data[LABELS_KEY])),
                ),
            ]
            post_transform = Compose(t)
            t_data = post_transform(data)

            # Split the data into patches of size self.patch_size
            # TODO not working for 2d data yet!
            new_data = PatchIterd(keys=[CommonKeys.PRED, CommonKeys.LABEL], patch_size=self.patch_size)(t_data)
            pred_list = []
            label_list = []
            coordinate_list = []

            for patch in new_data:
                actual_patch = patch[0]
                pred_list.append(actual_patch[CommonKeys.PRED])
                label_list.append(actual_patch[CommonKeys.LABEL])
                coordinate_list.append(actual_patch["patch_coords"])

            label_stack = torch.stack(label_list, 0)
            pred_stack = torch.stack(pred_list, 0)

            dice_loss = DiceLoss(include_background=True, reduction="none")
            with torch.no_grad():
                loss_per_label = dice_loss.forward(input=pred_stack, target=label_stack).squeeze()
                assert len(loss_per_label.shape) == 2
                # 1. dim: patch number, 2. dim: number of labels, e.g. [27,2]
                max_loss_position_per_label = torch.argmax(loss_per_label, dim=0)
                assert len(max_loss_position_per_label) == len(data[LABELS_KEY])

            # We now have the worst patches for each label, now sample clicks on them
            for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                patch_number = max_loss_position_per_label[idx]
                coordinates = coordinate_list[patch_number]

                tmp_gui = get_guidance_tensor_for_key_label(data, key_label, self.device)
                # Add guidance based on discrepancy
                data[key_label] = self.add_guidance_based_on_discrepancy(data, tmp_gui, key_label, coordinates)

            gc.collect()
        else:
            raise UserWarning("Unknown click strategy")


        return data


class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation
    """

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key == "pred":
                for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                    if key_label != "background":
                        data[f"pred_{key_label}"] = data[key][idx + 1, ...][None]
                        data[f"label_{key_label}"] = data["label"][idx + 1, ...][None]
            elif key != "pred":
                logger.info("This transform is only for pred key")
        return data


class FlipChanneld(MapTransform, InvertibleTransform, LazyTransform):
    """
    A transformation class to flip specified channels along the specified spatial axis in tensor-like data.
    This is an invertible transform and can be applied lazily.

    Args:
        keys: Keys specifying the data in the dictionary to be processed.
        spatial_axis: Spatial axis or axes along which flipping should occur.
        channels: Channels to be flipped.
        allow_missing_keys: If True, allows for keys in `keys` to be missing in the input dictionary.
        lazy: If True, the transform is applied lazily.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_axis: Sequence[int] | int | None = None,
        channels: Sequence[int] | int | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = Flip(spatial_axis=spatial_axis)
        self.channels = channels

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val



    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            for channel in self.channels:
                d[key][channel:channel+1] = self.flipper(d[key][channel:channel+1], lazy=lazy_)
        return d




    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            for channel in self.channels:
                d[key][channel:channel+1] = self.flipper.inverse(d[key][channel:channel+1])
        return d