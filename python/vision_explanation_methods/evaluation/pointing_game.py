# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

"""Defines a variety of explanation evaluation tools."""

from io import BytesIO
from typing import Any, List

import numpy as np
import requests
import torch
import torchvision.transforms as T
from captum.attr import visualization as viz
from matplotlib import pyplot as pl
from PIL import Image
from torch import Tensor

from ml_wrappers.common.constants import Device
from ..explanations import drise


class PointingGame:
    """A class for the high energy pointing game."""

    def __init__(self,
                 model: Any,
                 device=Device.AUTO.value) -> None:
        """Initialize the PointingGame.

        :param model: mlflow model
        :type model: Any
        :param device: optional parameter specifying the device to move the
            model to. If not specified, then cpu is the default
        :type device: str
        """
        self._device = torch.device(self._get_device(device))
        self._model = model

    def pointing_game(self,
                      imagelocation: str,
                      index: int,
                      threshold: float,
                      num_masks: int = 100):
        """
        Calculate the saliency scores for a given object detection prediction.

        The calculated value is a matrix of saliency scores. Values below
        the threshold are set to -1.

        :param imagelocation: Path of the image location
        :type imagelocation: str
        :param index: Index of the desired object within the given image to
            evaluate
        :type index: int
        :param threshold: threshold between 0 and 1 to determine saliency of a
            pixel. If saliency score is below the threshold, then the score is
            set to -1
        :type threshold: float
        :param num_masks: number of masks to run drise with
        :type num_masks: int
        :return: 2d matrix of highly salient pixels
        :rtype: List[Tensor]
        """
        image_open_pointer = imagelocation
        if (imagelocation.startswith("http://")
           or imagelocation.startswith("https://")):
            response = requests.get(imagelocation)
            image_open_pointer = BytesIO(response.content)

        test_image = Image.open(image_open_pointer).convert('RGB')

        img_input = (T.ToTensor()(test_image)
                     .unsqueeze(0).to(self._device))

        detections = self._model.predict(img_input)

        saliency_scores = drise.DRISE_saliency(
                model=self._model,
                # Repeated the tensor to test batching
                image_tensor=img_input,
                target_detections=detections,
                # This is how many masks to run -
                # more is slower but gives higher quality mask.
                number_of_masks=num_masks,
                mask_padding=None,
                device=self._device,
                # This is the resolution of the random masks.
                # High resolutions will give finer masks, but more need to be
                # run.
                mask_res=(2, 2),
                verbose=True  # Turns progress bar on/off.
            )

        temp = saliency_scores[0][index]['detection']

        temp[temp < threshold] = -1
        return temp

    def visualize_highly_salient_pixels(self,
                                        img,
                                        saliency_scores):
        """
        Create figure of highly salient pixels.

        :param imagelocation: Path of the image location
        :type imagelocation: str
        :param saliency_scores: 2D matrix representing the saliency scores
            of each pixel in an image
        :type saliency_scores: List[Tensor]
        :return: Overlay of the saliency scores on top of the image
        :rtype: Figure
        """
        fig, ax = pl.subplots(1, 1, figsize=(10, 10))

        viz.visualize_image_attr(
            np.transpose(
                saliency_scores.detach().cpu().numpy(),
                (1, 2, 0)),
            np.transpose(T.ToTensor()(img).detach().cpu().numpy(), (1, 2, 0)),
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            cmap=pl.cm.get_cmap("Blues"),
            title="Pointing Game Visualization",
            plt_fig_axis=(fig, ax),
            use_pyplot=False
        )

        return fig

    def calculate_gt_salient_pixel_overlap(self,
                                           saliency_scores: List[Tensor],
                                           gt_bbox: List):
        """
        Calculate percent of overlap between salient pixels and gt bbox.

        :param saliency_scores: 2D matrix representing the saliency scores
            of each pixel in an image
        :type saliency_scores: List[Tensor]
        :param gt_bbox: bounding box for ground truth prediction
        :type gt_bbox: List
        :return: return percent of salient pixel overlap with the ground truth
        :rtype: Float
        """
        saliency_scores = torch.tensor(saliency_scores)
        gt_bbox = torch.tensor(gt_bbox)

        gt_mask = torch.zeros_like(saliency_scores, dtype=torch.bool)
        gt_mask[gt_bbox[0]:gt_bbox[2], gt_bbox[1]:gt_bbox[3]] = True

        positive_mask = saliency_scores > 0
        positive_gt_mask = torch.logical_and(positive_mask, gt_mask)

        good = positive_gt_mask.sum().item()
        total = positive_mask.sum().item()

        return good / total

    def _get_device(self, device: str) -> str:
        """
        Set the device to run computations on to the desired value.

        If device were set to "auto", then the desired device will be cuda
        (GPU) if available. Otherwise, the device should be set to cpu.

        :param device: parameter specifying the device to move the model
            to.
        :type device: str
        :return: selected device to run computations on
        :rtype: str
        """
        if (device in set(member.value for member in Device)
           or type(device) == int or device is None):
            if device == Device.AUTO.value:
                return (Device.CUDA.value if torch.cuda.is_available()
                        else Device.CPU.value)
            return device.value
        else:
            raise ValueError("Selected device is invalid")
