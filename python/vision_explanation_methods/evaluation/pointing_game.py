# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

"""Defines a variety of explanation evaluation tools."""

from io import BytesIO
from typing import Any, List

import matplotlib.patches as patches
import numpy as np
import requests
import torch
import torchvision.transforms as T
from captum.attr import visualization as viz
from matplotlib import pyplot as pl
from ml_wrappers.common.constants import Device
from ml_wrappers.model.image_model_wrapper import _get_device
from PIL import Image
from torch import Tensor

from ..explanations import drise

NUM_MASKS = 100


class PointingGame:
    """A class for the high energy pointing game."""

    def __init__(self,
                 model: Any,
                 device=Device.AUTO.value) -> None:
        """Initialize the PointingGame.

        Note: gt = ground truth.

        :param model: mlflow model
        :type model: Any
        :param device: optional parameter specifying the device to move the
            model to. If not specified, then cpu is the default
        :type device: str
        """
        self._device = torch.device(_get_device(device))
        self._model = model

    def pointing_game(self,
                      imagelocation: str,
                      index: int,
                      threshold: float = .8,
                      num_masks: int = NUM_MASKS):
        """
        Calculate the saliency scores for a given object detection prediction.

        The calculated value is a matrix of saliency scores. Values below
        the threshold are set to -1. The goal here is to filter out
        insignificant saliency scores, and identify highly salient pixels.
        That is why it is called a pointing game - we want to "point", i.e.
        identify, all highly salient pixels. That way we can easily
        determine if these highly salient pixels overlap with the gt bounding
        box.

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
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold parameter must be a float \
                             between 0 and 1.')
        if num_masks < 0:
            raise ValueError('Number of masks parameter must be a \
                             positive int.')
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
                                        saliency_scores,
                                        gt_bbox: List = None):
        """
        Create figure of highly salient pixels.

        :param img: PIL test image
        :type img: PIL.Image
        :param saliency_scores: 2D matrix representing the saliency scores
            of each pixel in an image
        :type saliency_scores: List[Tensor]
        :param gt_bbox: bounding box for ground truth prediction. if none
            then no ground truth bounding box is drawn
        :type gt_bbox: List
        :return: Overlay of the saliency scores on top of the image
        :rtype: Figure
        """
        pl.clf()
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
        if gt_bbox is not None:
            x, y, width, height = gt_bbox
            rectangle = patches.Rectangle((x, y), width, height, linewidth=1,
                                          edgecolor='r', facecolor='none')
            # Add the rectangle patch to the axes
            ax.add_patch(rectangle)

        return fig

    def calculate_gt_salient_pixel_overlap(self,
                                           saliency_scores: List[Tensor],
                                           gt_bbox: List):
        """
        Calculate percent of overlap between salient pixels and gt bbox.

        Formula: number of salient pixels in the gt bbox /
                 number of pixels in the gt bbox

        :param saliency_scores: 2D matrix representing the saliency scores
            of each pixel in an image
        :type saliency_scores: List[Tensor]
        :param gt_bbox: bounding box for ground truth prediction
        :type gt_bbox: List
        :return: return percent of salient pixel overlap with the ground truth
        :rtype: Float
        """
        saliency_scores = torch.tensor(saliency_scores)[0]
        gt_bbox = torch.tensor(gt_bbox)

        gt_mask = torch.zeros_like(saliency_scores, dtype=torch.bool)
        gt_mask[gt_bbox[1]:gt_bbox[3]+1, gt_bbox[0]:gt_bbox[2]+1] = True

        positive_mask = torch.gt(saliency_scores, 0)
        positive_gt_mask = torch.logical_and(positive_mask, gt_mask)
        good = torch.count_nonzero(positive_gt_mask)
        total = torch.count_nonzero(gt_mask)

        return torch.div(good, total).item()
