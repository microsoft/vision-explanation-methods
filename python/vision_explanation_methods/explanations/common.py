# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Common functions and classes.

Used between different object detection explainability methods.
"""

import abc
from dataclasses import dataclass
from typing import List, Optional, Type

import torch


@dataclass
class DetectionRecord:
    """Data class to provide a common format for detections.

    :param bounding_boxes: Set of bounding boxes for image,
        shape [D, 4], where D is the number of detections. Assumed
        to be of format [Left, Top, Right, Bottom]
    :type bounding_boxes: torch.Tensor
    :param objectness_scores: Score of how likely detection is to be an
        object. Some models, don't specify object score - can be set to 1.0
        in this case. Shape [D]
    :type objectness_scores: torch.Tensor
    :param class_scores: Scores associated with the probability detections
        belong to each class. Shape [D, C], where C is the number of classes
    :type class_scores: torch.Tensor
    """

    def __init__(
            self,
            bounding_boxes: torch.Tensor,
            objectness_scores: torch.Tensor,
            class_scores: torch.Tensor
    ):
        """Initialize the DetectionRecord."""
        self.bounding_boxes = bounding_boxes
        self.objectness_scores = objectness_scores
        self.class_scores = class_scores

    def to(self, device: str) -> None:
        """Move tensors to compute device.

        :param device: Device to move tensors to, e.g. cpu, cuda:0
        :type device: str
        """
        self.bounding_boxes = self.bounding_boxes.to(device)
        self.class_scores = self.class_scores.to(device)
        self.objectness_scores = self.objectness_scores.to(device)

    def get_by_index(self, indicies: List[int]) -> Type["DetectionRecord"]:
        """Select a subset of detections from set of indices of those boxes.

        :param indicies: Indices of the subset of boxes to return
            for example, if you need boxes 0, 2, and 3, pass [0, 2, 3]
        :type indices: List of ints
        :return: A new detection record with only the detections at
            specified indices.
        :rtype: DetectionRecord
        """
        return DetectionRecord(
            bounding_boxes=self.bounding_boxes[indicies, :],
            objectness_scores=self.objectness_scores[indicies],
            class_scores=self.class_scores[indicies, :]
        )


class GeneralObjectDetectionModelWrapper(abc.ABC):
    """Prototype for that defines the interface for standard OD model."""

    @abc.abstractmethod
    def predict(self, x: torch.Tensor) -> List[DetectionRecord]:
        """Take a tensor and return a list of detection records.

        This is the only required method.

        :param x: Tensor of a batch of images. Shape [B, 3, W, H]
        :type x: torch.Tensor
        :return: List of Detections produced by wrapped model
        :rtype: List of DetectionRecords
        """
        raise NotImplementedError


def compute_intersections(
        boxes_a: torch.Tensor,
        boxes_b: torch.Tensor
) -> torch.Tensor:
    """Compute intersection between two lists of boxes.

    :param boxes_a: Tensor of M boxes in coordinates
        [left, top, right, bottom], shape [N, 4]
    :type boxes_a: Tensor
    :param boxes_b: Tensor of N boxes in coordinates
        [left, top, right, bottom], shape [M, 4]
    :type boxes_b: Tensor
    :return: Intersection matrix. Shape [N, M]. Entry (n, m) is the
        intersection between boxes_a[n] and boxes_b[m]
    :rtype: Tensor
    """
    number_of_boxes_a = boxes_a.shape[0]
    number_of_boxes_b = boxes_b.shape[0]

    unpacked_boxes_a = boxes_a.unsqueeze(1).repeat(
        1, number_of_boxes_b, 1)  # Shape [N, M, 4]
    unpacked_boxes_b = boxes_b.unsqueeze(0).repeat(
        number_of_boxes_a, 1, 1)  # Shape [N, M, 4]

    left = torch.max(unpacked_boxes_a[:, :, 0], unpacked_boxes_b[:, :, 0])
    right = torch.min(unpacked_boxes_a[:, :, 2], unpacked_boxes_b[:, :, 2])
    top = torch.max(unpacked_boxes_a[:, :, 1], unpacked_boxes_b[:, :, 1])
    bottom = torch.min(unpacked_boxes_a[:, :, 3], unpacked_boxes_b[:, :, 3])

    repeated_0 = torch.tensor(0.).repeat(left.shape[0], left.shape[1])

    # Negative widths/heights treated as 0 so area is 0
    widths = torch.max(repeated_0, right - left)
    heights = torch.max(repeated_0, bottom - top)

    return widths * heights


def compute_areas(boxes: torch.Tensor) -> torch.Tensor:
    """Compute the areas of a list of boxes.

    :param boxes:  Tensor of N boxes in coordinates [left, top, right, bottom],
        shape [N, 4]
    :type boxes: Tensor
    :return: Tensor of box areas, shape [N]
    :rtype: Tensor
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def compute_unions(
        boxes_a: torch.Tensor,
        boxes_b: torch.Tensor
) -> torch.Tensor:
    """Compute area of unions between two lists of boxes.

    :param boxes_a: Tensor of M boxes in coordinates
        [left, top, right, bottom], shape [N, 4]
    :type boxes_a: Tensor
    :param boxes_b: Tensor of N boxes in coordinates
        [left, top, right, bottom], shape [M, 4]
    :type boxes_b: Tensor
    :return: Intersection matrix. Shape [N, M]. Entry (n, m) is the
        intersection between boxes_a[m] and boxes_b[n]
    :rtype: Tensor
    """
    number_of_boxes_a = boxes_a.shape[0]
    number_of_boxes_b = boxes_b.shape[0]

    return (compute_areas(boxes_a).unsqueeze(1).repeat(1, number_of_boxes_b) +
            compute_areas(boxes_b).unsqueeze(0).repeat(number_of_boxes_a, 1) -
            compute_intersections(boxes_a, boxes_b))


def compute_IoUs(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute the Intersection over Union of two lists of boxes.

    :param boxes_a: Tensor of M boxes in coordinates
        [left, top, right, bottom], shape [N, 4]
    :type boxes_a: Tensor
    :param boxes_b: Tensor of N boxes in coordinates
        [left, top, right, bottom], shape [M, 4]
    :type boxes_b: Tensor
    :return: Intersection over union matrix. Shape [N, M]. Entry (n, m) is the
        IoU between boxes_a[m] and boxes_b[n]
    :rtype: Tensor
    """
    return (compute_intersections(boxes_a, boxes_b) /
            compute_unions(boxes_a, boxes_b))


def compute_affinity_matrix(
        detections_a: DetectionRecord,
        detections_b: DetectionRecord,
        exclude_class: Optional[bool] = False,
) -> torch.Tensor:
    """Compute the affinity scores between two sets of object detections.

    :param base_detections: Detections associated with unmasked image
    :type base_detections: DetectionRecord
    :param masked_detections: Detections associated with masked image
    :type masked_detections: Detection Record
    :return: Tensor of box affinity scores, of shape [N]
    :rtype: Tensor
    """
    # No detections in the masked image
    if detections_b is None:
        return torch.zeros((1, detections_a.bounding_boxes.shape[0]))

    if detections_b.bounding_boxes.shape[0] == 0:
        return torch.zeros((1, detections_a.bounding_boxes.shape[0]))

    detections_a.to("cpu")
    detections_b.to("cpu")

    iou_scores = compute_IoUs(detections_a.bounding_boxes,
                              detections_b.bounding_boxes)
    objectness_scores = (detections_a.objectness_scores.unsqueeze(1) @
                         detections_b.objectness_scores.unsqueeze(0))

    if exclude_class:
        class_affinities, class_normalization = 1.0, 1.0
    else:
        class_affinities = detections_a.class_scores @ torch.transpose(
            detections_b.class_scores, 0, 1)
        class_normalization = (
            torch.norm(detections_a.class_scores, p=2, dim=1).unsqueeze(1) @
            torch.norm(detections_b.class_scores, p=2, dim=1).unsqueeze(0))

    score_matrix = iou_scores * objectness_scores * (
        class_affinities / class_normalization)

    return score_matrix


def expand_class_scores(
        scores: torch.Tensor,
        labels: torch.Tensor,
        number_of_classes: int,
) -> torch.Tensor:
    """Extrapolate a full set of class scores.

    Many object detection models don't return a full set of class scores, but
    rather just a score for the predicted class. This is a helper function
    that approximates a full set of class scores by dividing the difference
    between 1.0 and the predicted class score among the remaning classes.

    :param scores: Set of class specific scores. Shape [D] where D is number
        of detections
    :type scores: torch.Tensor
    :param labels: Set of label indices corresponding to predicted class.
        Shape [D] where D is number of detections
    :type labels: torch.Tensor (ints)
    :param number_of_classes: Number of classes model predicts
    :type number_of_classes: int
    :return: A set of expanded scores, of shape [D, C], where C is number of
        classes
    :type: torch.Tensor
    """
    number_of_detections = scores.shape[0]

    expanded_scores = torch.ones(number_of_detections, number_of_classes + 1)

    for i, (score, label) in enumerate(zip(scores, labels)):

        residual = (1. - score.item()) / (number_of_classes)
        expanded_scores[i, :] *= residual
        expanded_scores[i, int(label.item())] = score

    return expanded_scores
