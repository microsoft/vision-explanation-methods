# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

"""Defines the Error Labeling Manager class."""

import cv2
import base64
import io
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, List, Optional
import torch
import torchmetrics
import torchmetrics.functional as metrics
import torchvision
from torch import Tensor

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import shap
from ml_wrappers import wrap_model
from responsibleai._interfaces import ModelExplanationData
from responsibleai._internal.constants import ExplainerManagerKeys as Keys
from responsibleai._internal.constants import (ListProperties, ManagerNames,
                                               Metadata)
from responsibleai._tools.shared.state_directory_management import \
    DirectoryManager
from responsibleai.exceptions import UserConfigValidationException
from responsibleai.managers.base_manager import BaseManager
from responsibleai_vision.common.constants import (CommonTags,
                                                   ExplainabilityDefaults,
                                                   ExplainabilityLiterals,
                                                   MLFlowSchemaLiterals,
                                                   ModelTask,
                                                   XAIPredictionLiterals)
from responsibleai_vision.utils.image_reader import (
    get_base64_string_from_path, get_image_from_path, is_automl_image_model)
from shap.plots import colors
from shap.utils._legacy import kmeans
from vision_explanation_methods.DRISE_runner import get_drise_saliency_map
from ml_wrappers.model.image_model_wrapper import (PytorchDRiseWrapper,
                                                   MLflowDRiseWrapper)
from PIL import Image, ImageDraw, ImageFont
from enum import Enum

LABELS = 'labels'


class ErrorLabelType(Enum):
    """
    Enum providing types of error labels.

    If none, then the detection is not an error. It is a
    correct prediction.
    """
    # the gt doesn't have a corresponding detection
    MISSING = "missing"

    # the model predicted detections, but there was nothing there
    BACKGROUND = "background"

    # the predicted class is correct, bounding box is not
    LOCALIZATION = "localization"

    # the predicted class is incorrect, the bounding box is correct
    CLASS_NAME = "class_name"

    # both the predicted class and bounding box are incorrect
    BOTH = "both"

    # the predicted class is correct, the bounding box is correct, but
    # the iou score is lower than another detection
    DUPLICATE_DETECTION = "duplicate_detection"

    MATCH = "match"


class ErrorLabeling(BaseManager):
    """Defines a wrapper class of Error Labeling for vision scenario.
    Only supported for object detection at this point.
    """
    def __init__(self,
                 task_type: str,
                 pred_y: str,
                 true_y: str,
                 iou_threshold: float = 0.5):
        """Creates an ErrorLabeling object.

        :param model: The model to explain.
            A model that implements sklearn.predict or sklearn.predict_proba
            or function that accepts a 2d ndarray.
        :type model: object
        :param evaluation_examples: A matrix of feature vector
            examples (# examples x # features) on which to explain the
            model's output, with an additional label column.
        :type evaluation_examples: pandas.DataFrame
        :param target_column: The name of the label column.
        :type target_column: str
        :param task_type: The task to run.
        :type task_type: str
        :param classes: Class names as a list of strings.
            The order of the class names should match that of the model
            output. Only required if explaining classifier.
        :type classes: list
        :param image_mode: The mode to open the image in.
            See pillow documentation for all modes:
            https://pillow.readthedocs.io/en/stable/handbook/concepts.html
        :type image_mode: str
        """
        self._is_run = False
        self._is_added = False
        self._task_type = task_type
        self._pred_y = pred_y
        self._true_y = true_y
        self._iou_threshold = iou_threshold

        self._match_matrix = [[None for i in range(len(pred_y))] for i in range(len(true_y))]

    def compute(self, **kwargs):
        """Compute the error analysis data.

        :param kwargs: The keyword arguments to pass to the compute method.
            Note that this method does not take any arguments currently.
        :type kwargs: dict
        """
        if not self._is_added:
            self.add()

        for detect_index in range(len(self._pred_y)):
            detect = self._pred_y[detect_index]
            matched = False
            background = True
            for gt_index in range(len(self._true_y)):
                gt = self._true_y[gt_index]
                iou_score = torchvision.ops.box_iou(Tensor(detect[1:5]).unsqueeze(0),
                                                    Tensor(gt[1:5]).unsqueeze(0))
                if iou_score > 0:
                    background = False
                if (self._iou_threshold <= iou_score):
                    # the detection and ground truth bb's must be overlapping
                    if detect[0] != gt[0]:
                        # the bb's line up, but labels do not
                        matched = True
                        self._prediction_error_labels[detect_index] = ErrorLabelType.CLASS_NAME
                    elif (gt_is_matched[gt_index] is not None):
                        # todo - check if should use conf score or iou score
                        if gt_is_matched[gt_index][0] <= iou_score:
                            # reset previously correct match
                            # todo - fix this so choose one w higher conf score to be consistent w MAP and NMS algorithsm 
                            matched = True 
                            prev_correct_match_index = gt_is_matched[gt_index][1]
                            gt_is_matched[prev_correct_match_index] = -1 # todo check this
                            gt_is_matched[gt_index] = (iou_score, detect_index)
                            self._prediction_error_labels[detect_index] = ErrorLabelType.DUPLICATE_DETECTION
                    else:
                        # this means bbs overlap, class names = (1st time)
                        matched = True
                        gt_is_matched[gt_index] = (iou_score, detect_index)
                        break
            if not matched:
                if background:
                    self._prediction_error_labels[detect_index] = ErrorLabelType.BACKGROUND
                else:
                    # if detect[0] 
                    self._prediction_error_labels[detect_index] = ErrorLabelType.BOTH

        for gt_index in range(len(self._true_y)):
            gt = self._true_y[gt_index]
            if gt is None:
                self._missing_labels[gt_index] = ErrorLabelType.MISSING
