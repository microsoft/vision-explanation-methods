# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

"""Defines the Error Labeling Manager class."""

from copy import deepcopy
from enum import Enum

import numpy as np
import torchvision
from torch import Tensor

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
    # this prediction must have a 0 iou score with all gt detections
    BACKGROUND = "background"

    # the predicted class is correct, bounding box does not have sufficient
    # overlap with ground truth (based on the iou threshold)
    LOCALIZATION = "localization"

    # the predicted class is incorrect, the bounding box is correct
    CLASS_NAME = "class_name"

    # both the predicted class and bounding box are incorrect
    CLASS_LOCALIZATION = "class_localization"

    # the predicted class is correct, the bounding box is correct, but
    # the iou score is lower than another detection
    DUPLICATE_DETECTION = "duplicate_detection"

    MATCH = "match"


class ErrorLabeling():
    """
    Defines a wrapper class of Error Labeling for vision scenario.

    Only supported for object detection at this point.
    """

    def __init__(self,
                 task_type: str,
                 pred_y: list,
                 true_y: list,
                 iou_threshold: float = 0.5):
        """
        Create an ErrorLabeling object.

        :param task_type: The task to run.
        :type task_type: str
        :param pred_y: predicted detections, nested list of 6 floats (class,
            bounding box, conf score). The bounding box will be located at
            indexes 1-4.
        :type pred_y: list
        :param true_y: ground truth detections, nested list of 6 floats (class,
            bounding box, is crowded). The bounding box will be located at
            indexes 1-4.
        :type true_y: list
        :param iou_threshold: required minimum for bounding box overlap
        :type iou_threshold: float
        """
        self._is_run = False
        self._is_added = False
        self._task_type = task_type
        self._pred_y = pred_y
        self._true_y = true_y
        self._iou_threshold = iou_threshold

    def compute_error_labels(self):
        """
        Compute labels for errors in an object detection prediction.

        Note: if a row does not have a match, that means that there is a
        missing gt detection

        :return: 2d matrix of error labels
        :rtype: NDArray
        """
        match_matrix = np.full((len(self._true_y), len(self._pred_y)),
                               None)
        # save original ordering of predictions
        original_indices = [i for i, _ in sorted(enumerate(self._pred_y),
                                                 key=lambda x: x[1][-1],
                                                 reverse=True)]

        # sort predictions by decreasing conf score
        # this is to stay consistent with NMS and MAP algorithms
        sorted_list = sorted(self._pred_y, key=lambda x: x[-1], reverse=True)

        if len(self._true_y) == 0:
            match_matrix = np.array(
                [[ErrorLabelType.BACKGROUND]
                 for _ in range(len(self._pred_y))]
            )
            return match_matrix

        for gt_index, gt in enumerate(self._true_y):
            for detect_index, detect in enumerate(sorted_list):
                iou_score = torchvision.ops.box_iou(
                    Tensor(detect[1:5]).unsqueeze(0).view(-1, 4),
                    Tensor(gt[1:5]).unsqueeze(0).view(-1, 4))

                if iou_score.item() == 0:
                    # if iou is 0, then prediction is detecting the background
                    match_matrix[gt_index][detect_index] = (
                        ErrorLabelType.BACKGROUND)
                elif self._iou_threshold <= iou_score:
                    # the detection and ground truth bb's are overlapping
                    if detect[0] != gt[0]:
                        # the bboxes line up, but labels do not
                        match_matrix[gt_index][detect_index] = (
                            ErrorLabelType.CLASS_NAME)
                    elif (ErrorLabelType.MATCH in
                          match_matrix[gt_index]):
                        # class name and bbox correct, but there is already a
                        # match with a higher confidence score (this is why
                        # it was imporant to sort by descending confidence
                        # scores as the first step)
                        match_matrix[gt_index][detect_index] = (
                            ErrorLabelType.DUPLICATE_DETECTION)
                    else:
                        # this means bboxes overlap, class names = (1st time)
                        match_matrix[gt_index][detect_index] = (
                            ErrorLabelType.MATCH)
                else:
                    if detect[0] != gt[0]:
                        # the bboxes don't line up, and labels do not
                        match_matrix[gt_index][detect_index] = (
                            ErrorLabelType.CLASS_LOCALIZATION)
                    else:
                        # the bboxes don't line up, but the labels are correct
                        match_matrix[gt_index][detect_index] = (
                            ErrorLabelType.LOCALIZATION)

        # resort the columns (so no longer ordered by descending conf
        # scores)
        match_matrix[gt_index] = [match_matrix[gt_index][i]
                                  for i in original_indices]
        return match_matrix

    def compute_error_list(self):
        """
        Determine a complete list of errors encountered during prediction.

        Note that it is possible to have more errors than actual objects
        in an image (because we account for missing detections and
        duplicate detections).

        :return: list of error labels
        :rtype: list
        """
        match_matrix = self.compute_error_labels()
        error_arr = self._remove_matches(deepcopy(match_matrix))
        dup_count = np.count_nonzero(match_matrix ==
                                     ErrorLabelType.DUPLICATE_DETECTION)
        error_list = [ErrorLabelType.DUPLICATE_DETECTION
                      for _ in range(dup_count)]

        if len(error_arr) == 0:
            return error_list

        diff = len(error_arr) - len(error_arr[0])
        if diff > 0:
            for _ in range(diff):
                error_list.append(ErrorLabelType.MISSING)

        order_of_errors = [ErrorLabelType.CLASS_NAME,
                           ErrorLabelType.LOCALIZATION,
                           ErrorLabelType.CLASS_LOCALIZATION,
                           ErrorLabelType.BACKGROUND]

        for err in order_of_errors:
            for gt_index, gt in enumerate(error_arr):
                for detect_index, detect in enumerate(gt):
                    if detect == err:
                        error_list.append(err)
                        error_arr = self._remove_rows_cols(error_arr,
                                                           set([gt_index]),
                                                           set([detect_index]))
                        if len(error_arr) == 0:
                            break

        return error_list

    def _remove_matches(self, arr: np.array):
        """
        Remove match rows and columns from a error labeling matrix.

        :param arr: np 2d array
        :type arr: np.array
        :return: array with removed rows and columns
        :rtype: np.array
        """
        rows_to_delete = set()
        cols_to_delete = set()

        for row, row_items in enumerate(arr):
            for col, value in enumerate(row_items):
                if value == ErrorLabelType.MATCH:
                    rows_to_delete.add(row)
                    cols_to_delete.add(col)

        modified_array = self._remove_rows_cols(arr,
                                                rows_to_delete,
                                                cols_to_delete)

        return modified_array

    def _remove_rows_cols(self,
                          arr: np.array,
                          rows_to_delete: set,
                          cols_to_delete: set):
        """
        Remove rows and columns from a given array.

        :param arr: np 2d array
        :type arr: np.array
        :param rows_to_delete: unique set of indexes of rows to remove
        :type rows_to_delete: set
        :param cols_to_delete: unique set of indexes of cols to remove
        :type cols_to_delete: set
        :return: array with removed rows and columns
        :rtype: np.array
        """
        # Delete rows
        modified_array = [row for row_index, row in enumerate(arr)
                          if row_index not in rows_to_delete]

        # Delete columns
        modified_array = [[value for col_index, value in enumerate(row)
                           if col_index not in cols_to_delete]
                          for row in modified_array]
        return modified_array
