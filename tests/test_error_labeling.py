# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

"""Test error labeling in the vision-explanation-methods package."""

import numpy as np
import pytest
from vision_explanation_methods.error_labeling.error_labeling import (
    ErrorLabeling, ErrorLabelType)


class TestErrorLabelingManager(object):
    """Testing error_labeling.py."""

    @pytest.mark.parametrize(("pred_y", "true_y", "iou_threshold", "result"), [
        # correct instance, prediction exactly the same
        ([[44, 162, 65, 365, 660, 0]],
         [[44, 162, 65, 365, 660, 0]],
         .5,
         np.array([[ErrorLabelType.MATCH]])),

        # correct instance, predictions exactly the same for multiple detects
        ([[44, 5, 5, 7, 7, 0], [44, 1, 1, 2, 2, 0]],
         [[44, 5, 5, 7, 7, 0], [44, 1, 1, 2, 2, 0]],
         .5,
          np.array([[ErrorLabelType.MATCH, ErrorLabelType.BACKGROUND],
                    [ErrorLabelType.BACKGROUND, ErrorLabelType.MATCH]])),

        # correct instance, predictions exactly the same for multiple detects
        # order is mixed up
        ([[44, 5, 5, 7, 7, 0], [44, 1, 1, 2, 2, 0]],
         [[44, 1, 1, 2, 2, 0], [44, 5, 5, 7, 7, 0]],
         .5,
         np.array([[ErrorLabelType.BACKGROUND, ErrorLabelType.MATCH],
                   [ErrorLabelType.MATCH, ErrorLabelType.BACKGROUND]])),

        # missing detection
        ([[44, 5, 5, 7, 7, 0]],
         [[44, 1, 1, 2, 2, 0], [44, 5, 5, 7, 7, 0]],
         .5,
         np.array([[ErrorLabelType.BACKGROUND], [ErrorLabelType.MATCH]])),

        # correct instance, prediction not = but w/in iou threshold
        ([[44, 162, 65, 365, 660, 0]],
         [[44, 162, 65, 365, 670, 0]],
         .5,
         np.array([[ErrorLabelType.MATCH]])),

        # class error with 100% overlap
        ([[44, 162, 65, 365, 660, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         np.array([[ErrorLabelType.CLASS_NAME]])),

        # class error with overlapping bb, but not completely
        ([[44, 162, 65, 365, 660, 0]],
         [[1, 170, 65, 365, 660, 0]],
         .5,
         np.array([[ErrorLabelType.CLASS_NAME]])),

        # complete miss, minor overlap with original (less than iou)
        ([[1, 162, 65, 1000, 1000, 0]],
         [[2, 162, 65, 1000, 200, 0]],
         .5,
         np.array([[ErrorLabelType.CLASS_LOCALIZATION]])),

        # duplicate detection,
        ([[1, 162, 65, 365, 660, 0], [1, 162, 65, 365, 660, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         np.array([[ErrorLabelType.MATCH, ErrorLabelType.DUPLICATE_DETECTION]])),

        # duplicate detection, detections not identical but w/in iou range.
        # same conf score
        ([[1, 162, 65, 365, 660, 0], [1, 162, 65, 365, 659, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         np.array([[ErrorLabelType.MATCH, ErrorLabelType.DUPLICATE_DETECTION]])),

        # duplicate detection, detections not identical but w/in iou range.
        # Detection with lower iou has higher conf score
        ([[1, 162, 65, 365, 660, 0], [1, 162, 65, 365, 659, 50]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         np.array([[ErrorLabelType.DUPLICATE_DETECTION, ErrorLabelType.MATCH]])),

        # Same as previous, but in a different order to ensure that conf score
        # prioritization is working correctly
        ([[1, 162, 65, 365, 659, 50], [1, 162, 65, 365, 660, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         np.array([[ErrorLabelType.MATCH, ErrorLabelType.DUPLICATE_DETECTION]])),

        # background error with same class
        ([[1, 50, 50, 100, 100, 0]],
         [[1, 350, 350, 100, 100, 0]],
         .5,
         np.array([[ErrorLabelType.BACKGROUND]])),

        # background error with different class
        ([[0, 50, 50, 100, 100, 0]],
         [[1, 350, 350, 100, 100, 0]],
         .5,
         np.array([[ErrorLabelType.BACKGROUND]])),

        # gt is empty
        ([[0, 1, 1, 20, 20, 0]],
         [],
         .5,
         np.array([[ErrorLabelType.BACKGROUND]])),

        # pred is empty
        ([],
         [[0, 1, 1, 20, 20, 0]],
         .5,
         np.array([[ErrorLabelType.MISSING]])),

        # edge case for background error (bb's touch, but iou is 0)
        ([[44, 5, 5, 1, 1, 0]],
         [[44, 6, 6, 1, 1, 0]],
         .5,
         np.array([[ErrorLabelType.BACKGROUND]])),
    ])
    def test_object_detection_image_labeling(self,
                                             pred_y,
                                             true_y,
                                             iou_threshold,
                                             result):
        """Compare _match_matrix attribute to expected result."""
        task_type = 'object_detection'
        mng = ErrorLabeling(task_type,
                            pred_y,
                            true_y,
                            iou_threshold)
        err = mng.compute_error_labels()
        assert (err == result).all()

    @pytest.mark.parametrize(("pred_y", "true_y", "iou_threshold", "result"), [
        # correct instance, prediction exactly the same
        ([[44, 162, 65, 365, 660, 0]],
         [[44, 162, 65, 365, 660, 0]],
         .5,
         []),

        # correct instance, predictions exactly the same for multiple detects
        ([[44, 5, 5, 7, 7, 0], [44, 1, 1, 2, 2, 0]],
         [[44, 5, 5, 7, 7, 0], [44, 1, 1, 2, 2, 0]],
         .5,
         []),

        # correct instance, predictions exactly the same for multiple detects
        # order is mixed up
        ([[44, 5, 5, 7, 7, 0], [44, 1, 1, 2, 2, 0]],
         [[44, 1, 1, 2, 2, 0], [44, 5, 5, 7, 7, 0]],
         .5,
         []),

        # missing detection
        ([[44, 5, 5, 7, 7, 0]],
         [[44, 1, 1, 2, 2, 0], [44, 5, 5, 7, 7, 0]],
         .5,
         [ErrorLabelType.MISSING]),

        # correct instance, prediction not = but w/in iou threshold
        ([[44, 162, 65, 365, 660, 0]],
         [[44, 162, 65, 365, 670, 0]],
         .5,
         []),

        # class error with 100% overlap
        ([[44, 162, 65, 365, 660, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         [ErrorLabelType.CLASS_NAME]),

        # class error with overlapping bb, but not completely
        ([[44, 162, 65, 365, 660, 0]],
         [[1, 170, 65, 365, 660, 0]],
         .5,
         [ErrorLabelType.CLASS_NAME]),

        # complete miss, minor overlap with original (less than iou)
        ([[1, 162, 65, 1000, 1000, 0]],
         [[2, 162, 65, 1000, 200, 0]],
         .5,
         [ErrorLabelType.CLASS_LOCALIZATION]),

        # duplicate detection,
        ([[1, 162, 65, 365, 660, 0], [1, 162, 65, 365, 660, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         [ErrorLabelType.DUPLICATE_DETECTION]),

        # duplicate detection, detections not identical but w/in iou range.
        # same conf score
        ([[1, 162, 65, 365, 660, 0], [1, 162, 65, 365, 659, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         [ErrorLabelType.DUPLICATE_DETECTION]),

        # duplicate detection, detections not identical but w/in iou range.
        # Detection with lower iou has higher conf score
        ([[1, 162, 65, 365, 660, 0], [1, 162, 65, 365, 659, 50]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         [ErrorLabelType.DUPLICATE_DETECTION]),

        # Same as previous, but in a different order to ensure that conf score
        # prioritization is working correctly
        ([[1, 162, 65, 365, 659, 50], [1, 162, 65, 365, 660, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         [ErrorLabelType.DUPLICATE_DETECTION]),

        # background error with same class
        ([[1, 50, 50, 100, 100, 0]],
         [[1, 350, 350, 100, 100, 0]],
         .5,
         [ErrorLabelType.BACKGROUND]),

        # background error with different class
        ([[0, 50, 50, 100, 100, 0]],
         [[1, 350, 350, 100, 100, 0]],
         .5,
         [ErrorLabelType.BACKGROUND]),

        # gt is empty
        ([[0, 1, 1, 20, 20, 0]],
         [],
         .5,
         [ErrorLabelType.BACKGROUND]),

        # pred is empty
        ([],
         [[0, 1, 1, 20, 20, 0]],
         .5,
         [ErrorLabelType.MISSING]),

        # edge case for background error (bb's touch, but iou is 0)
        ([[44, 5, 5, 1, 1, 0]],
         [[44, 6, 6, 1, 1, 0]],
         .5,
         [ErrorLabelType.BACKGROUND]),
    ])
    def test_object_detection_err_list_labeling(self,
                                                pred_y,
                                                true_y,
                                                iou_threshold,
                                                result):
        """Compare error list to expected result."""
        task_type = 'object_detection'
        mng = ErrorLabeling(task_type,
                            pred_y,
                            true_y,
                            iou_threshold)
        lst = mng.compute_error_list()
        assert (lst == result)
