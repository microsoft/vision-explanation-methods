# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import pytest

from responsibleai_vision import ModelTask
from vision_explanation_methods.error_labeling import (
    ErrorLabelType, ErrorLabeling)


class TestErrorLabelingManager(object):

    @pytest.mark.parametrize("pred_y, true_y, iou_threshold, result, result_missing", [
        # correct instance, prediction exactly the same
        ([[44, 162, 65, 365, 660, 0]],
         [[44, 162, 65, 365, 660, 0]],
         .5,
         [None],
         [None]),

        # correct instance, predictions exactly the same for multiple detects
        ([[44, 162, 65, 365, 660, 0], [44, 1, 4, 7, 10, 0]],
         [[44, 162, 65, 365, 660, 0], [44, 1, 4, 7, 10, 0]],
         .5,
         [None, None],
         [None, None]),

        # correct instance, prediction not = but w/in iou threshold
        ([[44, 162, 65, 365, 660, 0]],
         [[44, 162, 65, 365, 670, 0]],
         .5,
         [None],
         [None]),

        # class error
        ([[44, 162, 65, 365, 660, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         [ErrorLabelType.CLASS_NAME],
         [None]),

        # complete miss, minor overlap with original 5, 660, 0]],
         [[1, 162, 65, 1000, 1000, 0]],
         .5,
         [ErrorLabelType.BOTH],
         [None]),

        # duplicate detection, 
        ([[1, 162, 65, 365, 660, 0], [1, 162, 65, 365, 660, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         [None, ErrorLabelType.DUPLICATE_DETECTION],
         [None]),

        # duplicate detection, detections not identical but w/in iou range
        ([[1, 162, 65, 365, 660, 0], [1, 162, 65, 365, 659, 0]],
         [[1, 162, 65, 365, 660, 0]],
         .5,
         [None, ErrorLabelType.DUPLICATE_DETECTION],
         [None]),

        # background error with same class
        ([[1, 50, 50, 100, 100, 0]],
         [[1, 350, 350, 100, 100, 0]],
         .5,
         [ErrorLabelType.BACKGROUND],
         [ErrorLabelType.MISSING]),

        # background error with different class
        ([[0, 50, 50, 100, 100, 0]],
         [[1, 350, 350, 100, 100, 0]],
         .5,
         [ErrorLabelType.BACKGROUND],
         [ErrorLabelType.MISSING]),

        # complete miss
        ([[0, 1, 1, 20, 20, 0]],
         [[1, 19, 19, 10, 10, 0]],
         .5,
         [ErrorLabelType.BOTH],
         [ErrorLabelType.MISSING]),
    ])
    def test_object_detection_image_labeling(self,
                                             pred_y,
                                             true_y,
                                             iou_threshold,
                                             result,
                                             result_missing):
        task_type = ModelTask.OBJECT_DETECTION
        mng = ErrorLabeling(task_type,
                                   pred_y,
                                   true_y,
                                   iou_threshold)
        mng.compute()
        assert mng._prediction_error_labels == result
        assert mng._missing_labels == result_missing


# TestErrorLabeling().test_object_detection_image_labeling([[1, 50, 50, 100, 100, 0]],
#          [[1, 350, 350, 100, 100, 0]],.5,[None, None])