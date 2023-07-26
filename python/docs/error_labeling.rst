.. _error_labeling:

Error Labeling
==============

The Error Labeling Manager class is defined in the ``error_labeling.py`` file in the ``vision_explanation_methods/error_labeling`` directory. This class is used to label errors in the vision-explanation-methods package.

Error Labeling Manager Class
----------------------------

The Error Labeling Manager class is used to label errors in the vision-explanation-methods package. It uses the Intersection over Union (IoU) threshold to determine the overlap between the predicted and true bounding boxes. The class also uses the ErrorLabelType Enum to define the types of error labels.

The ErrorLabelType Enum provides the following types of error labels:

- MISSING: The ground truth doesn't have a corresponding detection.
- BACKGROUND: The model predicted detections, but there was nothing there. This prediction must have a 0 IoU score with all ground truth detections.
- LOCALIZATION: The predicted class is correct, but the bounding box does not have sufficient overlap with the ground truth (based on the IoU threshold).
- CLASS_NAME: The predicted class is incorrect, but the bounding box is correct.
- CLASS_LOCALIZATION: Both the predicted class and bounding box are incorrect.
- DUPLICATE_DETECTION: The predicted class is correct, the bounding box is correct, but the IoU score is lower than another detection.
- MATCH: The bounding boxes overlap and the class names match.

The Error Labeling Manager class provides the following methods:

- ``compute_error_labels()``: This method computes the error labels for the predicted and true bounding boxes.
- ``compute_error_list()``: This method determines a complete list of errors encountered during prediction.

Error Labeling in Object Detection
-----------------------------------

In object detection, the Error Labeling Manager class is used to label errors in the predictions. The class uses the IoU threshold to determine the overlap between the predicted and true bounding boxes. The class also uses the ErrorLabelType Enum to define the types of error labels.

The ErrorLabelType Enum provides the following types of error labels:

- MISSING: The ground truth doesn't have a corresponding detection.
- BACKGROUND: The model predicted detections, but there was nothing there. This prediction must have a 0 IoU score with all ground truth detections.
- LOCALIZATION: The predicted class is correct, but the bounding box does not have sufficient overlap with the ground truth (based on the IoU threshold).
- CLASS_NAME: The predicted class is incorrect, but the bounding box is correct.
- CLASS_LOCALIZATION: Both the predicted class and bounding box are incorrect.
- DUPLICATE_DETECTION: The predicted class is correct, the bounding box is correct, but the IoU score is lower than another detection.
- MATCH: The bounding boxes overlap and the class names match.

The Error Labeling Manager class provides the following methods:

- ``compute_error_labels()``: This method computes the error labels for the predicted and true bounding boxes.
- ``compute_error_list()``: This method determines a complete list of errors encountered during prediction.

.. note::
   The Error Labeling Manager class is used in the ``test_error_labeling.py`` file in the ``tests`` directory to test the error labeling in the vision-explanation-methods package.