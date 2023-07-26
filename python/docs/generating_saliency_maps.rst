.. _generating_saliency_maps:

Generating Saliency Maps for Object Detection Models
====================================================

This section provides an overview of the methods used to generate saliency maps for object detection models in the vision-explanation-methods package.

DRISE_runner.py
----------------

The DRISE_runner.py file contains the main function for generating saliency maps, `get_drise_saliency_map()`. This function takes in an image, a model, the number of classes, and a save name as parameters. It also accepts optional parameters for the number of masks, mask resolution, mask padding, device choice, and maximum figures.

The function begins by setting the device to either CUDA or CPU, depending on the availability of a GPU. If a model is not provided, the function loads a pre-trained Faster R-CNN model with a ResNet50 backbone. The image is then converted to a tensor and passed to the model for prediction.

The function then generates saliency scores using the DRISE method. The saliency scores are filtered to remove any scores containing NaN values. If no detections are found, the function raises a ValueError.

The function then generates a list of labels and a list of figures for each detection. Each figure is a visualization of the saliency map for the corresponding detection. The figures are saved as JPEG images and returned as base64 strings.

The function finally returns a tuple containing the list of figures, the save name, and the list of labels.

DRISE_saliency()
----------------

The `DRISE_saliency()` function in the drise.py file is used to compute the DRISE saliency map. This function takes in a model, an image tensor, target detections, and the number of masks as parameters. It also accepts optional parameters for mask resolution, mask padding, device, and verbosity.

The function begins by setting the mask padding and generating a list of mask records. Each mask record contains a mask and a list of affinity scores. The affinity scores are computed by comparing the target detections with the detections made on the masked image.

The function then fuses the masks based on their affinity scores to generate the saliency map.

PointingGame Class
------------------

The PointingGame class in the pointing_game.py file provides methods for evaluating the saliency maps. The `pointing_game()` method calculates the saliency scores for a given object detection prediction. The `calculate_gt_salient_pixel_overlap()` method calculates the overlap between the salient pixels and the ground truth bounding box.

Error Labeling
--------------

The error_labeling.py file provides methods for labeling the errors in the object detection predictions. The `ErrorLabelingManager` class manages the error labeling process. The `label_errors()` method labels the errors based on the intersection over union (IoU) scores between the ground truth and predicted bounding boxes.

.. note::
   The vision-explanation-methods package uses the DRISE method for generating saliency maps. DRISE is a black box explainability method for object detection models. It generates saliency maps by occluding parts of the image with random masks and observing the effect on the model's predictions.