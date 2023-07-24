.. _using_drise:

Using DRISE for Image Explanation
=================================

DRISE is a black box explainability method for object detection. It generates saliency maps for object detection models. 

The DRISE method is implemented in the ``drise.py`` file located in the ``vision_explanation_methods/explanations`` directory. 

The DRISE method is used in the ``DRISE_runner.py`` file located in the ``vision_explanation_methods`` directory. 

The DRISE method is used in the ``pointing_game.py`` file located in the ``vision_explanation_methods/evaluation`` directory. 

DRISE Implementation
--------------------

The DRISE method is implemented in the ``drise.py`` file. The implementation includes the following functions:

- ``DRISE_saliency``: This function computes the DRISE saliency map. It takes as input an object detection model, an image tensor, a list of target detections, the number of masks to use for saliency, the resolution of the mask before scale up, the amount to pad the mask before cropping, the device to use to run the function, and a boolean indicating whether to print verbose output. It returns a list of tensors, one tensor for each image. Each tensor is of shape [D, 3, W, H], and [i ,3 W, H] is the saliency map associated with detection i.

- ``DRISE_saliency_for_mlflow``: This function is similar to ``DRISE_saliency``, but it is designed to work with the MLflow tracking service. It takes as input a model, an image tensor, a list of target detections, the number of masks to use for saliency, the resolution of the mask before scale up, the amount to pad the mask before cropping, the device to use to run the function, and a boolean indicating whether to print verbose output. It returns a list of tensors, one tensor for each image. Each tensor is of shape [D, 3, W, H], and [i ,3 W, H] is the saliency map associated with detection i.

- ``generate_mask``: This function creates a random mask for image occlusion. It takes as input the lower resolution mask grid shape, the size of the image to be masked, and the amount to offset the mask. It returns an occlusion mask for the image, which has the same shape as the image.

- ``fuse_mask``: This function masks an image tensor. It takes as input an image tensor and a mask for the image. It returns the masked image.

- ``compute_affinity_scores``: This function computes the highest affinity score between two sets of detections. It takes as input a set of detections to get affinity scores for and a set of detections to score against. It returns a set of affinity scores associated with each detection.

Using DRISE
-----------

The DRISE method is used in the ``DRISE_runner.py`` file. The ``get_drise_saliency_map`` function in this file runs D-RISE on an image and visualizes the saliency maps. It takes as input the path of the image location, the model to use for D-RISE, the number of classes the model predicted, the path to save the output figure, the number of masks to use for saliency, the resolution of the mask before scale up, the amount to pad the mask before cropping, the device to use, and the maximum number of figures to generate. It returns a tuple of a list of Matplotlib figures, the path to where the output figure is saved, and a list of labels.

The DRISE method is also used in the ``pointing_game.py`` file. The ``PointingGame`` class in this file uses D-RISE to generate saliency maps for object detection models. The ``pointing_game`` function in this class finds saliency scores for the top 20% of salient pixels. It takes as input the filename of the image, the index of the detection to explain, the threshold for saliency, and the number of masks to use for saliency. It returns a saliency map for the image.

.. note:: The DRISE method requires the PyTorch library.