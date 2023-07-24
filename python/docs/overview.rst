.. _overview:

Overview
========

The vision-explanation-methods repository provides a set of tools and methods for generating saliency maps for object detection models, evaluating explanations, and error labeling. 

The repository includes the following main components:

- **DRISE Runner**: This module provides a method for generating saliency maps for object detection models. It uses the DRISE (Detection-based Rise) method to generate saliency maps for a given image and model. The generated saliency maps can be used to understand which parts of the image are most important for the model's predictions.

- **Pointing Game**: This module provides a variety of explanation evaluation tools. It includes a method for visualizing highly salient pixels and calculating the overlap between salient pixels and ground truth bounding boxes.

- **Error Labeling**: This module provides a class for error labeling in object detection models. It includes methods for calculating Intersection over Union (IoU) scores and assigning error labels based on the IoU scores and class labels.

- **Setup**: The setup file for the vision-explanation-methods package includes the package metadata and dependencies.

The repository also includes a set of guidelines for contributing to the project, a code of conduct, and a license file.

For more detailed information about each component and how to use them, please refer to the respective sections in the documentation.