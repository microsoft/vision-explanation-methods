"""Method for generating saliency maps for object detection models."""

import os
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
import torchvision.models.detection as detection
from captum.attr import visualization as viz
from ml_wrappers.model.image_model_wrapper import PytorchFasterRCNNWrapper
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .explanations import drise


def plot_img_bbox(ax: matplotlib.axes._subplots, box: numpy.ndarray,
                  label: str, color: str):
    """Plot predicted bounding box and label on the D-RISE saliency map.

    :param ax: Axis on which the d-rise saliency map was plotted
    :type ax: Matplotlib AxesSubplot
    :param box: Bounding box the model predicted
    :type box: numpy.ndarray
    :param label: Label the model predicted
    :type label: str
    :param color: Color of the bounding box based on predicted label
    :type color: single letter color string
    :return: Axis with the predicted bounding box and label plotted on top of
        d-rise saliency map
    :rtype: Matplotlib AxesSubplot
    """
    x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
    rect = matplotlib.patches.Rectangle((x, y),
                                        width,
                                        height,
                                        linewidth=2,
                                        edgecolor=color,
                                        facecolor='none',
                                        label=label)
    ax.add_patch(rect)
    frame = ax.get_position()
    ax.set_position([frame.x0, frame.y0, frame.width * 0.8, frame.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax


def get_instance_segmentation_model(num_classes: int):
    """Load in pre-trained Faster R-CNN model with resnet50 backbone.

    :param num_classes: Number of classes model predicted
    :type num_classes: int
    :return: Faster R-CNN PyTorch model
    :rtype: PyTorch model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_drise_saliency_map(
        imagelocation: str,
        model: Optional[object],
        numclasses: int,
        savename: str,
        nummasks: int = 25,
        maskres: Tuple[int, int] = (4, 4),
        maskpadding: Optional[int] = None,
        devicechoice: Optional[str] = None
):
    """Run D-RISE on image and visualize the saliency maps.

    :param imagelocation: Path of the image location
    :type imagelocation: str
    :param model: Input model for D-RISE. If None, Faster R-CNN model
        will be used.
    :type model: PyTorch model
    :param numclasses: Number of classes model predicted
    :type numclasses: int
    :param savename: Path of the saved output figure
    :type savename: str
    :param nummasks: Number of masks to use for saliency
    :type nummasks: int
    :param maskres: Resolution of mask before scale up
    :type maskres: Tuple of ints
    :param maskpadding: How much to pad the mask before cropping
    :type: Optional int
    :param devicechoice: Device to use to run the function
    :type devicechoice: str
    :param wrapperchoice: Choice to use fastrcnn wrapper or custom wrapper
    :type wrapperchoice: class
    :return: Tuple of Matplotlib figure, path to where the output
        figure is saved
    :rtype: Tuple of Matplotlib figure, str
    """
    if not devicechoice:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = devicechoice

    if not model:
        print("using pretrained fastercnn model")
        model = PytorchFasterRCNNWrapper(
            detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                              map_location=device), numclasses)

    test_image = Image.open(imagelocation).convert('RGB')

    detections = model.predict(
        T.ToTensor()(test_image).unsqueeze(0).repeat(2, 1, 1, 1).to(device))

    saliency_scores = drise.DRISE_saliency(
        model=model,
        # Repeated the tensor to test batching
        image_tensor=T.ToTensor()(test_image).repeat(2, 1, 1, 1).to(device),
        target_detections=detections,
        # This is how many masks to run -
        # more is slower but gives higher quality mask.
        number_of_masks=nummasks,
        mask_padding=maskpadding,
        device=device,
        # This is the resolution of the random masks.
        # High resolutions will give finer masks, but more need to be run.
        mask_res=maskres,
        verbose=True  # Turns progress bar on/off.
    )

    img_index = 0

    # Filter out saliency scores containing nan values
    saliency_scores = [saliency_scores[img_index][i]
                       for i in range(len(saliency_scores[img_index]))
                       if not torch.isnan(
                       saliency_scores[img_index][i]['detection']).any()]

    num_detections = len(saliency_scores)

    if num_detections == 0:  # If no objects have been detected...
        fail = Image.open(os.path.join("python", "vision_explanation_methods",
                                       "images", "notfound.jpg"))
        fail = fail.save(savename)
        return None, None

    fig, axis = plt.subplots(1, num_detections,
                             figsize=(num_detections*10, 10))

    label_list = []
    for i in range(num_detections):
        box = detections[img_index].bounding_boxes[i].detach().numpy()
        label = int(torch.argmax(detections[img_index].class_scores[i]))
        label_list.append(label)

        # There is more than one element to display, hence multiple subplots
        # Unclear why, but sometimes even with just one element,
        # axis needs to be indexed
        if num_detections > 1 or type(axis) == list:
            ax = axis[i]
        else:
            ax = axis

        viz.visualize_image_attr(
            numpy.transpose(
                saliency_scores[i]['detection'].cpu().detach().numpy(),
                (1, 2, 0)),
            numpy.transpose(T.ToTensor()(test_image).numpy(), (1, 2, 0)),
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            cmap=plt.cm.inferno,
            title="Detection " + str(i),
            plt_fig_axis=(fig, ax),
            use_pyplot=False
        )

        if num_detections > 1 or type(axis) == list:
            axis[i] = plot_img_bbox(axis[i], box, str(label), 'r')
        else:
            axis = plot_img_bbox(axis, box, str(label), 'r')

    fig.savefig(savename)
    return fig, savename, label_list
