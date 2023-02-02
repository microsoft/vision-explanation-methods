"""Method for generating saliency maps for object detection models."""

import os
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
import torchvision.models.detection as detection
from captum.attr import visualization as viz
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .explanations import common as od_common
from .explanations import drise


class PytorchFasterRCNNWrapper(
        od_common.GeneralObjectDetectionModelWrapper):
    """Wraps a PytorchFasterRCNN model with a predict API function.

    To be compatible with the D-RISE explainability method,
    all models must be wrapped to have the same output and input class and a
    predict function for object detection. This wrapper is customized for the
    FasterRCNN model from Pytorch, and can also be used with the RetinaNet or
    any other models with the same output class.

    :param model: Object detection model
    :type model: PytorchFasterRCNN model
    :param number_of_classes: Number of classes the model is predicting
    :type number_of_classes: int
    """

    def __init__(self, model, number_of_classes: int):
        """Initialize the PytorchFasterRCNNWrapper."""
        self._model = model
        self._number_of_classes = number_of_classes

    def predict(self, x: torch.Tensor) -> List[od_common.DetectionRecord]:
        """Create a list of detection records from the image predictions.

        :param x: Tensor of the image
        :type x: torch.Tensor
        :return: Baseline detections to get saliency maps for
        :rtype: List of Detection Records
        """
        raw_detections = self._model(x)

        def apply_nms(orig_prediction: dict, iou_thresh: float = 0.5):
            """Perform nms on the predictions based on their IoU.

            :param orig_prediction: Original model prediction
            :type orig_prediction: dict
            :param iou_thresh: iou_threshold for nms
            :type iou_thresh: float
            :return: Model prediction after nms is applied
            :rtype: dict
            """
            keep = torchvision.ops.nms(orig_prediction['boxes'],
                                       orig_prediction['scores'], iou_thresh)

            nms_prediction = orig_prediction
            nms_prediction['boxes'] = nms_prediction['boxes'][keep]
            nms_prediction['scores'] = nms_prediction['scores'][keep]
            nms_prediction['labels'] = nms_prediction['labels'][keep]
            return nms_prediction

        def filter_score(orig_prediction: dict, score_thresh: float = 0.5):
            """Filter out predictions with confidence scores < score_thresh.

            :param orig_prediction: Original model prediction
            :type orig_prediction: dict
            :param score_thresh: Score threshold to filter by
            :type score_thresh: float
            :return: Model predictions filtered out by score_thresh
            :rtype: dict
            """
            keep = orig_prediction['scores'] > score_thresh

            filter_prediction = orig_prediction
            filter_prediction['boxes'] = filter_prediction['boxes'][keep]
            filter_prediction['scores'] = filter_prediction['scores'][keep]
            filter_prediction['labels'] = filter_prediction['labels'][keep]
            return filter_prediction

        detections = []
        for raw_detection in raw_detections:
            raw_detection = apply_nms(raw_detection, 0.005)

            # Note that FasterRCNN doesn't return a score for each class, only
            # the predicted class. DRISE requires a score for each class.
            # We approximate the score for each class
            # by dividing (class score) evenly among the other classes.

            raw_detection = filter_score(raw_detection, 0.2)
            expanded_class_scores = od_common.expand_class_scores(
                raw_detection['scores'],
                raw_detection['labels'],
                self._number_of_classes)

            detections.append(
                od_common.DetectionRecord(
                    bounding_boxes=raw_detection['boxes'],
                    class_scores=expanded_class_scores,
                    objectness_scores=torch.tensor(
                        [1.0]*raw_detection['boxes'].shape[0]),
                )
            )

        return detections


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
        modellocation: Optional[str],
        numclasses: int,
        savename: str,
        nummasks: int = 25,
        maskres: Tuple[int, int] = (4, 4),
        maskpadding: Optional[int] = None,
        devicechoice: Optional[str] = None,
        wrapperchoice: Optional[object] = PytorchFasterRCNNWrapper
):
    """Run D-RISE on image and visualize the saliency maps.

    :param imagelocation: Path of the image location
    :type imagelocation: str
    :param model: Input model for D-RISE. If None, Faster R-CNN model
        will be used.
    :type model: PyTorch model
    :param modellocation: Path of the model weights. If None, pre-trained
        Faster R-CNN model will be used.
    :type modellocation: Optional str
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
        if not modellocation:
            # If user did not specify a model location,
            # we simply load in the pytorch pre-trained model.
            print("using pretrained fastercnn model")
            model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                      map_location=device)
            numclasses = 91

        else:
            print("loading user fastercnn model")
            model = get_instance_segmentation_model(numclasses)
            model.load_state_dict(
                torch.load(modellocation, map_location=device))
    else:
        if modellocation:
            print("loading any user model")
            model.load_state_dict(
                torch.load(modellocation, map_location=device))

    test_image = Image.open(imagelocation).convert('RGB')

    model = model.to(device)
    model.eval()

    if not wrapperchoice:
        wrapperchoice = PytorchFasterRCNNWrapper
    explainable_wrapper = wrapperchoice(model, numclasses)

    detections = explainable_wrapper.predict(
        T.ToTensor()(test_image).unsqueeze(0).repeat(2, 1, 1, 1).to(device))

    saliency_scores = drise.DRISE_saliency(
        model=explainable_wrapper,
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

    for i in range(num_detections):
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
            plt_fig_axis=(fig, axis[i]),
            use_pyplot=False
        )

        box = detections[img_index].bounding_boxes[i].detach().numpy()
        label = int(torch.argmax(detections[img_index].class_scores[i]))
        # There is more than one element to display, hence multiple subplots
        if num_detections > 1:
            axis[i] = plot_img_bbox(axis[i], box, str(label), 'r')
        elif type(axis) != list:
            axis = plot_img_bbox(axis, box, str(label), 'r')
        # Unclear why, but sometimes even with just one element,
        # axis needs to be indexed
        else:
            axis[i] = plot_img_bbox(axis[i], box, str(label), 'r')
        fig.savefig(savename)

    return fig, savename
