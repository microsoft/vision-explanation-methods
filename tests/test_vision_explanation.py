# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Test functions for the vision-explanation-methods package."""

import logging
import os
import urllib.request as request_file

import pytest
import torchvision.models.detection as d
import vision_explanation_methods.DRISE_runner as dr
from ml_wrappers.model.image_model_wrapper import PytorchDRiseWrapper
from vision_explanation_methods.evaluation.pointing_game import PointingGame

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

try:
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
except ImportError:
    module_logger.debug('Could not import torchvision packages, required' +
                        'if using a PyTorch computer vision model')

try:
    import torch
except ImportError:
    module_logger.debug('Could not import torch, required if using a' +
                        'PyTorch model')

# execute tests from the root folder as follows:
# pytest tests/test_vision_explanation.py

BASE_DIR = "./python/vision_explanation_methods/images/"
NUM_CLASSES = 5
TRANSFORM_SIZE = 400


def download_assets(filepath, force=False):
    """Download Faster R-CNN model."""
    if force or not os.path.exists(filepath):
        request_file.urlretrieve(
            "https://publictestdatasets.blob.core.windows.net" +
            "/models/fastrcnn.pt",
            os.path.join(filepath))
    else:
        print('Found' + filepath)

    return filepath


def test_vision_explain_preloaded():
    """End to end testing for saliency map generation function."""
    # unseen test image
    imgpath = os.path.join('python', 'vision_explanation_methods', 'images',
                           'cartons.jpg')
    savepath = os.path.join('python', 'vision_explanation_methods', 'res',
                            'testoutput_preloaded.jpg')
    # save tested result in res

    # run the main function for saliency map generation
    res = dr.get_drise_saliency_map(imagelocation=imgpath,
                                    model=None,
                                    numclasses=90,
                                    savename=savepath,
                                    max_figures=2)

    # assert that result is a tuple of figure, location, and labels.
    assert (len(res) == 3)

    # assert that first element in result is a string
    assert (isinstance(res[0][0], str))

    # assert that figure has been saved in proper location.
    assert (os.path.exists(res[1]+"0"+".jpg"))

    # assert that labels returned are in a list.
    assert (isinstance(res[2], list))

    print("Test1 passed for multiple detections")

    # just one carton
    imgpath2 = os.path.join('python', 'vision_explanation_methods', 'images',
                            'justmilk.jpg')
    savepath2 = os.path.join('python', 'vision_explanation_methods', 'res',
                             'testoutput_preloaded2.jpg')

    # run the main function for saliency map generation
    # in the case of just a single item in photo
    res2 = dr.get_drise_saliency_map(imagelocation=imgpath2,
                                     model=None,
                                     numclasses=90,
                                     savename=savepath2,
                                     max_figures=2)

    # assert that result is a tuple of figure, location, and labels.
    assert (len(res2) == 3)

    # assert that first element in result is a string
    assert (isinstance(res2[0][0], str))

    # assert that figure has been saved in proper location.
    assert (os.path.exists(res2[1]+"0"+".jpg"))

    # assert that labels returned are in a list.
    assert (isinstance(res2[2], list))

    print("Test1 passed for single detection")

    # delete files created during testing
    for elt in [savepath+"0"+".jpg", savepath2+"0"+".jpg"]:
        os.remove(elt)


def _get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@pytest.mark.parametrize("use_transforms", [True, False])
def test_vision_explain_loadmodel(use_transforms):
    """End to end testing for saliency map generation function."""
    # unseen test image
    imgpath = os.path.join('python', 'vision_explanation_methods', 'images',
                           'cartons.jpg')
    # load fastrcnn model
    modelpath = os.path.join('python', 'vision_explanation_methods', 'models',
                             'fastrcnn.pt')
    savepath = os.path.join('python', 'vision_explanation_methods', 'res',
                            'testoutput_loadedmodel.jpg')
    # save tested result in res

    # use helper function from above to fetch model
    _ = download_assets(modelpath)

    # run the main function for saliency map generation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _get_instance_segmentation_model(NUM_CLASSES)
    model.load_state_dict(torch.load(modelpath, device))
    model.to(device)

    if use_transforms:
        resize_transforms = torchvision.transforms.Resize(size=TRANSFORM_SIZE)
        model = PytorchDRiseWrapper(model=model,
                                    number_of_classes=NUM_CLASSES,
                                    transforms=resize_transforms)
    else:
        model = PytorchDRiseWrapper(model=model,
                                    number_of_classes=NUM_CLASSES)

    res = dr.get_drise_saliency_map(imagelocation=imgpath,
                                    model=model,
                                    numclasses=NUM_CLASSES,
                                    savename=savepath,
                                    max_figures=2)

    # assert that result is a tuple of figure, location, and labels.
    assert (len(res) == 3)

    # assert that first element in result is a string
    assert (isinstance(res[0][0], str))

    # assert that figure has been saved in proper location.
    assert (os.path.exists(res[1]+"0"+".jpg"))

    # assert that labels returned are in a list.
    assert (isinstance(res[2], list))

    print("Test2 passed for multiple detections")

    # just one carton
    imgpath2 = os.path.join('python', 'vision_explanation_methods', 'images',
                            'justmilk.jpg')
    savepath2 = os.path.join('python', 'vision_explanation_methods', 'res',
                             'testoutput_loadedmodel2.jpg')

    # run the main function for saliency map generation
    # in the case of just a single item in photo
    res2 = dr.get_drise_saliency_map(imagelocation=imgpath2,
                                     model=model,
                                     numclasses=NUM_CLASSES,
                                     savename=savepath2,
                                     max_figures=2)

    # assert that result is a tuple of figure, location, and labels.
    assert (len(res2) == 3)

    # assert that first element in result is a string
    assert (isinstance(res2[0][0], str))

    # assert that figure has been saved in proper location.
    assert (os.path.exists(res2[1]+"0"+".jpg"))

    # assert that labels returned are in a list.
    assert (isinstance(res2[2], list))

    print("Test2 passed for single detection")

    # delete files created during testing
    for elt in [savepath+"0"+".jpg", savepath2+"0"+".jpg"]:
        os.remove(elt)


class TestPointingGame(object):
    """Testing error_labeling.py."""

    @pytest.mark.parametrize(("img_fname", "gt_bbox",
                              "threshold", "num_masks"), [
        # correct instance, prediction exactly the same

        # 1 object in the image
        (os.path.join(BASE_DIR, "2.jpg"),
         [247, 192, 355, 493],
         .8,
         100),

        # multiple objects in the image (test defaults to first)
        (os.path.join(BASE_DIR, "128.jpg"),
         [134, 257, 222, 415],
         .8,
         100),

        # Invalid threshold
        (os.path.join(BASE_DIR, "2.jpg"),
         [247, 192, 355, 493],
         -10,
         100),

        # Invalid nummasks
        (os.path.join(BASE_DIR, "2.jpg"),
         [247, 192, 355, 493],
         .8,
         -100),
    ])
    def test_pointing_game(self,
                           img_fname,
                           gt_bbox,
                           threshold,
                           num_masks):
        """
        Test calculate_gt_salient_pixel_overlap.

        :param img_fname: Path of the image location
        :type img_fname: str
        :param gt_bbox: 4 ints representing the x, y, width, height of ground
            truth bounding box
        :type gt_bbox: list of ints
        :param threshold: threshold between 0 and 1 to determine saliency of a
            pixel. If saliency score is below the threshold, then the score is
            set to -1
        :type threshold: float
        :param num_masks: number of masks to run drise with
        :type num_masks: int
        """
        # get fasterrcnn model
        model = d.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        detection_model = PytorchDRiseWrapper(model=model,
                                              number_of_classes=87)

        # find saliency scores for top 20% of salient pixels
        pg = PointingGame(detection_model)

        if not 0 <= threshold <= 1:
            with pytest.raises(
                    ValueError,
                    match='Threshold parameter must be a float \
                             between 0 and 1.'):
                salient_scores = pg.pointing_game(img_fname,
                                                  0,
                                                  threshold=threshold,
                                                  num_masks=num_masks)

        elif not num_masks > 0:
            with pytest.raises(
                    ValueError,
                    match='Number of masks parameter must be a \
                             positive int.'):
                salient_scores = pg.pointing_game(img_fname,
                                                  0,
                                                  threshold=threshold,
                                                  num_masks=num_masks)

        else:
            salient_scores = pg.pointing_game(img_fname,
                                              0,
                                              threshold=threshold,
                                              num_masks=num_masks)
            overlap = pg.calculate_gt_salient_pixel_overlap(salient_scores,
                                                            gt_bbox)

            # calculated salient pixel overlap equals expected brute force
            good = 0
            total = 0
            for iindex, i in enumerate(salient_scores[0]):
                for jindex, j in enumerate(i):
                    if (j > 0 and gt_bbox[1] <= iindex <= gt_bbox[3]
                       and gt_bbox[0] <= jindex <= gt_bbox[2]):
                        good += 1
                    if (gt_bbox[1] <= iindex <= gt_bbox[3]
                       and gt_bbox[0] <= jindex <= gt_bbox[2]):
                        total += 1
            overlap_check = good / total

            assert round(overlap, 2) == round(overlap_check, 2)

    def test_vision_explain_evaluation(self):
        """End to end testing for explanation evaluation."""
        # pointing game run
        imgpath = os.path.join('python', 'vision_explanation_methods',
                               'images', '128.jpg')
        # load fastrcnn model
        modelpath = os.path.join('python', 'vision_explanation_methods',
                                 'models', 'fastrcnn.pt')
        # save tested result in res

        # use helper function from above to fetch model
        _ = download_assets(modelpath)

        # run the main function for saliency map generation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = _get_instance_segmentation_model(NUM_CLASSES)
        model.load_state_dict(torch.load(modelpath, device))
        model.to(device)
        model.eval()
        detection_model = PytorchDRiseWrapper(model=model,
                                              number_of_classes=1000)

        # find saliency scores for top 20% of salient pixels
        # do this for the second object in an image
        pg = PointingGame(detection_model)
        index = 1
        s = pg.pointing_game(imgpath, index)

        # check that the saliency map exists and has 3 channels
        assert len(s) == 3

        # calculate overlap
        v = pg.calculate_gt_salient_pixel_overlap(s, [244, 139, 428, 519])

        # check that this is a percent value
        assert 0 < v < 1
