# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Test functions for the vision-explanation-methods package."""

import logging
import os
import pytest
import urllib.request as request_file

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


def test_vision_explain_loadmodel():
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
    model = _get_instance_segmentation_model(5)
    model.load_state_dict(torch.load(modelpath, device))
    model.to(device)

    res = dr.get_drise_saliency_map(imagelocation=imgpath,
                                    model=PytorchDRiseWrapper(
                                        model=model,
                                        number_of_classes=5),
                                    numclasses=5,
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
                                     model=PytorchDRiseWrapper(
                                          model=model,
                                          number_of_classes=5),
                                     numclasses=5,
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

    @pytest.mark.parametrize(("pred_y", "true_y", "iou_threshold", "result"), [
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
    def test_pointing_game(img_fname=str,
                           gt_bbox="list"[int],
                           threshold=float,
                           num_masks=int):
        """Test calculate_gt_salient_pixel_overlap."""
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
                    match='Threshold parameter not a float \
                             between 0 and 1.'):
                salient_scores = pg.pointing_game(img_fname,
                                                  0,
                                                  threshold=threshold,
                                                  num_masks=num_masks)

        if not num_masks > 0:
            with pytest.raises(
                    ValueError,
                    match='Number of masks parameter not a \
                             positive int.'):
                salient_scores = pg.pointing_game(img_fname,
                                                  0,
                                                  threshold=threshold,
                                                  num_masks=num_masks)

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
                if j > 0:
                    if (gt_bbox[1] <= iindex <= gt_bbox[3]
                       and gt_bbox[0] <= jindex <= gt_bbox[2]):
                        good += 1
                if (gt_bbox[1] <= iindex <= gt_bbox[3]
                   and gt_bbox[0] <= jindex <= gt_bbox[2]):
                    total += 1
        overlap_check = good/total

        assert round(overlap, 2) == round(overlap_check, 2)

    def test_vision_explain_evaluation():
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
        model = _get_instance_segmentation_model(5)
        model.load_state_dict(torch.load(modelpath, device))
        model.to(device)
        model.eval()
        detection_model = PytorchDRiseWrapper(model=model,
                                              number_of_classes=1000)

        # find saliency scores for top 20% of salient pixels
        # do this for the second object in an image
        pg = PointingGame(detection_model)
        index = 0
        s = pg.pointing_game(imgpath, index)

        # check that the saliency map exists and has 3 channels
        assert (len(s) == 3)

        # calculate overlap
        v = pg.calculate_gt_salient_pixel_overlap(s, [244, 139, 428, 519])

        # check that this is a percent value
        assert (0 < v < 1)
