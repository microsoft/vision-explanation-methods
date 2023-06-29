# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

"""Test error labeling in the vision-explanation-methods package."""

import os
import torchvision.models.detection as d
from ml_wrappers.model.image_model_wrapper import PytorchDRiseWrapper
from vision_explanation_methods.evaluation.pointing_game import PointingGame


class TestPointingGame(object):
    """Testing pointing_game.py."""
    def test_pointing_game(self):
        BASE_DIR = "./python/vision_explanation_methods/images/"
        img_fname = os.path.join(BASE_DIR, "2.jpg")

        # get fasterrcnn model
        model = d.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        model.to("cuda")
        detection_model = PytorchDRiseWrapper(model=model,
                                              number_of_classes=87)

        # find saliency scores for top 20% of salient pixels
        pg = PointingGame(detection_model)
        salient_scores = pg.pointing_game(img_fname, 0, .8)
        gt_bbox = [247, 192, 355, 493]

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
                    total += 1
        overlap_check = good/total

        assert round(overlap, 2) == round(overlap_check, 2)
