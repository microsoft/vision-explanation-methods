# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Test functions for the vision-explanation-methods package."""

import logging
import os
import urllib.request as request_file

import matplotlib.pyplot as plt
import vision_explanation_methods.DRISE_runner as dr

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

try:
    import torch
except ImportError:
    module_logger.debug('Could not import torch, required if using a' +
                        'PyTorch model')

# execute tests from the root folder as follows:
# pytest tests/test_vision_explanation.py


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
                                    numclasses=5,
                                    savename=savepath)

    # assert that result is a tuple of figure, location, and labels.
    assert(len(res) == 3)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(isinstance(res[0], type(fig)))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res[1]))

    # assert that labels returned are in a list.
    assert(isinstance(res[2], list))

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
                                     numclasses=5,
                                     savename=savepath2)

    # assert that result is a tuple of figure, location, and labels.
    assert(len(res2) == 3)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(isinstance(res2[0], type(fig)))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res2[1]))

    # assert that labels returned are in a list.
    assert(isinstance(res2[2], list))

    print("Test1 passed for single detection")

    # delete files created during testing
    for elt in [savepath, savepath2]:
        os.remove(elt)


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
    model = (torch.load(modelpath,
                        map_location='cuda' if torch.cuda.is_available()
                        else 'cpu'))
    res = dr.get_drise_saliency_map(imagelocation=imgpath,
                                    model=model,
                                    numclasses=5,
                                    savename=savepath)

    # assert that result is a tuple of figure, location, and labels.
    assert(len(res) == 3)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(isinstance(res[0], type(fig)))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res[1]))

    # assert that labels returned are in a list.
    assert(isinstance(res[2], list))

    print("Test2 passed for multiple detections")

    # just one carton
    imgpath2 = os.path.join('python', 'vision_explanation_methods', 'images',
                            'justmilk.jpg')
    savepath2 = os.path.join('python', 'vision_explanation_methods', 'res',
                             'testoutput_loadedmodel2.jpg')

    # run the main function for saliency map generation
    # in the case of just a single item in photo
    res2 = dr.get_drise_saliency_map(imagelocation=imgpath2,
                                     model=None,
                                     numclasses=5,
                                     savename=savepath2)

    # assert that result is a tuple of figure, location, and labels.
    assert(len(res2) == 3)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(isinstance(res2[0], type(fig)))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res2[1]))

    # assert that labels returned are in a list.
    assert(isinstance(res2[2], list))

    print("Test2 passed for single detection")

    # delete files created during testing
    for elt in [savepath, savepath2, modelpath]:
        os.remove(elt)
