# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Test functions for the vision-explanation-methods package."""

import os
import urllib.request as request_file

import matplotlib.pyplot as plt
import vision_explanation_methods.DRISE_runner as dr

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
    imgpath = os.path.join('python/vision_explanation_methods/images',
                           'cartons.jpg')  # unseen test image
    savepath = os.path.join('python/vision_explanation_methods/res',
                            'testoutput_preloaded.jpg')
    # save tested result in res

    # run the main function for saliency map generation
    res = dr.get_drise_saliency_map(imgpath, None, None, None, savepath)

    # assert that result is a tuple of figure and its location.
    assert(len(res) == 2)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(type(res[0]) == type(fig))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res[1]))

    print("Test1 passed")


def test_vision_explain_loadmodel():
    """End to end testing for saliency map generation function."""
    imgpath = os.path.join('python/vision_explanation_methods/images',
                           'cartons.jpg')  # unseen test image
    modelpath = os.path.join('python/vision_explanation_methods/models',
                             'fastrcnn.pt')  # load fastrcnn model
    savepath = os.path.join('python/vision_explanation_methods/res',
                            'testoutput_loadedmodel.jpg')
    # save tested result in res

    # use helper function from above to fetch model
    _ = download_assets(modelpath)

    # run the main function for saliency map generation
    res = dr.get_drise_saliency_map(imgpath, None, modelpath, 5, savepath)

    # assert that result is a tuple of figure and its location.
    assert(len(res) == 2)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(type(res[0]) == type(fig))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res[1]))

    print("Test2 passed")
