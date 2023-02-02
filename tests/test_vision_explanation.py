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
    imgpath = os.path.join('python','vision_explanation_methods','images',
                           'cartons.jpg')  # unseen test image
    savepath = os.path.join('python','vision_explanation_methods','res',
                            'testoutput_preloaded.jpg')
    # save tested result in res

    # run the main function for saliency map generation
    res = dr.get_drise_saliency_map(imgpath, None, None, None, savepath)

    # assert that result is a tuple of figure, location, and labels.
    assert(len(res) == 3)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(type(res[0]) == type(fig))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res[1]))

    #assert that labels returned are in a list.
    assert(type(res[2]) == type(["cup"]))

    print("Test1 passed for multiple detections")

    imgpath2 = os.path.join('python','vision_explanation_methods','images',
                           'justmilk.jpg')  # just one carton
    savepath2 = os.path.join('python','vision_explanation_methods','res',
                            'testoutput_preloaded2.jpg') 

    # run the main function for saliency map generation
    # in the case of just a single item in photo
    res2 = dr.get_drise_saliency_map(imgpath2, None, None, None, savepath2)

    # assert that result is a tuple of figure, location, and labels.
    assert(len(res2) == 3)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(type(res2[0]) == type(fig))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res2[1]))

    #assert that labels returned are in a list.
    assert(type(res2[2]) == type(["cup"]))

    print("Test1 passed for single detection")

    #delete files created during testing
    for elt in [savepath, savepath2]: 
        os.remove(elt)



def test_vision_explain_loadmodel():
    """End to end testing for saliency map generation function."""
    imgpath = os.path.join('python','vision_explanation_methods','images',
                           'cartons.jpg')  # unseen test image
    modelpath = os.path.join('python','vision_explanation_methods','models',
                             'fastrcnn.pt')  # load fastrcnn model
    savepath = os.path.join('python','vision_explanation_methods','res',
                            'testoutput_loadedmodel.jpg')
    # save tested result in res

    # use helper function from above to fetch model
    _ = download_assets(modelpath)

    # run the main function for saliency map generation
    res = dr.get_drise_saliency_map(imgpath, None, modelpath, 5, savepath)

    # assert that result is a tuple of figure, location, and labels.
    assert(len(res) == 3)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(type(res[0]) == type(fig))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res[1]))


    #assert that labels returned are in a list.
    assert(type(res[2]) == type(["cup"]))

    print("Test2 passed for multiple detections")

    imgpath2 = os.path.join('python','vision_explanation_methods','images',
                           'justmilk.jpg')  # just one carton
    savepath2 = os.path.join('python','vision_explanation_methods','res',
                            'testoutput_loadedmodel2.jpg') 

    # run the main function for saliency map generation
    # in the case of just a single item in photo
    res2 = dr.get_drise_saliency_map(imgpath2, None, None, None, savepath2)

    # assert that result is a tuple of figure, location, and labels.
    assert(len(res2) == 3)

    # assert that first element in result is a figure.
    fig, axis = plt.subplots(2, 2)
    assert(type(res2[0]) == type(fig))

    # assert that figure has been saved in proper location.
    assert(os.path.exists(res2[1]))

    #assert that labels returned are in a list.
    assert(type(res2[2]) == type(["cup"]))

    print("Test2 passed for single detection")

    #delete files created during testing
    for elt in [savepath, savepath2, modelpath]: 
        os.remove(elt)

