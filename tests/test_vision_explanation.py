import pytest
import sys
import os
import matplotlib.pyplot as plt
import vision_explanation_methods.DRISE_runner as dr
import urllib.request as request_file


#execute tests from the root folder as follows: pytest path/to/test_filename.py


def download_assets(filepath,force=False):
    if force or not os.path.exists(filepath):
        request_file.urlretrieve(
                        "https://publictestdatasets.blob.core.windows.net/models/fastrcnn.pt",
                        os.path.join(filepath))
    else:
        print('Found' + filepath)

    return filepath


def test_vision_explain_preloaded():
    """
    end to end testing for saliency map generation function
    """
    imgpath = os.path.join('python','vision_explanation_methods','images', 'cartons.jpg') #unseen test image
    savepath = os.path.join('python','vision_explanation_methods/res','testoutput_preloaded.jpg')
    #save tested result in res

    res = dr.final_visualization(imgpath, None, None, savepath) #run the main function for saliency map generation

    #assert that result is a tuple of figure and its location.
    assert(len(res)==2)

    #assert that first element in result is a figure.
    fig,axis = plt.subplots(2,2)
    assert(type(res[0]) == type(fig))

    #assert that figure has been saved in proper location.
    assert(os.path.exists(res[1]))

    print("Test1 passed")


def test_vision_explain_loadmodel():
    """
    end to end testing for saliency map generation function
    """
    imgpath = os.path.join('python','vision_explanation_methods','images', 'cartons.jpg') #unseen test image
    modelpath = os.path.join('python','vision_explanation_methods','models','fastrcnn.pt') #load fastrcnn model. 
    savepath = os.path.join('python','vision_explanation_methods','res','testoutput_loadedmodel.jpg')
    #save tested result in res

    _ = download_assets(modelpath) #use helper function from above to fetch model

    res = dr.final_visualization(imgpath, modelpath, 5, savepath) #run the main function for saliency map generation

    #assert that result is a tuple of figure and its location.
    assert(len(res)==2)

    #assert that first element in result is a figure.
    fig,axis = plt.subplots(2,2)
    assert(type(res[0]) == type(fig))

    #assert that figure has been saved in proper location.
    assert(os.path.exists(res[1]))

    print("Test2 passed")

