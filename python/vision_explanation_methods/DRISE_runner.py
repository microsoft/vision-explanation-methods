import torch
import torchvision.models.detection as detection
from torchvision import transforms as T
from captum.attr import visualization as viz
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image as Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os
import numpy as np
from bs4 import BeautifulSoup
import argparse

from sys import path

from .explanations import drise as drise
from .explanations import common as od_common



class PytorchFasterRCNNWrapper(od_common.GeneralObjectDetectionModelWrapper):
    """
    To be compatible with the drise explainability method, all models must be wrapped to have
    the same output and input class.
    This wrapper is customized for the FasterRCNN model from Pytorch, and can
    also be used with the RetinaNet or any other models with the same output class.
    """
    
    def __init__(self, model, number_of_classes):
        self._model = model
        self._number_of_classes = number_of_classes

    # This is the only method that needs to be impelmented. It takes a tensor and 
    # returns a list of detection records.


    def predict(self, x):
        raw_detections = self._model(x)

        def apply_nms(orig_prediction, iou_thresh=0.5): 
            keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

            nms_prediction = orig_prediction
            nms_prediction['boxes'] = nms_prediction['boxes'][keep]
            nms_prediction['scores'] = nms_prediction['scores'][keep]
            nms_prediction['labels'] = nms_prediction['labels'][keep]
            return nms_prediction
        
        def filter_score(orig_prediction, score_thresh=0.2):
            keep = orig_prediction['scores'] > 0.2
            filter_prediction = orig_prediction
            filter_prediction['boxes'] = filter_prediction['boxes'][keep]
            filter_prediction['scores'] = filter_prediction['scores'][keep]
            filter_prediction['labels'] = filter_prediction['labels'][keep]
            return filter_prediction
        
        detections = [] 
        for raw_detection in raw_detections:
            raw_detection = apply_nms(raw_detection,0.005)
            
            # Note that FasterRCNN doesn't return a socre for each class, only the predicted class
            # DRISE requires a score for each class. We approximate the score for each class
            # by dividing the (1.0 - class score) evenly among the other classes.
            
            raw_detection = filter_score(raw_detection, 0.2) # filter by class scores
            expanded_class_scores = od_common.expand_class_scores(raw_detection['scores'],
                                                                  raw_detection['labels'],
                                                                  self._number_of_classes)
            detections.append(
                od_common.DetectionRecord(
                    bounding_boxes=raw_detection['boxes'],
                    class_scores=expanded_class_scores,
                    objectness_scores=torch.tensor([1.0]*raw_detection['boxes'].shape[0]),
                    
                )
            )
        
        return detections


def plot_img_bbox(ax, box, label, color):
    """
    helper function to plot final visualizations
    """
    x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
    rect = patches.Rectangle((x, y),
                            width, height,
                            linewidth = 2,
                            edgecolor = color,
                            facecolor = 'none',
                            label=label)
    ax.add_patch(rect)
    frame = ax.get_position()
    ax.set_position([frame.x0, frame.y0, frame.width * 0.8, frame.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax


def get_instance_segmentation_model(num_classes):
    """
    To load in recycling pre-trained model -
    load an instance segmentation model pre-trained on COCO
    """
    model2= torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model2.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model2.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model2


def final_visualization(imagelocation,modellocation,numclasses,savename,nummasks=25,maskres=(4,4),devicechoice=None):
    """
    Parse user input and call drise on specified model.
    Save visualizations.
    """
    
    if not devicechoice:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = devicechoice

    if not modellocation:
        # If user did not specify a model location, we simply load in the pytorch pre-trained model.
        print("using pretrained model")
        model = detection.fasterrcnn_resnet50_fpn(pretrained=True,map_location=device)
        numclasses = 91
        
    else:
        print("loading user model")
        model = get_instance_segmentation_model(numclasses)
        model.load_state_dict(torch.load(modellocation,map_location=device))
        
    test_image = Image.open(imagelocation).convert('RGB')
    
    model = model.to(device)
    model.eval()

    print(PytorchFasterRCNNWrapper)
    explainable_wrapper = PytorchFasterRCNNWrapper(model, numclasses)

    detections = explainable_wrapper.predict(T.ToTensor()(test_image).unsqueeze(0).repeat(2, 1, 1, 1).to(device))
    saliency_scores = drise.DRISE_saliency(
        model=explainable_wrapper,
        image_tensor=T.ToTensor()(test_image).repeat(2, 1, 1, 1).to(device), # Repeated the tensor to test batching.
        target_detections=detections,
        number_of_masks=nummasks, # This is how many masks to run - more is slower but gives higher quality mask.
        device=device,
        mask_res=maskres, # This is the resolution of the random masks. High resolutions will give finer masks, but more need to be run.
        verbose=True # Turns progress bar on/off.
    ) 

    img_index = 0
    saliency_scores = [saliency_scores[img_index][i] for i in range(len(saliency_scores[img_index])) 
    if torch.isnan(saliency_scores[img_index][i]['detection']).any() == False] #exclude scores containing NaN
    num_detections = len(saliency_scores)

    if num_detections == 0: #if no objects have been detected... 
        fail = Image.open(os.path.join("images","notfound.png")) 
        fail = fail.save(savename)
        return None,None

    fig, axis = plt.subplots(1, num_detections,figsize= (num_detections*10,10))

    # color and label mappings 
    cmap = plt.cm.get_cmap('tab20c', len(detections[0].class_scores[0]))
    

    for i in range(num_detections):
        viz.visualize_image_attr(
            numpy.transpose(saliency_scores[i]['detection'].cpu().detach().numpy(), (1, 2, 0)), #drise runner usually has access to an images folder
            # The [0][0] means first image, first detection.
            numpy.transpose(T.ToTensor()(test_image).numpy(), (1, 2, 0)),
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            cmap=plt.cm.inferno,
            title="Detection " + str(i),
            plt_fig_axis = (fig, axis[i]),
            use_pyplot = False
        )
        
        box = detections[img_index].bounding_boxes[i].detach().numpy() 
        label = int(torch.argmax(detections[img_index].class_scores[i]))  
        if num_detections>1: #if there is more than one element to display, hence multiple subplots
            #applicable only to recycling dataset.
            axis[i] = plot_img_bbox(axis[i],box,str(label),'r')
        elif type(axis)!=list: 
            axis = plot_img_bbox(axis,box,str(label),'r')
        else: #unclear why, but sometimes even with just one element axis needs to be indexed
            axis[i] = plot_img_bbox(axis[i],box,str(label),'r')
        fig.savefig(savename)
    return fig,savename



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagelocation", default='images/cartons.jpg',\
        help = "image subdirectory. Default: images/cartons.jpg", type=str)
    parser.add_argument("--modellocation", default=None ,help = "fine-tuned model subdirectory. Default: pre-trained FastRCNN from Pytorch")
    parser.add_argument("--numclasses", default=91 ,help = "number of classes. Default: 91",type=int) #interestingly, not enforcing int made it a float ;v;
    parser.add_argument("--savename", default='res/outputmaps.jpg' ,help = "exported Filename. Default: res/outputmaps.jpg", type=str)
    
    parser.add_argument("--nummasks", default=25 ,help = "number of masks. Default: 25", type=int)
    parser.add_argument("--maskres", default=(4,4) ,help = "mask resolution. Default: (4,4) ", type=tuple)
    parser.add_argument("--device",default=None, help="enforce certain device. Default: cuda:0 if available, cpu if not.", type=str)

    args = parser.parse_args()

    #generate(4,5)

    res = final_visualization(args.imagelocation, args.modellocation, args.numclasses, args.savename, args.nummasks, args.maskres, args.device)