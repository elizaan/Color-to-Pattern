import sys
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import numpy as np

# Specify the path to model config and checkpoint file
config_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
checkpoint_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')

def get_plot_area_bbox(result):
    """Get best plot area bbox without any visualization"""
    class_names = model.CLASSES
    if 'plot_area' not in class_names:
        print("plot_area not in class_names")
        return None
        
    plot_area_index = class_names.index('plot_area')
    plot_area_detections = result[plot_area_index]
    
    if len(plot_area_detections) == 0:
        print("plot_area not detected")
        return None
        
    # Select the bounding box with the highest confidence score
    best_bbox = plot_area_detections[0]
    for bbox in plot_area_detections:
        if bbox[4] > best_bbox[4]:
            best_bbox = bbox
            
    return best_bbox

# Loop through the sample images
for i in range(1, 5):
    img = f'./sample{i}.jpg'
    plot_area_extracted_img = f'./plot_area_extracted_img{i}.jpg'

    # Run detection
    result = inference_detector(model, img)
    
    # Get best plot area bbox
    best_bbox = get_plot_area_bbox(result)
    
    if best_bbox is not None:
        x1, y1, x2, y2, score = best_bbox
        
        # Extract the plot area directly from original image
        img_tmp = cv2.imread(img)  # Read original image instead of visualization
        plot_area = img_tmp[int(y1):int(y2), int(x1):int(x2)]
        
        cv2.imwrite(plot_area_extracted_img, plot_area)
        print(f"plot_area_extracted_img saved at {plot_area_extracted_img}")
    else:
        print(f"Could not process image {i} - no plot area detected")