import sys
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import numpy as np
import os
from pathlib import Path

input_dir = './data/pmc_2022/pmc_coco/plots_detection/bar_images'
output_area_dir = './extended_output/area_extracted'
output_initial_dir = './extended_output/initial_result'

# input_dir = './data/pmc_2022/pmc_coco/plots_detection/bar_test'
# output_area_dir = './output/area_extracted'
# output_initial_dir = './output/initial_result'

os.makedirs(output_area_dir, exist_ok=True)
os.makedirs(output_initial_dir, exist_ok=True)
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Specify the path to model config and checkpoint file
config_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
checkpoint_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')

def show_plot_area_only(img, result, out_file=None):
    """Custom visualization function to show only plot area bbox without label"""
    img = mmcv.imread(img)
    img = img.copy()
    
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
            
    x1, y1, x2, y2, score = best_bbox
    
    # Draw only the best plot area bbox
    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color=(72, 101, 241),
        thickness=2
    )
    
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    
    return best_bbox


for i in  image_files:
    # img = f'./sample{i}.jpg'
    # img = f'./data/pmc_2022/plots_detection/bar_test/{all the files in this folder}.jpg'
    # plot_area_extracted_img = f'./output/area_extracted/{all the files in this folder}_output.jpg'
    # sample_result_img = f'./output/initial_result/result{i}.jpg'
    img = os.path.join(input_dir, i)
    plot_area_extracted_img = os.path.join(output_area_dir, f'{Path(i).stem}_output.jpg')
    sample_result_img = os.path.join(output_initial_dir, f'{Path(i).stem}_result.jpg')

    # Run detection
    result = inference_detector(model, img)
    
    # Show and save only plot area detection
    best_bbox = show_plot_area_only(img, result, out_file=sample_result_img)
    
    if best_bbox is not None:
        x1, y1, x2, y2, score = best_bbox
        
        # Extract the plot area
        img_tmp = cv2.imread(sample_result_img)
        plot_area = img_tmp[int(y1):int(y2), int(x1):int(x2)]
        
        cv2.imwrite(plot_area_extracted_img, plot_area)
        print(f"plot_area_extracted_img saved at {plot_area_extracted_img}")
    else:
        print(f"Could not process image {i} - no plot area detected")