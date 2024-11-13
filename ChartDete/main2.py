# import sys
# from mmdet.apis import init_detector, inference_detector
# import mmcv
# import cv2

# # Specify the path to model config and checkpoint file
# config_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
# checkpoint_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'

# # build the model from a config file and a checkpoint file
# # model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = './sample.jpg'  # or img = mmcv.imread(img), which will only load it once
  
# plot_area_extracted_img ='./plot_area_extracted_img.jpg'

# model = init_detector(config_file, checkpoint_file, device='cpu')
# result = inference_detector(model, img)
# # print("result", result)

# # visualize the results in a new window
# model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='./sample_result.jpg')


# class_names = model.CLASSES
# # print("class_names", class_names)

# if 'plot_area' not in class_names:
#     print("plot_area not in class_names")
#     sys.exit()   
    
# # plot_area_index = class_names.index('plot_area')
# # print("plot_area_index", plot_area_index)   

# # print("the shape of result", len(result), len(result[0]), len(result[0][plot_area_index]))

# plot_area_index = class_names.index('plot_area')
# plot_area_detections = result[plot_area_index]

# if len(plot_area_detections) == 0:
#     print("plot_area not detected")
#     sys.exit()

# # Select the bounding box with the highest confidence score
# best_bbox = plot_area_detections[0]
# for bbox in plot_area_detections:
#     if bbox[4] > best_bbox[4]:
#         best_bbox = bbox

# x1, y1, x2, y2, score = best_bbox


# # bbox = result[0][plot_area_index]
# # print("bbox", bbox)

# # if len(bbox) == 0:
# #     print("plot_area not detected")
# #     sys.exit()
    
# # x1, y1, x2, y2, score = bbox

# img_tmp = cv2.imread('./sample_result.jpg')
# plot_area = img_tmp[int(y1):int(y2), int(x1):int(x2)]

# cv2.imwrite(plot_area_extracted_img, plot_area)
# print("plot_area_extracted_img saved at", plot_area_extracted_img)


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

# Loop through the sample images
for i in range(1, 5):
    img = f'./sample{i}.jpg'
    plot_area_extracted_img = f'./plot_area_extracted_img{i}.jpg'
    sample_result_img = f'./sample_result{i}.jpg'

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