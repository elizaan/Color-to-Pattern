import sys
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2

# Specify the path to model config and checkpoint file
config_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
checkpoint_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'

# build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = './sample.jpg'  # or img = mmcv.imread(img), which will only load it once
  
plot_area_extracted_img ='./plot_area_extracted_img.jpg'

model = init_detector(config_file, checkpoint_file, device='cpu')
result = inference_detector(model, img)
# print("result", result)

# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='./sample_result.jpg')


class_names = model.CLASSES
# print("class_names", class_names)

if 'plot_area' not in class_names:
    print("plot_area not in class_names")
    sys.exit()   
    
# plot_area_index = class_names.index('plot_area')
# print("plot_area_index", plot_area_index)   

# print("the shape of result", len(result), len(result[0]), len(result[0][plot_area_index]))

plot_area_index = class_names.index('plot_area')
plot_area_detections = result[plot_area_index]

if len(plot_area_detections) == 0:
    print("plot_area not detected")
    sys.exit()

# Select the bounding box with the highest confidence score
best_bbox = plot_area_detections[0]
for bbox in plot_area_detections:
    if bbox[4] > best_bbox[4]:
        best_bbox = bbox

x1, y1, x2, y2, score = best_bbox


# bbox = result[0][plot_area_index]
# print("bbox", bbox)

# if len(bbox) == 0:
#     print("plot_area not detected")
#     sys.exit()
    
# x1, y1, x2, y2, score = bbox

img_tmp = cv2.imread('./sample_result.jpg')
plot_area = img_tmp[int(y1):int(y2), int(x1):int(x2)]

cv2.imwrite(plot_area_extracted_img, plot_area)
print("plot_area_extracted_img saved at", plot_area_extracted_img)
