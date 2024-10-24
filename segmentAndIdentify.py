import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.filters import sobel
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage import graph



image = cv2.imread('test.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

edges = sobel(gray_image)

# segments = slic(lab_image, n_segments=200, compactness=30, sigma=1, start_label=1)

# segmented_image = label2rgb(segments, image, kind='avg', bg_label=0, alpha=0.5)

# edges_colored = np.dstack([edges]*3)
# edges_colored = (edges_colored * [255, 0, 0]).astype(np.uint8)
#
# overlay_image = cv2.addWeighted(segmented_image, 0.7, edges_colored, 0.3, 0)
#


# Use Gaussian blur to smooth the image and reduce noise
blurred_image = cv2.GaussianBlur(lab_image, (5, 5), 0)

# Output edge information separately
plt.figure(figsize=(10, 10))
plt.imshow(edges, cmap='gray')  # The edge image is in grayscale
plt.axis('off')
plt.savefig('output_edges.png', bbox_inches='tight', pad_inches=0)
plt.show()

# SLIC superpixel segmentation
# Use a high compactness value to reduce unnecessary segmentation and retain smooth edges
n_segments = 150  # Number of superpixels
compactness = 50  # A higher compactness reduces segmentation in gradient areas and makes edges smoother

segments = slic(blurred_image, n_segments=n_segments, compactness=compactness, sigma=1, start_label=1)

# Merge superpixels based on color similarity
rag = graph.rag_mean_color(image, segments)

threshold = 30
new_segments = graph.cut_threshold(segments, rag, threshold)

num_segments = len(np.unique(new_segments))
print(num_segments)

plt.figure(figsize=(10, 10))
plt.imshow(label2rgb(new_segments, image, kind='avg', bg_label=0, alpha=0.5))
plt.axis('off')
plt.show()

# num_segments = len(np.unique(segments))
# print(num_segments)
# # Generate a transparent overlay of the superpixels
# segmented_image = label2rgb(segments, image, kind='avg', bg_label=0, alpha=0.5)
#
# # Display and save the segmentation result
# plt.figure(figsize=(10, 10))
# plt.imshow(segmented_image)
# plt.axis('off')
# plt.savefig('output_segmented_chart_lab_smooth.png', bbox_inches='tight', pad_inches=0)
# plt.show()
#
#
# segments = slic(image, n_segments=150, compactness=50)
#
# # 创建区域邻接图 (RAG)
# rag = graph.rag_mean_color(image, segments)
#
# # 基于颜色的阈值合并超像素块
# threshold = 30  # 设定一个颜色相似性的阈值
# new_segments = graph.cut_threshold(segments, rag, threshold)
#
# # 显示合并后的图像
# plt.figure(figsize=(10, 10))
# plt.imshow(label2rgb(new_segments, image, kind='avg', bg_label=0, alpha=0.5))
# plt.axis('off')
# plt.show()