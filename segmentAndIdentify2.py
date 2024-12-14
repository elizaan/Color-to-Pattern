import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb, rgb2lab, deltaE_ciede2000
from skimage.filters import sobel
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage import graph

image = cv2.imread('./ChartDete/output/area_extracted/PMC2999541___g006_output.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lab_image = rgb2lab(image_rgb)

edges = sobel(gray_image)

blurred_image = cv2.GaussianBlur(image_rgb, (5, 5), 0)

n_segments = 150
compactness = 50
segments = slic(blurred_image, n_segments=n_segments, compactness=compactness, sigma=1, start_label=1)

# Merge superpixels based on color similarity
rag = graph.rag_mean_color(image_rgb, segments)
threshold = 30
new_segments = graph.cut_threshold(segments, rag, threshold)

boundary_image = mark_boundaries(image_rgb, new_segments)

plt.figure(figsize=(10, 10))
plt.imshow(boundary_image)
plt.axis('off')
plt.show()

def is_color_consistent(colors, threshold=15):
    std_dev = np.std(colors, axis=0)
    return np.all(std_dev < threshold)

def is_black_or_white(color_lab):
    L, a, b = color_lab
    if L < 5 and abs(a) < 2 and abs(b) < 2:
        return "black"

    elif L > 95 and abs(a) < 2 and abs(b) < 2:
        return "white"

    return "neither"

valid_superpixels_lab = []
valid_superpixel_count = 0

for seg_val in np.unique(new_segments):
    mask = (new_segments == seg_val)
    superpixel_colors = lab_image[mask]

    mean_color = np.mean(superpixel_colors, axis=0)

    if is_black_or_white(mean_color) != "neither":
        continue

    if is_color_consistent(superpixel_colors):
        valid_superpixel_count += 1

        valid_superpixels_lab.append(mean_color)

print(f"Number of valid superpixels (after removing background and checking consistency): {valid_superpixel_count}")

used_superpixels = set()
closest_color_diffs = []

def find_closest_color(lab_colors, used_superpixels):
    min_diff = float('inf')
    closest_pair = None

    for i in range(len(lab_colors)):
        if i in used_superpixels:
            continue
        for j in range(i + 1, len(lab_colors)):
            if j in used_superpixels:
                continue
            lab1 = lab_colors[i].reshape(1, 3)
            lab2 = lab_colors[j].reshape(1, 3)
            diff = deltaE_ciede2000(lab1, lab2)
            if diff > 0 and diff < min_diff:  # 排除完全相同的颜色
                min_diff = diff
                closest_pair = (i, j)

    return closest_pair, min_diff

while len(used_superpixels) < len(valid_superpixels_lab) - 1:
    closest_pair, min_diff = find_closest_color(valid_superpixels_lab, used_superpixels)
    if closest_pair is not None:
        i, j = closest_pair
        closest_color_diffs.append(min_diff)
        used_superpixels.add(i)
        used_superpixels.add(j)

if len(closest_color_diffs) > 0:
    mean_color_diff = np.mean(closest_color_diffs)
    print(f"Mean CIEDE2000 color difference between the closest superpixels: {mean_color_diff}")
else:
    print("No significant color differences found between superpixels.")

plt.figure(figsize=(10, 10))
plt.imshow(label2rgb(new_segments, image_rgb, kind='avg', bg_label=0, alpha=0.5))
plt.axis('off')
plt.show()
