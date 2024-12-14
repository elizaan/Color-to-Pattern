import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb, rgb2lab, deltaE_ciede2000
from skimage.filters import sobel
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage import graph
import matplotlib.patches as mpatches
import os
from pathlib import Path

input_dir = './ChartDete/output/area_extracted/'
output_dir = './output/patterned_bars'

os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def create_pattern(pattern_type, width, height):
    """Create different pattern textures"""
    pattern = np.zeros((height, width))
    
    if pattern_type == "stripes":
        for i in range(height):
            pattern[i, :] = i % 20 < 10
    elif pattern_type == "dots":
        for i in range(0, height, 10):
            for j in range(0, width, 10):
                pattern[i:i+5, j:j+5] = 1
    elif pattern_type == "crosshatch":
        for i in range(height):
            for j in range(width):
                pattern[i, j] = (i % 20 < 10) ^ (j % 20 < 10)
    elif pattern_type == "grid":
        pattern[::10, :] = 1
        pattern[:, ::10] = 1
                
    return pattern

# [Previous imports and functions remain the same until apply_pattern_to_segments]

def apply_pattern_to_segments(image_rgb, segments, patterns=["stripes", "dots", "crosshatch", "grid"]):
    height, width = image_rgb.shape[:2]
    result = image_rgb.copy()
    
    # Create a mapping of segment labels to patterns
    unique_segments = np.unique(segments)
    segment_patterns = {}
    
    for idx, seg_val in enumerate(unique_segments):
        if seg_val == 0:  # Skip background
            continue
        pattern_type = patterns[idx % len(patterns)]
        segment_patterns[seg_val] = pattern_type
    
    # Apply patterns to each segment
    for seg_val, pattern_type in segment_patterns.items():
        mask = segments == seg_val
        if not np.any(mask):
            continue
            
        # Get region boundaries
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
            
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        region_height = y_max - y_min + 1
        region_width = x_max - x_min + 1
        
        # Create and scale pattern
        pattern = create_pattern(pattern_type, region_width, region_height)
        
        # Apply pattern to the region
        for channel in range(3):
            region = result[y_min:y_max+1, x_min:x_max+1, channel]
            region_mask = mask[y_min:y_max+1, x_min:x_max+1]
            pattern_values = 0.7 + 0.3 * pattern
            region[region_mask] = (region[region_mask] * pattern_values[region_mask]).astype(np.uint8)
    
    return result, segment_patterns

# [Rest of the code remains the same]

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
            if diff > 0 and diff < min_diff:
                min_diff = diff
                closest_pair = (i, j)

    return closest_pair, min_diff

# Main execution
if __name__ == "__main__":
    for idx, image_file in enumerate(image_files, 1):  # start counting from 1
        try:
            # Construct input path
            input_path = os.path.join(input_dir, image_file)
            
            # Load and preprocess image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {input_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lab_image = rgb2lab(image_rgb)
            
            # Edge detection and blurring
            edges = sobel(gray_image)
            blurred_image = cv2.GaussianBlur(image_rgb, (5, 5), 0)
            
            # Segment image
            n_segments = 150
            compactness = 50
            segments = slic(blurred_image, n_segments=n_segments, compactness=compactness, sigma=1, start_label=1)
            
            # Merge superpixels
            rag = graph.rag_mean_color(image_rgb, segments)
            threshold = 30
            new_segments = graph.cut_threshold(segments, rag, threshold)
            
            # Create visualizations with patterns
            patterned_image, pattern_mapping = apply_pattern_to_segments(image_rgb, new_segments)
            
            # Create figure for this image
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(image_rgb)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(mark_boundaries(image_rgb, new_segments))
            plt.title('Segmentation Boundaries')
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(patterned_image)
            plt.title('Segmentation with Patterns')
            plt.axis('off')
            
            # Add legend for patterns
            pattern_labels = list(set(pattern_mapping.values()))
            legend_elements = [mpatches.Patch(facecolor='gray', 
                                            label=pattern,
                                            hatch='/' if pattern == 'stripes' else 
                                                  'o' if pattern == 'dots' else
                                                  'x' if pattern == 'crosshatch' else
                                                  '+')
                              for pattern in pattern_labels]
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save the figure
            output_path = os.path.join(output_dir, f'{idx}_{idx}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()  # Close the figure to free memory
            
            # Process superpixels
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
            
            print(f"Processing image {idx}: {image_file}")
            print(f"Number of valid superpixels: {valid_superpixel_count}")
            
            # Find closest colors
            used_superpixels = set()
            closest_color_diffs = []
            
            while len(used_superpixels) < len(valid_superpixels_lab) - 1:
                closest_pair, min_diff = find_closest_color(valid_superpixels_lab, used_superpixels)
                if closest_pair is not None:
                    i, j = closest_pair
                    closest_color_diffs.append(min_diff)
                    used_superpixels.add(i)
                    used_superpixels.add(j)
            
            if len(closest_color_diffs) > 0:
                mean_color_diff = np.mean(closest_color_diffs)
                print(f"Mean CIEDE2000 color difference: {mean_color_diff}\n")
            else:
                print("No significant color differences found\n")
                
        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}\n")
            continue

    print("Processing completed!")