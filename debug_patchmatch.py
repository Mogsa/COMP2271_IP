import cv2
import numpy as np
import os
from patchmatch_inpainting import Inpainter, PATCHMATCH_AVAILABLE

# Create a directory for debug images
debug_dir = "debug_patchmatch_output"
os.makedirs(debug_dir, exist_ok=True)

# Choose an image to process - try a different image with more clearly visible artifacts
image_path = "/Users/morgan/Documents/GitHub/COMP2271_IP/driving_images/im025-snow.jpg"
print(f"Processing {image_path}")

# Read the image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read image at {image_path}")
    exit(1)

# Save original image for comparison
original_path = os.path.join(debug_dir, "1_original.jpg")
cv2.imwrite(original_path, image)
print(f"Original image saved to {original_path}")

# Print PatchMatch availability
print(f"PatchMatch library available: {PATCHMATCH_AVAILABLE}")

# Create inpainter instance
patchmatch_inpainter = Inpainter(use_patchmatch=True, patch_size=5)
opencv_inpainter = Inpainter(use_patchmatch=False)

# Step 1: Automatically detect artifacts
print("\nStep 1: Detecting artifacts...")
mask = patchmatch_inpainter.detect_artifacts(image)

# Save the mask
mask_path = os.path.join(debug_dir, "2_detected_mask.jpg")
cv2.imwrite(mask_path, mask)
print(f"Generated mask saved to {mask_path}")

# Step 2: Visualize mask on the original image
mask_vis = image.copy()
mask_vis[mask > 0] = [0, 0, 255]  # Mark mask in red
mask_vis_path = os.path.join(debug_dir, "3_mask_visualization.jpg")
cv2.imwrite(mask_vis_path, mask_vis)
print(f"Mask visualization saved to {mask_vis_path}")

# Step 3: We'll use the automatically detected mask (if it's not empty)
# Check if the detected mask has any non-zero pixels
if np.any(mask > 0):
    print("\nStep 3: Using automatically detected mask for inpainting...")
    inpaint_mask = mask
else:
    # If no mask was detected, force detection with manual_mask=True parameter
    print("\nStep 3: No mask detected automatically, forcing detection...")
    inpaint_mask = patchmatch_inpainter.detect_artifacts(image, manual_mask=True)
    mask_path = os.path.join(debug_dir, "4_forced_mask.jpg")
    cv2.imwrite(mask_path, inpaint_mask)
    
    # Visualize forced mask
    forced_vis = image.copy()
    forced_vis[inpaint_mask > 0] = [0, 0, 255]
    forced_vis_path = os.path.join(debug_dir, "5_forced_visualization.jpg")
    cv2.imwrite(forced_vis_path, forced_vis)
    print(f"Forced mask visualization saved to {forced_vis_path}")

# Step 4: Inpaint with different methods
print("\nStep 4: Applying different inpainting methods...")

# OpenCV Inpainting - Telea algorithm
opencv_telea = cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_TELEA)
opencv_telea_path = os.path.join(debug_dir, "6_opencv_telea.jpg")
cv2.imwrite(opencv_telea_path, opencv_telea)
print(f"OpenCV Telea inpainting saved to {opencv_telea_path}")

# OpenCV Inpainting - Navier-Stokes algorithm
opencv_ns = cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_NS)
opencv_ns_path = os.path.join(debug_dir, "7_opencv_ns.jpg")
cv2.imwrite(opencv_ns_path, opencv_ns)
print(f"OpenCV Navier-Stokes inpainting saved to {opencv_ns_path}")

# OpenCV auto select method
opencv_result = opencv_inpainter.inpaint_image(image, inpaint_mask)
opencv_auto_path = os.path.join(debug_dir, "8_opencv_auto.jpg")
cv2.imwrite(opencv_auto_path, opencv_result)
print(f"OpenCV auto-select inpainting saved to {opencv_auto_path}")

# PatchMatch inpainting (if available)
if PATCHMATCH_AVAILABLE:
    patchmatch_result = patchmatch_inpainter.inpaint_image(image, inpaint_mask)
    patchmatch_path = os.path.join(debug_dir, "9_patchmatch.jpg")
    cv2.imwrite(patchmatch_path, patchmatch_result)
    print(f"PatchMatch inpainting saved to {patchmatch_path}")

# Step 5: Create comparison image
print("\nStep 5: Creating comparison image...")

def create_comparison_image(images, titles, height=250):
    # Resize images to a common height
    resized_images = []
    
    for img in images:
        h, w = img.shape[:2]
        aspect = w / h
        new_width = int(height * aspect)
        resized = cv2.resize(img, (new_width, height))
        resized_images.append(resized)
    
    # Split into rows if too many images
    max_width = 1500
    images_per_row = []
    current_width = 0
    current_row = []
    current_row_titles = []
    
    for i, img in enumerate(resized_images):
        if current_width + img.shape[1] > max_width and current_row:
            images_per_row.append((current_row, current_row_titles))
            current_row = []
            current_row_titles = []
            current_width = 0
        
        current_row.append(img)
        current_row_titles.append(titles[i])
        current_width += img.shape[1]
    
    if current_row:
        images_per_row.append((current_row, current_row_titles))
    
    # Create rows
    rows = []
    for row_images, row_titles in images_per_row:
        # Concatenate images horizontally
        row = np.hstack(row_images)
        
        # Add titles
        row_with_titles = np.zeros((height + 30, row.shape[1], 3), dtype=np.uint8)
        row_with_titles[30:, :, :] = row
        
        # Add title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        font_thickness = 1
        
        x_offset = 0
        for i, title in enumerate(row_titles):
            img_width = row_images[i].shape[1]
            text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
            text_x = x_offset + (img_width - text_size[0]) // 2
            cv2.putText(row_with_titles, title, (text_x, 20), font, font_scale, font_color, font_thickness)
            x_offset += img_width
        
        rows.append(row_with_titles)
    
    # Concatenate rows vertically
    result = np.vstack(rows)
    return result

# Prepare images and titles for comparison
comparison_images = [image, mask_vis, opencv_telea, opencv_ns, opencv_result]
comparison_titles = ["Original", "Detected Mask", "OpenCV-Telea", "OpenCV-NS", "OpenCV-Auto"]

# Add PatchMatch result if available
if PATCHMATCH_AVAILABLE:
    comparison_images.append(patchmatch_result)
    comparison_titles.append("PatchMatch")

# Create comparison image
comparison = create_comparison_image(comparison_images, comparison_titles)
comparison_path = os.path.join(debug_dir, "10_comparison.jpg")
cv2.imwrite(comparison_path, comparison)
print(f"Comparison image saved to {comparison_path}")

print("\nDone debugging PatchMatch inpainting!")