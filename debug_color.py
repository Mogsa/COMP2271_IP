import cv2
import os
import numpy as np
from color import EnhancedColorCorrector

# Create a directory for debug images
debug_dir = "debug_color_output"
os.makedirs(debug_dir, exist_ok=True)

# Initialize the color corrector with debug mode on
corrector = EnhancedColorCorrector(debug=True)

# Process a single image for debugging
image_path = "/Users/morgan/Documents/GitHub/COMP2271_IP/driving_images/im001-snow.jpg"
print(f"Processing {image_path}")

# Read the image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read image at {image_path}")
    exit(1)

# Save original image for comparison
original_path = os.path.join(debug_dir, "original.jpg")
cv2.imwrite(original_path, image)

# Analyze the image
metrics = corrector.analyze_image(image)
print("Image Analysis Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")

# Step 1: White Balance
white_balanced = corrector.auto_white_balance(image)
wb_path = os.path.join(debug_dir, "1_white_balanced.jpg")
cv2.imwrite(wb_path, white_balanced)
print(f"White balanced image saved to {wb_path}")

# Step 2: Brightness Correction
brightness_corrected = corrector.correct_brightness(white_balanced, metrics)
bright_path = os.path.join(debug_dir, "2_brightness_corrected.jpg")
cv2.imwrite(bright_path, brightness_corrected)
print(f"Brightness corrected image saved to {bright_path}")

# Step 3: Contrast Enhancement
contrast_enhanced = corrector.enhance_contrast(brightness_corrected, metrics)
contrast_path = os.path.join(debug_dir, "3_contrast_enhanced.jpg")
cv2.imwrite(contrast_path, contrast_enhanced)
print(f"Contrast enhanced image saved to {contrast_path}")

# Final result using full auto_correct function
final = corrector.auto_correct(image)
final_path = os.path.join(debug_dir, "4_final_result.jpg")
cv2.imwrite(final_path, final)
print(f"Final result saved to {final_path}")

# Create a comparison image with all steps side by side
def create_comparison_image(images, titles):
    # Resize images to a common height
    height = 300
    resized_images = []
    
    for img in images:
        h, w = img.shape[:2]
        aspect = w / h
        new_width = int(height * aspect)
        resized = cv2.resize(img, (new_width, height))
        resized_images.append(resized)
    
    # Concatenate images horizontally
    result = np.hstack(resized_images)
    
    # Add titles
    result_with_titles = np.zeros((height + 30, result.shape[1], 3), dtype=np.uint8)
    result_with_titles[30:, :, :] = result
    
    # Add title text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    font_thickness = 1
    
    x_offset = 0
    for i, title in enumerate(titles):
        img_width = resized_images[i].shape[1]
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        text_x = x_offset + (img_width - text_size[0]) // 2
        cv2.putText(result_with_titles, title, (text_x, 20), font, font_scale, font_color, font_thickness)
        x_offset += img_width
    
    return result_with_titles

# Create comparison image
comparison_images = [image, white_balanced, brightness_corrected, contrast_enhanced, final]
comparison_titles = ["Original", "White Balance", "Brightness", "Contrast", "Final"]
comparison = create_comparison_image(comparison_images, comparison_titles)

comparison_path = os.path.join(debug_dir, "comparison.jpg")
cv2.imwrite(comparison_path, comparison)
print(f"Comparison image saved to {comparison_path}")

print("Done debugging color correction!")