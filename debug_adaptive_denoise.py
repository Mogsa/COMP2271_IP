import cv2
import numpy as np
import os
from advanced_denoise import AdvancedDenoiser

def visualize_masks(image_path, output_dir):
    """
    Visualize the masks used in adaptive edge-aware denoising.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Initialize denoiser with debug mode
    denoiser = AdvancedDenoiser(debug=True)
    
    # 1. Non-black mask
    non_black = np.logical_or.reduce((
        image[:,:,0] > 5,
        image[:,:,1] > 5,
        image[:,:,2] > 5
    ))
    non_black_vis = non_black.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, f"{filename}_non_black_mask.png"), non_black_vis)
    
    # 2. Apply pre-denoising for better edge detection
    pre_denoised = cv2.bilateralFilter(image, 5, 25, 25)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_pre_denoised.jpg"), pre_denoised)
    
    # 3. Convert to grayscale and apply Gaussian blur before edge detection
    gray = cv2.cvtColor(pre_denoised, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_blurred_gray.jpg"), blurred)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(blurred)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_enhanced_gray.jpg"), enhanced_gray)
    
    # 1. Gradient-based approach (Sobel)
    grad_x = cv2.Sobel(enhanced_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(enhanced_gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_sobel_gradient.jpg"), sobel)
    
    # Use much higher threshold for better noise resistance
    _, sobel_thresh = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY) # Increased from 70 to 120
    cv2.imwrite(os.path.join(output_dir, f"{filename}_sobel_threshold.jpg"), sobel_thresh)
    
    # Clean up the gradient image
    kernel_clean = np.ones((3, 3), np.uint8)
    sobel_cleaned = cv2.morphologyEx(sobel_thresh, cv2.MORPH_CLOSE, kernel_clean)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_sobel_cleaned.jpg"), sobel_cleaned)
    
    # 2. Canny-based edge detection with much higher thresholds
    edges1 = cv2.Canny(sobel, 150, 300)  # Significantly increased threshold for very strong edges only
    edges2 = cv2.Canny(enhanced_gray, 100, 200)  # Increased threshold for medium edges
    
    # 3. Combine only Canny results (excluding sobel_cleaned which is too sensitive)
    combined_edges_all = cv2.bitwise_or(edges1, edges2)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_combined_all_edges.jpg"), combined_edges_all)
    
    # Save individual edge detections
    cv2.imwrite(os.path.join(output_dir, f"{filename}_edges1.png"), edges1)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_edges2.png"), edges2)
    
    # Use the combined edges from all methods
    edges = combined_edges_all
    
    # Apply morphological filtering to clean and connect edges
    kernel_connect = np.ones((3, 3), np.uint8)
    connected_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_connect)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_connected_edges.png"), connected_edges)
    
    # Filter out small regions
    contours, _ = cv2.findContours(connected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_edges = np.zeros_like(connected_edges)
    
    # Create visualization with areas colored by size
    contour_vis = np.zeros((connected_edges.shape[0], connected_edges.shape[1], 3), dtype=np.uint8)
    
    # Count contours by size for debugging
    small_count = 0
    medium_count = 0
    large_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Color code by area for visualization - updated thresholds
        if area > 100:
            color = (0, 255, 0)  # Green for large areas
            large_count += 1
        elif area > 50:  # Increased from 25 to 50
            color = (255, 255, 0)  # Yellow for medium areas
            medium_count += 1
        else:
            color = (0, 0, 255)  # Red for small areas (noise)
            small_count += 1
            
        cv2.drawContours(contour_vis, [contour], 0, color, -1)
        
        # Only keep contours above threshold for actual processing
        if area > 50:  # Increased from 25 to 50 to match the main algorithm
            cv2.drawContours(filtered_edges, [contour], 0, 255, -1)
    
    # Add contour count data to the visualization
    font = cv2.FONT_HERSHEY_SIMPLEX
    contour_count_img = contour_vis.copy()
    cv2.putText(contour_count_img, f'Large: {large_count}', (20, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(contour_count_img, f'Medium: {medium_count}', (20, 60), font, 0.7, (255, 255, 0), 2)
    cv2.putText(contour_count_img, f'Small: {small_count}', (20, 90), font, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_contour_counts.jpg"), contour_count_img)
    
    cv2.imwrite(os.path.join(output_dir, f"{filename}_contour_areas.jpg"), contour_vis)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_filtered_edges.png"), filtered_edges)
    
    # For comparison, save original combined edges too
    cv2.imwrite(os.path.join(output_dir, f"{filename}_combined_edges.png"), edges)
    
    # 5. Dilated edges
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(filtered_edges, kernel, iterations=2)
    edge_mask = dilated_edges > 0
    edge_mask_vis = edge_mask.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, f"{filename}_dilated_edge_mask.png"), edge_mask_vis)
    
    # 6. Brightness masks
    ycrcb_bright = cv2.cvtColor(pre_denoised, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_bright[:,:,0]
    cv2.imwrite(os.path.join(output_dir, f"{filename}_y_channel.png"), y_channel)
    
    # Create brightness masks with adjusted thresholds
    bright_mask = y_channel > 170
    mid_mask = (y_channel > 110) & (y_channel <= 170)
    dark_mask = y_channel <= 110
    
    # Save brightness masks
    bright_mask_vis = bright_mask.astype(np.uint8) * 255
    mid_mask_vis = mid_mask.astype(np.uint8) * 255
    dark_mask_vis = dark_mask.astype(np.uint8) * 255
    
    cv2.imwrite(os.path.join(output_dir, f"{filename}_bright_mask.png"), bright_mask_vis)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_mid_mask.png"), mid_mask_vis)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_dark_mask.png"), dark_mask_vis)
    
    # 7. Final result after denoising
    denoised = denoiser.adaptive_edge_aware_denoising(image)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_denoised.jpg"), denoised)
    
    # Create individual mask visualizations with original image overlay
    # This helps to see exactly what parts of the image are being processed differently
    
    # Create RGB masks for visualization
    edge_rgb = np.zeros_like(image)
    edge_rgb[edge_mask] = [0, 255, 0]  # Green for edges
    
    bright_rgb = np.zeros_like(image)
    bright_rgb[bright_mask] = [255, 0, 0]  # Red for bright areas
    
    mid_rgb = np.zeros_like(image)
    mid_rgb[mid_mask] = [0, 0, 255]  # Blue for mid areas
    
    dark_rgb = np.zeros_like(image)
    dark_rgb[dark_mask] = [255, 255, 0]  # Yellow for dark areas
    
    # Overlay masks on original image with transparency
    alpha = 0.3
    edge_overlay = cv2.addWeighted(image, 1, edge_rgb, alpha, 0)
    bright_overlay = cv2.addWeighted(image, 1, bright_rgb, alpha, 0)
    mid_overlay = cv2.addWeighted(image, 1, mid_rgb, alpha, 0)
    dark_overlay = cv2.addWeighted(image, 1, dark_rgb, alpha, 0)
    
    # Save individual overlays
    cv2.imwrite(os.path.join(output_dir, f"{filename}_edge_overlay.jpg"), edge_overlay)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_bright_overlay.jpg"), bright_overlay)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_mid_overlay.jpg"), mid_overlay)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_dark_overlay.jpg"), dark_overlay)
    
    # Save the brightness analysis image
    y_display = cv2.normalize(y_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    y_heatmap = cv2.applyColorMap(y_display, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_brightness_heatmap.jpg"), y_heatmap)
    
    # Create bilateral filter visualizations for each brightness level
    # For bright areas - extremely aggressive bilateral filter
    d_bright = 15  # Larger diameter
    sigma_color_bright = 150  # Much stronger smoothing
    sigma_space_bright = 150  # Much more broad effect
    bright_result = cv2.bilateralFilter(image, d_bright, sigma_color_bright, sigma_space_bright)
    
    # For mid-brightness areas - moderately aggressive bilateral filter
    d_mid = 11  # Medium diameter
    sigma_color_mid = 100  # Stronger smoothing
    sigma_space_mid = 100
    mid_result = cv2.bilateralFilter(image, d_mid, sigma_color_mid, sigma_space_mid)
    
    # For dark areas - gentle bilateral filter (keep mostly unchanged)
    d_dark = 7  # Smaller diameter
    sigma_color_dark = 60  # Gentle smoothing
    sigma_space_dark = 60
    dark_result = cv2.bilateralFilter(image, d_dark, sigma_color_dark, sigma_space_dark)
    
    # Save bilateral results for each brightness level
    cv2.imwrite(os.path.join(output_dir, f"{filename}_bilateral_bright.jpg"), bright_result)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_bilateral_mid.jpg"), mid_result)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_bilateral_dark.jpg"), dark_result)
    
    # NLM strengths for different brightness levels
    base_h_y = max(15, int(25 * (1.0 - 0.85 * 0.2)))
    h_bright = base_h_y * 2.0  # Extremely strong for bright areas
    h_mid = base_h_y * 1.5     # Stronger for mid-brightness
    h_dark = base_h_y * 0.8    # Gentle for dark areas
    h_c = max(40, int(60 * (1.0 - 0.85 * 0.2)))  # Chromatic channel strength
    
    # Create text file with all algorithm parameters
    with open(os.path.join(output_dir, f"{filename}_parameters.txt"), 'w') as f:
        f.write(f"Image: {image_path}\n\n")
        f.write("Brightness Distribution:\n")
        f.write(f"Bright pixels (Y > 180): {np.sum(bright_mask)/np.sum(non_black)*100:.1f}%\n")
        f.write(f"Mid pixels (120 < Y <= 180): {np.sum(mid_mask)/np.sum(non_black)*100:.1f}%\n")
        f.write(f"Dark pixels (Y <= 120): {np.sum(dark_mask)/np.sum(non_black)*100:.1f}%\n\n")
        
        f.write("Bilateral Filter Parameters:\n")
        f.write(f"Bright areas: d={d_bright}, sigma_color={sigma_color_bright}, sigma_space={sigma_space_bright}\n")
        f.write(f"Mid areas: d={d_mid}, sigma_color={sigma_color_mid}, sigma_space={sigma_space_mid}\n")
        f.write(f"Dark areas: d={d_dark}, sigma_color={sigma_color_dark}, sigma_space={sigma_space_dark}\n\n")
        
        f.write("NLM Denoising Parameters:\n")
        f.write(f"Base h: {base_h_y}\n")
        f.write(f"Bright areas h: {h_bright}\n")
        f.write(f"Mid areas h: {h_mid}\n")
        f.write(f"Dark areas h: {h_dark}\n")
        f.write(f"Chromatic channels h: {h_c}\n\n")
        
        f.write("Blending Weights (Bilateral/NLM):\n")
        f.write(f"Bright areas: 70% bilateral, 30% NLM\n")
        f.write(f"Mid areas: 50% bilateral, 50% NLM\n")
        f.write(f"Dark areas: 30% bilateral, 70% NLM\n")
    
    print(f"Saved visualizations to {output_dir}")
    return denoised

def main():
    """
    Process a sample image to visualize adaptive denoising masks.
    """
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Sample image path - choose a snow and rain image to compare
    snow_image = os.path.join(current_dir, "driving_images", "im010-snow.jpg")
    rain_image = os.path.join(current_dir, "driving_images", "im060-rain.jpg")
    
    # Output directory
    output_dir = os.path.join(current_dir, "debug_outputs")
    
    # Process images
    print("Processing snow image...")
    snow_denoised = visualize_masks(snow_image, output_dir)
    
    print("Processing rain image...")
    rain_denoised = visualize_masks(rain_image, output_dir)
    
    # Save before/after comparison
    if snow_denoised is not None and rain_denoised is not None:
        snow_orig = cv2.imread(snow_image)
        rain_orig = cv2.imread(rain_image)
        
        # Create comparison images
        snow_comparison = np.hstack((snow_orig, snow_denoised))
        rain_comparison = np.hstack((rain_orig, rain_denoised))
        
        cv2.imwrite(os.path.join(output_dir, "snow_comparison.jpg"), snow_comparison)
        cv2.imwrite(os.path.join(output_dir, "rain_comparison.jpg"), rain_comparison)
        
        print("Saved comparison images")

if __name__ == "__main__":
    main()