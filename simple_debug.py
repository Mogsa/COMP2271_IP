#!/usr/bin/env python3
"""
Simple script to debug the image processing pipeline steps.
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time

# Add the project directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from perspective import PerspectiveCorrector
from color import EnhancedColorCorrector
from denoise import ImageDenoiser
from patchmatch_inpainting import Inpainter

def process_and_debug_image(image_path, output_dir):
    """
    Process a single image with detailed debug output for each step.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save debug output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image filename
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    print(f"Processing image: {filename}")
    
    # Read original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Save original image
    original_path = os.path.join(output_dir, f"00_original{ext}")
    cv2.imwrite(original_path, original_image)
    
    # Initialize processors
    perspective_corrector = PerspectiveCorrector(debug=True)
    color_corrector = EnhancedColorCorrector(debug=True)
    denoiser = ImageDenoiser(debug=True)
    inpainter = Inpainter(use_patchmatch=True, patch_size=5)
    
    # Step 1: Perspective Correction
    print("\n=== Step 1: Perspective Correction ===")
    start_time = time.time()
    perspective_image = perspective_corrector.process_image(image_path)
    if perspective_image is None:
        print(f"Error: Perspective correction failed for {image_path}")
        return
    
    perspective_time = time.time() - start_time
    print(f"Perspective correction time: {perspective_time:.2f}s")
    
    # Save perspective corrected image
    perspective_path = os.path.join(output_dir, f"01_perspective{ext}")
    cv2.imwrite(perspective_path, perspective_image)
    
    # Create a before/after comparison
    h1, w1 = original_image.shape[:2]
    h2, w2 = perspective_image.shape[:2]
    
    # Resize to same height if needed
    if h1 != h2:
        scale = h2 / h1
        original_resized = cv2.resize(original_image, (int(w1 * scale), h2))
    else:
        original_resized = original_image
    
    # Create comparison image
    comparison = np.hstack((original_resized, perspective_image))
    comparison_path = os.path.join(output_dir, f"01a_perspective_comparison{ext}")
    cv2.imwrite(comparison_path, comparison)
    
    # Step 2: Color Correction
    print("\n=== Step 2: Color Correction ===")
    start_time = time.time()
    color_image = color_corrector.auto_correct(perspective_image)
    color_time = time.time() - start_time
    print(f"Color correction time: {color_time:.2f}s")
    
    # Save color corrected image
    color_path = os.path.join(output_dir, f"02_color{ext}")
    cv2.imwrite(color_path, color_image)
    
    # Create a before/after comparison for color
    comparison = np.hstack((perspective_image, color_image))
    comparison_path = os.path.join(output_dir, f"02a_color_comparison{ext}")
    cv2.imwrite(comparison_path, comparison)
    
    # Step 3: Different Denoising Methods
    print("\n=== Step 3: Denoising Methods ===")
    
    # Normal denoising (detail preserving)
    start_time = time.time()
    denoised_normal = denoiser.detail_preserving_denoising(color_image)
    normal_time = time.time() - start_time
    print(f"Normal denoising time: {normal_time:.2f}s")
    
    # Bilateral filter
    start_time = time.time()
    denoised_bilateral = denoiser.bilateral_filter_denoising(
        color_image, d=3, sigma_color=35, sigma_space=35
    )
    bilateral_time = time.time() - start_time
    print(f"Bilateral filter time: {bilateral_time:.2f}s")
    
    # Non-local means
    start_time = time.time()
    denoised_nlm = denoiser.non_local_means_denoising(
        color_image, h_luminance=5, h_color=5, template_size=3, search_size=9
    )
    nlm_time = time.time() - start_time
    print(f"Non-local means time: {nlm_time:.2f}s")
    
    # Save denoised images
    cv2.imwrite(os.path.join(output_dir, f"03a_denoised_normal{ext}"), denoised_normal)
    cv2.imwrite(os.path.join(output_dir, f"03b_denoised_bilateral{ext}"), denoised_bilateral)
    cv2.imwrite(os.path.join(output_dir, f"03c_denoised_nlm{ext}"), denoised_nlm)
    
    # Create a comparison of all denoising methods
    # Resize to same height if needed
    h, w = color_image.shape[:2]
    target_height = 300
    
    scale = target_height / h
    original_small = cv2.resize(color_image, (int(w * scale), target_height))
    normal_small = cv2.resize(denoised_normal, (int(w * scale), target_height))
    bilateral_small = cv2.resize(denoised_bilateral, (int(w * scale), target_height))
    nlm_small = cv2.resize(denoised_nlm, (int(w * scale), target_height))
    
    # Create labels for each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 255, 255)
    thickness = 1
    
    label_height = 30
    
    # Add labels to the top of each image
    original_labeled = np.zeros((target_height + label_height, original_small.shape[1], 3), dtype=np.uint8)
    original_labeled[label_height:, :, :] = original_small
    cv2.putText(original_labeled, "Original", (10, 20), font, fontScale, color, thickness)
    
    normal_labeled = np.zeros((target_height + label_height, normal_small.shape[1], 3), dtype=np.uint8)
    normal_labeled[label_height:, :, :] = normal_small
    cv2.putText(normal_labeled, "Detail-Preserving", (10, 20), font, fontScale, color, thickness)
    
    bilateral_labeled = np.zeros((target_height + label_height, bilateral_small.shape[1], 3), dtype=np.uint8)
    bilateral_labeled[label_height:, :, :] = bilateral_small
    cv2.putText(bilateral_labeled, "Bilateral", (10, 20), font, fontScale, color, thickness)
    
    nlm_labeled = np.zeros((target_height + label_height, nlm_small.shape[1], 3), dtype=np.uint8)
    nlm_labeled[label_height:, :, :] = nlm_small
    cv2.putText(nlm_labeled, "Non-Local Means", (10, 20), font, fontScale, color, thickness)
    
    # Combine horizontally
    row1 = np.hstack((original_labeled, normal_labeled))
    row2 = np.hstack((bilateral_labeled, nlm_labeled))
    denoising_comparison = np.vstack((row1, row2))
    
    # Save the comparison
    cv2.imwrite(os.path.join(output_dir, f"03d_denoising_comparison.png"), denoising_comparison)
    
    # Use normal denoising for the pipeline
    denoised_image = denoised_normal
    
    # Step 4: Inpainting for artifact removal
    print("\n=== Step 4: Inpainting ===")
    start_time = time.time()
    
    # Generate mask for inpainting (detect artifacts)
    mask = inpainter.detect_artifacts(denoised_image, threshold=15)
    mask_path = os.path.join(output_dir, f"04a_inpaint_mask.png")
    cv2.imwrite(mask_path, mask)
    
    # Create a mask visualization
    mask_vis = denoised_image.copy()
    mask_vis[mask > 0] = [0, 0, 255]  # Mark mask in red
    mask_vis_path = os.path.join(output_dir, f"04b_mask_visualization{ext}")
    cv2.imwrite(mask_vis_path, mask_vis)
    
    # Apply inpainting
    inpainted_image = inpainter.inpaint_image(denoised_image, mask)
    inpainted_path = os.path.join(output_dir, f"04c_inpainted{ext}")
    cv2.imwrite(inpainted_path, inpainted_image)
    
    # Create a before/after comparison for inpainting
    comparison = np.hstack((denoised_image, inpainted_image))
    comparison_path = os.path.join(output_dir, f"04d_inpainting_comparison{ext}")
    cv2.imwrite(comparison_path, comparison)
    
    inpainting_time = time.time() - start_time
    print(f"Inpainting time: {inpainting_time:.2f}s")
    
    # Final result
    final_path = os.path.join(output_dir, f"05_final_result{ext}")
    cv2.imwrite(final_path, inpainted_image)
    
    # Create a full pipeline comparison
    # Original -> Perspective -> Color -> Denoised -> Inpainted
    h, w = original_image.shape[:2]
    # Resize all images to the same height
    target_height = 200
    
    scale = target_height / h
    original_small = cv2.resize(original_image, (int(w * scale), target_height))
    
    h, w = perspective_image.shape[:2]
    scale = target_height / h
    perspective_small = cv2.resize(perspective_image, (int(w * scale), target_height))
    
    h, w = color_image.shape[:2]
    scale = target_height / h
    color_small = cv2.resize(color_image, (int(w * scale), target_height))
    
    h, w = denoised_image.shape[:2]
    scale = target_height / h
    denoised_small = cv2.resize(denoised_image, (int(w * scale), target_height))
    
    h, w = inpainted_image.shape[:2]
    scale = target_height / h
    inpainted_small = cv2.resize(inpainted_image, (int(w * scale), target_height))
    
    # Create labels for each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 255, 255)
    thickness = 1
    
    label_height = 30
    
    # Add labels to the top of each image
    original_labeled = np.zeros((target_height + label_height, original_small.shape[1], 3), dtype=np.uint8)
    original_labeled[label_height:, :, :] = original_small
    cv2.putText(original_labeled, "Original", (10, 20), font, fontScale, color, thickness)
    
    perspective_labeled = np.zeros((target_height + label_height, perspective_small.shape[1], 3), dtype=np.uint8)
    perspective_labeled[label_height:, :, :] = perspective_small
    cv2.putText(perspective_labeled, "Perspective", (10, 20), font, fontScale, color, thickness)
    
    color_labeled = np.zeros((target_height + label_height, color_small.shape[1], 3), dtype=np.uint8)
    color_labeled[label_height:, :, :] = color_small
    cv2.putText(color_labeled, "Color", (10, 20), font, fontScale, color, thickness)
    
    denoised_labeled = np.zeros((target_height + label_height, denoised_small.shape[1], 3), dtype=np.uint8)
    denoised_labeled[label_height:, :, :] = denoised_small
    cv2.putText(denoised_labeled, "Denoised", (10, 20), font, fontScale, color, thickness)
    
    inpainted_labeled = np.zeros((target_height + label_height, inpainted_small.shape[1], 3), dtype=np.uint8)
    inpainted_labeled[label_height:, :, :] = inpainted_small
    cv2.putText(inpainted_labeled, "Inpainted", (10, 20), font, fontScale, color, thickness)
    
    # Combine horizontally
    pipeline = np.hstack((original_labeled, perspective_labeled, color_labeled, denoised_labeled, inpainted_labeled))
    
    # Save the pipeline visualization
    pipeline_path = os.path.join(output_dir, f"06_full_pipeline.png")
    cv2.imwrite(pipeline_path, pipeline)
    
    # Create a detailed report
    report_path = os.path.join(output_dir, "processing_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Image Processing Report for {filename}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Processing Times:\n")
        f.write(f"- Perspective Correction: {perspective_time:.2f}s\n")
        f.write(f"- Color Enhancement: {color_time:.2f}s\n")
        f.write(f"- Detail Preserving Denoising: {normal_time:.2f}s\n")
        f.write(f"- Bilateral Filter Denoising: {bilateral_time:.2f}s\n")
        f.write(f"- Non-Local Means Denoising: {nlm_time:.2f}s\n")
        f.write(f"- Inpainting: {inpainting_time:.2f}s\n")
        f.write(f"- Total Pipeline Time: {perspective_time + color_time + normal_time + inpainting_time:.2f}s\n\n")
        
        f.write("Denoising Statistics:\n")
        f.write(f"- Estimated Noise Level: {denoiser.estimate_noise_level(color_image):.2f}\n\n")
        
        f.write("Inpainting Statistics:\n")
        f.write(f"- Inpainted Pixels: {cv2.countNonZero(mask)}\n")
        f.write(f"- Percentage of Image: {cv2.countNonZero(mask)/(mask.shape[0]*mask.shape[1])*100:.2f}%\n\n")
        
        f.write("File Paths:\n")
        f.write(f"- Original Image: {original_path}\n")
        f.write(f"- Perspective Corrected: {perspective_path}\n")
        f.write(f"- Color Enhanced: {color_path}\n")
        f.write(f"- Detail Preserving Denoised: {os.path.join(output_dir, f'03a_denoised_normal{ext}')}\n")
        f.write(f"- Bilateral Filter Denoised: {os.path.join(output_dir, f'03b_denoised_bilateral{ext}')}\n")
        f.write(f"- Non-Local Means Denoised: {os.path.join(output_dir, f'03c_denoised_nlm{ext}')}\n")
        f.write(f"- Inpainting Mask: {mask_path}\n")
        f.write(f"- Final Result: {final_path}\n")
    
    print(f"\nProcessing completed. Debug output saved to {output_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug detailed image processing pipeline.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="debug_steps", help="Output directory for debug files")
    
    args = parser.parse_args()
    
    # Full path for output directory
    if not os.path.isabs(args.output):
        output_dir = os.path.join("/Users/morgan/Documents/GitHub/COMP2271_IP/processed_images", args.output)
    else:
        output_dir = args.output
    
    process_and_debug_image(args.input, output_dir)