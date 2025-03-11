#!/usr/bin/env python3
"""
Complete image processing pipeline:
1. Perspective correction
2. Color/contrast enhancement
3. Multi-stage denoising
4. Inpainting (removal of artifacts like black circles)

This script applies all four processing steps in sequence to produce enhanced images.
"""

import os
import argparse
import cv2
import numpy as np
from perspective import PerspectiveCorrector
from color import EnhancedColorCorrector
from denoise import ImageDenoiser
from advanced_denoise import AdvancedDenoiser
from patchmatch_inpainting import Inpainter

def process_single_image(image_path, output_dir, debug=False):
    """
    Process a single image through the complete pipeline:
    1. Perspective correction
    2. Color correction
    3. Multi-stage denoising
    4. Inpainting
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output images
        debug: Enable debug output
    """
    # Extract filename from path
    filename = os.path.basename(image_path)
    
    print(f"Processing: {filename}")
    
    # Initialize processors
    perspective_corrector = PerspectiveCorrector(debug=debug)
    color_corrector = EnhancedColorCorrector(debug=debug)
    # Use advanced denoiser instead of the original one
    advanced_denoiser = AdvancedDenoiser(debug=debug)
    inpainter = Inpainter(use_patchmatch=True, patch_size=5)
    
    # Step 1: Perspective Correction
    print(f"  Step 1: Applying perspective correction...")
    perspective_image = perspective_corrector.process_image(image_path)
    if perspective_image is None:
        print(f"Error: Perspective correction failed for {image_path}")
        return False
    
    # Save intermediate result (optional)
    perspective_dir = os.path.join(output_dir, "1_perspective")
    os.makedirs(perspective_dir, exist_ok=True)
    perspective_path = os.path.join(perspective_dir, filename)
    cv2.imwrite(perspective_path, perspective_image)
    
    # Step 2: Apply advanced detail-preserving denoising
    print(f"  Step 2: Applying advanced detail-preserving denoising...")
    # Use the new advanced denoiser with high detail preservation (0.7)
    denoised_image = advanced_denoiser.denoise_image(perspective_image, preserve_detail_level=0.7)
    
    # Save intermediate result (optional)
    denoised_dir = os.path.join(output_dir, "2_denoised")
    os.makedirs(denoised_dir, exist_ok=True)
    denoised_path = os.path.join(denoised_dir, filename)
    cv2.imwrite(denoised_path, denoised_image)
    
    # Step 3: Color Correction
    print(f"  Step 3: Applying color enhancement...")
    color_image = color_corrector.auto_correct(denoised_image)
    
    # Save intermediate result (optional)
    color_dir = os.path.join(output_dir, "3_color")
    os.makedirs(color_dir, exist_ok=True)
    color_path = os.path.join(color_dir, filename)
    cv2.imwrite(color_path, color_image)
    
    # Step 4: Apply PatchMatch-based inpainting
    print(f"  Step 4: Applying PatchMatch inpainting to remove artifacts...")
    try:
        # Detect artifacts (focusing on top-right region where black circles typically appear)
        # Use a lower threshold (15) for detecting dark regions
        mask = inpainter.detect_artifacts(color_image, threshold=15)
        
        # Apply inpainting
        inpainted_image = inpainter.inpaint_image(color_image, mask)
        
        # Save the inpainted result
        final_dir = os.path.join(output_dir, "4_final")
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, filename)
        cv2.imwrite(final_path, inpainted_image)
        
        # Also save to main output directory for convenience
        main_output_path = os.path.join(output_dir, filename)
        cv2.imwrite(main_output_path, inpainted_image)
        
        print(f"  PatchMatch inpainting completed successfully")
        
        # For debugging, save the mask
        if debug:
            mask_dir = os.path.join(output_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)
            mask_path = os.path.join(mask_dir, f"mask_{filename}")
            cv2.imwrite(mask_path, mask)
            
            # Also save a visualization of the mask
            mask_vis = color_image.copy()
            mask_vis[mask > 0] = [0, 0, 255]  # Mark mask in red
            mask_vis_path = os.path.join(mask_dir, f"mask_vis_{filename}")
            cv2.imwrite(mask_vis_path, mask_vis)
    
    except Exception as e:
        print(f"  Warning: Inpainting failed: {str(e)}")
        print(f"  Using color-corrected image as final result instead.")
        
        # Save color-corrected image as final result when inpainting fails
        final_dir = os.path.join(output_dir, "4_final")
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, filename)
        cv2.imwrite(final_path, color_image)
        
        # Also save to main output directory
        main_output_path = os.path.join(output_dir, filename)
        cv2.imwrite(main_output_path, color_image)
    
    print(f"  Processing complete for {filename}")
    return True

def process_directory(input_dir, output_dir, debug=False):
    """
    Process all images in a directory through the complete pipeline.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        debug: Enable debug output
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    
    for ext in image_extensions:
        pattern = os.path.join(input_dir, ext)
        image_paths.extend(glob.glob(pattern))
    
    # Make sure we found some images
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    # Process each image
    success_count = 0
    total_count = len(image_paths)
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{total_count}")
        success = process_single_image(image_path, output_dir, debug)
        if success:
            success_count += 1
    
    # Report success rate
    accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
    print(f"Processing complete! Successfully processed {success_count}/{total_count} images ({accuracy:.2f}%).")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Complete image processing pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Input image path or directory.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    
    args = parser.parse_args()
    
    # Process input
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.debug)
    else:
        # Process single image
        process_single_image(args.input, args.output, args.debug)

if __name__ == "__main__":
    import glob
    main()