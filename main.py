#!/usr/bin/env python3
"""
Complete image processing pipeline:
1. Perspective correction
2. Color/contrast enhancemen
3. Multi-stage denoising
4. Inpainting (removal of artifacts like black circles)

This script applies processing steps to produce enhanced images.
"""

import os
import sys
import argparse
import glob

# Check for required packages
try:
    import cv2
    import numpy as np
    import scipy
    from perspective import PerspectiveCorrector
    from color import EnhancedColorCorrector
    from advanced_denoise import AdvancedDenoiser
    from patchmatch_inpainting import Inpainter
except ImportError as e:
    print(f"Error: Missing dependency - {str(e)}")
    print("Please install all required dependencies:")
    print("  - opencv-python")
    print("  - numpy")
    print("  - scipy")
    print("  - scikit-image")
    sys.exit(1)


def process_single_image(image_path, output_dir, denoise_method="bm3d", debug=False):
    """
    Processes an image through the complete enhancement pipeline.

    The pipeline consists of:
    - Perspective correction to fix camera angle
    - Noise removal (using various methods)
    - Color correction and enhancemen
    - Artifact removal via inpainting
    Parameters:
    -----------
    image_path : str
        Path to the input image
    output_dir : str
        Where to save the processed image
    denoise_method : str, optional
        Which denoising algorithm to use
    debug : bool, optional
        Print extra info during processing
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

    # Remove noise from the image
    # Each method has its own tradeoffs between speed and quality
    noise_methods = {
        "bm3d": "BM3D filtering",
        "nlm": "Non-local means",
        "median_only": "Median filter",
        "multi_stage": "Multi-stage",
        "adaptive_edge": "Adaptive edge-aware",
    }

    noise_method_name = noise_methods.get(denoise_method, denoise_method)
    print(f"  Step 2: Removing noise with {noise_method_name}...")

    # 0.8 works well for most images
    # Experiment with this value
    detail_level = 0.8

    # Apply selected denoising method
    denoised_image = advanced_denoiser.denoise_image(
        perspective_image,
        preserve_detail_level=detail_level,
        method=denoise_method,
        median_first=(denoise_method == "bm3d"),
        median_kernel_size=3,
    )

    # Step 3: Color Correction
    print(f"  Step 3: Applying color enhancement...")
    color_image = color_corrector.auto_correct(denoised_image)

    # Step 4: Apply PatchMatch-based inpainting
    print(f"  Step 4: Applying PatchMatch inpainting...")
    try:
        # Detect artifacts in top-right region where black circles appear
        # Use a lower threshold (15) for detecting dark regions
        mask = inpainter.detect_artifacts(color_image, threshold=15)

        # Apply inpainting
        inpainted_image = inpainter.inpaint_image(color_image, mask)

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

        # Save the inpainted image as final resul
        result_path = os.path.join(output_dir, filename)
        cv2.imwrite(result_path, inpainted_image)

        print(f"  Pipeline completed successfully")

    except Exception as e:
        print(f"  Warning: Inpainting failed: {str(e)}")
        print(f"  Using color-corrected image as final result instead.")

        # Save color-corrected image as final resul
        result_path = os.path.join(output_dir, filename)
        cv2.imwrite(result_path, color_image)

    print(f"  Processing complete for {filename}")
    return True


def process_directory(input_dir, output_dir, denoise_method="bm3d", debug=False):
    """
    Batch process all images in a folder.

    Finds all JPG images in the given directory and runs them through
    the enhancement pipeline, saving results to the output folder.

    Parameters:
    -----------
    input_dir : str
        Source directory with images
    output_dir : str
        Where to save processed images
    denoise_method : str
        Noise removal algorithm to use
    debug : bool
        Show extra debug info
    """
    # Make sure output dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all the jpg files
    images = glob.glob(os.path.join(input_dir, "*.jpg"))

    # Bail if we didn't find any
    if len(images) == 0:
        print(f"No images found in {input_dir}!")
        return

    # Keep track of success/failure
    good_count = 0
    total = len(images)

    # Process each image one by one
    for idx, img in enumerate(images):
        print(f"Image {idx+1} of {total}: {os.path.basename(img)}")
        if process_single_image(img, output_dir, denoise_method, debug):
            good_count += 1

    # Show a summary when we're done
    success_rate = (good_count / total) * 100 if total > 0 else 0
    msg = f"Done! {good_count}/{total} images processed"
    msg += f" successfully ({success_rate:.1f}%)."
    print(msg)


def main():
    # Set up command line options
    parser = argparse.ArgumentParser(description="Road Image Enhancement Pipeline")

    # Required argument - where are the input images?
    parser.add_argument("input_dir", help="Folder containing images to process")

    # Optional - what denoising method to use
    parser.add_argument(
        "--denoise",
        choices=["bm3d", "nlm", "median_only", "multi_stage", "adaptive_edge"],
        default="bm3d",  # BM3D is slower but gives best results
        help="Algorithm to use for noise removal",
    )

    # For detailed outpu
    parser.add_argument(
        "--debug", action="store_true", help="Show extra debug information"
    )

    # Parse args
    args = parser.parse_args()

    # Output always goes to Results folder
    output_dir = "Results"

    # Make sure it exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Do the actual work
    process_directory(args.input_dir, output_dir, args.denoise, args.debug)


if __name__ == "__main__":
    main()
