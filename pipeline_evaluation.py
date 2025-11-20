#!/usr/bin/env python3
"""
Pipeline Evaluation Script

This script evaluates each step of the image processing pipeline individually:
1. Perspective correction
2. Denoising
3. Color correction
4. Inpainting

It processes images through each stage separately and runs classification
on the results to measure the impact of each processing step.
"""

import os
import sys
import subprocess
import shutil
import argparse
import time
import numpy as np
import cv2

# Check for required packages
try:
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


def create_output_dirs():
    """Create output directories for each pipeline stage."""
    output_dirs = {
        "perspective": "Results_perspective_only",
        "denoise": "Results_denoise_only",
        "color": "Results_color_only",
        "inpaint": "Results_inpaint_only",
        "full": "Results_full_pipeline",
    }
    
    for dir_name in output_dirs.values():
        os.makedirs(dir_name, exist_ok=True)
        
    return output_dirs


def perspective_only(image_path, output_dir, debug=False):
    """Apply only perspective correction."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Initialize perspective corrector
    perspective_corrector = PerspectiveCorrector(debug=debug)
    
    # Apply perspective correction
    corrected_image = perspective_corrector.process_image(image_path)
    if corrected_image is None:
        print(f"Error: Perspective correction failed for {image_path}")
        return False
    
    # Save the output
    cv2.imwrite(output_path, corrected_image)
    return True


def denoise_only(image_path, output_dir, method="bm3d", debug=False):
    """Apply only denoising."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return False
    
    # Initialize denoiser
    denoiser = AdvancedDenoiser(debug=debug)
    
    # Apply denoising
    denoised_image = denoiser.denoise_image(
        image,
        preserve_detail_level=0.8,
        method=method,
        median_first=(method == "bm3d"),
        median_kernel_size=3,
    )
    
    # Save the output
    cv2.imwrite(output_path, denoised_image)
    return True


def color_only(image_path, output_dir, debug=False):
    """Apply only color correction."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return False
    
    # Initialize color corrector
    color_corrector = EnhancedColorCorrector(debug=debug)
    
    # Apply color correction
    color_corrected = color_corrector.auto_correct(image)
    
    # Save the output
    cv2.imwrite(output_path, color_corrected)
    return True


def inpaint_only(image_path, output_dir, debug=False):
    """Apply only inpainting."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return False
    
    # Initialize inpainter
    inpainter = Inpainter(use_patchmatch=True, patch_size=5)
    
    try:
        # Detect artifacts
        mask = inpainter.detect_artifacts(image, threshold=15)
        
        # Apply inpainting
        inpainted_image = inpainter.inpaint_image(image, mask)
        
        # Save debug images if requested
        if debug:
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save mask
            mask_path = os.path.join(debug_dir, f"mask_{filename}")
            cv2.imwrite(mask_path, mask)
            
            # Save mask visualization
            mask_vis = image.copy()
            mask_vis[mask > 0] = [0, 0, 255]  # Mark mask in red
            mask_vis_path = os.path.join(debug_dir, f"mask_vis_{filename}")
            cv2.imwrite(mask_vis_path, mask_vis)
        
        # Save the output
        cv2.imwrite(output_path, inpainted_image)
        return True
    
    except Exception as e:
        print(f"Error during inpainting: {e}")
        # Save original image as fallback
        cv2.imwrite(output_path, image)
        return False


def full_pipeline(image_path, output_dir, denoise_method="bm3d", debug=False):
    """Apply the full pipeline."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Initialize processors
    perspective_corrector = PerspectiveCorrector(debug=debug)
    color_corrector = EnhancedColorCorrector(debug=debug)
    denoiser = AdvancedDenoiser(debug=debug)
    inpainter = Inpainter(use_patchmatch=True, patch_size=5)
    
    # Step 1: Perspective Correction
    corrected_image = perspective_corrector.process_image(image_path)
    if corrected_image is None:
        print(f"Error: Perspective correction failed for {image_path}")
        return False
    
    # Step 2: Denoising
    denoised_image = denoiser.denoise_image(
        corrected_image,
        preserve_detail_level=0.8,
        method=denoise_method,
        median_first=(denoise_method == "bm3d"),
        median_kernel_size=3,
    )
    
    # Step 3: Color Correction
    color_image = color_corrector.auto_correct(denoised_image)
    
    # Step 4: Inpainting
    try:
        mask = inpainter.detect_artifacts(color_image, threshold=15)
        inpainted_image = inpainter.inpaint_image(color_image, mask)
        
        # Save debug images if requested
        if debug:
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save mask
            mask_path = os.path.join(debug_dir, f"mask_{filename}")
            cv2.imwrite(mask_path, mask)
            
            # Save mask visualization
            mask_vis = color_image.copy()
            mask_vis[mask > 0] = [0, 0, 255]  # Mark mask in red
            mask_vis_path = os.path.join(debug_dir, f"mask_vis_{filename}")
            cv2.imwrite(mask_vis_path, mask_vis)
        
        # Save the output
        cv2.imwrite(output_path, inpainted_image)
        return True
    
    except Exception as e:
        print(f"Error during inpainting: {e}")
        # Save color corrected image as fallback
        cv2.imwrite(output_path, color_image)
        return True


def process_directory(input_dir, output_dirs, denoise_method="bm3d", debug=False):
    """Process all images in a directory through each pipeline stage."""
    # Get all jpg files
    images = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    # Process counts
    total = len(images)
    counts = {stage: 0 for stage in output_dirs.keys()}
    
    # Process each image
    for i, img_file in enumerate(images):
        img_path = os.path.join(input_dir, img_file)
        print(f"Processing image {i+1}/{total}: {img_file}")
        
        # Apply each stage individually
        if perspective_only(img_path, output_dirs["perspective"], debug):
            counts["perspective"] += 1
        
        if denoise_only(img_path, output_dirs["denoise"], denoise_method, debug):
            counts["denoise"] += 1
        
        if color_only(img_path, output_dirs["color"], debug):
            counts["color"] += 1
        
        if inpaint_only(img_path, output_dirs["inpaint"], debug):
            counts["inpaint"] += 1
        
        if full_pipeline(img_path, output_dirs["full"], denoise_method, debug):
            counts["full"] += 1
    
    # Report success rates
    print("\nProcessing complete:")
    for stage, count in counts.items():
        success_rate = (count / total) * 100
        print(f"  {stage.capitalize()}: {count}/{total} images ({success_rate:.1f}%)")


def run_classification(output_dirs, classifier_model="classifier.model"):
    """Run classification on all processed outputs and report accuracy."""
    results = {}
    
    print("\nRunning classification on each pipeline stage output:")
    
    for stage, dir_path in output_dirs.items():
        print(f"\nClassifying {stage} images...")
        
        # Run the classifier
        cmd = ["python", "classify.py", f"--data={dir_path}", f"--model={classifier_model}"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  universal_newlines=True)
        stdout, stderr = process.communicate()
        
        # Extract accuracy from output
        for line in stdout.split('\n'):
            if line.startswith("Accuracy is"):
                accuracy = float(line.split()[-1])
                results[stage] = accuracy
                break
        
        print(f"  {stage.capitalize()} accuracy: {results.get(stage, 'N/A')}")
    
    return results


def main():
    """Main function to process images and run classification."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate each image processing pipeline stage individually"
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        default="driving_images",
        help="Directory containing input images"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="classifier.model",
        help="Path to the classifier model"
    )
    
    parser.add_argument(
        "--denoise", 
        choices=["bm3d", "nlm", "median_only", "multi_stage", "adaptive_edge"],
        default="bm3d",
        help="Denoising method to use"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug output and visualizations"
    )
    
    args = parser.parse_args()
    
    # Create output directories for each stage
    output_dirs = create_output_dirs()
    
    # Process images through each stage
    start_time = time.time()
    process_directory(args.input, output_dirs, args.denoise, args.debug)
    processing_time = time.time() - start_time
    
    print(f"\nTotal processing time: {processing_time:.2f} seconds")
    
    # Run classification on the results
    classification_results = run_classification(output_dirs, args.model)
    
    # Print summary table
    print("\nPipeline Stage Evaluation Summary:")
    print("=" * 50)
    print(f"{'Stage':<20} | {'Accuracy':<10}")
    print("-" * 50)
    
    # Get baseline result (raw images)
    cmd = ["python", "classify.py", f"--data={args.input}", f"--model={args.model}"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              universal_newlines=True)
    stdout, stderr = process.communicate()
    
    # Extract baseline accuracy
    baseline_accuracy = None
    for line in stdout.split('\n'):
        if line.startswith("Accuracy is"):
            baseline_accuracy = float(line.split()[-1])
            break
    
    # Print baseline
    print(f"{'Original (baseline)':<20} | {baseline_accuracy:.4f}")
    
    # Print results for each stage
    for stage in ["perspective", "denoise", "color", "inpaint", "full"]:
        accuracy = classification_results.get(stage, "N/A")
        if isinstance(accuracy, float):
            # Calculate improvement
            improvement = ""
            if baseline_accuracy is not None:
                diff = (accuracy - baseline_accuracy) * 100
                if diff > 0:
                    improvement = f" (+{diff:.2f}%)"
                else:
                    improvement = f" ({diff:.2f}%)"
            
            print(f"{stage.capitalize():<20} | {accuracy:.4f}{improvement}")
        else:
            print(f"{stage.capitalize():<20} | {accuracy}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()