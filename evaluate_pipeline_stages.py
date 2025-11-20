#!/usr/bin/env python3
"""
Pipeline Stage Evaluation Script

This script evaluates how each stage of the image processing pipeline
contributes to classification accuracy.
"""

import os
import sys
import subprocess
import argparse
import cv2
import numpy as np

# Prevent PyPatchMatch from recompiling
os.environ['SKIP_PATCHMATCH_COMPILE'] = '1'

# Import pipeline components
try:
    from perspective import PerspectiveCorrector
    from color import EnhancedColorCorrector
    from advanced_denoise import AdvancedDenoiser
    from patchmatch_inpainting import Inpainter
except ImportError as e:
    print(f"Error: Missing dependency - {str(e)}")
    print("Please install all required dependencies.")
    sys.exit(1)

# Define the output directories
OUTPUT_DIRS = {
    "perspective": "Results_perspective",
    "denoise": "Results_denoise",
    "color": "Results_color",
    "inpaint": "Results_inpaint",
    "full": "Results_full"
}

def setup_directories():
    """Create output directories for each pipeline stage."""
    for dir_name in OUTPUT_DIRS.values():
        os.makedirs(dir_name, exist_ok=True)

def process_perspective(image_path, output_dir):
    """Apply only perspective correction."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Apply perspective correction
    corrector = PerspectiveCorrector(debug=False)
    result = corrector.process_image(image_path)
    
    if result is not None:
        cv2.imwrite(output_path, result)
        return True
    return False

def process_denoise(image_path, output_dir, method="bm3d"):
    """Apply only denoising."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Apply denoising
    denoiser = AdvancedDenoiser(debug=False)
    result = denoiser.denoise_image(
        img,
        preserve_detail_level=0.8,
        method=method
    )
    
    cv2.imwrite(output_path, result)
    return True

def process_color(image_path, output_dir):
    """Apply only color correction."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Apply color correction
    corrector = EnhancedColorCorrector(debug=False)
    result = corrector.auto_correct(img)
    
    cv2.imwrite(output_path, result)
    return True

def process_inpaint(image_path, output_dir):
    """Apply only inpainting."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Apply inpainting
    inpainter = Inpainter(use_patchmatch=False)  # Use OpenCV inpainting instead
    try:
        mask = inpainter.detect_artifacts(img)
        result = inpainter.inpaint_image(img, mask)
        cv2.imwrite(output_path, result)
        return True
    except Exception as e:
        print(f"Inpainting error for {filename}: {e}")
        cv2.imwrite(output_path, img)  # Save original as fallback
        return False

def process_full_pipeline(image_path, output_dir, denoise_method="bm3d"):
    """Apply the full pipeline."""
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Step 1: Perspective Correction
    corrector = PerspectiveCorrector(debug=False)
    perspective_result = corrector.process_image(image_path)
    if perspective_result is None:
        return False
    
    # Step 2: Denoising
    denoiser = AdvancedDenoiser(debug=False)
    denoised_result = denoiser.denoise_image(
        perspective_result,
        preserve_detail_level=0.8,
        method=denoise_method
    )
    
    # Step 3: Color Correction
    color_corrector = EnhancedColorCorrector(debug=False)
    color_result = color_corrector.auto_correct(denoised_result)
    
    # Step 4: Inpainting
    inpainter = Inpainter(use_patchmatch=False)  # Use OpenCV inpainting instead
    try:
        mask = inpainter.detect_artifacts(color_result)
        final_result = inpainter.inpaint_image(color_result, mask)
        cv2.imwrite(output_path, final_result)
    except Exception:
        # Fall back to color-corrected result
        cv2.imwrite(output_path, color_result)
    
    return True

def process_directory(input_dir, denoise_method="bm3d"):
    """Process all images through each pipeline stage."""
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Ensure we have a reasonable number of files
    if not image_files:
        print(f"No images found in {input_dir}")
        return False
    
    # Process all images
    selected_images = sorted(image_files)
    
    # Process each image through each stage
    for i, filename in enumerate(selected_images):
        image_path = os.path.join(input_dir, filename)
        print(f"Processing image {i+1}/{len(selected_images)}: {filename}")
        
        # Process image through each stage
        process_perspective(image_path, OUTPUT_DIRS["perspective"])
        process_denoise(image_path, OUTPUT_DIRS["denoise"], denoise_method)
        process_color(image_path, OUTPUT_DIRS["color"])
        process_inpaint(image_path, OUTPUT_DIRS["inpaint"])
        process_full_pipeline(image_path, OUTPUT_DIRS["full"], denoise_method)
    
    return True

def run_classification(stage_dir, model_path):
    """Run classifier on processed images and return accuracy."""
    cmd = ["python", "classify.py", f"--data={stage_dir}", f"--model={model_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract accuracy from output
    for line in result.stdout.split('\n'):
        if "Accuracy is" in line:
            accuracy = float(line.split()[-1])
            return accuracy
    
    return None

def evaluate_stages(input_dir, model_path):
    """Run classification on each stage output and report results."""
    print("\nEvaluating classification accuracy for each pipeline stage:")
    
    # Run classification on original images first (baseline)
    baseline = run_classification(input_dir, model_path)
    print(f"Original images (baseline): {baseline:.4f}")
    
    results = {}
    for stage, dir_path in OUTPUT_DIRS.items():
        accuracy = run_classification(dir_path, model_path)
        results[stage] = accuracy
        
        if accuracy is not None and baseline is not None:
            change = (accuracy - baseline) * 100
            change_str = f"+{change:.2f}%" if change >= 0 else f"{change:.2f}%"
            print(f"{stage.capitalize()} only: {accuracy:.4f} ({change_str})")
        else:
            print(f"{stage.capitalize()} only: {accuracy}")
    
    return baseline, results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate each image processing pipeline stage"
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
        help="Path to classifier model"
    )
    
    parser.add_argument(
        "--denoise",
        choices=["bm3d", "nlm", "median_only", "multi_stage", "adaptive_edge"],
        default="bm3d",
        help="Denoising method to use"
    )
    
    # Removed subset parameter to process all images by default
    
    args = parser.parse_args()
    
    # Create output directories
    setup_directories()
    
    # Process images through each stage
    print(f"Processing all images from {args.input}...")
    if process_directory(args.input, args.denoise):
        # Evaluate classification accuracy
        baseline, results = evaluate_stages(args.input, args.model)
        
        # Print summary table
        print("\nSummary of Pipeline Stage Impact on Classification:")
        print("="*60)
        print(f"{'Stage':<15} | {'Accuracy':<10} | {'Change':<10} | {'Impact':<15}")
        print("-"*60)
        
        print(f"{'Original':<15} | {baseline:.4f} | {'---':<10} | {'Baseline':<15}")
        
        for stage in ["perspective", "denoise", "color", "inpaint", "full"]:
            acc = results.get(stage)
            if acc is not None and baseline is not None:
                change = (acc - baseline) * 100
                change_str = f"+{change:.2f}%" if change >= 0 else f"{change:.2f}%"
                
                # Determine impact
                if change > 10:
                    impact = "Major positive"
                elif change > 5:
                    impact = "Positive"
                elif change > 1:
                    impact = "Slight positive"
                elif change > -1:
                    impact = "Neutral"
                elif change > -5:
                    impact = "Slight negative"
                else:
                    impact = "Negative"
                
                print(f"{stage.capitalize():<15} | {acc:.4f} | {change_str:<10} | {impact:<15}")
            else:
                print(f"{stage.capitalize():<15} | {'N/A':<10} | {'N/A':<10} | {'Unknown':<15}")
        
        print("="*60)
    else:
        print("Processing failed.")

if __name__ == "__main__":
    main()