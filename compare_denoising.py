#!/usr/bin/env python3
"""
Script to compare different denoising methods.
This script processes sample images with each denoising technique and
creates a side-by-side comparison for visual evaluation.
"""

import cv2
import numpy as np
import os
import argparse
import glob
from time import time

# Try importing with error handling to avoid crashes
try:
    from advanced_denoise import AdvancedDenoiser
    from perspective import PerspectiveCorrector
    ADVANCED_METHODS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import advanced methods: {e}")
    print("Will only use basic OpenCV methods for denoising comparison")
    ADVANCED_METHODS_AVAILABLE = False

def analyze_brightness(image):
    """Analyze brightness distribution of an image."""
    # Convert to YCrCb for brightness analysis
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:,:,0]  # Y channel represents brightness
    
    # Create brightness masks
    bright_mask = y_channel > 180
    mid_mask = (y_channel > 120) & (y_channel <= 180)
    dark_mask = y_channel <= 120
    
    # Calculate percentage of each brightness level
    total_pixels = image.shape[0] * image.shape[1]
    bright_pct = np.sum(bright_mask) / total_pixels * 100
    mid_pct = np.sum(mid_mask) / total_pixels * 100
    dark_pct = np.sum(dark_mask) / total_pixels * 100
    
    return {
        "bright_pct": bright_pct,
        "mid_pct": mid_pct,
        "dark_pct": dark_pct,
        "mean_brightness": np.mean(y_channel)
    }

def process_image_with_method(image_path, method, detail_level=0.8, debug=False):
    """Process an image with a specific denoising method and return the result."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
        
    # Analyze brightness for debugging
    if debug:
        brightness = analyze_brightness(image)
        print(f"{os.path.basename(image_path)} brightness: " +
              f"Bright {brightness['bright_pct']:.1f}%, " +
              f"Mid {brightness['mid_pct']:.1f}%, " +
              f"Dark {brightness['dark_pct']:.1f}%, " +
              f"Mean {brightness['mean_brightness']:.1f}")
        
        # If image is very dark or very bright, show warning
        if brightness['mean_brightness'] < 50:
            print(f"WARNING: Image is very dark (mean brightness {brightness['mean_brightness']:.1f})")
        elif brightness['mean_brightness'] > 200:
            print(f"WARNING: Image is very bright (mean brightness {brightness['mean_brightness']:.1f})")
    
    # For basic OpenCV methods
    if method == 'opencv_fastNlMeans':
        start_time = time()
        result = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        proc_time = time() - start_time
        return result, proc_time
    
    elif method == 'opencv_bilateral':
        start_time = time()
        result = cv2.bilateralFilter(image, 9, 75, 75)
        proc_time = time() - start_time
        return result, proc_time
    
    elif method == 'opencv_gaussian':
        start_time = time()
        result = cv2.GaussianBlur(image, (5, 5), 0)
        proc_time = time() - start_time
        return result, proc_time
    
    elif method == 'opencv_median':
        start_time = time()
        result = cv2.medianBlur(image, 3)
        proc_time = time() - start_time
        return result, proc_time
    
    # Advanced denoising methods
    else:
        # Create processor
        denoiser = AdvancedDenoiser(debug=debug)
        
        # Apply the selected denoising method
        start_time = time()
        result = denoiser.denoise_image(
            image, 
            preserve_detail_level=detail_level,
            method=method,
            median_first=(method == 'bm3d'),
            median_kernel_size=3
        )
        proc_time = time() - start_time
        
        return result, proc_time

def create_comparison_image(image_path, output_path, detail_level=0.8, methods=None, debug=False):
    """
    Create a comparison image showing results of different denoising methods.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the comparison image
        detail_level: Detail preservation level (0.0-1.0)
        methods: List of denoising methods to compare
        debug: Whether to print debug information
    """
    if methods is None:
        # Include only OpenCV methods and advanced methods (excluding basic ones)
        methods = [
            'opencv_fastNlMeans', 'opencv_bilateral', 'opencv_median', 'opencv_gaussian',
            'bm3d', 'multi_stage', 'adaptive_edge'
        ]
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Create a denoiser
    denoiser = AdvancedDenoiser(debug=debug)
    
    # Process with each method
    results = {}
    times = {}
    
    for method in methods:
        print(f"Applying {method} denoising...")
        result, proc_time = process_image_with_method(image_path, method, detail_level, debug)
        results[method] = result
        times[method] = proc_time
    
    # Create comparison image
    h, w = image.shape[:2]
    num_methods = len(methods)
    result_h = h
    result_w = w * (num_methods + 1)  # +1 for original
    
    # Create canvas
    comparison = np.zeros((result_h, result_w, 3), dtype=np.uint8)
    
    # Add titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    
    # Add original image to canvas
    comparison[:h, :w] = image
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, color, thickness)
    
    # Add processed images
    for i, method in enumerate(methods):
        start_x = (i + 1) * w
        comparison[:h, start_x:start_x+w] = results[method]
        cv2.putText(comparison, 
                   f"{method} ({times[method]:.2f}s)", 
                   (start_x + 10, 30), 
                   font, font_scale, color, thickness)
    
    # Save the comparison image
    cv2.imwrite(output_path, comparison)
    
    print(f"Comparison image saved to {output_path}")
    for method in methods:
        print(f"{method} processing time: {times[method]:.2f}s")

def create_comparison_grid(input_dir, output_path, num_samples=5, detail_level=0.8, methods=None, debug=False):
    """
    Create a grid of comparison images from multiple samples.
    
    Args:
        input_dir: Directory containing input images
        output_path: Path to save the comparison grid image
        num_samples: Number of sample images to include
        detail_level: Detail preservation level (0.0-1.0)
        methods: List of denoising methods to compare
        debug: Whether to print debug information
    """
    if methods is None:
        # Include only OpenCV methods and advanced methods (excluding basic ones)
        methods = [
            'opencv_fastNlMeans', 'opencv_bilateral', 'opencv_median', 'opencv_gaussian',
            'bm3d', 'multi_stage', 'adaptive_edge'
        ]
    
    # Get sample images
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_files:
        print(f"No jpg images found in {input_dir}")
        return
    
    # Select evenly distributed samples
    if len(image_files) > num_samples:
        step = len(image_files) // num_samples
        selected_images = image_files[::step][:num_samples]
    else:
        selected_images = image_files[:num_samples]
    
    # Process each sample with each method
    sample_results = {}
    for i, image_path in enumerate(selected_images):
        print(f"Processing sample {i+1}/{len(selected_images)}: {os.path.basename(image_path)}")
        sample_results[image_path] = {}
        
        # Original image
        image = cv2.imread(image_path)
        if image is None:
            continue
        sample_results[image_path]['original'] = image
        
        # Process with each method
        for method in methods:
            result, _ = process_image_with_method(image_path, method, detail_level, debug)
            sample_results[image_path][method] = result
    
    # Create the grid
    # Calculate grid dimensions
    num_cols = len(methods) + 1  # +1 for original
    num_rows = len(sample_results)
    
    # Get a sample image to determine dimensions
    sample_path = list(sample_results.keys())[0]
    sample_img = sample_results[sample_path]['original']
    
    # Determine thumbnail size (resized if needed)
    thumb_w = min(300, sample_img.shape[1])
    scale = thumb_w / sample_img.shape[1]
    thumb_h = int(sample_img.shape[0] * scale)
    
    # Create canvas with padding and labels
    padding = 10
    label_h = 30
    grid_w = num_cols * (thumb_w + padding) + padding
    
    # Calculate grid height with enough space
    grid_h = padding + 80  # Top margin for title and headers
    for row in range(num_rows):
        grid_h += thumb_h + label_h + padding
    
    # Add extra padding to ensure we have enough room
    grid_h += 50
    
    print(f"Creating grid with dimensions: {grid_w}x{grid_h}")
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    title_color = (0, 0, 0)  # Black text
    
    cv2.putText(grid, 
               "Denoising Method Comparison", 
               (padding, padding + 20), 
               font, font_scale, title_color, thickness)
    
    # Labels for columns
    col_labels = ['Original'] + methods
    for col, label in enumerate(col_labels):
        x = padding + col * (thumb_w + padding)
        y = padding + 60
        cv2.putText(grid, 
                   label, 
                   (x, y), 
                   font, 0.7, title_color, 1)
    
    # Add thumbnails
    for row, (image_path, results) in enumerate(sample_results.items()):
        filename = os.path.basename(image_path)
        # Add filename on left
        y = padding + 80 + row * (thumb_h + label_h + padding)
        cv2.putText(grid, 
                   f"{filename}", 
                   (padding, y + 15), 
                   font, 0.5, title_color, 1)
        
        # Add thumbnails for each method
        for col, method in enumerate(['original'] + methods):
            if method in results:
                img = results[method]
                # Always resize to ensure consistent dimensions
                img = cv2.resize(img, (thumb_w, thumb_h))
                
                # Place in grid
                x = padding + col * (thumb_w + padding)
                y = padding + 80 + row * (thumb_h + label_h + padding)
                
                try:
                    grid[y:y+thumb_h, x:x+thumb_w] = img
                except ValueError as e:
                    print(f"Error placing image in grid: {e}")
                    print(f"Grid shape: {grid.shape}, Image shape: {img.shape}")
                    print(f"Position: y={y}, y+thumb_h={y+thumb_h}, x={x}, x+thumb_w={x+thumb_w}")
    
    # Save the grid
    cv2.imwrite(output_path, grid)
    print(f"Comparison grid saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different denoising methods.")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("--output", default="Results/denoising_comparison.jpg", 
                       help="Output path for comparison grid image")
    parser.add_argument("--samples", type=int, default=5, 
                       help="Number of sample images to use (default: 5)")
    parser.add_argument("--detail", type=float, default=0.8, 
                       help="Detail preservation level (0.0-1.0, default: 0.8)")
    parser.add_argument("--methods", 
                       default="opencv_fastNlMeans,opencv_bilateral,opencv_median,opencv_gaussian,bm3d,multi_stage,adaptive_edge",
                       help="Comma-separated list of denoising methods to compare")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Parse methods
    methods = args.methods.split(',')
    
    # Make sure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating comparison grid with {args.samples} samples...")
    print(f"Denoising methods: {', '.join(methods)}")
    
    create_comparison_grid(
        args.input_dir, 
        args.output, 
        num_samples=args.samples,
        detail_level=args.detail,
        methods=methods,
        debug=args.debug
    )
    
    print("Comparison complete!")