#!/usr/bin/env python3
"""
Run the complete pipeline: 
1. Perspective correction
2. Color/contrast enhancement 
3. Denoising
4. Inpainting
5. Classification

This updated version integrates all processing steps in the new pipeline.
"""

import os
import subprocess
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Run the full image processing pipeline and classification")
    parser.add_argument("--input", default="driving_images", 
                      help="Directory containing input images (default: driving_images)")
    parser.add_argument("--output", default="Final_Images", 
                      help="Directory for processed images (default: Final_Images)")
    parser.add_argument("--intermediate", action="store_true",
                      help="Save intermediate results from each processing step")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with additional output")
    parser.add_argument("--model", default="classifier.model", 
                      help="Path to the classifier model (default: classifier.model)")
    
    args = parser.parse_args()
    
    # Make sure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Step 1: Run the pipeline to process images
    print("="*80)
    print("Step 1: Running the complete image processing pipeline...")
    print("="*80)
    
    pipeline_cmd = [
        "python", "pipeline.py", 
        "--input", args.input,
        "--output", args.output
    ]
    
    if args.intermediate:
        pipeline_cmd.append("--intermediate")
    
    if args.debug:
        pipeline_cmd.append("--debug")
    
    try:
        subprocess.run(pipeline_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running pipeline: {e}")
        return
    
    pipeline_time = time.time() - start_time
    print(f"Pipeline completed in {pipeline_time:.2f} seconds")
    
    # Step 2: Run classification on processed images
    print("\n" + "="*80)
    print("Step 2: Running classification on processed images...")
    print("="*80)
    
    classification_start = time.time()
    
    classify_cmd = [
        "python", "classify.py",
        "--data", args.output,
        "--model", args.model
    ]
    
    try:
        subprocess.run(classify_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running classification: {e}")
        return
    
    classification_time = time.time() - classification_start
    print(f"Classification completed in {classification_time:.2f} seconds")
    
    # Print overall statistics
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("Overall results:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("="*80)
    
    print("\nComplete pipeline execution finished successfully!")

if __name__ == "__main__":
    main()