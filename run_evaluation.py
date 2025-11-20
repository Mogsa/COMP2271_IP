#!/usr/bin/env python3
"""
Simplified wrapper to run the pipeline evaluation with reduced complexity.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline evaluation"
    )
    
    parser.add_argument(
        "--subset",
        type=int,
        default=15,
        help="Number of images to process"
    )
    
    args = parser.parse_args()
    
    # Step 1: Run basic evaluation with smaller subset to get meaningful results
    print(f"Running evaluation with {args.subset} images...")
    cmd = ["python", "evaluate_pipeline_stages.py", f"--subset={args.subset}"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    print("\n--- RESULTS FROM PIPELINE EVALUATION ---\n")
    # Extract and print the most relevant parts of the output
    lines = process.stdout.strip().split('\n')
    summary_started = False
    
    for line in lines:
        if "Evaluating classification accuracy for each pipeline stage:" in line:
            summary_started = True
        
        if summary_started:
            print(line)
    
    print("\nThe full processed images for each stage are available in:")
    for key in ["perspective", "denoise", "color", "inpaint", "full"]:
        dir_name = f"Results_{key}"
        print(f"- {dir_name}/: Images with only {key} processing applied")

if __name__ == "__main__":
    main()