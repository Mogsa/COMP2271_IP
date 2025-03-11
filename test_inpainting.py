#!/usr/bin/env python3
"""
Simple script to test the inpainting functionality.
"""

import os
import argparse
import cv2
import numpy as np
from patchmatch_inpainting import Inpainter

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test inpainting functionality")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="inpainted_output.jpg", help="Output image path")
    parser.add_argument("--method", choices=["patchmatch", "opencv"], default="patchmatch", 
                        help="Inpainting method")
    parser.add_argument("--patch-size", type=int, default=5, help="Patch size for PatchMatch")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    
    args = parser.parse_args()
    
    # Read input image
    print(f"Reading image: {args.input}")
    image = cv2.imread(args.input)
    if image is None:
        print(f"Failed to read image: {args.input}")
        exit(1)
    
    # Create inpainter
    print(f"Using method: {args.method} with patch size: {args.patch_size}")
    inpainter = Inpainter(
        use_patchmatch=(args.method == "patchmatch"),
        patch_size=args.patch_size
    )
    
    # Detect artifacts
    print("Detecting artifacts...")
    
    # Find the actual black circles, not using a manual mask
    mask = inpainter.detect_artifacts(image, threshold=15)
    
    # Count non-zero pixels in mask to check if anything was detected
    non_zero = cv2.countNonZero(mask)
    print(f"Detected {non_zero} pixels to inpaint ({non_zero/(mask.shape[0]*mask.shape[1])*100:.2f}% of image)")
    
    if non_zero == 0:
        print("No artifacts detected. Exiting.")
        exit(0)
    
    # Inpaint image
    print("Inpainting image...")
    result = inpainter.inpaint_image(image, mask)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save result
    print(f"Saving result to: {args.output}")
    cv2.imwrite(args.output, result)
    
    # Save debug images if requested
    if args.debug:
        base_name = os.path.splitext(args.output)[0]
        
        # Save mask
        mask_path = f"{base_name}_mask.png"
        print(f"Saving mask to: {mask_path}")
        cv2.imwrite(mask_path, mask)
        
        # Visualize mask on image
        mask_vis = image.copy()
        mask_vis[mask > 0] = [0, 0, 255]  # Mark mask in red
        mask_vis_path = f"{base_name}_mask_vis.png"
        print(f"Saving mask visualization to: {mask_vis_path}")
        cv2.imwrite(mask_vis_path, mask_vis)
        
        # Save before/after comparison
        comparison = np.hstack((image, result))
        comparison_path = f"{base_name}_comparison.png"
        print(f"Saving before/after comparison to: {comparison_path}")
        cv2.imwrite(comparison_path, comparison)
    
    print("Inpainting completed successfully!")

if __name__ == "__main__":
    main()