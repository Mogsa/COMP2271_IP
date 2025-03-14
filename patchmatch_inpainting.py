"""
Implementation of inpainting using OpenCV with options for PatchMatch inpainting when available.
"""

import os
import cv2
import numpy as np
import subprocess
from typing import Optional, Tuple, Union

# Import required libraries for OpenCV fallback inpainting
import subprocess
import sys
import os
import os.path as osp

# Check if we can compile and use PyPatchMatch
PATCHMATCH_AVAILABLE = False
try:
    # Try to import the patch_match module
    sys.path.append(os.path.join(os.path.dirname(__file__), 'PyPatchMatch'))
    import patch_match
    PATCHMATCH_AVAILABLE = True
    print("Successfully loaded PyPatchMatch for inpainting.")
except ImportError as e:
    print(f"Could not import PyPatchMatch: {e}")
    print("Using OpenCV inpainting instead of PyPatchMatch.")
except Exception as e:
    print(f"Error loading PyPatchMatch: {e}")
    print("Using OpenCV inpainting instead of PyPatchMatch.")


class Inpainter:
    """Class for inpainting operations using various methods."""

    def __init__(self, use_patchmatch: bool = True, patch_size: int = 5):
        """
        Initialize the inpainter.
        
        Args:
            use_patchmatch: Whether to use PatchMatch inpainting when available
            patch_size: Size of patches for PatchMatch algorithm
        """
        self.use_patchmatch = use_patchmatch and PATCHMATCH_AVAILABLE
        self.patch_size = patch_size
        
    def detect_artifacts(self, image: np.ndarray, 
                         focus_region: Optional[Tuple[int, int, int, int]] = None,
                         manual_mask: bool = False,
                         threshold: int = 20) -> np.ndarray:
        """
        Detect artifacts in the image, focusing on black circles in the specified region.
        
        Args:
            image: Input image
            focus_region: Region to focus on (x, y, width, height) - if None, use top-right quarter
            manual_mask: If True, create a manual mask in the top-right for testing purposes
            threshold: Brightness threshold for detecting dark regions (0-255)
            
        Returns:
            Binary mask where white pixels (255) represent artifacts
        """
        # Make a copy of the image
        img = image.copy()
        h, w = img.shape[:2]
        
        # Option to create a manual mask for testing purposes
        if manual_mask:
            # Analyze the image to find the actual black circle
            # First convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Create a binary mask of very dark pixels
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours of dark regions
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area, largest first
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Create the mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # If we found any contours, take the largest one that's in the top-right quarter
            # and has reasonable circularity
            for contour in contours:
                # Calculate centroid
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Check if contour is in top-right quarter
                if cx > w/2 and cy < h/2:
                    # Calculate circularity
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter == 0:
                        continue
                    
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # If it's reasonably circular and large enough, use it
                    if circularity > 0.4 and area > 100:
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        # Dilate to ensure full coverage
                        kernel = np.ones((5, 5), np.uint8)
                        mask = cv2.dilate(mask, kernel, iterations=1)
                        return mask
            
            # If we didn't find a suitable contour, create a small circle in the top-right
            # (This is just a fallback for testing)
            if cv2.countNonZero(mask) == 0:
                center = (w - 50, 50)  # Top-right position
                cv2.circle(mask, center, 20, 255, -1)  # Draw a filled circle
            
            return mask
        
        # Default to top-right quarter if no focus region is provided
        if focus_region is None:
            focus_region = (w // 2, 0, w // 2, h // 2)
        
        # Extract the focus region
        x, y, width, height = focus_region
        focus_img = img[y:y+height, x:x+width]
        
        # Convert to grayscale
        gray = cv2.cvtColor(focus_img, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple thresholds to catch different artifact intensities
        masks = []
        
        # Threshold 1: Very dark regions (black artifacts)
        _, binary1 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        masks.append(binary1)
        
        # Threshold 2: Medium dark regions
        _, binary2 = cv2.threshold(gray, threshold+10, 255, cv2.THRESH_BINARY_INV)
        masks.append(binary2)
        
        # Combine all masks
        combined_mask = np.zeros_like(gray)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for the artifacts
        mask = np.zeros_like(gray)
        
        # Filter contours based on area and circularity
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Avoid division by zero
            if perimeter == 0:
                continue
                
            # Calculate circularity: 4*pi*area/perimeter^2
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # More relaxed criteria to catch more potential artifacts
            if area > 100 and circularity > 0.4:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Create full-size mask
        full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        full_mask[y:y+height, x:x+width] = mask
        
        # Dilate the mask to ensure complete coverage of artifacts
        kernel = np.ones((5, 5), np.uint8)
        full_mask = cv2.dilate(full_mask, kernel, iterations=1)
        
        return full_mask
    
    def inpaint_image(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                     auto_detect: bool = True) -> np.ndarray:
        """
        Inpaint an image to remove artifacts.
        
        Args:
            image: Input image
            mask: Binary mask where white pixels (255) represent areas to inpaint.
                  If None and auto_detect is True, mask will be generated automatically.
            auto_detect: Whether to automatically detect artifacts if mask is None
            
        Returns:
            Inpainted image
        """
        # Generate mask if not provided
        if mask is None and auto_detect:
            mask = self.detect_artifacts(image)
        
        # If no mask is provided and auto_detect is False, return the original image
        if mask is None:
            return image.copy()
        
        # Ensure mask has proper shape
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Check if PatchMatch is available and should be used
        if self.use_patchmatch and PATCHMATCH_AVAILABLE:
            try:
                # Use PatchMatch inpainting
                print("Using PyPatchMatch for inpainting...")
                # Convert OpenCV BGR to RGB for PyPatchMatch
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Call the PatchMatch inpainting function
                result_rgb = patch_match.inpaint(
                    image_rgb,
                    mask,
                    patch_size=self.patch_size
                )
                
                # Convert back to BGR for OpenCV
                result = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                return result
            except Exception as e:
                print(f"PyPatchMatch inpainting failed: {e}")
                print("Falling back to OpenCV inpainting...")
        
        # If PatchMatch is not available or failed, use OpenCV inpainting methods
        # We'll try multiple inpainting methods and return the best result
        
        # Method 1: Telea algorithm - fast but less accurate
        telea_result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # Method 2: Navier-Stokes algorithm - slower but sometimes better quality
        ns_result = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        
        # Method 3: Fast Marching Method - another approach
        try:
            # Extended inpainting with larger radius
            fmm_result = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        except Exception:
            # If it fails, use telea_result as fallback
            fmm_result = telea_result
            
        # Choose the result based on variance in the inpainted region
        # Higher variance usually means more natural-looking results
        
        # Create a mask for evaluating the inpainted region
        dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        # Calculate variance in the inpainted regions
        telea_var = cv2.meanStdDev(telea_result, mask=dilated_mask)[1].mean()
        ns_var = cv2.meanStdDev(ns_result, mask=dilated_mask)[1].mean()
        fmm_var = cv2.meanStdDev(fmm_result, mask=dilated_mask)[1].mean()
        
        # Choose the result with the highest variance
        if telea_var >= ns_var and telea_var >= fmm_var:
            return telea_result
        elif ns_var >= telea_var and ns_var >= fmm_var:
            return ns_result
        else:
            return fmm_result


# For standalone testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inpainting functionality")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--method", choices=["patchmatch", "opencv"], default="patchmatch", 
                        help="Inpainting method")
    parser.add_argument("--patch-size", type=int, default=5, help="Patch size for PatchMatch")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    
    args = parser.parse_args()
    
    # Read input image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Failed to read image: {args.input}")
        exit(1)
    
    # Create inpainter
    inpainter = Inpainter(
        use_patchmatch=(args.method == "patchmatch"),
        patch_size=args.patch_size
    )
    
    # Detect artifacts
    mask = inpainter.detect_artifacts(image)
    
    # Inpaint image
    result = inpainter.inpaint_image(image, mask)
    
    # Save result
    cv2.imwrite(args.output, result)
    
    # Save debug images if requested
    if args.debug:
        output_dir = os.path.dirname(args.output)
        base_name = os.path.splitext(os.path.basename(args.output))[0]
        
        # Save mask
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), mask)
        
        # Visualize mask on image
        mask_vis = image.copy()
        mask_vis[mask > 0] = [0, 0, 255]  # Mark mask in red
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask_vis.png"), mask_vis)
        
    print(f"Inpainting completed. Result saved to {args.output}")