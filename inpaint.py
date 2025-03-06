import cv2
import numpy as np
# Comment out matplotlib to avoid compatibility issues
# import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import distance_transform_edt

class PatchMatchInpainting:
    """
    Implementation of PatchMatch-based image inpainting algorithm.
    
    Based on:
    Barnes, C., Shechtman, E., Finkelstein, A., & Goldman, D. B. (2009). 
    PatchMatch: A randomized correspondence algorithm for structural image editing.
    ACM Transactions on Graphics (ToG), 28(3), 24.
    
    And extension work by:
    Criminisi, A., Pérez, P., & Toyama, K. (2004).
    Region filling and object removal by exemplar-based image inpainting.
    IEEE Transactions on image processing, 13(9), 1200-1212.
    """
    
    def __init__(self, patch_size=9, alpha=0.1, iterations=5, search_ratio=0.5, debug=False):
        """
        Initialize the PatchMatch inpainting algorithm.
        
        Args:
            patch_size: Size of patches used for matching (odd number)
            alpha: Blend factor for inpainting (0-1)
            iterations: Number of refinement iterations
            search_ratio: Ratio of image to search for patches (0-1)
            debug: Whether to display debug visualizations
        """
        assert patch_size % 2 == 1, "Patch size must be odd"
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.alpha = alpha
        self.iterations = iterations
        self.search_ratio = search_ratio
        self.debug = debug
        
    def _detect_black_circle(self, image, expected_radius_ratio=0.1):
        """
        Automatically detect the black circle in the top right of the image.
        
        Args:
            image: Input image
            expected_radius_ratio: Expected radius of the circle as a ratio of image width
        
        Returns:
            Mask of the detected circle (255 for the circle, 0 elsewhere)
        """
        height, width = image.shape[:2]
        expected_radius = int(width * expected_radius_ratio)
        
        # Focus on the top right corner
        roi_size = expected_radius * 3
        roi_x = max(0, width - roi_size)
        roi_y = 0
        roi_width = min(roi_size, width - roi_x)
        roi_height = min(roi_size, height // 3)
        
        # Extract ROI
        roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        
        # Convert to grayscale if it's a color image
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
            
        # Threshold to find dark regions
        _, thresh = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if not contours:
            return mask
            
        # Find the most circular contour
        best_circularity = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter out small contours
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Calculate circularity (1 for perfect circle)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > best_circularity:
                best_circularity = circularity
                best_contour = contour
        
        if best_contour is not None and best_circularity > 0.6:
            # Adjust contour coordinates to the original image
            best_contour = best_contour + np.array([roi_x, roi_y])
            
            # Draw the contour on the mask
            cv2.drawContours(mask, [best_contour], 0, 255, -1)
            
            if self.debug:
                # Debug visualization disabled due to matplotlib compatibility issues
                # debug_img = image.copy()
                # cv2.drawContours(debug_img, [best_contour], 0, (0, 255, 0), 2)
                print(f"Detected circle with circularity: {best_circularity:.2f}")
        
        return mask
    
    def _create_distance_field(self, mask):
        """
        Create a distance field from the mask boundary.
        
        Args:
            mask: Binary mask (255 for holes, 0 for valid regions)
        
        Returns:
            Distance field and confidence map
        """
        # Normalize mask to 0-1
        mask_norm = mask.astype(np.float32) / 255.0
        
        # Calculate distance from boundary
        dist = distance_transform_edt(mask_norm)
        
        # Normalize distances to 0-1
        if np.max(dist) > 0:
            dist = dist / np.max(dist)
        
        # Create confidence map (inverse of the mask)
        confidence = 1.0 - mask_norm
        
        return dist, confidence
    
    def _compute_priority(self, distance_field, confidence_map, gradient_magnitude):
        """
        Compute filling priority according to Criminisi's algorithm.
        
        Args:
            distance_field: Distance to mask boundary
            confidence_map: Confidence values (1 for known pixels, 0 for unknown)
            gradient_magnitude: Magnitude of image gradients
        
        Returns:
            Priority map
        """
        # Normalize gradient magnitude to 0-1
        if np.max(gradient_magnitude) > 0:
            data_term = gradient_magnitude / np.max(gradient_magnitude)
        else:
            data_term = np.zeros_like(gradient_magnitude)
        
        # Calculate priority as described in Criminisi's paper
        priority = confidence_map * (data_term + 0.001)
        
        # Ensure priority is only calculated for boundary pixels
        boundary = np.logical_and(distance_field < 0.01, distance_field > 0)
        priority[~boundary] = 0
        
        return priority
    
    def _get_patch(self, image, center_y, center_x):
        """
        Extract a patch from the image centered at (center_x, center_y).
        Handles image boundaries by mirroring.
        
        Args:
            image: Input image
            center_y, center_x: Center coordinates of the patch
        
        Returns:
            Extracted patch
        """
        height, width = image.shape[:2]
        
        # Calculate patch boundaries
        start_y = center_y - self.half_patch
        end_y = center_y + self.half_patch + 1
        start_x = center_x - self.half_patch
        end_x = center_x + self.half_patch + 1
        
        # Handle boundaries by mirroring
        y_indices = np.clip(np.arange(start_y, end_y), 0, height - 1)
        x_indices = np.clip(np.arange(start_x, end_x), 0, width - 1)
        
        # Extract patch
        if len(image.shape) == 3:
            patch = image[np.ix_(y_indices, x_indices, np.arange(image.shape[2]))]
        else:
            patch = image[np.ix_(y_indices, x_indices)]
        
        return patch
    
    def _find_best_match(self, image, mask, target_y, target_x, exclude_mask=None):
        """
        Find the best matching patch in the image for the target position.
        
        Args:
            image: Input image
            mask: Binary mask (255 for holes, 0 for valid regions)
            target_y, target_x: Target position for inpainting
            exclude_mask: Optional mask of regions to exclude from search
        
        Returns:
            Coordinates of the best matching patch (y, x)
        """
        height, width = image.shape[:2]
        
        # Extract target patch
        target_patch = self._get_patch(image, target_y, target_x)
        target_mask = self._get_patch(mask, target_y, target_x)
        
        # Create validity mask (1 for valid pixels, 0 for holes)
        valid_mask = (target_mask == 0).astype(np.float32)
        
        # If all pixels are invalid, use a simple mask
        if np.sum(valid_mask) == 0:
            valid_mask = np.ones_like(target_mask, dtype=np.float32)
        
        best_distance = float('inf')
        best_match = (0, 0)
        
        # Limit search range to speed up the algorithm
        search_height = int(height * self.search_ratio)
        search_width = int(width * self.search_ratio)
        
        # Random search points
        num_points = 1000
        
        # Generate random search locations
        y_coords = np.random.randint(self.half_patch, height - self.half_patch, num_points)
        x_coords = np.random.randint(self.half_patch, width - self.half_patch, num_points)
        
        for i in range(num_points):
            y, x = y_coords[i], x_coords[i]
            
            # Skip if in excluded region
            if exclude_mask is not None and exclude_mask[y, x] > 0:
                continue
            
            # Skip if patch overlaps with the hole
            source_mask = self._get_patch(mask, y, x)
            if np.any(source_mask > 0):
                continue
            
            # Extract source patch
            source_patch = self._get_patch(image, y, x)
            
            # Calculate distance (weighted by valid pixels)
            diff = (target_patch - source_patch) ** 2
            if len(diff.shape) == 3:
                diff = np.sum(diff, axis=2)
            weighted_diff = diff * valid_mask
            distance = np.sum(weighted_diff) / np.sum(valid_mask)
            
            # Update best match
            if distance < best_distance:
                best_distance = distance
                best_match = (y, x)
        
        # If no valid match found, use random valid position
        if best_distance == float('inf'):
            valid_positions = np.where(np.logical_and(
                np.logical_and(
                    np.arange(self.half_patch, height - self.half_patch)[:, None] >= self.half_patch,
                    np.arange(self.half_patch, height - self.half_patch)[:, None] < height - self.half_patch
                ),
                np.logical_and(
                    np.arange(self.half_patch, width - self.half_patch)[None, :] >= self.half_patch,
                    np.arange(self.half_patch, width - self.half_patch)[None, :] < width - self.half_patch
                )
            ))
            if len(valid_positions[0]) > 0:
                idx = np.random.randint(0, len(valid_positions[0]))
                best_match = (valid_positions[0][idx], valid_positions[1][idx])
            else:
                best_match = (height // 2, width // 2)
        
        return best_match
    
    def _update_image(self, image, mask, target_y, target_x, source_y, source_x):
        """
        Update the image by copying a patch from the source to the target.
        
        Args:
            image: Input image
            mask: Binary mask (255 for holes, 0 for valid regions)
            target_y, target_x: Target position for inpainting
            source_y, source_x: Source position to copy from
        
        Returns:
            Updated image and mask
        """
        # Get patches
        target_patch = self._get_patch(image, target_y, target_x)
        target_mask = self._get_patch(mask, target_y, target_x)
        source_patch = self._get_patch(image, source_y, source_x)
        
        # Create update region
        update_region = (target_mask > 0)
        
        # Update image
        result = target_patch.copy()
        if len(result.shape) == 3:
            for c in range(result.shape[2]):
                result[:,:,c][update_region] = source_patch[:,:,c][update_region]
        else:
            result[update_region] = source_patch[update_region]
        
        # Update the image
        y_start = target_y - self.half_patch
        y_end = target_y + self.half_patch + 1
        x_start = target_x - self.half_patch
        x_end = target_x + self.half_patch + 1
        
        # Ensure we're within image boundaries
        height, width = image.shape[:2]
        y_start = max(0, y_start)
        y_end = min(height, y_end)
        x_start = max(0, x_start)
        x_end = min(width, x_end)
        
        # Get the corresponding section of the result
        result_y_start = self.half_patch - (target_y - y_start)
        result_y_end = self.patch_size - (target_y + self.half_patch + 1 - y_end)
        result_x_start = self.half_patch - (target_x - x_start)
        result_x_end = self.patch_size - (target_x + self.half_patch + 1 - x_end)
        
        # Update the image and mask
        updated_image = image.copy()
        updated_mask = mask.copy()
        
        # Calculate the blend factor based on distance to center
        y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
        dist_to_center = np.sqrt((y_grid - target_y)**2 + (x_grid - target_x)**2)
        max_dist = self.half_patch * np.sqrt(2)
        blend_factor = np.clip(1.0 - dist_to_center / max_dist, 0, 1) * self.alpha
        
        # Apply the update with blending
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                section = updated_image[y_start:y_end, x_start:x_end, c]
                result_section = result[result_y_start:result_y_end, result_x_start:result_x_end, c]
                mask_section = updated_mask[y_start:y_end, x_start:x_end]
                
                # Only update holes
                update = mask_section > 0
                section[update] = (1 - blend_factor[update]) * section[update] + blend_factor[update] * result_section[update]
                updated_image[y_start:y_end, x_start:x_end, c] = section
        else:
            section = updated_image[y_start:y_end, x_start:x_end]
            result_section = result[result_y_start:result_y_end, result_x_start:result_x_end]
            mask_section = updated_mask[y_start:y_end, x_start:x_end]
            
            # Only update holes
            update = mask_section > 0
            section[update] = (1 - blend_factor[update]) * section[update] + blend_factor[update] * result_section[update]
            updated_image[y_start:y_end, x_start:x_end] = section
        
        # Update the mask (reduce hole size)
        center_mask = np.zeros_like(updated_mask)
        center_mask[target_y, target_x] = 255
        center_dist = distance_transform_edt(1 - center_mask / 255.0)
        center_dist = np.exp(-center_dist / (self.half_patch / 2))
        
        # Reduce the mask values
        updated_mask = np.maximum(0, updated_mask - (center_dist * 255 * 0.5).astype(np.uint8))
        
        return updated_image, updated_mask
    
    def inpaint(self, image, mask=None, max_iterations=1000):
        """
        Inpaint the image using the PatchMatch algorithm.
        
        Args:
            image: Input image
            mask: Optional binary mask (255 for holes, 0 for valid regions)
                  If None, automatically detect black circle
            max_iterations: Maximum number of iterations
        
        Returns:
            Inpainted image
        """
        # If no mask provided, detect black circle
        if mask is None:
            mask = self._detect_black_circle(image)
            
        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 128).astype(np.uint8) * 255
            
        if np.sum(mask) == 0:
            print("No regions to inpaint")
            return image
            
        # Initialize
        current_image = image.copy()
        current_mask = mask.copy()
        
        if self.debug:
            # Debug visualization disabled due to matplotlib compatibility issues
            print(f"Starting inpainting. Mask sum: {np.sum(mask)}")
        
        # Main inpainting loop
        iteration = 0
        while np.sum(current_mask) > 0 and iteration < max_iterations:
            # Calculate distance field and confidence map
            distance_field, confidence_map = self._create_distance_field(current_mask)
            
            # Calculate image gradients
            if len(image.shape) == 3:
                gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = current_image
            
            gradients = np.gradient(gray.astype(np.float32))
            gradient_magnitude = np.sqrt(gradients[0]**2 + gradients[1]**2)
            
            # Compute priority
            priority = self._compute_priority(distance_field, confidence_map, gradient_magnitude)
            
            # Find point with highest priority
            if np.max(priority) > 0:
                target_point = np.unravel_index(np.argmax(priority), priority.shape)
                target_y, target_x = target_point
            else:
                # If no priority points, find a point on the boundary
                boundary = np.logical_and(distance_field < 0.01, distance_field > 0)
                if np.any(boundary):
                    boundary_points = np.where(boundary)
                    idx = np.random.randint(0, len(boundary_points[0]))
                    target_y, target_x = boundary_points[0][idx], boundary_points[1][idx]
                else:
                    # If no boundary points, find any point in the mask
                    mask_points = np.where(current_mask > 0)
                    if len(mask_points[0]) > 0:
                        idx = np.random.randint(0, len(mask_points[0]))
                        target_y, target_x = mask_points[0][idx], mask_points[1][idx]
                    else:
                        break
            
            # Find best match
            source_y, source_x = self._find_best_match(current_image, current_mask, target_y, target_x)
            
            # Update image and mask
            current_image, current_mask = self._update_image(
                current_image, current_mask, target_y, target_x, source_y, source_x
            )
            
            # Increment iteration counter
            iteration += 1
            
            # Debug output every few iterations
            if self.debug and (iteration % 20 == 0 or np.sum(current_mask) == 0):
                # Debug visualization disabled due to matplotlib compatibility issues
                print(f"Inpainting iteration {iteration}, remaining mask sum: {np.sum(current_mask)}")
        
        if self.debug:
            # Debug visualization disabled due to matplotlib compatibility issues
            print(f"Inpainting completed in {iteration} iterations")
        
        return current_image
    
    def process_image(self, image_path, output_path=None, detect_circle=True, mask_path=None,
                     show_result=False):
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            detect_circle: Whether to automatically detect the black circle
            mask_path: Path to a mask image (optional)
            show_result: Whether to display the result
        
        Returns:
            Inpainted image
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return None
        
        # Get the mask
        if mask_path is not None:
            # Use provided mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Error: Could not read mask at {mask_path}")
                return None
        elif detect_circle:
            # Detect black circle
            mask = self._detect_black_circle(image)
        else:
            # No mask
            mask = None
        
        # Inpaint the image
        result = self.inpaint(image, mask)
        
        # Save the result if output path is specified
        if output_path:
            cv2.imwrite(output_path, result)
        
        # Show the result if requested (disabled due to matplotlib compatibility issues)
        if show_result and self.debug:
            print("Result visualization disabled due to matplotlib compatibility issues")
        
        return result
    
    def process_directory(self, input_dir, output_dir, detect_circle=True, mask_dir=None):
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for inpainted images
            detect_circle: Whether to automatically detect black circles
            mask_dir: Directory containing mask images (optional)
        
        Returns:
            Number of successfully processed images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                     glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(input_dir, "*.png"))
        
        # Make sure we found some images
        if not image_paths:
            print(f"No images found in {input_dir}")
            return 0
        
        # Process each image
        success_count = 0
        total_count = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                # Generate output path
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, filename)
                
                # Get mask path if mask directory is provided
                mask_path = None
                if mask_dir is not None:
                    base_name = os.path.splitext(filename)[0]
                    mask_candidates = [
                        os.path.join(mask_dir, f"{base_name}.png"),
                        os.path.join(mask_dir, f"{base_name}.jpg"),
                        os.path.join(mask_dir, f"{base_name}.jpeg")
                    ]
                    for candidate in mask_candidates:
                        if os.path.exists(candidate):
                            mask_path = candidate
                            break
                
                # Process the image
                print(f"Processing image {i+1}/{total_count}: {filename}")
                self.process_image(
                    image_path, 
                    output_path=output_path,
                    detect_circle=detect_circle,
                    mask_path=mask_path,
                    show_result=False
                )
                
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # Report success rate
        accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
        print(f"Processing complete. Successfully processed {success_count}/{total_count} images ({accuracy:.2f}%).")
        
        return success_count


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PatchMatch-based image inpainting")
    parser.add_argument("--input", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--output", type=str, required=True, help="Output image path or directory")
    parser.add_argument("--patch-size", type=int, default=9, help="Patch size (odd number)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of refinement iterations")
    parser.add_argument("--alpha", type=float, default=0.1, help="Blend factor (0-1)")
    parser.add_argument("--search-ratio", type=float, default=0.5, help="Search ratio (0-1)")
    parser.add_argument("--no-detect", action="store_false", dest="detect_circle", 
                       help="Disable automatic black circle detection")
    parser.add_argument("--mask", type=str, help="Path to mask image or directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualizations")
    
    args = parser.parse_args()
    
    inpainter = PatchMatchInpainting(
        patch_size=args.patch_size,
        alpha=args.alpha,
        iterations=args.iterations,
        search_ratio=args.search_ratio,
        debug=args.debug
    )
    
    if os.path.isdir(args.input):
        # Process directory
        inpainter.process_directory(
            args.input, 
            args.output, 
            detect_circle=args.detect_circle,
            mask_dir=args.mask if os.path.isdir(args.mask) else None
        )
    else:
        # Process single image
        inpainter.process_image(
            args.input, 
            args.output,
            detect_circle=args.detect_circle,
            mask_path=args.mask if not os.path.isdir(args.mask) else None,
            show_result=True
        )