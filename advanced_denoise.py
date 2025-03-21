import cv2
import numpy as np
import os
import glob
import argparse
from time import time
from bm3d import BM3D

class AdvancedDenoiser:
    """
    Advanced image denoising class with multi-stage processing pipeline.
    """
    
    def __init__(self, debug=False):
        """
        Initialize the denoiser.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
    
    def bm3d_denoising(self, image, strength=0.7, apply_median_first=True, median_kernel_size=3):
        """
        Apply BM3D denoising algorithm, optionally with median filtering first.
        
        Args:
            image: Input image (numpy array)
            strength: Denoising strength (0.0-1.0)
            apply_median_first: Whether to apply median filtering before BM3D
            median_kernel_size: Size of the median filter kernel
                
        Returns:
            Denoised image
        """
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,
            image[:,:,1] > 5,
            image[:,:,2] > 5
        ))

        # Only process if there are non-black pixels
        if not np.any(non_black):
            return image
            
        # Apply median filtering first if requested
        if apply_median_first:
            if self.debug:
                print(f"Applying median filter with kernel size {median_kernel_size}x{median_kernel_size}")
                
            # Apply median filter to the whole image
            median_filtered = cv2.medianBlur(image, median_kernel_size)
            
            # Apply the median filter only to non-black regions
            img_to_process = image.copy()
            img_to_process[non_black] = median_filtered[non_black]
        else:
            img_to_process = image

        # Calculate sigma based on strength
        # Higher strength = higher sigma (more aggressive denoising)
        # Use lower sigma if median filter was applied first
        base_sigma = 8.0 if apply_median_first else 10.0
        sigma = base_sigma + (20.0 * (1.0 - strength))

        if self.debug:
            print(f"Applying BM3D with sigma={sigma}")

        # Create BM3D denoiser
        denoiser = BM3D(
            sigma=sigma,
            block_size=8,
            max_blocks=16,
            hard_threshold=2.7
        )

        # Apply BM3D
        result = denoiser.denoise(img_to_process)

        # Apply the denoised image only to non-black regions
        final = image.copy()
        final[non_black] = result[non_black]

        return final

    
    def nlm_color_channels(self, image, preserve_detail_level=0.7):
        """
        Apply Non-Local Means denoising to each color channel separately.
        Enhanced version with edge-aware processing and color space conversion.
        Much more aggressive noise removal while preserving key structures.
        
        Args:
            image: Input image (numpy array)
            preserve_detail_level: Level of detail preservation (0.0-1.0)
            
        Returns:
            Denoised image with each channel processed separately
        """
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,
            image[:,:,1] > 5,
            image[:,:,2] > 5
        ))
        
        if not np.any(non_black):
            return image
        
        # Step 1: Initial median filter to remove strong noise spikes
        # This helps NLM work better by removing outliers
        median_filtered = cv2.medianBlur(image, 3)
        
        # Step 2: Convert to YCrCb color space for better processing
        ycrcb = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Find edges to preserve - calculate on Y channel
        # First sharpen to enhance edges
        sharpen_kernel = np.array([[-1,-1,-1], 
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        y_sharp = cv2.filter2D(y, -1, sharpen_kernel)
        
        # Better edge detection
        edge_detector = cv2.Canny(y_sharp, 30, 90)
        # Dilate edges to protect edge areas
        kernel = np.ones((2, 2), np.uint8)
        dilated_edges = cv2.dilate(edge_detector, kernel, iterations=1)
        edge_mask = dilated_edges > 0
        
        # Much more aggressive parameter for flat areas
        # Higher h value = stronger noise reduction but less detail
        # Adjusted to truly remove noise in flat areas
        h_detail = max(5, int(10 * (1.0 - preserve_detail_level)))  
        h_flat = max(20, int(30 * (1.0 - preserve_detail_level)))   
        
        # Template and search window sizes
        template_window_size = 7   
        search_window_size = 21    
        
        if self.debug:
            print(f"Applying enhanced NLM with detail h={h_detail}, flat h={h_flat}")
        
        # Process Y channel (luminance) with special care for details
        y_result = np.zeros_like(y)
        
        # Process edge areas with gentler denoising
        y_edges = y.copy()
        cv2.fastNlMeansDenoising(
            src=y_edges,
            dst=y_edges,
            h=float(h_detail),
            templateWindowSize=5,  # Keep small template near edges
            searchWindowSize=15
        )
        
        # Process flat areas with MUCH more aggressive denoising
        y_flat = y.copy()
        cv2.fastNlMeansDenoising(
            src=y_flat,
            dst=y_flat,
            h=float(h_flat),
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
        
        # Combine results based on edge mask
        y_result[edge_mask] = y_edges[edge_mask]
        y_result[~edge_mask] = y_flat[~edge_mask]
        
        # Process Cr and Cb channels - color information can be denoised VERY aggressively
        cr_h = int(25 * (1.0 - preserve_detail_level * 0.5))  # Much more aggressive for chrominance
        cb_h = int(25 * (1.0 - preserve_detail_level * 0.5))
        
        cr_denoised = cr.copy()
        cb_denoised = cb.copy()
        
        cv2.fastNlMeansDenoising(
            src=cr,
            dst=cr_denoised,
            h=float(cr_h),
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        cv2.fastNlMeansDenoising(
            src=cb,
            dst=cb_denoised,
            h=float(cb_h),
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Merge channels back
        denoised_ycrcb = cv2.merge([y_result, cr_denoised, cb_denoised])
        result = cv2.cvtColor(denoised_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # Step 3: Apply bilateral filter as a final step to further smooth while preserving edges
        d = 5  # Smaller diameter for speed
        sigma_color = 50  # Large value for strong smoothing effect
        sigma_space = 50  # Large value for more broad effect
        bilateral_result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
        
        # Combine original near edges, bilateral result elsewhere
        final_result = image.copy()
        final_result[non_black & edge_mask] = result[non_black & edge_mask]  # Keep NLM only near edges
        final_result[non_black & ~edge_mask] = bilateral_result[non_black & ~edge_mask]  # Bilateral elsewhere
        
        return final_result

    def multi_stage_denoising(self, image, preserve_detail_level=0.7):
        """
        Apply multi-stage denoising with detail preservation.
        1. Median filter (3x3) for initial noise removal
        2. Non-local means applied mainly to color channels in YCbCr space
        3. Bilateral filter for final smoothing while preserving edges
        
        Args:
            image: Input image (numpy array)
            preserve_detail_level: Level of detail preservation (0.0-1.0)
            
        Returns:
            Denoised image
        """
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,
            image[:,:,1] > 5,
            image[:,:,2] > 5
        ))
        
        if not np.any(non_black):
            return image  # If image is all black, return as is
        
        if self.debug:
            print(f"Processing with detail level: {preserve_detail_level}")
        
        # STAGE 1: Apply median filter (3x3) for initial denoising
        median_filtered = cv2.medianBlur(image, 3)
        
        # Apply median filter only to non-black regions
        median_result = image.copy()
        median_result[non_black] = median_filtered[non_black]
        
        if self.debug:
            print("Median filter (3x3) applied")
                
        # STAGE 2: Convert to YCbCr color space
        ycrcb = cv2.cvtColor(median_result, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # STAGE 3: Calculate strength parameters - focus more on color channels
        # Less aggressive on luminance (inverse of what we had before)
        h_luminance = int(10 * (1.0 - preserve_detail_level))
        h_luminance = max(3, min(h_luminance, 15))
        
        # More aggressive on chrominance 
        h_chrominance = int(15 * (1.0 - preserve_detail_level))
        h_chrominance = max(7, min(h_chrominance, 30))
        
        if self.debug:
            print(f"Denoising parameters - Luminance: {h_luminance}, Chrominance: {h_chrominance}")
        
        # Denoise Y channel (luminance) - less aggressively
        y_denoised = cv2.fastNlMeansDenoising(
            src=y,
            dst=None,
            h=float(h_luminance),
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Denoise Cr and Cb channels (chrominance) - more aggressively
        cr_denoised = cv2.fastNlMeansDenoising(
            src=cr,
            dst=None,
            h=float(h_chrominance),
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        cb_denoised = cv2.fastNlMeansDenoising(
            src=cb,
            dst=None,
            h=float(h_chrominance),
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Recombine the channels
        ycrcb_denoised = cv2.merge([y_denoised, cr_denoised, cb_denoised])
        
        # Convert back to BGR
        nlm_result = cv2.cvtColor(ycrcb_denoised, cv2.COLOR_YCrCb2BGR)
        
        # STAGE 4: Apply bilateral filter for edge preservation and final smoothing
        # Parameters adjusted by detail level
        d = max(5, int(7 * (1 - preserve_detail_level)))
        sigma_color = max(15, 50 * (1 - preserve_detail_level))
        sigma_space = max(15, 50 * (1 - preserve_detail_level))
        
        bilateral = cv2.bilateralFilter(nlm_result, d, sigma_color, sigma_space)
        
        # Apply only to non-black regions
        result = image.copy()
        result[non_black] = bilateral[non_black]
        
        if self.debug:
            print("Multi-stage denoising complete (Median -> NLM-YCrCb -> Bilateral)")
        
        return result
    
    def adaptive_edge_aware_denoising(self, image, preserve_detail_level=0.85):
        """
        Heavy-duty noise removal while preserving critical driving features.
        Brightness-aware denoising: stronger for bright areas, gentler for dark areas and edges.
        Uses improved edge detection to better identify structural elements like cars, lane markings, 
        and road boundaries while avoiding detecting noise as edges.
        
        Args:
            image: Input image (numpy array)
            preserve_detail_level: Level of detail preservation (0.0-1.0)
                                  Higher values preserve more details
        
        Returns:
            Denoised image with preserved edges and details
        """
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,
            image[:,:,1] > 5,
            image[:,:,2] > 5
        ))
        
        if not np.any(non_black):
            return image  # If image is all black, return as is
        
        # Step 1: Initial pre-denoising to make edge detection better
        # First apply a gentle bilateral filter to reduce noise while preserving structure
        pre_denoised = cv2.bilateralFilter(image, 5, 25, 25)
        
        # Step 2: Edge detection on the pre-denoised image - improved to better find strong edges
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(pre_denoised, cv2.COLOR_BGR2GRAY)
        
        # Apply stronger Gaussian blur before edge detection to further reduce noise influence
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # First enhance contrast to make edges more prominent
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(blurred)
        
        # Try two different approaches to edge detection and combine them
        
        # 1. Gradient-based approach (Sobel)
        grad_x = cv2.Sobel(enhanced_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced_gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Use much higher threshold for better noise resistance
        _, sobel_thresh = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY) # Increased from 70 to 120
        
        # Clean up the gradient image
        kernel_clean = np.ones((3, 3), np.uint8)
        sobel_cleaned = cv2.morphologyEx(sobel_thresh, cv2.MORPH_CLOSE, kernel_clean)
        
        # 2. Canny-based edge detection with much higher thresholds
        edges1 = cv2.Canny(sobel, 150, 300)  # Significantly increased threshold for very strong edges only
        edges2 = cv2.Canny(enhanced_gray, 100, 200)  # Increased threshold for medium edges
        
        # Combine edge detection results - prioritize Canny over Sobel since Sobel is too sensitive
        edges = cv2.bitwise_or(edges1, edges2) # Removed sobel_cleaned from the combination
        
        # Apply morphological filtering to connect nearby edges and remove small noise
        kernel_connect = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_connect)
        
        # Remove small noise by area filtering
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_edges = np.zeros_like(edges)
        
        # Much more aggressive area filtering to only keep major structures
        for contour in contours:
            area = cv2.contourArea(contour)
            # Significantly increased threshold to only keep real objects, not noise
            if area > 50:  # Increased from 25 to 50 - only keep substantial contours
                cv2.drawContours(filtered_edges, [contour], 0, 255, -1)
        
        # Dilate edges to ensure we preserve enough context around them
        kernel = np.ones((5, 5), np.uint8)  # Larger kernel to protect wider edge regions
        dilated_edges = cv2.dilate(filtered_edges, kernel, iterations=2)
        
        # Create edge mask - these are the areas where we'll use gentler denoising
        edge_mask = dilated_edges > 0
        
        # Step 3: Create brightness mask for adaptive strength
        # Convert to YCrCb for brightness analysis
        ycrcb_bright = cv2.cvtColor(pre_denoised, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_bright[:,:,0]  # Y channel represents brightness
        
        # Create brightness masks (3 levels) - adjusted thresholds for better segmentation
        # Bright areas (Y > 170) - strong denoising
        bright_mask = y_channel > 170
        # Mid-brightness areas (110 < Y <= 170) - moderate denoising
        mid_mask = (y_channel > 110) & (y_channel <= 170)
        # Dark areas (Y <= 110) - gentle denoising
        dark_mask = y_channel <= 110
        
        if self.debug:
            print(f"Brightness distribution: Bright: {np.sum(bright_mask)/np.sum(non_black)*100:.1f}%, " 
                  f"Mid: {np.sum(mid_mask)/np.sum(non_black)*100:.1f}%, "
                  f"Dark: {np.sum(dark_mask)/np.sum(non_black)*100:.1f}%")
        
        # Step 4: Process different brightness regions with varying strength
        # For bright areas - extremely aggressive bilateral filter
        d_bright = 15  # Larger diameter
        sigma_color_bright = 150  # Much stronger smoothing
        sigma_space_bright = 150  # Much more broad effect
        bright_result = cv2.bilateralFilter(image, d_bright, sigma_color_bright, sigma_space_bright)
        
        # For mid-brightness areas - moderately aggressive bilateral filter
        d_mid = 11  # Medium diameter
        sigma_color_mid = 100  # Stronger smoothing
        sigma_space_mid = 100
        mid_result = cv2.bilateralFilter(image, d_mid, sigma_color_mid, sigma_space_mid)
        
        # For dark areas - gentle bilateral filter (keep mostly unchanged)
        d_dark = 7  # Smaller diameter
        sigma_color_dark = 60  # Gentle smoothing
        sigma_space_dark = 60
        dark_result = cv2.bilateralFilter(image, d_dark, sigma_color_dark, sigma_space_dark)
        
        # Combine bilateral results based on brightness masks
        bilateral_result = np.zeros_like(image)
        bilateral_result[bright_mask] = bright_result[bright_mask]
        bilateral_result[mid_mask] = mid_result[mid_mask]
        bilateral_result[dark_mask] = dark_result[dark_mask]
        
        # Step 5: For edge areas, use a modified LAB guided filter
        # Convert to LAB color space for guided filtering
        lab = cv2.cvtColor(pre_denoised, cv2.COLOR_BGR2LAB)
        lab_channels = cv2.split(lab)
        
        # Apply guided filter to each LAB channel - edge preserving smoothing
        radius = max(1, int(2 * (1.0 - preserve_detail_level)))  # Small radius to preserve detail
        eps = max(0.01, 0.1 * (1.0 - preserve_detail_level))     # Small epsilon for detail preservation
        
        guided_lab_channels = []
        for ch in range(3):
            # Use luminance channel as guide for all channels
            filtered = cv2.ximgproc.guidedFilter(
                guide=lab_channels[0],  # L channel as guide
                src=lab_channels[ch],
                radius=radius,
                eps=eps,
                dDepth=-1
            )
            guided_lab_channels.append(filtered)
        
        # Merge channels and convert back to BGR
        guided_lab = cv2.merge(guided_lab_channels)
        guided_result = cv2.cvtColor(guided_lab, cv2.COLOR_LAB2BGR)
        
        # Step 6: Apply NLM with different strengths based on brightness
        # Convert to YCrCb for better color handling
        ycrcb = cv2.cvtColor(pre_denoised, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Base NLM strength adjusted by detail preservation level - increase base strength
        base_h_y = max(15, int(25 * (1.0 - preserve_detail_level * 0.2)))
        
        # NLM strengths for different brightness levels - modify Y channel
        h_bright = base_h_y * 2.0  # Extremely strong for bright areas
        h_mid = base_h_y * 1.5     # Stronger for mid-brightness
        h_dark = base_h_y * 0.8    # Gentle for dark areas
        
        # Process Y channel with brightness-adaptive NLM
        # Create mask versions that work with uint8 images
        bright_mask_u8 = bright_mask.astype(np.uint8) * 255
        mid_mask_u8 = mid_mask.astype(np.uint8) * 255
        dark_mask_u8 = dark_mask.astype(np.uint8) * 255
        
        # Process each brightness region separately with different strengths
        y_bright = cv2.fastNlMeansDenoising(y, None, float(h_bright), 7, 21)
        y_mid = cv2.fastNlMeansDenoising(y, None, float(h_mid), 7, 21)
        y_dark = cv2.fastNlMeansDenoising(y, None, float(h_dark), 7, 21)
        
        # Combine Y channel results based on brightness masks
        y_denoised = np.zeros_like(y)
        y_denoised[bright_mask] = y_bright[bright_mask]
        y_denoised[mid_mask] = y_mid[mid_mask]
        y_denoised[dark_mask] = y_dark[dark_mask]
        
        # Process Cr and Cb channels - be extremely aggressive on color regardless of brightness
        h_c = max(40, int(60 * (1.0 - preserve_detail_level * 0.2)))
        
        cr_denoised = cv2.fastNlMeansDenoising(
            src=cr,
            dst=None,
            h=float(h_c),
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        cb_denoised = cv2.fastNlMeansDenoising(
            src=cb,
            dst=None,
            h=float(h_c),
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Recombine the channels
        ycrcb_denoised = cv2.merge([y_denoised, cr_denoised, cb_denoised])
        nlm_result = cv2.cvtColor(ycrcb_denoised, cv2.COLOR_YCrCb2BGR)
        
        # Step 7: Apply adaptive blending based on brightness and edges
        # Blend bilateral and NLM results with different weights for brightness levels
        non_edge_blend = np.zeros_like(image)
        
        # For bright areas, favor bilateral filter more (smoother result)
        alpha_bright = 0.7  # 70% bilateral, 30% NLM
        non_edge_blend[bright_mask] = cv2.addWeighted(
            bilateral_result[bright_mask], alpha_bright, 
            nlm_result[bright_mask], 1.0-alpha_bright, 0
        )
        
        # For mid-brightness, even blend
        alpha_mid = 0.5  # 50% bilateral, 50% NLM
        non_edge_blend[mid_mask] = cv2.addWeighted(
            bilateral_result[mid_mask], alpha_mid, 
            nlm_result[mid_mask], 1.0-alpha_mid, 0
        )
        
        # For dark areas, favor NLM more (better detail preservation)
        alpha_dark = 0.3  # 30% bilateral, 70% NLM
        non_edge_blend[dark_mask] = cv2.addWeighted(
            bilateral_result[dark_mask], alpha_dark, 
            nlm_result[dark_mask], 1.0-alpha_dark, 0
        )
        
        # Combine edge and non-edge results
        result = image.copy()
        result[edge_mask & non_black] = guided_result[edge_mask & non_black]  # Edge areas
        result[~edge_mask & non_black] = non_edge_blend[~edge_mask & non_black]  # Non-edge areas
        
        # Step 7: Final contrast enhancement to compensate for smoothing
        # This helps bring out details that may have been reduced by aggressive denoising
        alpha = 1.15  # Contrast control (1.0 means no change)
        beta = 5      # Brightness control (0 means no change)
        
        # Apply contrast enhancement
        final_result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        if self.debug:
            print("Heavy-duty adaptive edge-aware denoising completed")
            
        return final_result
    
    def bilateral_filtering(self, image, preserve_detail_level=0.7):
        """
        Apply bilateral filtering for noise reduction while preserving edges.
        
        Args:
            image: Input image (numpy array)
            preserve_detail_level: Level of detail preservation (0.0-1.0)
                                  Higher values preserve more details
                                  
        Returns:
            Denoised image
        """
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,
            image[:,:,1] > 5,
            image[:,:,2] > 5
        ))
        
        # Only process if there are non-black pixels
        if not np.any(non_black):
            return image
        
        # Calculate bilateral filter parameters based on detail preservation level
        # Lower detail preservation level = stronger filtering
        d = max(5, int(9 * (1 - preserve_detail_level)))  # Filter size (diameter)
        sigma_color = max(25, int(75 * (1 - preserve_detail_level)))  # Color space filter sigma
        sigma_space = max(25, int(75 * (1 - preserve_detail_level)))  # Coordinate space filter sigma
        
        if self.debug:
            print(f"Applying bilateral filter with parameters: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
        
        # Apply bilateral filtering to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        # Apply the filtered image only to non-black regions
        result = image.copy()
        result[non_black] = filtered[non_black]
        
        return result
        
    def denoise_image(self, image, preserve_detail_level=0.7, method='bm3d', median_first=True, median_kernel_size=3):
        """
        Main method to denoise an image. This is the recommended entry point.
        
        Args:
            image: Input image (numpy array)
            preserve_detail_level: Level of detail preservation (0.0-1.0)
                                  Higher values preserve more details
            method: Denoising method to use ('bm3d', 'nlm', 'multi_stage', 'median_only', 'adaptive_edge', 'bilateral')
            median_first: Whether to apply median filtering before BM3D (only for 'bm3d' method)
            median_kernel_size: Size of the median filter kernel (only for 'bm3d' method)
                                  
        Returns:
            Denoised image
        """
        if method == 'bm3d':
            if self.debug:
                print(f"Using BM3D denoising method {'with' if median_first else 'without'} median filtering")
            return self.bm3d_denoising(
                image, 
                preserve_detail_level, 
                apply_median_first=median_first,
                median_kernel_size=median_kernel_size
            )
        elif method == 'nlm':
            if self.debug:
                print("Using NLM denoising method")
            return self.nlm_color_channels(image, preserve_detail_level)
        elif method == 'median_only':
            if self.debug:
                print(f"Using median filter only with kernel size {median_kernel_size}x{median_kernel_size}")
            # Create a mask for non-black pixels
            non_black = np.logical_or.reduce((
                image[:,:,0] > 5,
                image[:,:,1] > 5,
                image[:,:,2] > 5
            ))
            # Apply median filter
            median_filtered = cv2.medianBlur(image, median_kernel_size)
            # Apply only to non-black regions
            result = image.copy()
            result[non_black] = median_filtered[non_black]
            return result
        elif method == 'bilateral':
            if self.debug:
                print("Using bilateral filter denoising method")
            return self.bilateral_filtering(image, preserve_detail_level)
        elif method == 'adaptive_edge':
            if self.debug:
                print("Using adaptive edge-aware denoising method")
            return self.adaptive_edge_aware_denoising(image, preserve_detail_level)
        else:
            if self.debug:
                print("Using multi-stage denoising method")
            return self.multi_stage_denoising(image, preserve_detail_level)
    
    def process_image(self, image_path, output_path, detail_level=0.7, method='bm3d', median_first=True, median_kernel_size=3):
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
            detail_level: Detail preservation level (0.0-1.0)
            method: Denoising method to use ('bm3d', 'nlm', 'multi_stage', 'median_only')
            median_first: Whether to apply median filtering before BM3D (only for 'bm3d' method)
            median_kernel_size: Size of the median filter kernel (3, 5, 7, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image at {image_path}")
                return False
                
            # Apply denoising with the selected method
            start_time = time()
            result = self.denoise_image(
                image, 
                detail_level, 
                method, 
                median_first=median_first,
                median_kernel_size=median_kernel_size
            )
            processing_time = time() - start_time
            
            # Save the result
            cv2.imwrite(output_path, result)
            
            if self.debug:
                median_info = f" with median filter ({median_kernel_size}x{median_kernel_size})" if method == 'bm3d' and median_first else ""
                print(f"Processed {image_path} -> {output_path} in {processing_time:.2f}s using {method} method{median_info}")
                
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False
    
    def process_directory(self, input_dir, output_dir, detail_level=0.7, method='bm3d', median_first=True, median_kernel_size=3):
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory to save processed images
            detail_level: Detail preservation level (0.0-1.0)
            method: Denoising method to use ('bm3d', 'nlm', 'multi_stage', 'median_only')
            median_first: Whether to apply median filtering before BM3D (only for 'bm3d' method)
            median_kernel_size: Size of the median filter kernel (3, 5, 7, etc.)
            
        Returns:
            Success rate (percentage of successfully processed images)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
        total_files = len(image_files)
        if total_files == 0:
            print(f"No image files found in {input_dir}")
            return 0
        
        # Process each image
        successful = 0
        for image_path in image_files:
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
            
            if self.process_image(
                image_path, 
                output_path, 
                detail_level, 
                method, 
                median_first=median_first,
                median_kernel_size=median_kernel_size
            ):
                successful += 1
        
        # Calculate success rate
        success_rate = (successful / total_files) * 100
        
        median_info = f" with median filter ({median_kernel_size}x{median_kernel_size})" if method == 'bm3d' and median_first else ""
        print(f"Processed {successful} of {total_files} images ({success_rate:.1f}%) using {method} method{median_info}")
        return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced image denoising.")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory.")
    parser.add_argument("--output", type=str, required=True, help="Output image or directory.")
    parser.add_argument("--detail", type=float, default=0.7, help="Detail preservation level (0.0-1.0).")
    parser.add_argument("--method", type=str, default="bm3d", 
                        choices=["bm3d", "nlm", "multi_stage", "median_only", "bilateral", "adaptive_edge"], 
                        help="Denoising method to use.")
    parser.add_argument("--median-first", dest="median_first", action="store_true",
                        help="Apply median filtering before BM3D (default: True).")
    parser.add_argument("--no-median-first", dest="median_first", action="store_false",
                        help="Don't apply median filtering before BM3D.")
    parser.add_argument("--median-kernel", type=int, default=3, choices=[3, 5, 7, 9],
                        help="Size of median filter kernel (3, 5, 7, or 9).")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    
    # Set default for median_first
    parser.set_defaults(median_first=True)
    
    args = parser.parse_args()
    
    denoiser = AdvancedDenoiser(debug=args.debug)
    
    # Check if input is a directory or a single file
    if os.path.isdir(args.input):
        denoiser.process_directory(
            args.input, 
            args.output, 
            args.detail, 
            args.method,
            median_first=args.median_first,
            median_kernel_size=args.median_kernel
        )
    else:
        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        denoiser.process_image(
            args.input, 
            args.output, 
            args.detail, 
            args.method,
            median_first=args.median_first,
            median_kernel_size=args.median_kernel
        )