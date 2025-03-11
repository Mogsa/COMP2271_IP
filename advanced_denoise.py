import cv2
import numpy as np
import os
import glob
import argparse
from time import time

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
    
    def nlm_color_channels(self, image, preserve_detail_level=0.7):
        """
        Apply Non-Local Means denoising to each color channel separately.
        
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
        
        # Adjust NLM parameters based on detail preservation level
        h = int(10 * (1.0 - preserve_detail_level * 0.5))  # Lower h for more detail preservation
        h = max(3, min(h, 15))  # Keep h in reasonable range
        
        template_window_size = 7
        search_window_size = 21
        
        # Process each channel separately
        result = image.copy()
        
        if self.debug:
            print(f"Applying NLM to color channels with h={h}")
        
        for channel in range(3):
            # Extract channel
            img_channel = image[:,:,channel].copy()
            
            # Create a mask for this channel (where pixels are non-zero and non-black)
            channel_mask = np.logical_and(img_channel > 5, non_black)
            
            # Only process if there are pixels to process
            if np.any(channel_mask):
                # Apply NLM denoising to this channel
                denoised_channel = cv2.fastNlMeansDenoising(
                    src=img_channel,
                    dst=None,
                    h=float(h),
                    templateWindowSize=template_window_size,
                    searchWindowSize=search_window_size
                )
                
                # Update only non-black pixels in the result
                result[:,:,channel][channel_mask] = denoised_channel[channel_mask]
        
        return result

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
    
    def denoise_image(self, image, preserve_detail_level=0.7):
        """
        Main method to denoise an image. This is the recommended entry point.
        
        Args:
            image: Input image (numpy array)
            preserve_detail_level: Level of detail preservation (0.0-1.0)
                                  Higher values preserve more details
                                  
        Returns:
            Denoised image
        """
        # Use the multi-stage denoising method
        return self.multi_stage_denoising(image, preserve_detail_level)
    
    def process_image(self, image_path, output_path, detail_level=0.7):
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
            detail_level: Detail preservation level (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image at {image_path}")
                return False
                
            # Apply multi-stage denoising
            start_time = time()
            result = self.multi_stage_denoising(image, detail_level)
            processing_time = time() - start_time
            
            # Save the result
            cv2.imwrite(output_path, result)
            
            if self.debug:
                print(f"Processed {image_path} -> {output_path} in {processing_time:.2f}s")
                
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False
    
    def process_directory(self, input_dir, output_dir, detail_level=0.7):
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory to save processed images
            detail_level: Detail preservation level (0.0-1.0)
            
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
            
            if self.process_image(image_path, output_path, detail_level):
                successful += 1
        
        # Calculate success rate
        success_rate = (successful / total_files) * 100
        
        print(f"Processed {successful} of {total_files} images ({success_rate:.1f}%)")
        return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced image denoising.")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory.")
    parser.add_argument("--output", type=str, required=True, help="Output image or directory.")
    parser.add_argument("--detail", type=float, default=0.7, help="Detail preservation level (0.0-1.0).")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    
    args = parser.parse_args()
    
    denoiser = AdvancedDenoiser(debug=args.debug)
    
    # Check if input is a directory or a single file
    if os.path.isdir(args.input):
        denoiser.process_directory(args.input, args.output, args.detail)
    else:
        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        denoiser.process_image(args.input, args.output, args.detail)