import cv2
import numpy as np
import os
import glob
# Comment out matplotlib to avoid compatibility issues
# import matplotlib.pyplot as plt
from time import time

class ImageDenoiser:
    def __init__(self, debug=False):
        self.debug = debug
        
    def _get_non_black_mask(self, image):
        """Create a mask for non-black pixels."""
        # Check if pixels are non-black in any channel
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,  # Blue
            image[:,:,1] > 5,  # Green
            image[:,:,2] > 5   # Red
        ))
        
        # Create a mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[non_black] = 255
        
        # Optional: Use morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def estimate_noise_level(self, image):
        """Estimate the noise level of the image."""
        # Get non-black mask
        mask = self._get_non_black_mask(image)
        if np.sum(mask) == 0:
            return 0  # No non-black pixels
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian filter
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        
        # Calculate standard deviation in the non-black regions
        # This is a simple measure of the noise level
        std_dev = np.std(lap[mask > 0])
        
        return std_dev
    
    def non_local_means_denoising(self, image, h_luminance=15, h_color=15, template_size=7, search_size=21):
        """Apply Non-Local Means denoising to non-black regions of the image."""
        # Get non-black mask
        mask = self._get_non_black_mask(image)
        if np.sum(mask) == 0:
            return image.copy()  # No non-black pixels
            
        # Noise level estimation can be used to adapt parameters
        noise_level = self.estimate_noise_level(image)
        
        # Adjust h parameter based on estimated noise level
        # For heavy noise, increase h
        if noise_level > 15:
            h_luminance = min(20, h_luminance * 1.5)
            h_color = min(20, h_color * 1.5)
        elif noise_level < 5:
            h_luminance = max(3, h_luminance * 0.7)
            h_color = max(3, h_color * 0.7)
            
        # Apply Non-Local Means denoising
        start_time = time()
        denoised = cv2.fastNlMeansDenoisingColored(
            image, 
            None, 
            h_luminance,  # Filter strength for luminance
            h_color,      # Filter strength for color
            template_size,  # Template window size
            search_size     # Search window size
        )
        processing_time = time() - start_time
        
        # Preserve the original black background
        result = image.copy()
        result[mask > 0] = denoised[mask > 0]
        
        if self.debug:
            print(f"Non-Local Means denoising - Time: {processing_time:.2f}s, Noise level: {noise_level:.2f}")
            
        return result
    
    def bilateral_filter_denoising(self, image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filtering to non-black regions of the image."""
        # Get non-black mask
        mask = self._get_non_black_mask(image)
        if np.sum(mask) == 0:
            return image.copy()  # No non-black pixels
            
        # Noise level estimation
        noise_level = self.estimate_noise_level(image)
        
        # Adjust parameters based on noise level
        if noise_level > 15:
            sigma_color = min(100, sigma_color * 1.3)
            sigma_space = min(100, sigma_space * 1.3)
        elif noise_level < 5:
            sigma_color = max(30, sigma_color * 0.7)
            sigma_space = max(30, sigma_space * 0.7)
            
        # Apply bilateral filter
        start_time = time()
        denoised = cv2.bilateralFilter(
            image, 
            d,           # Diameter of pixel neighborhood
            sigma_color, # Filter sigma in the color space
            sigma_space  # Filter sigma in the coordinate space
        )
        processing_time = time() - start_time
        
        # Preserve the original black background
        result = image.copy()
        result[mask > 0] = denoised[mask > 0]
        
        if self.debug:
            print(f"Bilateral filter denoising - Time: {processing_time:.2f}s, Noise level: {noise_level:.2f}")
            
        return result
    
    def wavelet_denoising(self, image, sigma=10, wavelet='db4', levels=2):
        """Apply wavelet-based denoising."""
        try:
            import pywt
            from skimage.restoration import denoise_wavelet
        except ImportError:
            print("Wavelet denoising requires PyWavelets and scikit-image libraries.")
            print("Install with: pip install PyWavelets scikit-image")
            return image.copy()
            
        # Get non-black mask
        mask = self._get_non_black_mask(image)
        if np.sum(mask) == 0:
            return image.copy()  # No non-black pixels
            
        # Noise level estimation
        noise_level = self.estimate_noise_level(image)
        
        # Adjust sigma based on noise level
        if noise_level > 15:
            sigma = min(25, sigma * 1.5)
        elif noise_level < 5:
            sigma = max(2, sigma * 0.7)
            
        # Convert to [0, 1] range for wavelet denoising
        image_norm = image.astype(np.float32) / 255.0
        
        # Apply wavelet denoising
        start_time = time()
        denoised = denoise_wavelet(
            image_norm,
            wavelet=wavelet,
            mode='soft',
            wavelet_levels=levels,
            sigma=sigma/255.0,
            multichannel=True,
            convert2ycbcr=True  # Better color handling
        )
        
        # Convert back to [0, 255] range
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
        processing_time = time() - start_time
        
        # Preserve the original black background
        result = image.copy()
        result[mask > 0] = denoised[mask > 0]
        
        if self.debug:
            print(f"Wavelet denoising - Time: {processing_time:.2f}s, Noise level: {noise_level:.2f}")
            
        return result
    
    def guided_filter_denoising(self, image, radius=8, eps=100):
        """Apply guided filter for edge-preserving denoising."""
        # Get non-black mask
        mask = self._get_non_black_mask(image)
        if np.sum(mask) == 0:
            return image.copy()  # No non-black pixels
            
        # Noise level estimation
        noise_level = self.estimate_noise_level(image)
        
        # Adjust parameters based on noise level
        if noise_level > 15:
            eps = min(200, eps * 1.3)
        elif noise_level < 5:
            eps = max(50, eps * 0.7)
            
        # Apply guided filter
        start_time = time()
        denoised = cv2.ximgproc.guidedFilter(
            image,   # Guidance image (same as input)
            image,   # Filtering input (same as guidance)
            radius,  # Radius of filtering kernel
            eps      # Regularization term
        )
        processing_time = time() - start_time
        
        # Preserve the original black background
        result = image.copy()
        result[mask > 0] = denoised[mask > 0]
        
        if self.debug:
            print(f"Guided filter denoising - Time: {processing_time:.2f}s, Noise level: {noise_level:.2f}")
            
        return result
    
    def denoise_tv_bregman(self, image, weight=30, max_iter=100):
        """Apply Total Variation Bregman denoising."""
        try:
            from skimage.restoration import denoise_tv_bregman
        except ImportError:
            print("TV Bregman denoising requires scikit-image library.")
            print("Install with: pip install scikit-image")
            return image.copy()
            
        # Get non-black mask
        mask = self._get_non_black_mask(image)
        if np.sum(mask) == 0:
            return image.copy()  # No non-black pixels
            
        # Noise level estimation
        noise_level = self.estimate_noise_level(image)
        
        # Adjust weight based on noise level
        if noise_level > 15:
            weight = min(50, weight * 1.3)
        elif noise_level < 5:
            weight = max(10, weight * 0.7)
            
        # Convert to [0, 1] range
        image_norm = image.astype(np.float32) / 255.0
        
        # Process each channel separately
        channels = cv2.split(image_norm)
        denoised_channels = []
        
        start_time = time()
        for channel in channels:
            # Create a mask for this channel
            channel_mask = np.zeros_like(channel)
            channel_mask[mask > 0] = 1
            
            # Apply TV Bregman denoising
            denoised_channel = denoise_tv_bregman(
                channel,
                weight=weight,
                max_iter=max_iter,
                eps=1e-5
            )
            
            # Preserve zeros
            denoised_channel = denoised_channel * channel_mask
            denoised_channels.append(denoised_channel)
            
        # Merge channels
        denoised = cv2.merge(denoised_channels)
        
        # Convert back to [0, 255] range
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
        processing_time = time() - start_time
        
        # Preserve the original black background
        result = image.copy()
        result[mask > 0] = denoised[mask > 0]
        
        if self.debug:
            print(f"TV Bregman denoising - Time: {processing_time:.2f}s, Noise level: {noise_level:.2f}")
            
        return result
    
    def multi_method_denoising(self, image, methods=None):
        """Apply multiple denoising methods and combine results for better output."""
        # Default methods if not specified
        if methods is None:
            methods = ['nlm', 'bilateral']
            
        results = []
        
        # Apply specified denoising methods
        for method in methods:
            if method == 'nlm':
                # Non-local means
                denoised = self.non_local_means_denoising(image)
                results.append(denoised)
                
            elif method == 'bilateral':
                # Bilateral filter
                denoised = self.bilateral_filter_denoising(image)
                results.append(denoised)
                
            elif method == 'wavelet':
                # Wavelet-based denoising
                denoised = self.wavelet_denoising(image)
                results.append(denoised)
                
            elif method == 'guided':
                # Guided filter
                denoised = self.guided_filter_denoising(image)
                results.append(denoised)
                
            elif method == 'tv':
                # Total Variation denoising
                denoised = self.denoise_tv_bregman(image)
                results.append(denoised)
        
        # If no methods were applied successfully, return original image
        if not results:
            return image.copy()
            
        # Average the results (this is a simple way to combine methods)
        # For better results, more sophisticated fusion could be implemented
        combined = np.zeros_like(image, dtype=np.float32)
        for result in results:
            combined += result.astype(np.float32)
        combined /= len(results)
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        # Preserve the original black background
        mask = self._get_non_black_mask(image)
        final = image.copy()
        final[mask > 0] = combined[mask > 0]
        
        return final
    
    def process_image(self, image_path, method='auto', output_path=None, show_comparison=False, _image=None):
        """Process a single image with the specified denoising method.
        
        Args:
            image_path: Path to the image file (can be None if _image is provided)
            method: Denoising method to use ('auto', 'nlm', 'bilateral', etc.)
            output_path: Path to save the output image (optional)
            show_comparison: Whether to display a comparison of before/after
            _image: Pre-loaded image data (bypasses loading from image_path)
        """
        # If image data is provided directly, use it instead of loading from path
        if _image is not None:
            image = _image
        else:
            # Read the image from path
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image at {image_path}")
                return None
            
        # Make a copy of the original for comparison
        original = image.copy()
        
        # Estimate noise level for automatic method selection
        noise_level = self.estimate_noise_level(image)
        
        # Select method based on noise level if 'auto' is specified
        if method == 'auto':
            if noise_level > 15:
                # Heavy noise: combine multiple methods
                method = 'multi'
            elif noise_level > 8:
                # Medium noise: non-local means works well
                method = 'nlm'
            else:
                # Light noise: bilateral filter is faster and good enough
                method = 'bilateral'
                
        # Apply the selected denoising method
        if method == 'nlm':
            denoised = self.non_local_means_denoising(image)
        elif method == 'bilateral':
            denoised = self.bilateral_filter_denoising(image)
        elif method == 'wavelet':
            denoised = self.wavelet_denoising(image)
        elif method == 'guided':
            denoised = self.guided_filter_denoising(image)
        elif method == 'tv':
            denoised = self.denoise_tv_bregman(image)
        elif method == 'multi':
            # For heavy noise, combine multiple methods
            denoised = self.multi_method_denoising(image, methods=['nlm', 'wavelet'])
        else:
            print(f"Unknown method: {method}, using non-local means")
            denoised = self.non_local_means_denoising(image)
            
        # Save the result if output path is specified
        if output_path:
            cv2.imwrite(output_path, denoised)
            
        # Show comparison if requested - disabled for compatibility
        if show_comparison or self.debug:
            # Skip visualization to avoid matplotlib compatibility issues
            if self.debug:
                print(f"Denoising completed with method: {method}")
                print(f"Noise level estimate: {noise_level:.2f}")
            # plt.figure(figsize=(12, 6))
            # plt.subplot(121)
            # plt.title('Original Image')
            # plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.subplot(122)
            # plt.title(f'Denoised Image (Method: {method})')
            # plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()
            
        return denoised
    
    def process_directory(self, input_dir, output_dir, method='auto'):
        """Process all images in a directory."""
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
        
        for image_path in image_paths:
            try:
                # Generate output path
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, filename)
                
                # Process the image
                self.process_image(image_path, method=method, output_path=output_path)
                
                success_count += 1
                print(f"Processed {success_count}/{total_count}: {filename}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                
        # Report success rate
        accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
        print(f"Processing complete. Successfully processed {success_count}/{total_count} images ({accuracy:.2f}%).")
        
        return accuracy
    
    def compare_methods(self, image_path, methods=None):
        """Compare different denoising methods on a single image."""
        # Default methods if not specified
        if methods is None:
            methods = ['original', 'nlm', 'bilateral', 'wavelet', 'guided', 'tv', 'multi']
            
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return
            
        print("Comparing denoising methods (visualization disabled for compatibility)")
        
        # Process with each method and print timing info
        for method in methods:
            if method == 'original':
                print(f"- Original image")
                continue
                
            # Apply denoising method
            start_time = time()
            
            if method == 'nlm':
                self.non_local_means_denoising(image)
                title = 'Non-Local Means'
            elif method == 'bilateral':
                self.bilateral_filter_denoising(image)
                title = 'Bilateral Filter'
            elif method == 'wavelet':
                self.wavelet_denoising(image)
                title = 'Wavelet'
            elif method == 'guided':
                try:
                    self.guided_filter_denoising(image)
                    title = 'Guided Filter'
                except:
                    print(f"- Guided Filter: Not available in this OpenCV installation")
                    continue
            elif method == 'tv':
                self.denoise_tv_bregman(image)
                title = 'TV Bregman'
            elif method == 'multi':
                self.multi_method_denoising(image)
                title = 'Combined Methods'
            else:
                print(f"- Unknown method: {method}")
                continue
                
            processing_time = time() - start_time
            print(f"- {title}: {processing_time:.2f}s")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced image denoising preserving black backgrounds.")
    parser.add_argument("--input", type=str, help="Input image path or directory.")
    parser.add_argument("--output", type=str, help="Output image path or directory.")
    parser.add_argument("--method", type=str, default="auto", 
                        choices=["auto", "nlm", "bilateral", "wavelet", "guided", "tv", "multi"],
                        help="Denoising method to use.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with visualizations.")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare different denoising methods on the input image.")
    
    args = parser.parse_args()
    
    denoiser = ImageDenoiser(debug=args.debug)
    
    if args.compare and args.input:
        # Compare different methods
        denoiser.compare_methods(args.input)
    elif os.path.isdir(args.input):
        # Process directory
        if not args.output:
            print("Error: Output directory must be specified when processing a directory.")
        else:
            denoiser.process_directory(args.input, args.output, method=args.method)
    elif args.input and args.output:
        # Process single image
        denoiser.process_image(args.input, method=args.method, output_path=args.output, show_comparison=True)
    else:
        print("Error: Please provide input and output paths.")
        parser.print_help()