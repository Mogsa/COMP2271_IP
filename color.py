import cv2
import numpy as np
import os
import glob
# Comment out matplotlib to avoid NumPy compatibility issues
# import matplotlib.pyplot as plt

class EnhancedColorCorrector:
    def __init__(self, debug=False):
        self.debug = debug
    
    def analyze_image(self, image):
        """Analyze the image and determine the correction needed."""
        # Only consider non-black pixels
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,  # Blue
            image[:,:,1] > 5,  # Green
            image[:,:,2] > 5   # Blue
        ))
        
        metrics = {}
        
        if np.any(non_black):
            # Get non-black regions
            b_vals = image[:,:,0][non_black]
            g_vals = image[:,:,1][non_black]
            r_vals = image[:,:,2][non_black]
            
            # Calculate metrics
            metrics['mean_b'] = np.mean(b_vals)
            metrics['mean_g'] = np.mean(g_vals)
            metrics['mean_r'] = np.mean(r_vals)
            
            metrics['std_b'] = np.std(b_vals)
            metrics['std_g'] = np.std(g_vals)
            metrics['std_r'] = np.std(r_vals)
            
            # Calculate average luminance
            metrics['luminance'] = 0.299 * metrics['mean_r'] + 0.587 * metrics['mean_g'] + 0.114 * metrics['mean_b']
            
            # Determine if image has color cast
            channel_means = [metrics['mean_b'], metrics['mean_g'], metrics['mean_r']]
            max_idx = np.argmax(channel_means)
            min_idx = np.argmin(channel_means)
            
            color_names = ['blue', 'green', 'red']
            metrics['dominant_color'] = color_names[max_idx]
            metrics['recessive_color'] = color_names[min_idx]
            
            # Calculate color imbalance
            max_diff = max([
                abs(metrics['mean_r'] - metrics['mean_g']),
                abs(metrics['mean_r'] - metrics['mean_b']),
                abs(metrics['mean_g'] - metrics['mean_b'])
            ])
            
            metrics['color_imbalance'] = max_diff
            
            # Determine if there's a significant color cast
            if max_diff > 15:
                metrics['has_color_cast'] = True
            else:
                metrics['has_color_cast'] = False
                
            # Check contrast
            metrics['contrast'] = max(metrics['std_r'], metrics['std_g'], metrics['std_b'])
            
            # Get histogram information
            for i, channel in enumerate(['b', 'g', 'r']):
                # Calculate histogram
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                
                # Find significant range (5% to 95% percentile)
                cumsum = np.cumsum(hist)
                if cumsum[-1] > 0:  # Avoid division by zero
                    cumsum = cumsum / cumsum[-1]
                    
                    # Find 5th and 95th percentiles
                    idx_5 = np.searchsorted(cumsum, 0.05)
                    idx_95 = np.searchsorted(cumsum, 0.95)
                    
                    metrics[f'{channel}_range'] = idx_95 - idx_5
                else:
                    metrics[f'{channel}_range'] = 0
        else:
            # Default metrics for completely black images
            metrics = {
                'mean_b': 0, 'mean_g': 0, 'mean_r': 0,
                'std_b': 0, 'std_g': 0, 'std_r': 0,
                'luminance': 0, 'dominant_color': 'none',
                'recessive_color': 'none', 'color_imbalance': 0,
                'has_color_cast': False, 'contrast': 0,
                'b_range': 0, 'g_range': 0, 'r_range': 0
            }
        
        return metrics
    
    def auto_white_balance(self, image):
        """Apply automatic white balance to correct color casts."""
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,
            image[:,:,1] > 5,
            image[:,:,2] > 5
        ))
        
        # Create a mask array
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[non_black] = 255
        
        # Apply white balance only to non-black pixels
        result = image.copy()
        
        if np.any(non_black):
            # Split the image into channels
            b, g, r = cv2.split(image)
            
            # Calculate the average values for non-black regions
            b_avg = np.mean(b[non_black])
            g_avg = np.mean(g[non_black])
            r_avg = np.mean(r[non_black])
            
            # Calculate the overall average
            avg = (b_avg + g_avg + r_avg) / 3
            
            # Calculate scaling factors (avoid division by zero)
            b_scale = avg / b_avg if b_avg > 0 else 1
            g_scale = avg / g_avg if g_avg > 0 else 1
            r_scale = avg / r_avg if r_avg > 0 else 1
            
            # Apply scaling
            b_balanced = np.clip(b * b_scale, 0, 255).astype(np.uint8)
            g_balanced = np.clip(g * g_scale, 0, 255).astype(np.uint8)
            r_balanced = np.clip(r * r_scale, 0, 255).astype(np.uint8)
            
            # Merge the balanced channels
            balanced = cv2.merge([b_balanced, g_balanced, r_balanced])
            
            # Apply the balanced image only to non-black regions
            result = image.copy()
            result[non_black] = balanced[non_black]
        
        return result
    
    def correct_brightness(self, image, metrics):
        """Adjust brightness based on image analysis."""
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,
            image[:,:,1] > 5,
            image[:,:,2] > 5
        ))
        
        # Only process if there are non-black pixels
        if not np.any(non_black):
            return image
        
        result = image.copy()
        
        # Determine target brightness (standard middle gray)
        target_brightness = 128
        current_brightness = metrics['luminance']
        
        # Calculate adjustment factor
        if current_brightness < 40:
            # Very dark image: stronger correction
            gamma = 0.5
        elif current_brightness < 90:
            # Dark image: moderate correction
            gamma = 0.7
        elif current_brightness > 180:
            # Very bright image: stronger correction
            gamma = 1.5
        elif current_brightness > 140:
            # Bright image: moderate correction
            gamma = 1.2
        else:
            # Image brightness is good
            return image
        
        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 if i > 0 else 0
            for i in np.arange(0, 256)
        ]).astype(np.uint8)
        
        # Apply the correction only to non-black pixels
        result = cv2.LUT(image, table)
        
        return result
    
    def enhance_contrast(self, image, metrics):
        """Enhance contrast while preserving colors."""
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce((
            image[:,:,0] > 5,
            image[:,:,1] > 5,
            image[:,:,2] > 5
        ))
        
        # Only process if there are non-black pixels and contrast is low
        if not np.any(non_black) or metrics['contrast'] > 40:
            return image
        
        # Convert to LAB color space (L=lightness, A=green-red, B=blue-yellow)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge back the channels
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply the enhanced image only to non-black regions
        result = image.copy()
        result[non_black] = enhanced[non_black]
        
        return result
    
    def auto_correct(self, image):
        """Apply automatic color correction based on image analysis."""
        # Analyze the image first
        metrics = self.analyze_image(image)
        
        # Original image for comparison
        original = image.copy()
        
        # Step 1: White Balance Correction
        if metrics['has_color_cast']:
            image = self.auto_white_balance(image)
        
        # Step 2: Brightness Correction
        image = self.correct_brightness(image, metrics)
        
        # Step 3: Contrast Enhancement
        # Check if any channel has low dynamic range
        low_range = any([
            metrics['r_range'] < 100,
            metrics['g_range'] < 100,
            metrics['b_range'] < 100
        ])
        
        if low_range or metrics['contrast'] < 40:
            image = self.enhance_contrast(image, metrics)
        
        
        return image
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) 
        
        success_count = 0
        total_count = len(image_paths)
        
        for image_path in image_paths:
            try:
                # Read the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Could not read image at {image_path}")
                    continue
                
                # Apply color correction
                corrected = self.auto_correct(image)
                
                # Save the corrected image
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, corrected)
                
                success_count += 1
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
        print(f"Processing complete. Successfully processed {success_count}/{total_count} images ({accuracy:.2f}%).")
        
        return accuracy


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced automatic color correction without reference learning.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for corrected images.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with visualizations.")
    
    args = parser.parse_args()
    
    corrector = EnhancedColorCorrector(debug=args.debug)
    accuracy = corrector.process_directory(args.input, args.output)
    
    print(f"Overall accuracy: {accuracy:.2f}%")