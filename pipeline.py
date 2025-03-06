import cv2
import numpy as np
import os
import glob
import argparse
from perspective import PerspectiveCorrector
from color import EnhancedColorCorrector
from denoise import ImageDenoiser
from inpaint import PatchMatchInpainting

class ImagePipeline:
    def __init__(self, debug=False):
        self.debug = debug
        self.perspective_corrector = PerspectiveCorrector(debug=debug)
        self.color_corrector = EnhancedColorCorrector(debug=debug)
        self.denoiser = ImageDenoiser(debug=debug)
        self.inpainter = PatchMatchInpainting(debug=debug)
        
    def process_image(self, image_path, perspective_output_dir=None, color_output_dir=None, denoise_output_dir=None):
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            perspective_output_dir: Optional directory to save intermediate perspective-corrected images
            color_output_dir: Optional directory to save intermediate color-corrected images
            denoise_output_dir: Optional directory to save intermediate denoised images
            
        Returns:
            Fully processed image
        """
        # Step 1: Perspective correction
        print(f"Applying perspective correction to {os.path.basename(image_path)}...")
        perspective_corrected = self.perspective_corrector.process_image(image_path)
        
        if perspective_corrected is None:
            print(f"Error: Perspective correction failed for {image_path}")
            return None
        
        # Optionally save the intermediate result
        if perspective_output_dir:
            os.makedirs(perspective_output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            output_path = os.path.join(perspective_output_dir, filename)
            cv2.imwrite(output_path, perspective_corrected)
        
        # Step 2: Color, contrast, and brightness enhancement
        print(f"Applying color enhancement to {os.path.basename(image_path)}...")
        color_corrected = self.color_corrector.auto_correct(perspective_corrected)
        
        # Optionally save the intermediate result
        if color_output_dir:
            os.makedirs(color_output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            output_path = os.path.join(color_output_dir, filename)
            cv2.imwrite(output_path, color_corrected)
        
        # Step 3: Detail-preserving noise removal
        print(f"Applying adaptive denoising to {os.path.basename(image_path)}...")
        # Use bilateral filtering which is most compatible
        denoised = self.denoiser.bilateral_filter_denoising(color_corrected)
        
        # Optionally save the intermediate result
        if denoise_output_dir:
            os.makedirs(denoise_output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            output_path = os.path.join(denoise_output_dir, filename)
            cv2.imwrite(output_path, denoised)
        
        # Step 4: Inpainting black spots (such as the black circle in the top-right corner)
        print(f"Applying inpainting to {os.path.basename(image_path)}...")
        # Use simpler approach for inpainting to avoid scipy compatibility issues
        try:
            final_result = self.inpainter.inpaint(denoised, max_iterations=100)
        except Exception as e:
            print(f"Inpainting failed: {str(e)}, returning denoised result instead.")
            final_result = denoised
        
        return final_result
    
    def process_directory(self, input_dir, output_dir, save_intermediate=False):
        """
        Process all images in a directory through the complete pipeline.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save fully processed images
            save_intermediate: Whether to save intermediate results for each processing step
            
        Returns:
            Processing accuracy (percentage of successfully processed images)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create intermediate directories if needed
        perspective_dir = None
        color_dir = None
        denoise_dir = None
        
        if save_intermediate:
            perspective_dir = os.path.join(os.path.dirname(output_dir), "perspective_corrected")
            color_dir = os.path.join(os.path.dirname(output_dir), "color_corrected")
            denoise_dir = os.path.join(os.path.dirname(output_dir), "denoised")
            os.makedirs(perspective_dir, exist_ok=True)
            os.makedirs(color_dir, exist_ok=True)
            os.makedirs(denoise_dir, exist_ok=True)
        
        # Get all image files
        image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                     glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(input_dir, "*.png"))
        
        success_count = 0
        total_count = len(image_paths)
        
        for image_path in image_paths:
            try:
                # Process the image through the full pipeline
                final_image = self.process_image(image_path, perspective_dir, color_dir, denoise_dir)
                
                if final_image is not None:
                    # Save the fully processed image
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, final_image)
                    success_count += 1
                    print(f"Fully processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
        print(f"Pipeline complete. Successfully processed {success_count}/{total_count} images ({accuracy:.2f}%).")
        
        return accuracy

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing pipeline for perspective and color correction.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for fully processed images.")
    parser.add_argument("--intermediate", action="store_true", help="Save intermediate perspective-corrected images.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with visualizations.")
    
    args = parser.parse_args()
    
    pipeline = ImagePipeline(debug=args.debug)
    accuracy = pipeline.process_directory(args.input, args.output, args.intermediate)
    
    print(f"Overall pipeline accuracy: {accuracy:.2f}%")