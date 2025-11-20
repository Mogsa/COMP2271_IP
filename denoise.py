import cv2
import numpy as np
import os
import glob
from skimage import exposure


class ImageDenoiser:
    def __init__(self, debug=False):
        self.debug = debug

    def bilateral_filter_denoising(self, image):
        """
        Apply bilateral filter for denoising while preserving edges.

        Args:
            image: Input image (numpy array)

        Returns:
            Denoised image
        """
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce(
            (image[:, :, 0] > 5, image[:, :, 1] > 5, image[:, :, 2] > 5)
        )

        # Only process if there are non-black pixels
        if not np.any(non_black):
            return image

        # Apply bilateral filtering to reduce noise while preserving edges
        # Parameters: src, d, sigmaColor, sigmaSpace
        filtered = cv2.bilateralFilter(image, 9, 75, 75)

        # Apply the filtered image only to non-black regions
        result = image.copy()
        result[non_black] = filtered[non_black]

        return result

    def nlm_denoising(self, image):
        """
        Apply Non-Local Means denoising.
        This is more powerful than bilateral filtering but slower.

        Args:
            image: Input image (numpy array)

        Returns:
            Denoised image
        """
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce(
            (image[:, :, 0] > 5, image[:, :, 1] > 5, image[:, :, 2] > 5)
        )

        # Only process if there are non-black pixels
        if not np.any(non_black):
            return image

        # Apply Non-Local Means denoising
        # Parameters: src, h, templateWindowSize, searchWindowSize
        filtered = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Apply the filtered image only to non-black regions
        result = image.copy()
        result[non_black] = filtered[non_black]

        return result

    def auto_denoise(self, image):
        """
        Automatically select the best denoising algorithm based on image content.

        Args:
            image: Input image (numpy array)

        Returns:
            Denoised image
        """
        # Simple heuristic - use bilateral filter for most cases as it's faster
        # and works well for the driving images dataset
        return self.bilateral_filter_denoising(image)

    def detail_preserving_denoise(self, image, strength=0.7):
        """
        Apply a detail-preserving denoising algorithm that combines
        bilateral filtering with edge enhancement.

        Args:
            image: Input image (numpy array)
            strength: Denoising strength (0.0-1.0)

        Returns:
            Denoised image
        """
        # Create a mask for non-black pixels
        non_black = np.logical_or.reduce(
            (image[:, :, 0] > 5, image[:, :, 1] > 5, image[:, :, 2] > 5)
        )

        # Only process if there are non-black pixels
        if not np.any(non_black):
            return image

        # Step 1: Apply initial gentle denoising with small-kernel bilateral filter
        # This preserves edges better than standard Gaussian filtering
        guided = cv2.bilateralFilter(image, 5, 75, 75)

        # Step 2: Apply bilateral filtering for stronger noise reduction
        # but still preserving edges
        d = int(9 * strength)  # Filter size based on strength
        sigma_color = 75 * strength
        sigma_space = 75 * strength
        bilateral = cv2.bilateralFilter(guided, d, sigma_color, sigma_space)

        # Step 3: Enhance edges to recover details
        # Create a sharpen kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(bilateral, -1, kernel)

        # Mix the bilateral and sharpened result based on strength
        alpha = 0.2 + (0.3 * strength)  # How much sharpening to apply
        enhanced = cv2.addWeighted(bilateral, 1 - alpha, sharpen, alpha, 0)

        # Apply contrast enhancement to the result
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge back the channels
        merged = cv2.merge([cl, a, b])
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Apply the enhanced image only to non-black regions
        final = image.copy()
        final[non_black] = result[non_black]

        if self.debug:
            # Show comparison of original vs denoised
            comparison = np.hstack((image, final))
            cv2.imshow("Original vs Denoised", comparison)
            cv2.waitKey(0)

        return final

    def denoise_image(self, image, preserve_detail_level=0.7):
        """
        Apply denoising to an image with the specified detail preservation level.

        Args:
            image: Input image (numpy array)
            preserve_detail_level: Level of detail preservation (0.0-1.0)
                                  Higher values preserve more details

        Returns:
            Denoised image
        """
        # For high detail preservation levels, use the detail-preserving method
        if preserve_detail_level > 0.5:
            return self.detail_preserving_denoise(image, preserve_detail_level)
        # For lower detail preservation (stronger denoising), use NLM
        elif preserve_detail_level > 0.3:
            return self.nlm_denoising(image)
        # For very low detail preservation, use bilateral filter
        else:
            return self.bilateral_filter_denoising(image)

    def process_directory(self, input_dir, output_dir, preserve_detail_level=0.7):
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save denoised images
            preserve_detail_level: Level of detail preservation (0.0-1.0)

        Returns:
            Processing accuracy (percentage of successfully processed images)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get all image files
        image_paths = (
            glob.glob(os.path.join(input_dir, "*.jpg"))
            + glob.glob(os.path.join(input_dir, "*.jpeg"))
            + glob.glob(os.path.join(input_dir, "*.png"))
        )

        success_count = 0
        total_count = len(image_paths)

        for image_path in image_paths:
            try:
                # Read the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Could not read image at {image_path}")
                    continue

                # Apply denoising
                denoised = self.denoise_image(image, preserve_detail_level)

                # Save the denoised image
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, denoised)

                success_count += 1
                print(f"Denoised: {filename}")

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

        accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
        print(
            f"Processing complete. Successfully processed {success_count}/{total_count} images ({accuracy:.2f}%)."
        )

        return accuracy


# Function to maintain backward compatibility
def denoise_preserve_details(image_path, output_path):
    """
    Legacy function that uses the class-based implementation.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Create a denoiser and process the image
    denoiser = ImageDenoiser(debug=False)
    result = denoiser.detail_preserving_denoise(img, strength=0.7)

    # Enhance contrast
    p2, p98 = np.percentile(result, (2, 98))
    contrast_enhanced = exposure.rescale_intensity(result, in_range=(p2, p98))

    # Save the result
    cv2.imwrite(output_path, contrast_enhanced)
    return contrast_enhanced


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detail-preserving image denoising.")
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing images."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for denoised images.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with visualizations."
    )
    parser.add_argument(
        "--detail", type=float, default=0.7, help="Detail preservation level (0.0-1.0)."
    )

    args = parser.parse_args()

    denoiser = ImageDenoiser(debug=args.debug)
    accuracy = denoiser.process_directory(args.input, args.output, args.detail)

    print(f"Overall accuracy: {accuracy:.2f}%")
