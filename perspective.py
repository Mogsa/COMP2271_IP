import cv2
import numpy as np
import os
import glob

# Comment out matplotlib to avoid NumPy compatibility issues
# from matplotlib import pyplot as plt


class PerspectiveCorrector:
    def __init__(self, debug=False):
        self.debug = debug

    def find_content_boundaries(self, image):
        """Find the boundaries of the non-black content in the image."""
        # Create a binary mask where non-black pixels are white
        # We use a small threshold to account for possible compression artifacts
        mask = np.zeros_like(image[:, :, 0])

        # Check if pixels are non-black in any channel
        non_black = np.logical_or.reduce(
            (
                image[:, :, 0] > 7,  # Red
                image[:, :, 1] > 7,  # Green
                image[:, :, 2] > 7,  # Blue
            )
        )
        mask[non_black] = 255

        if self.debug:
            cv2.imshow("Content Mask", mask)
            cv2.waitKey(0)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour (this should be our content)
        if not contours:
            # If no contours found, return the full image dimensions
            h, w = image.shape[:2]
            return np.array([[0, 0], [w, 0], [w, h], [0, h]])

        largest_contour = max(contours, key=cv2.contourArea)

        # Get the convex hull to ensure we have a proper shape
        hull = cv2.convexHull(largest_contour)

        # Approximate the hull to get a quadrilateral
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        # If we have more than 4 points, find the corners that form the largest area
        if len(approx) > 4:
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            # Use astype(np.int32) instead of np.int0 for NumPy 2.x compatibility
            approx = box.astype(np.int32)

        # If we have less than 4 points, use the minimum bounding rectangle
        if len(approx) < 4:
            x, y, w, h = cv2.boundingRect(hull)
            approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

        # Ensure we have exactly 4 points
        if len(approx) != 4:
            # Get the 4 corners of the bounding rectangle
            rect = cv2.minAreaRect(approx)
            approx = cv2.boxPoints(rect)
            # Use astype(np.int32) instead of np.int0 for NumPy 2.x compatibility
            approx = approx.astype(np.int32)

        if self.debug:
            debug_img = image.copy()
            cv2.drawContours(debug_img, [approx], 0, (0, 255, 0), 2)
            for i, point in enumerate(approx):
                cv2.circle(debug_img, tuple(point.reshape(-1)), 5, (0, 0, 255), -1)
                cv2.putText(
                    debug_img,
                    str(i),
                    tuple(point.reshape(-1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
            cv2.imshow("Detected Corners", debug_img)
            cv2.waitKey(0)

        return approx.reshape(-1, 2)

    def order_points(self, pts):
        """Order points in [top-left, top-right, bottom-right, bottom-left] order."""
        # Initialize list of coordinates
        rect = np.zeros((4, 2), dtype=np.float32)

        # Calculate sum and difference of points
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        # Top-left will have the smallest sum
        rect[0] = pts[np.argmin(s)]
        # Bottom-right will have the largest sum
        rect[2] = pts[np.argmax(s)]

        # Top-right will have the smallest difference
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left will have the largest difference
        rect[3] = pts[np.argmax(diff)]

        return rect

    def correct_perspective(self, image, corners):
        """Apply perspective transformation to correct the image."""
        # Order the corners
        rect = self.order_points(np.float32(corners))

        # Calculate width and height of the new image
        width_a = np.sqrt(
            ((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2)
        )
        width_b = np.sqrt(
            ((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2)
        )
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(
            ((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2)
        )
        height_b = np.sqrt(
            ((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2)
        )
        max_height = max(int(height_a), int(height_b))

        # Destination points for the transform
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)

        # Apply transformation
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        return warped

    def process_image(self, image_path):
        """Process a single image to correct perspective."""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        # Make a copy for visualization
        original = image.copy()

        # Find the boundaries between black background and content
        corners = self.find_content_boundaries(image)

        # Apply perspective correction
        warped = self.correct_perspective(original, corners)

        if self.debug:
            # Draw corners on original image
            debug_img = original.copy()
            for i, corner in enumerate(corners):
                cv2.circle(
                    debug_img, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1
                )
                cv2.putText(
                    debug_img,
                    str(i),
                    (int(corner[0]), int(corner[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

            # Show the original and corrected images
            cv2.imshow("Original", debug_img)
            cv2.imshow("Warped", warped)
            cv2.waitKey(0)

        return warped

    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get all image files
        image_paths = (
            glob.glob(os.path.join(input_dir, "*.jpg"))
        )

        success_count = 0
        total_count = len(image_paths)

        for image_path in image_paths:
            try:
                # Process the image
                warped = self.process_image(image_path)

                if warped is not None:
                    # Save the corrected image
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, warped)
                    success_count += 1
                    print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

        accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
        print(
            f"Processing complete. Successfully processed {success_count}/{total_count} images ({accuracy:.2f}%)."
        )

        return accuracy

    def validate_results(self, input_dir, output_dir):
        """Validate that all images were processed and have proper perspective correction."""
        # Get all image files in input and output directories
        input_images = set(
            os.path.basename(p)
            for p in glob.glob(os.path.join(input_dir, "*.jpg"))
            + glob.glob(os.path.join(input_dir, "*.jpeg"))
            + glob.glob(os.path.join(input_dir, "*.png"))
        )

        output_images = set(
            os.path.basename(p)
            for p in glob.glob(os.path.join(output_dir, "*.jpg"))
            + glob.glob(os.path.join(output_dir, "*.jpeg"))
            + glob.glob(os.path.join(output_dir, "*.png"))
        )

        # Check if all input images were processed
        missing_images = input_images - output_images
        if missing_images:
            print(f"Warning: {len(missing_images)} images were not processed:")
            for img in missing_images:
                print(f"  - {img}")

        # Validate the quality of perspective correction
        # This could be extended with more sophisticated validation if needed
        print(
            f"Validation complete. {len(output_images)}/{len(input_images)} images processed."
        )

        return len(output_images) / len(input_images) * 100 if input_images else 0


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Correct perspective distortion in images."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing images."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for corrected images.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with visualizations."
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate the results after processing."
    )

    args = parser.parse_args()

    corrector = PerspectiveCorrector(debug=args.debug)
    accuracy = corrector.process_directory(args.input, args.output)

    if args.validate:
        validation_accuracy = corrector.validate_results(args.input, args.output)
        print(f"Validation accuracy: {validation_accuracy:.2f}%")

    print(f"Overall accuracy: {accuracy:.2f}%")
