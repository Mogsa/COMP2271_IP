import cv2
import os
from perspective import PerspectiveCorrector

# Create a directory for debug images
debug_dir = "debug_perspective_output"
os.makedirs(debug_dir, exist_ok=True)

# Initialize the perspective corrector with debug mode on
corrector = PerspectiveCorrector(debug=True)

# Process a single image for debugging
image_path = "/Users/morgan/Documents/GitHub/COMP2271_IP/driving_images/im001-snow.jpg"
print(f"Processing {image_path}")

# Process the image and save the result
warped = corrector.process_image(image_path)
if warped is not None:
    output_path = os.path.join(debug_dir, "im001-snow-perspective.jpg")
    cv2.imwrite(output_path, warped)
    print(f"Saved result to {output_path}")

print("Done debugging perspective correction!")