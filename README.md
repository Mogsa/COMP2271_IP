# Image Processing Pipeline for Driving Images

This project implements a complete image processing pipeline for enhancing driving images taken in adverse weather conditions (rain and snow).

## Processing Pipeline

The pipeline consists of four main stages:

1. **Perspective Correction** - Corrects the viewing angle of images
2. **Denoising** - Removes noise using advanced techniques like BM3D
3. **Color Enhancement** - Improves color balance and contrast
4. **Inpainting** - Removes artifacts like black circles from images

## Installation

### Dependencies

This project requires the following dependencies:
- Python 3.6+
- OpenCV (cv2)
- NumPy
- SciPy
- scikit-image

### Setting up a virtual environment (recommended)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy scipy scikit-image
```

## Usage

Run the processing pipeline using the command:

```bash
python3 main.py [input_directory]
```

For example:

```bash
python3 main.py unseen_test_imgs
```

This will process all images in the specified directory and save the results in a directory called `Results`.

### Additional options

- `--denoise [method]`: Select the denoising method to use
  - `bm3d`: Block Matching 3D filtering (default, best quality but slower)
  - `nlm`: Non-local means denoising (preserves details better)
  - `median_only`: Simple median filter (fastest)
  - `multi_stage`: Multi-stage denoising with adaptive filtering
  - `adaptive_edge`: Custom edge-aware denoising optimized for low-resolution driving images

- `--debug`: Enable debug output and save intermediate results

### Examples

```bash
# Use non-local means denoising for better detail preservation
python3 main.py unseen_test_imgs --denoise nlm

# Use median filter for faster processing
python3 main.py unseen_test_imgs --denoise median_only

# Use the custom edge-aware method optimized for low-resolution driving images
python3 main.py unseen_test_imgs --denoise adaptive_edge
```

## Output

Processed images will be saved in the `Results` directory with the same filenames as the input images.