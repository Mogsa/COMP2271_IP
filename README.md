# Image Processing Pipeline for Driving Images

A comprehensive image processing pipeline designed to enhance driving images captured in adverse weather conditions (rain and snow). This pipeline applies a series of computer vision techniques to improve image quality, correct perspective distortions, remove noise, and enhance visibility.

**COMP2271 Computer Science Project**

## Features

- **Multi-stage Processing Pipeline**: Four-stage enhancement process optimized for driving images
- **Multiple Denoising Algorithms**: Choice of five different denoising methods including BM3D and adaptive edge-aware filtering
- **Automated Artifact Removal**: PatchMatch-based inpainting for removing image artifacts
- **Batch Processing**: Process entire directories of images automatically
- **Weather Condition Handling**: Specifically optimized for rain and snow conditions
- **Flexible Configuration**: Command-line options for customizing the processing pipeline

## Pipeline Stages

The processing pipeline consists of four sequential stages:

### 1. Perspective Correction
Automatically detects and corrects camera viewing angles by:
- Identifying content boundaries in the image
- Detecting quadrilateral regions representing the road surface
- Applying perspective transformation to normalize the viewing angle

### 2. Advanced Denoising
Removes noise from images using one of five available methods:
- **BM3D** (Default): Block Matching 3D filtering - highest quality, slower processing
- **Non-local Means (NLM)**: Excellent detail preservation with good noise reduction
- **Median Filter**: Fast processing with basic noise removal
- **Multi-stage**: Combines multiple denoising techniques adaptively
- **Adaptive Edge-Aware**: Custom method optimized for low-resolution driving images

### 3. Color Enhancement
Improves image appearance through:
- Automatic white balance correction
- Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Color cast removal
- Adaptive brightness adjustment

### 4. Inpainting
Removes artifacts and damaged regions:
- Automatic artifact detection in dark regions
- PatchMatch-based inpainting algorithm
- Preserves surrounding texture and structure

## Installation

### Requirements

- Python 3.6 or higher
- OpenCV (cv2)
- NumPy
- SciPy
- scikit-image

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd COMP2271_IP
```

2. Create and activate a virtual environment (recommended):
```bash
# Create virtual environment
python3 -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install opencv-python numpy scipy scikit-image
```

## Usage

### Basic Usage

Process all images in a directory:

```bash
python3 main.py <input_directory>
```

Example:
```bash
python3 main.py driving_images
```

Processed images will be saved to the `Results` directory.

### Advanced Options

#### Denoising Method Selection

Choose a specific denoising algorithm with the `--denoise` flag:

```bash
# Use BM3D (default - best quality)
python3 main.py driving_images --denoise bm3d

# Use Non-local Means (good detail preservation)
python3 main.py driving_images --denoise nlm

# Use Median filter (fastest)
python3 main.py driving_images --denoise median_only

# Use Multi-stage denoising
python3 main.py driving_images --denoise multi_stage

# Use Adaptive edge-aware method
python3 main.py driving_images --denoise adaptive_edge
```

#### Debug Mode

Enable debug output to see intermediate results and processing details:

```bash
python3 main.py driving_images --debug
```

This will:
- Display verbose processing information
- Save artifact detection masks
- Show step-by-step progress

### Processing Individual Stages

The repository includes separate modules for testing individual pipeline stages:

- `perspective.py` - Perspective correction only
- `color.py` - Color enhancement only
- `advanced_denoise.py` - Denoising only
- `patchmatch_inpainting.py` - Inpainting only

## Project Structure

```
COMP2271_IP/
├── main.py                       # Main pipeline script
├── perspective.py                # Perspective correction module
├── color.py                      # Color enhancement module
├── advanced_denoise.py           # Advanced denoising module
├── bm3d.py                       # BM3D implementation
├── denoise.py                    # Basic denoising utilities
├── patchmatch_inpainting.py      # PatchMatch inpainting module
├── classify.py                   # Image classification utilities
├── classifier.model              # Pre-trained classifier model
├── pipeline_evaluation.py        # Pipeline evaluation scripts
├── evaluate_pipeline_stages.py   # Stage-by-stage evaluation
├── run_evaluation.py             # Evaluation runner
├── driving_images/               # Sample input images
├── Results/                      # Default output directory
├── Results_perspective/          # Perspective correction results
├── Results_color/                # Color enhancement results
├── Results_denoise/              # Denoising results
├── Results_inpaint/              # Inpainting results
├── Results_full_pipeline/        # Complete pipeline results
└── PyPatchMatch/                 # PatchMatch implementation
```

## Denoising Methods Comparison

| Method | Speed | Quality | Detail Preservation | Best For |
|--------|-------|---------|---------------------|----------|
| BM3D | Slow | Excellent | Very Good | Final production images |
| NLM | Medium | Very Good | Excellent | Images with fine details |
| Median | Fast | Good | Fair | Quick previews, testing |
| Multi-stage | Medium | Very Good | Good | Balanced quality/speed |
| Adaptive Edge | Medium | Good | Very Good | Low-resolution driving images |

## Output

All processed images are saved to the `Results` directory by default, maintaining the original filenames. Each processing stage can also save intermediate results to separate directories for analysis and debugging.

## Example Results

The pipeline processes images through all four stages:
- Input: Raw driving image with weather effects (snow/rain)
- After Perspective: Corrected viewing angle
- After Denoising: Noise removed while preserving details
- After Color: Enhanced contrast and color balance
- Final Output: Clean, enhanced image with artifacts removed

## Evaluation

The project includes evaluation scripts to assess pipeline performance:

```bash
# Evaluate complete pipeline
python3 pipeline_evaluation.py

# Evaluate individual stages
python3 evaluate_pipeline_stages.py

# Run full evaluation suite
python3 run_evaluation.py
```

## About

This project was developed as part of the COMP2271 module coursework.
