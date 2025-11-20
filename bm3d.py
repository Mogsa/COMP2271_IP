import numpy as np
import cv2
from scipy import fftpack


class BM3D:
    """Simplified BM3D implementation optimized for integration with existing pipeline"""

    def __init__(self, sigma=25.0, block_size=8, max_blocks=16, hard_threshold=2.7):
        self.sigma = sigma
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.hard_threshold = hard_threshold * sigma

    def _find_similar_blocks(self, image, ref_y, ref_x, window_size=33):
        """Find blocks similar to reference block"""
        h, w = image.shape
        half_bs = self.block_size // 2
        half_ws = window_size // 2

        ref_block = image[
            ref_y : ref_y + self.block_size, ref_x : ref_x + self.block_size
        ]

        # Search boundaries
        y_start = max(0, ref_y - half_ws)
        y_end = min(h - self.block_size, ref_y + half_ws)
        x_start = max(0, ref_x - half_ws)
        x_end = min(w - self.block_size, ref_x + half_ws)

        # Find similar blocks
        blocks = []
        for y in range(y_start, y_end + 1, 3):  # Step size 3 for speed
            for x in range(x_start, x_end + 1, 3):
                # Skip reference block
                if y == ref_y and x == ref_x:
                    continue

                # Get block
                block = image[y : y + self.block_size, x : x + self.block_size]

                # Calculate distance
                dist = np.sum((block - ref_block) ** 2) / (self.block_size**2)

                if dist < 3 * self.sigma**2:  # Threshold for similarity
                    blocks.append((y, x, dist))

        # Add reference block
        blocks.append((ref_y, ref_x, 0))

        # Sort by distance
        blocks.sort(key=lambda x: x[2])

        # Return top matches
        return [(y, x) for y, x, _ in blocks[: self.max_blocks]]

    def _process_group(self, image, coords):
        """Process a group of similar blocks"""
        # Extract blocks
        stack = np.zeros(
            (len(coords), self.block_size, self.block_size), dtype=np.float32
        )
        for i, (y, x) in enumerate(coords):
            stack[i] = image[y : y + self.block_size, x : x + self.block_size]

        # 3D transform (DCT in all dimensions)
        dct_stack = np.zeros_like(stack)
        for i in range(len(coords)):
            dct_stack[i] = fftpack.dct(
                fftpack.dct(stack[i].T, norm="ortho").T, norm="ortho"
            )
        dct_3d = fftpack.dct(dct_stack, axis=0, norm="ortho")

        # Hard thresholding
        dct_3d[np.abs(dct_3d) < self.hard_threshold] = 0

        # Inverse transform
        idct_stack = fftpack.idct(dct_3d, axis=0, norm="ortho")
        result = np.zeros_like(idct_stack)
        for i in range(len(coords)):
            result[i] = fftpack.idct(
                fftpack.idct(idct_stack[i].T, norm="ortho").T, norm="ortho"
            )

        return result

    def denoise(self, image):
        """Main denoising function"""
        if len(image.shape) == 3:
            # For color images, convert to YCrCb and process each channel
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            # Process Y channel (luminance)
            y_denoised = self._denoise_channel(y)

            # Process Cr and Cb channels (chrominance) with higher sigma
            cr_denoised = self._denoise_channel(cr, self.sigma * 1.5)
            cb_denoised = self._denoise_channel(cb, self.sigma * 1.5)

            # Merge channels
            result_ycrcb = cv2.merge([y_denoised, cr_denoised, cb_denoised])
            result = cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)
            return result
        else:
            # For grayscale images
            return self._denoise_channel(image)

    def _denoise_channel(self, channel, sigma=None):
        """Denoise a single channel"""
        if sigma is None:
            sigma = self.sigma

        h, w = channel.shape
        result = np.zeros_like(channel, dtype=np.float32)
        weight = np.zeros_like(channel, dtype=np.float32)

        # Add border padding to avoid edge artifacts
        pad_size = self.block_size
        padded_channel = cv2.copyMakeBorder(
            channel, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT_101
        )

        padded_h, padded_w = padded_channel.shape
        padded_result = np.zeros_like(padded_channel, dtype=np.float32)
        padded_weight = np.zeros_like(padded_channel, dtype=np.float32)

        # Process blocks on padded image
        for y in range(0, padded_h - self.block_size + 1, 6):  # Step size 6 for speed
            for x in range(0, padded_w - self.block_size + 1, 6):
                # Find similar blocks
                coords = self._find_similar_blocks(padded_channel, y, x)

                # Process group
                denoised_blocks = self._process_group(padded_channel, coords)

                # Aggregate results
                for i, (by, bx) in enumerate(coords):
                    padded_result[
                        by : by + self.block_size, bx : bx + self.block_size
                    ] += denoised_blocks[i]
                    padded_weight[
                        by : by + self.block_size, bx : bx + self.block_size
                    ] += 1

        # Average results on padded image
        valid_mask = padded_weight > 0
        padded_result[valid_mask] /= padded_weight[valid_mask]

        # Extract the original image region
        result = padded_result[pad_size : pad_size + h, pad_size : pad_size + w]

        return np.clip(result, 0, 255).astype(np.uint8)
