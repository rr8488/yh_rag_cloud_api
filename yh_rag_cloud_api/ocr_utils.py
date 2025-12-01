import cv2
import numpy as np


def preprocess_for_ocr(image_path: str) -> np.ndarray:
    """
    Loads an image, preprocesses it for better OCR accuracy,
    and returns the processed image as a NumPy array.
    """
    try:
        # 1. Load the image
        image = cv2.imread(image_path)

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. Binarization (Adaptive Thresholding)
        # This is the most critical step for your yellow-header problem
        binary = cv2.adaptiveThreshold(
            gray,
            255,  # Max value (white)
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size (neighborhood)
            2  # Constant subtracted from the mean
        )

        # 4. Denoising (Optional, but good)
        denoised = cv2.medianBlur(binary, 3)

        # Return the processed image data (as a NumPy array)
        return denoised

    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        # In case of error, you might return the original (gray) or None
        # Returning None is often better to signal a failure
        return None