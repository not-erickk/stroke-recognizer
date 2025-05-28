import cv2
import numpy as np
from PIL import Image


def run_flow(image: Image):
    # Convert to grayscale and binarize
    gray = np.array(image.convert('L'))  # Convert to grayscale
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # # Thinning/skeletonization
    # skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    #
    # # Edge enhancement
    # edges = cv2.Canny(skeleton, 50, 150)
    #
    # # Combine skeleton and edges
    # result = cv2.bitwise_or(skeleton, edges)
    #
    # # Invert back to original format
    result = binary
    result = 255 - result
    return Image.fromarray(result).convert('RGB')


def test_flow():
    run_flow()

if __name__ == '__main__':
    test_flow()