import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from utils.paths import from_inputs


def run_flow(image: Image, threshold: int) -> Image.Image:
    """
    Apply preprocessing filters to make the image more readable by Tesseract OCR.
    Args:
        image: Input image.
        threshold: Binarization threshold for converting the image to black and white
    Returns:
        Processed PIL Image
    """

    # Load the image
    image = image.convert('L')  # Convert to grayscale
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Remove noise
    image = image.filter(ImageFilter.MedianFilter(size=3))

    image = image.point(lambda x: 0 if x < threshold else 255, '1')


    return image

def test_flow(image: Image, threshold: int) -> Image.Image:
    return run_flow(image, threshold)

if __name__ == '__main__':
    binarization_threshold =100
    original_img = Image.open(from_inputs('paty.raw.jpg'))
    preprocessed = test_flow(original_img, binarization_threshold)
    preprocessed.show()