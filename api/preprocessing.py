import os
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


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
    input_path = '/home/not_erickk/Projects/trabajo_Terminal/dataset/plantilla/files_raw'
    output_path = '/home/not_erickk/Projects/trabajo_Terminal/dataset/plantilla/files_preprocessed'

    first = True
    for img_file in sorted(os.listdir(input_path), key=lambda k: np.random.random()):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            input_file = os.path.join(input_path, img_file)
            output_file = os.path.join(output_path, img_file)

            binarization_threshold =130
            original_img = Image.open(input_file)
            preprocessed = test_flow(original_img, binarization_threshold)
            if first:
                preprocessed.show()
                time.sleep(5)
                first = False
            preprocessed.save(output_file)
            print(f"Processed and saved to {output_file}")