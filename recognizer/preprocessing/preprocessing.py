from PIL import Image, ImageEnhance, ImageFilter
from utils.paths import from_inputs


def run_flow(img_path: str, params=None) -> Image.Image:
    """
    Apply preprocessing filters to make the image more readable by Tesseract OCR.
    Args:
        img_path: Path to the input image
        params: Dictionary containing preprocessing parameters
    Returns:
        Processed PIL Image
    """
    if params is None:
        raise Exception("No parameters provided")

    # Load the image
    image = Image.open(img_path).convert('L')  # Convert to grayscale
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    threshold = params['binarization_threshold']
    image = image.point(lambda x: 0 if x < threshold else 255, '1')

    # Remove noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return image

def test_flow(img_path: str, params=None) -> Image.Image:
    return run_flow(img_path, params)

if __name__ == '__main__':
    params = {
        'binarization_threshold': 1
    }
    img = test_flow(from_inputs('lovec.jpg'), params)
    img.show()