import pytesseract
from PIL import Image, ImageDraw
from typing import List, Tuple
from PIL.ImageFile import ImageFile

from recognizer.preprocessing import preprocessing
from utils.paths import from_inputs


def run_flow(image: Image, psm: int = 6, oem: int = 3) -> (
        Tuple)[List[Tuple[int, int, int, int]], List[Image.Image], list]:
    """
    Extract character-level bounding boxes from a handwritten word image using Tesseract OCR.
    Returns both the bounding boxes and cropped character images.
    """
    # Configure Tesseract for character-level detection
    custom_config = rf'--psm {psm} --oem {oem} -c textord_min_linesize=1.0 ' \
                   r'-c preserve_interword_spaces=1 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # Get character-level data from Tesseract
    char_boxes = pytesseract.image_to_boxes(image, config=custom_config)

    # Parse character boxes and extract character images
    boxes = []
    char_images = []
    chars = []
    for box in char_boxes.splitlines():
        b = box.split()
        if len(b) == 6:  # Standard box format: char left bottom right top page
            char, left, bottom, right, top, _ = b
            # Convert coordinates
            left, bottom, right, top = map(int, [left, bottom, right, top])
            # Convert to x, y, width, height format
            x = left
            y = image.height - top  # Invert y-coordinate
            w = right - left
            h = top - bottom
            boxes.append((x, y, w, h))

            # Crop character image
            char_img = image.crop((left, image.height - top, right, image.height - bottom))
            char_images.append(char_img)
            chars.append(char)

    return boxes, char_images, chars

def visualize_boxes(image_path: str, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    """
    Visualize character bounding boxes on the image.
    """
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Draw boxes
    for x, y, w, h in boxes:
        draw.rectangle(
            [x, y, x + w, y + h],
            outline='red',
            width=3  # Thinner lines for better visualization
        )
    
    return image

def test_flow(img: Image, params: dict):
    """
    Run the boxes test for a given image and threshold, save output image, and return result info.
    """
    psm = params['page_segmentation_mode']
    oem = params['ocr_engine_mode']
    boxes, char_images, chars = run_flow(image, psm, oem)
    print(f'find {len(boxes)} boxes')
    # result_img = visualize_boxes(input_image, boxes)
    return boxes, char_images

if __name__ == '__main__':
    #fine tuned
    params = {
        'binarization_threshold': 20,
        'page_segmentation_mode': 6,
        'ocr_engine_mode': 3
    }
    img_path = from_inputs('paty.jpg')
    image = preprocessing.run_flow(img_path, params)
    boxes, char_images = test_flow(image, params)
    for img in char_images:
        img.show()