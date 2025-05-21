import pytesseract
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
from typing import List, Tuple

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the image to improve character detection.
    """
    # Convert to grayscale
    image = image.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Binarize the image
    threshold = 20
    image = image.point(lambda x: 0 if x < threshold else 255, '1')
    
    # Remove noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image

def get_char_boxes(image_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Extract character-level bounding boxes from a handwritten word image using Tesseract OCR.
    """
    # Load and preprocess the image
    image = Image.open(image_path)
    processed_image = preprocess_image(image)
    
    # Configure Tesseract for character-level detection
    custom_config = r'--psm 6 --oem 3 -c textord_min_linesize=1.0 ' \
                   r'-c preserve_interword_spaces=1 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # Get character-level data from Tesseract
    char_boxes = pytesseract.image_to_boxes(processed_image, config=custom_config)
    
    # Parse character boxes
    boxes = []
    for box in char_boxes.splitlines():
        b = box.split()
        if len(b) == 6:  # Standard box format: char left bottom right top page
            char, left, bottom, right, top, _ = b
            # Convert coordinates
            left, bottom, right, top = map(int, [left, bottom, right, top])
            # Convert to x, y, width, height format
            x = left
            y = processed_image.height - top  # Invert y-coordinate
            w = right - left
            h = top - bottom
            boxes.append((x, y, w, h))
    
    return boxes

def visualize_boxes(image_path: str, boxes: List[Tuple[int, int, int, int]], 
                   output_path: str = None) -> Image.Image:
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
    
    if output_path:
        image.save(output_path)
    
    return image

def main():
    """Example usage"""
    image_path = 'inputs/paty.jpg'  # Path to the input image
    
    # Get character boxes
    boxes = get_char_boxes(image_path)
    
    # Visualize and save results
    output_path = "output_boxes.jpg"
    visualized_image = visualize_boxes(image_path, boxes, output_path)
    
    print(f"Found {len(boxes)} character boxes")
    print("Boxes coordinates (x, y, width, height):")
    for box in boxes:
        print(box)

if __name__ == "__main__":
    main()