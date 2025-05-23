import pytesseract
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
from typing import List, Tuple
import sys
import os

def preprocess_image(image: Image.Image, binarization_threshold: int = 20) -> Image.Image:
    """
    Preprocess the image to improve character detection.
    """
    # Convert to grayscale
    image = image.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Binarize the image
    threshold = binarization_threshold
    image = image.point(lambda x: 0 if x < threshold else 255, '1')
    
    # Remove noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image

def get_char_boxes(image_path: str, threshold: int = 20, psm: int = 6, oem: int = 3) -> List[Tuple[int, int, int, int]]:
    """
    Extract character-level bounding boxes from a handwritten word image using Tesseract OCR.
    """
    # Load and preprocess the image
    image = Image.open(image_path)
    processed_image = preprocess_image(image, threshold)
    
    # Configure Tesseract for character-level detection
    custom_config = rf'--psm {psm} --oem {oem} -c textord_min_linesize=1.0 ' \
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

def run_boxes_test(input_image: str, params: dict, output_path: str) -> dict:
    """
    Run the boxes test for a given image and threshold, save output image, and return result info.
    """
    threshold = params['binarization_threshold']
    psm = params['page_segmentation_mode']
    oem = params['ocr_engine_mode']
    boxes = get_char_boxes(input_image, threshold, psm, oem)
    visualize_boxes(input_image, boxes, output_path)
    return {
        "num_boxes": len(boxes),
        "boxes": boxes
    }

def main():
    """Allow running as a script with CLI args: input_image, binarization_threshold, output_path"""
    import argparse
    parser = argparse.ArgumentParser(description="Run boxes test.")
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--binarization_threshold', type=int, default=20)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    result = run_boxes_test(args.input_image, args.binarization_threshold, args.output_path)
    print(f"Found {result['num_boxes']} character boxes")
    print("Boxes coordinates (x, y, width, height):")
    for box in result['boxes']:
        print(box)

if __name__ == "__main__":
    main()