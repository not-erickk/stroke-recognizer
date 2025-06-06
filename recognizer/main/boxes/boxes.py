import pytesseract
from PIL import Image, ImageDraw
from typing import List, Tuple
from api import preprocessing


class Boxes:

    def __init__(self, preprocessed_img: Image.Image, original_img: Image.Image):
        self.image = preprocessed_img
        self.original_image = original_img
        self.preview_img = preprocessed_img.copy().convert("RGBA")
        self.output = []

    def run_flow(self, psm: int = 8, oem: int = 3) -> [Image, List[Tuple[int, int, int, int]], List[Image.Image], list]:
        """
        Extract character-level bounding boxes from a handwritten word image using Tesseract OCR.
        Returns both the bounding boxes and cropped character images.
        """
        # Configure Tesseract for character-level detection
        custom_config = rf'--psm {psm} --oem {oem} -c textord_min_linesize=1.0 -l spa ' \
        r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzáéíóúüñÁÉÍÓÚÜÑABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # Get character-level data from Tesseract
        char_boxes = pytesseract.image_to_boxes(self.image, config=custom_config)

        # Parse character boxes and extract character images
        for box in char_boxes.splitlines():
            b = box.split()
            if len(b) == 6:  # Standard box format: char left bottom right top page
                char, left, bottom, right, top, _ = b
                # Convert coordinates
                left, bottom, right, top = map(int, [left, bottom, right, top])
                # Convert to x, y, width, height format
                x = left
                y = self.image.height - top  # Invert y-coordinate
                w = right - left
                h = top - bottom
                self.output.append({
                    'box': (x, y, w, h),
                    'char': char,
                    'cropped_img': self.image.crop((left, self.image.height - top, right, self.image.height - bottom))
                })

        return self.output

    def visualize_boxes(self) -> Image.Image:
        """
        Visualize character bounding boxes on the image.
        """
        self.preview_img = self.image.copy().convert("RGBA")
        draw = ImageDraw.Draw(self.preview_img)

        # Draw boxes
        for char_data in self.output:
            x, y, w, h = char_data['box']
            draw.rectangle(
                [x, y, x + w, y + h],
                outline='red',
                width=3  # Thinner lines for better visualization
            )

        return self.preview_img

    def test_flow(self, params: dict):
        """
        Run the boxes test for a given image and threshold, save output image, and return result info.
        """
        psm = params['page_segmentation_mode']
        oem = params['ocr_engine_mode']
        output = self.run_flow(psm, oem)
        return output

if __name__ == '__main__':
    #fine tuned
    params = {
        'binarization_threshold': 75,
        'page_segmentation_mode': 6,
        'ocr_engine_mode': 3
    }
    threshold = params['binarization_threshold']
    img_path = '/home/not_erickk/Projects/trabajo_Terminal/dataset/plantilla/files/19voces.jpeg'
    original_img = Image.open(img_path)
    image = preprocessing.run_flow(original_img, threshold)
    boxes_mod = Boxes(image, original_img)
    output = boxes_mod.test_flow(params)
    boxes_mod.visualize_boxes().show()



    print([out['char'] for out in output])
    # for char_data in output:
    #     char_data['cropped_img'].show()