import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Dict, Any

from matplotlib import pyplot as plt

from recognizer.main.boxes.boxes import Boxes
from recognizer.midprocessing import midprocessing
from api import preprocessing
from utils import plot_ink, paths
from utils.paths import from_inputs, from_models
from utils.utils import scale_and_pad, detokenize, text_to_tokens, load_and_pad_img

# Load the model once when module is imported
MODEL_PATH = from_models('small-p-cpu')
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dev')
logger.setLevel(tf.compat.v1.logging.DEBUG)

def run_flow(image: Image, prompt: str):
    # Scale and pad the image
    scaled_image, _, _, _, _ = scale_and_pad(image)

    # Convert binary image to RGB if needed
    if scaled_image.mode == '1':
        scaled_image = scaled_image.convert('RGB')

    # Prepare the image for the model
    image_encoded = tf.reshape(tf.io.encode_jpeg(tf.cast(np.array(scaled_image), tf.uint8)), (1, 1))

    # Create the prompt
    input_text = tf.constant([prompt], dtype=tf.string)

    # Run inference
    model = tf.saved_model.load(MODEL_PATH)
    serving_func = model.signatures['serving_default']
    output = serving_func(**{
        'input_text': input_text,
        'image/encoded': image_encoded
    })

    # Process the output
    ink_data = detokenize(text_to_tokens(output['output_0'].numpy()[0][0].decode()))

    return ink_data


def save_test_outputs(char_images: list, test_uuid: str, output_dir: str) -> Dict[str, Any]:
    """
    Save character images and process them, organizing by test UUID.
    Args:
        char_images: List of character images
        test_uuid: UUID of the test run
        output_dir: Base directory for outputs
    Returns:
        Dictionary with test results
    """
    test_dir = os.path.join(output_dir, test_uuid[:8])
    os.makedirs(test_dir, exist_ok=True)

    results = []
    for idx, char_img in enumerate(char_images):
        # Save the character image
        img_path = os.path.join(test_dir, f"{test_uuid[:8]}_{idx}.jpg")
        char_img.save(img_path)

        # Process the character
        result = run_flow(char_img)
        result['image_path'] = img_path
        result['char_index'] = idx
        results.append(result)

    return {
        "test_uuid": test_uuid,
        "num_characters": len(char_images),
        "results": results
    }


def test_flow(boxes_data: list, params: dict) -> list:
    results = []
    for i, data in enumerate(boxes_data):
        prompt = f'{params['prompt']}{data['char']}'
        img = data['cropped_img']
        logger.debug(f'Recognizing char {i}, prompt: "{prompt}"')
        ink_data = run_flow(img, prompt)
        fig, ax = plt.subplots()
        plot_ink(ink_data, ax, input_image=load_and_pad_img(img))
        plt.savefig(paths.from_tests(f'temp/{data['char']}{i}'))
        plt.close()
        plt.show()
        results.append((ink_data, img))
    return results


if __name__ == '__main__':
    #fine tuned
    params = {
        'binarization_threshold': 20,
        'page_segmentation_mode': 8,
        'ocr_engine_mode': 3,
        'prompt': 'Derender the ink: '
    }
    threshold, psm, oem = params['binarization_threshold'], params['page_segmentation_mode'], params['ocr_engine_mode']
    img = Image.open(from_inputs('casa.jpg'))
    logger.debug(f'Preprocessing picture')
    preprocessed_img = preprocessing.run_flow(img, threshold)
    # preprocessed_img.show('Preprocessed image')

    logger.debug(f'Running boxes module')
    boxes_mod = Boxes(preprocessed_img, img)
    output = boxes_mod.run_flow(psm, oem)
    logger.debug(f'{len(output)} boxes detected')
    # boxes_mod.visualize_boxes().show()

    logger.debug(f'Midprocessing pictures')
    for i, data in enumerate(output):
        midprocessed_img = midprocessing.run_flow(data['cropped_img'])
        output[i]['cropped_img'] = midprocessed_img
        midprocessed_img.show('Midprocessed image')

    logger.debug(f'Running strokes module')
    results = test_flow(output, params)
    # for ink_data, image in results:
    #     fig, ax = plt.subplots()
    #     plot_ink(ink_data, ax, input_image=load_and_pad_img(image))
    #     plt.show()

