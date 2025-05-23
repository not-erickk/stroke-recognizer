import os

import numpy as np
import tensorflow as tf
from PIL import Image
import uuid
from typing import Dict, Any, Tuple

from matplotlib import pyplot as plt

from recognizer.preprocessing import preprocessing
from recognizer.main.boxes import boxes
from utils import plot_ink
from utils.paths import from_inputs, from_models
from utils.utils import scale_and_pad, detokenize, text_to_tokens, load_and_pad_img

# Load the model once when module is imported
MODEL_PATH = from_models('small-p-cpu')


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


def test_flow(input_images: list, chars: list, params: dict) -> list:
    results = []
    # for image in input_images:
    #     image.show()
    # for i, image in enumerate(input_images):
    #     prompt = f'{params['prompt']}{i}'
    with input_images[1] as image:
        prompt = f'{params['prompt']}{1}'
        results.append((run_flow(image, prompt), image))
    return results


if __name__ == '__main__':
    #fine tuned
    params = {
        'binarization_threshold': 60,
        'page_segmentation_mode': 9,
        'ocr_engine_mode': 3,
        'prompt': 'Derender the ink: '
    }
    psm, oem = params['page_segmentation_mode'], params['ocr_engine_mode']
    img_path = from_inputs('paty.raw.jpg')
    image = preprocessing.run_flow(img_path, params)
    # image.show()
    boxes, char_images, chars = boxes.run_flow(image, psm, oem)
    if len(char_images) > 1:
    #     char_images[1].show()
        pass
    else:
        raise Exception(f'{len(char_images)} chars? wtf')
    results = test_flow(char_images, chars, params)
    for ink_data, image in results:
        fig, ax = plt.subplots()
        plot_ink(ink_data, ax, input_image=load_and_pad_img(image))
        plt.show()

