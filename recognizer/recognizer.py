import os
import uuid
from typing import Dict, Any, List
from PIL import Image

from recognizer.main.boxes.boxes import run_flow as run_boxes
from recognizer.main.strokes.strokes import run_flow as run_strokes, save_test_outputs
from recognizer.preprocessing.preprocessing import run_flow as run_preprocessing

def run_flow(image_path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main flow of the recognizer module.
    Args:
        image_path: Path to the image to process
        params: Dictionary of parameters for each sub-module
    Returns:
        Dictionary containing all processing results
    """
    if params is None:
        params = {}

    # Load the image
    image = Image.open(image_path)

    # Run preprocessing (dummy for now)
    preprocess_params = params.get('preprocessing', {})
    preprocessed_image = run_preprocessing(image, preprocess_params)

    # Extract character boxes and images
    boxes_params = params.get('boxes')
    boxes, char_images = run_boxes(image_path, **boxes_params)

    # Process each character through the strokes module
    strokes_params = params.get('strokes', {})
    strokes_results = []
    for char_img in char_images:
        result = run_strokes(char_img, strokes_params)
        strokes_results.append(result)

    return {
        "boxes": boxes,
        "char_images": char_images,
        "strokes_results": strokes_results,
        "num_characters": len(char_images)
    }

def test_flow(input_image: str, params: dict, output_path: str) -> Dict[str, Any]:
    """
    Test entry point for the recognizer module.
    Args:
        input_image: Path to the input image
        params: Test parameters for all sub-modules
        output_path: Base path for test outputs
    Returns:
        Dictionary containing test results
    """
    # Generate test UUID
    test_uuid = str(uuid.uuid4())

    # Run the main flow
    result = run_flow(input_image, params)

    # Save character images and get stroke analysis
    output_dir = os.path.join(os.path.dirname(output_path), 'strokes')
    strokes_results = save_test_outputs(result['char_images'], test_uuid, output_dir)

    # Combine all results
    return {
        "test_uuid": test_uuid,
        "boxes_info": {
            "num_boxes": len(result['boxes']),
            "boxes": result['boxes']
        },
        "strokes_info": strokes_results
    }

if __name__ == '__main__':
    run_flow('')

