import os
import base64
import io
from roboflow import Roboflow
from PIL import Image, ImageDraw
import json
import tempfile

import preprocessing

# Configuración de Roboflow
API_KEY = "G6QW6ebEy90X8t57vTDA"
PROJECT_NAME = "detect-and-classify-object-detection-izxgz"
MODEL_VERSION = 8
BIN_THRESHOLD = 100
CONFIDENCE_THRESHOLD = 40
OVERLAP_THRESHOLD = 40


def initialize_model(local: bool = False):
    """Inicializa el modelo de Roboflow"""



    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(PROJECT_NAME)
    model = project.version(MODEL_VERSION).model
    return model


def base64_to_image(base64_string):
    """
    Convierte una cadena base64 a un objeto PIL Image
    
    Args:
        base64_string: String de imagen codificada en base64
        
    Returns:
        PIL.Image: Objeto de imagen
    """
    # Remover el prefijo data:image/xxx;base64, si existe
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Decodificar base64
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image = preprocessing.run_flow(image, BIN_THRESHOLD)
    image.show()
    return image


def perform_inference(model, image_data):
    """
    Realiza inferencia sobre una imagen
    
    Args:
        model: Modelo de Roboflow
        image_data: Puede ser una ruta de archivo o una cadena base64
        confidence: Umbral de confianza (0-100)
        overlap: Umbral de solapamiento (0-100)
        
    Returns:
        dict: Resultados de la predicción
    """
    if isinstance(image_data, str) and not os.path.isfile(image_data):
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image = base64_to_image(image_data)
                image.save(tmp_file.name)
                temp_path = tmp_file.name

            prediction = model.predict(temp_path, confidence=CONFIDENCE_THRESHOLD, overlap=OVERLAP_THRESHOLD)

            os.unlink(temp_path)

        except Exception as e:
            raise Exception(f"Error procesando imagen base64: {str(e)}")
    else:
        prediction = model.predict(image_data, confidence=CONFIDENCE_THRESHOLD, overlap=OVERLAP_THRESHOLD)

    prediction_json = prediction.json()
    return prediction, prediction_json


def save_prediction_image(prediction, output_path="prediction.jpg"):
    """
    Guarda la imagen con las predicciones anotadas
    
    Args:
        prediction: Objeto de predicción de Roboflow
        output_path: Ruta donde guardar la imagen
        
    Returns:
        str: Ruta donde se guardó la imagen
    """
    prediction.save(output_path=output_path)
    return output_path


def process_multiple_images(model, input_folder, output_folder):
    """
    Procesa m�ltiples im�genes de una carpeta

    Args:
        model: Modelo de Roboflow
        input_folder: Carpeta con im�genes de entrada
        output_folder: Carpeta para guardar resultados
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Obtener lista de im�genes
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in os.listdir(input_folder)
              if any(f.lower().endswith(ext) for ext in image_extensions)]

    results = {}

    for image_name in images:
        image_path = os.path.join(input_folder, image_name)
        print(f"Procesando: {image_name}")

        try:
            # Realizar inferencia
            prediction, prediction_json = perform_inference(model, image_path)

            # Guardar imagen con predicciones
            output_path = os.path.join(output_folder, f"pred_{image_name}")
            save_prediction_image(prediction, output_path)

            # Guardar resultados JSON
            json_output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_results.json")
            with open(json_output_path, 'w') as f:
                json.dump(prediction_json, f, indent=2)

            results[image_name] = {
                "status": "success",
                "predictions": len(prediction_json.get("predictions", [])),
                "output_image": output_path,
                "json_results": json_output_path
            }

        except Exception as e:
            print(f"Error procesando {image_name}: {str(e)}")
            results[image_name] = {
                "status": "error",
                "error": str(e)
            }

    return results


def execute(base64_image_string):
    """
    Función principal que procesa una imagen codificada en base64
    
    Args:
        base64_image_string: String de imagen codificada en base64
        
    Returns:
        dict: Resultados de la predicción con el siguiente formato:
            {
                "status": "success" o "error",
                "predictions": lista de predicciones,
                "num_detections": número de detecciones,
                "error": mensaje de error (si aplica)
            }
    """
    try:
        model = initialize_model()
        prediction, prediction_json = perform_inference(model, base64_image_string)

        predictions = prediction_json.get('predictions', [])

        # original_image = base64_to_image(base64_image_string)
        # preview_img = original_image.copy().convert("RGBA")
        # draw = ImageDraw.Draw(preview_img)
        #
        # for pred in predictions:
        #     x, y = pred.get('x', 0), pred.get('y', 0)
        #     w, h = pred.get('width', 0), pred.get('height', 0)
        #     x1 = int(x - w / 2)
        #     y1 = int(y - h / 2)
        #     x2 = int(x + w / 2)
        #     y2 = int(y + h / 2)
        #     draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        #     draw.text((x1, y1 - 20), pred.get('class', ''), fill='red')
        #
        # buffered = io.BytesIO()
        # preview_img.save(buffered, format="PNG")
        # base64_analysis_img = base64.b64encode(buffered.getvalue()).decode()

        formatted_predictions = []
        for pred in predictions:
            formatted_pred = {
                "class": pred.get('class'),
                "confidence": pred.get('confidence', 0),
                "x": pred.get('x'),
                "y": pred.get('y'),
                "width": pred.get('width'),
                "height": pred.get('height')
            }
            formatted_predictions.append(formatted_pred)

        return {
            "status": "success",
            "predictions": formatted_predictions,
            "num_detections": len(predictions),
            "analysis_img": 'fake_img'
        }


    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "predictions": [],
            "num_detections": 0
        }
