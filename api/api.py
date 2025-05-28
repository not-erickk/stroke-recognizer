from flask import Flask, request, jsonify
import phpserialize
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'assignmentId' not in data:
            return jsonify({"error": "Missing required fields: 'image' and 'assignmentId'"}), 400

        image_data = data['image']
        assignment_id = data['assignmentId']

        # Deserializar los datos de la imagen PHP
        try:
            # Decodificar base64 si los datos vienen en ese formato
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data)

            # Deserializar los datos usando phpserialize
            deserialized_data = phpserialize.loads(image_data)

            # Crear un objeto PIL Image desde los bytes deserializados
            image = Image.open(BytesIO(deserialized_data))
            image.show()
            # Aquí ya puedes usar 'image' que es un objeto PIL Image
            # TODO: Add your analysis logic here using image

        except phpserialize.Error as e:
            return jsonify({"error": "Error deserializing PHP data: " + str(e)}), 400
        except Exception as e:
            return jsonify({"error": "Error processing image: " + str(e)}), 400

        return jsonify({
            "message": "Analysis completed",
            "assignmentId": assignment_id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == 'main':
    app.run(debug=True)
