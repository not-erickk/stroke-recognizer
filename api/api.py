from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        return jsonify({
            'status': 'SUCCESS'
        }), 200
    except Exception:
        return jsonify({"status": 'ERROR'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
