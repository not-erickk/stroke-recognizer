import time
import threading
from flask import Flask, request, jsonify
import requests
import logging
from requests import RequestException
import call_model
from recognizer.postprocessing import postprocessor

# Configure logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        field = missing_field(data)
        if field:
            return jsonify({"status": 'ERROR', 'msg': f'{field} is missing' }), 400

        # Start webhook in background thread
        thread = threading.Thread(target=webhook_worker, args=(data,))
        thread.start()

        return jsonify({
            'status': 'SUCCESS'
        }), 200
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        return jsonify({
            "status": 'ERROR',
            'msg': 'Internal Server Error'
        }), 500

def send_webhook(url: str, analysis: dict, max_retries: int=10, retry_delay: int=10) -> bool:
    data = {
        'analysis_data': analysis
    }
    for attempt in range(max_retries):
        try:
            webhook_response = requests.post(url, json=data)
            if webhook_response.status_code == 200:
                logger.info(f"Webhook call successful on attempt {attempt + 1}")
                return True
            logger.error(
                f"Webhook call failed with status code {webhook_response.status_code}: {webhook_response.text}")
        except RequestException as e:
            logger.error(f"Webhook call failed with error: {str(e)}")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    return False

def webhook_worker(data: dict) -> None:


    url, encoded_img, objective_word = data['webhook'], data['encodedImage'], data['objectiveWord']
    raw_prediction = call_model.execute(encoded_img)
    analysis = postprocessor.analyze(objective_word, raw_prediction)
    success = send_webhook(url, analysis)
    if not success:
        logger.error("Webhook call failed after all retries")


def missing_field(data: dict):
    if 'encodedImage' not in data:
        return 'encodedImage'
    if 'webhook' not in data:
        return 'webhook'
    if 'objectiveWord' not in data:
        return 'objectiveWord'
    return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
