import json
import math
import os
import uuid
import time
from datetime import datetime
from importlib import import_module
from PIL import Image, ImageTk
import tkinter as tk

from recognizer.main.boxes.boxes import Boxes

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'data/history.json')
HISTORY_DIR = os.path.join(os.path.dirname(__file__), 'data/history')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'data/outputs')

# Ensure directories exist
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH) as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

def exec_test(test_set_name: str, params: dict, func, input_img_path: str):
    test_uuid = str(uuid.uuid4())
    output_path = os.path.join(OUTPUTS_DIR, test_set_name, f"{test_uuid}.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t0 = time.time()
    result = func(input_img_path, params, output_path)
    t1 = time.time()
    # Save params and result
    params_record = {
        'uuid': test_uuid,
        'params': params,
        'result': result,
        'start_time': datetime.fromtimestamp(t0).isoformat(),
        'end_time': datetime.fromtimestamp(t1).isoformat(),
        'output_path': output_path
    }
    return params_record

def main():
    while True:
        print("Select an option:")
        print("A) Run test-set")
        print("B) Show results")
        opt = input("Enter A or B: ").strip().upper()
        if opt == 'A':
            config = load_config()
            test_sets = config['test_sets']
            print("Available test-sets:")
            for i, ts in enumerate(test_sets):
                print(f"{i+1}. {ts['name']}")
            idx = int(input("Select test-set: ")) - 1
            run_test_set(test_sets[idx])
        elif opt == 'B':
            show_results()
        else:
            print("Invalid option.")
        print('\n' + '-'*40)

if __name__ == "__main__":
    main()
