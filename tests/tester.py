import json
import math
import os
import sys
import traceback
import uuid
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import tkinter.scrolledtext as scrolledtext

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

def run_test_set(test_set):
    def param_values_to_test(p: dict):
        p_value = p['value']
        if type(p_value) != list:
            return 1

        if p['type'] == 'range':
            if len(p_value) != 3:
                raise Exception(f"param {p['name']} range must have start, stop and step values")
            start, stop, step = p_value
            if step < 1:
                raise Exception(f"param {p['name']} step must be at least 1")
            if start >= stop:
                raise Exception(f"param {p['name']} range is backwards")
            if start == stop:
                return 1
            return math.ceil((stop - start + 1) / step)
        else:
            return len(p_value)

    print(f"Running test-set: {test_set['name']}")
    entry_mod, entry_func = test_set['entry_point'].rsplit('.', 1)
    params_config = test_set['params_config']
    mod = import_module(entry_mod)
    func = getattr(mod, entry_func)
    total_tests = math.prod([param_values_to_test(p) for p in params_config])
    test_records = []
    input_img_path = os.path.join(os.path.dirname(__file__), f'inputs/{test_set['input_image']}')

    def iterative_param_building(params=None, param_idx: int = 0):
        if params is None:
            params = {}

        if param_idx >= len(params_config):
            params_record = exec_test(test_set['name'], params.copy(), func, input_img_path)
            with open(os.path.join(HISTORY_DIR, f"{params_record['uuid']}.json"), 'w') as f:
                json.dump(params_record, f, indent=2)
            test_records.append(params_record)
            print(f"Test {len(test_records)}/{total_tests}: output saved to {params_record['output_path']}")
            return

        p_config = params_config[param_idx]
        p_value = p_config['value']

        # Handle list values
        if isinstance(p_value, list):
            for value in p_value:
                params[p_config['name']] = value
                iterative_param_building(params, param_idx + 1)
        # Handle range type
        elif p_config['type'] == 'range':
            start, stop, step = p_value
            for value in range(start, stop + 1, step):
                params[p_config['name']] = value
                iterative_param_building(params, param_idx + 1)
        # Handle non-list type (single values)
        else:
            params[p_config['name']] = p_value
            iterative_param_building(params, param_idx + 1)

    iterative_param_building()
    # Update history.json
    history = load_history()
    history.append({
        'test_set': test_set['name'],
        'timestamp': datetime.now().isoformat(),
        'tests': [r['uuid'] for r in test_records]
    })
    save_history(history)
    print("Test-set complete.")

def show_results():
    history = load_history()
    if not history:
        print("No test-sets found.")
        return
    print("Available test-sets:")
    for i, h in enumerate(history):
        print(f"{i+1}. {h['test_set']} at {h['timestamp']}")
    idx = int(input("Select test-set to view: ")) - 1
    test_uuids = history[idx]['tests']
    print("Options:")
    print("1. Open test-set results viewer")
    print("2. Delete this test-set and all its files")
    opt = input("Enter 1 or 2: ").strip()
    if opt != '2':
        records = []
        for uuid_ in test_uuids:
            with open(os.path.join(HISTORY_DIR, f"{uuid_}.json")) as f:
                records.append(json.load(f))
        # Tkinter scrollable viewer
        root = tk.Tk()
        root.title("Test-set Results Viewer")
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for i, rec in enumerate(records):
            img = Image.open(rec['output_path'])
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            img_label = tk.Label(scroll_frame, image=img_tk)
            img_label.image = img_tk
            img_label.grid(row=i, column=0, padx=10, pady=10)
            text = f"Test #{i+1}\nParams: {rec['params']}\nStart: {rec['start_time']}\nEnd: {rec['end_time']}"
            text_label = tk.Label(scroll_frame, text=text, justify='left')
            text_label.grid(row=i, column=1, sticky='w')
        root.mainloop()
    else:
        confirm = input("Are you sure you want to delete this test-set and all its files? (y/n): ").strip().lower()
        if confirm == 'y':
            # Delete all output and params files
            for uuid_ in test_uuids:
                # Delete params file
                params_path = os.path.join(HISTORY_DIR, f"{uuid_}.json")
                if os.path.exists(params_path):
                    os.remove(params_path)
                # Delete output image (path is in params file, but if missing, try default location)
                # Try to get output_path from params file if it exists
                output_path = None
                if os.path.exists(params_path):
                    try:
                        with open(params_path) as f:
                            rec = json.load(f)
                            output_path = rec.get('output_path')
                    except Exception:
                        pass
                if not output_path:
                    # Fallback: try default location
                    output_path = os.path.join(OUTPUTS_DIR, history[idx]['test_set'], f"{uuid_}.jpg")
                if os.path.exists(output_path):
                    os.remove(output_path)
            # Remove record from history
            del history[idx]
            save_history(history)
            print("Test-set and all associated files deleted.")
        else:
            print("Deletion cancelled.")

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
