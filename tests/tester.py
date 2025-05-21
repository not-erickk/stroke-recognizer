import json
import os
import sys
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

def run_test_set(test_set):
    print(f"Running test-set: {test_set['name']}")
    entry_mod, entry_func = test_set['entry_point'].rsplit('.', 1)
    mod = import_module(entry_mod)
    func = getattr(mod, entry_func)
    param = test_set['params'][0]
    param_name = param['name']
    start, stop, step = param['range']
    input_image = os.path.join(os.path.dirname(__file__), f'inputs/{test_set['input_image']}')

    test_records = []
    for value in range(start, stop+1, step):
        test_uuid = str(uuid.uuid4())
        output_path = os.path.join(OUTPUTS_DIR, test_set['name'], f"{test_uuid}.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        params = {param_name: value}
        t0 = time.time()
        result = func(input_image, value, output_path)
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
        with open(os.path.join(HISTORY_DIR, f"{test_uuid}.json"), 'w') as f:
            json.dump(params_record, f, indent=2)
        test_records.append(params_record)
        print(f"Test {value}: output saved to {output_path}")
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

def main():
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

if __name__ == "__main__":
    main()
