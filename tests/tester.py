import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any
import importlib
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import itertools

class TestSetViewer:
    def __init__(self, root, history_data=None):
        self.root = root
        self.root.title("Test Set Results Viewer")
        self.current_page = 0
        self.items_per_page = 1
        self.history_data = history_data or []
        
        # Main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image frame
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Parameters frame
        self.params_frame = ttk.Frame(self.main_frame)
        self.params_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Navigation buttons
        self.nav_frame = ttk.Frame(self.root)
        self.nav_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(self.nav_frame, text="Previous", command=self.prev_page).pack(side=tk.LEFT)
        ttk.Button(self.nav_frame, text="Next", command=self.next_page).pack(side=tk.RIGHT)
        
        self.update_display()
    
    def update_display(self):
        if not self.history_data:
            return
            
        # Clear previous content
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        # Get current item
        start_idx = self.current_page * self.items_per_page
        if start_idx >= len(self.history_data):
            return
            
        item = self.history_data[start_idx]
        
        # Display image
        try:
            image_path = os.path.join('tests/data/outputs', item['uuid'] + '.jpg')
            img = Image.open(image_path)
            # Resize image to fit the frame
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(self.image_frame, image=photo)
            label.image = photo  # Keep a reference
            label.pack()
        except Exception as e:
            ttk.Label(self.image_frame, text=f"Error loading image: {e}").pack()
        
        # Display parameters
        ttk.Label(self.params_frame, text=f"Test {start_idx + 1}/{len(self.history_data)}",
                 font=('Arial', 12, 'bold')).pack(anchor='w', pady=5)
        
        params_path = os.path.join('tests/data/history', item['uuid'] + '.json')
        try:
            with open(params_path, 'r') as f:
                params = json.load(f)
                for key, value in params.items():
                    ttk.Label(self.params_frame, 
                            text=f"{key}: {value}").pack(anchor='w')
        except Exception as e:
            ttk.Label(self.params_frame, text=f"Error loading parameters: {e}").pack()
            
        # Show timestamps
        ttk.Label(self.params_frame, text=f"\nStart: {item['start_time']}").pack(anchor='w')
        ttk.Label(self.params_frame, text=f"End: {item['end_time']}").pack(anchor='w')
    
    def next_page(self):
        if (self.current_page + 1) * self.items_per_page < len(self.history_data):
            self.current_page += 1
            self.update_display()
    
    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_display()

def load_config() -> Dict:
    """Load test configuration from config.json"""
    with open('tests/config.json', 'r') as f:
        return json.load(f)

def load_history() -> List[Dict]:
    """Load test history from history.json"""
    try:
        with open('tests/data/history.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_history(history: List[Dict]):
    """Save test history to history.json"""
    with open('tests/data/history.json', 'w') as f:
        json.dump(history, f, indent=2)

def run_test_set(test_set_config: Dict) -> List[Dict]:
    """Run a test set with all parameter combinations"""
    # Import the module dynamically
    module_name = test_set_config['module']
    function_name = test_set_config['entry_point']
    module = importlib.import_module(module_name)
    test_function = getattr(module, function_name)
    
    # Generate all parameter combinations
    param_names = [p['name'] for p in test_set_config['parameters']]
    param_values = [p['values'] for p in test_set_config['parameters']]
    
    results = []
    
    # For each combination of parameters
    for values in itertools.product(*param_values):
        params = dict(zip(param_names, values))
        test_id = str(uuid.uuid4())
        
        # Record start time
        start_time = datetime.now().isoformat()
        
        # Run the test
        output_path = os.path.join('tests/data/outputs', f'{test_id}.jpg')
        result = test_function('inputs/paty.jpg', output_path=output_path, **params)
        
        # Record end time
        end_time = datetime.now().isoformat()
        
        # Save parameters
        params_path = os.path.join('tests/data/history', f'{test_id}.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Record test metadata
        results.append({
            'uuid': test_id,
            'start_time': start_time,
            'end_time': end_time,
            'test_set': test_set_config['name']
        })
    
    return results

def main():
    # Load configuration
    config = load_config()
    
    while True:
        print("\nTest Set Manager")
        print("1. Run test set")
        print("2. Show results")
        print("3. Exit")
        
        choice = input("Select an option: ")
        
        if choice == "1":
            print("\nAvailable test sets:")
            for i, test_set in enumerate(config['test_sets'], 1):
                print(f"{i}. {test_set['name']}")
            
            try:
                idx = int(input("Select test set number: ")) - 1
                if 0 <= idx < len(config['test_sets']):
                    # Run selected test set
                    results = run_test_set(config['test_sets'][idx])
                    
                    # Update history
                    history = load_history()
                    history.extend(results)
                    save_history(history)
                    
                    # Show results
                    root = tk.Tk()
                    viewer = TestSetViewer(root, results)
                    root.mainloop()
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input")
                
        elif choice == "2":
            history = load_history()
            if not history:
                print("No test history found")
                continue
            
            # Group by test set
            test_sets = {}
            for item in history:
                test_sets.setdefault(item['test_set'], []).append(item)
            
            print("\nAvailable test sets:")
            test_set_names = list(test_sets.keys())
            for i, name in enumerate(test_set_names, 1):
                print(f"{i}. {name} ({len(test_sets[name])} tests)")
            
            try:
                idx = int(input("Select test set number: ")) - 1
                if 0 <= idx < len(test_set_names):
                    # Show selected test set results
                    root = tk.Tk()
                    viewer = TestSetViewer(root, test_sets[test_set_names[idx]])
                    root.mainloop()
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input")
                
        elif choice == "3":
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('tests/data/outputs', exist_ok=True)
    os.makedirs('tests/data/history', exist_ok=True)
    main()
