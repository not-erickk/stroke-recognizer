from PIL import Image, ImageTk
import tkinter as tk

def configurate():
    root = tk.Tk()
    root.title("test viewer")
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview, takefocus=True)
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
    return root, scroll_frame

def draw_record(img: Image, scroll_frame: tk.Frame, label: str, pos: int = 0):
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(scroll_frame, image=img_tk)
    img_label.image = img_tk
    img_label.grid(row=pos, column=0, padx=10, pady=10)
    text_label = tk.Label(scroll_frame, text=label, justify='left')
    text_label.grid(row=pos, column=1, sticky='w')
    return scroll_frame

def show(records: list):
    root, scroll_frame = configurate()
    for i, rec in enumerate(records):
        label = f"Test #{i + 1}\nParams: {rec['params']}"
        img = Image.open(rec['output_path'])
        draw_record(img, scroll_frame, label, i)
    root.mainloop()

def show(img: Image, params: dict):
    root, scroll_frame = configurate()
    label = f"Test Params: {params}"
    draw_record(img, scroll_frame, label)
    root.mainloop()
