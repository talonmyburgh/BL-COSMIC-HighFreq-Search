import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk  # Import PIL for image resizing
import os
import glob
import numpy as np

class ImageDisplayApp:
    def __init__(self, root, image_folder):
        self.root = root
        self.image_folder = image_folder
        self.image_paths = glob.glob(os.path.join(image_folder, "*_antennas.png"))
        self.current_index = 0
        self.enter_indices = []

        self.label = Label(root)
        self.label.pack()

        self.update_image()

        self.root.bind('<Return>', self.on_enter)
        self.root.bind('<space>', self.on_space)
        self.root.bind('<backspace>', self.on_backspace)

        print("Press <enter> to save the filename of the file you're looking at")
        print("Press <space> to skip the file")
        print("Press <backspace> to go back a file")

    def update_image(self):
        if self.current_index < len(self.image_paths):
            img_path = self.image_paths[self.current_index]

            # Open image using PIL
            image = Image.open(img_path)

            # # Resize image to fit within the window dimensions
            # window_width = self.root.winfo_width()
            # window_height = self.root.winfo_height()

            # if image.width > window_width or image.height > window_height:
            #     # Calculate aspect ratio
            #     aspect_ratio = min(window_width / image.width, window_height / image.height)
            #     new_width = int(image.width * aspect_ratio)
            #     new_height = int(image.height * aspect_ratio)
            # print(image.width)
            # print(image.height)
            image = image.resize((int(image.width / 8), int(image.height / 8)))

            # Convert Image object to PhotoImage object
            image = ImageTk.PhotoImage(image)

            self.label.configure(image=image)
            self.label.image = image  # Keep reference to avoid garbage collection
        else:
            self.root.quit()

    def on_enter(self, event):
        self.enter_indices.append(self.current_index)
        self.current_index += 1
        self.update_image()

    def on_space(self, event):
        self.current_index += 1
        self.update_image()

    def on_backspace(self, even):
        self.current_index -= 1
        self.update_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Display App")

    # Replace 'images' with your actual folder path containing images
    image_folder = '/home/noah/Desktop/BL/candidate_plots'

    app = ImageDisplayApp(root, image_folder)

    root.mainloop()

    print("Indices where Enter was pressed:")
    print(app.enter_indices)
    np.savetxt(os.path.join(app.image_folder, "saved_filenames.csv"), np.array(app.image_paths[app.enter_indices]))
