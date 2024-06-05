import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class FERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Recognition")
        
        self.model = tf.keras.models.load_model('fer_model.h5')
        
        self.label = tk.Label(root, text="Upload an image for emotion recognition", font=('Helvetica', 18))
        self.label.pack(pady=20)
        
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)
        
        self.result_label = tk.Label(root, text="", font=('Helvetica', 18))
        self.result_label.pack(pady=20)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path).convert('L').resize((48, 48))
            image = np.array(image).reshape(1, 48, 48, 1) / 255.0
            
            self.display_image(file_path)
            self.predict_emotion(image)
    
    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((200, 200), Image.LANCZOS)  # Changed from Image.ANTIALIAS to Image.LANCZOS
        photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=photo)
        self.image_label.image = photo
    
    def predict_emotion(self, image):
        emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        prediction = self.model.predict(image)
        emotion = emotion_dict[np.argmax(prediction)]
        
        self.result_label.config(text=f"Predicted Emotion: {emotion}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FERApp(root)
    root.mainloop()
