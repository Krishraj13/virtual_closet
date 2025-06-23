from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained CNN model
model = load_model('fashion_mnist_cnn_model.h5')

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    img = Image.open(file).convert('L').resize((28, 28))  # Grayscale & Resize
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result = f"{class_names[pred_class]} ({confidence:.2f}% confident)"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
