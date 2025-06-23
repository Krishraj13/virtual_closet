👕 Virtual Closet – AI-Powered Fashion Item Identifier
An AI-powered Flask web application that predicts the type of fashion item from an image using a Convolutional Neural Network (CNN), demonstrating core Machine Learning and Deep Learning principles.

🚀 Features
✅ Upload grayscale 28x28 images and get instant predictions
✅ Powered by an AI model trained on Fashion MNIST
✅ Supports classification for the following fashion categories:

📁 Testing Images Folder Structure



testing_images/
├── Ankle_boot/
├── Bag/
├── Coat/
├── Dress/
├── Pullover/
├── Sandal/
├── Shirt/
├── Sneaker/
├── Trouser/
└── T-shirt_top/
✅ Includes pre-trained model for immediate use
✅ Minimal Flask-based web interface
✅ Sample test images provided for easy demo

🛠️ Tech Stack
Python 3

TensorFlow / Keras (Deep Learning)

Flask (Web Framework)

HTML, CSS

📂 Project Structure


virtual_closet/
├── testing_images/            # Sample test images for all categories
├── app.py                     # Flask backend (runs the web app)
├── fashion_mnist_cnn_model.h5 # Pre-trained CNN model
├── index.html                 # Frontend webpage
├── model.py                   # Model training script (re-trains the AI model)
├── requirements.txt           # Project dependencies
└── README.md
⚙️ Setup Instructions (Local Testing)
Clone the repository


git clone https://github.com/Krishraj13/virtual_closet.git
cd virtual_closet
Install dependencies

bash
Copy
pip install -r requirements.txt
Train the AI Model (Optional)

bash
Copy
python model.py
The model will be saved as fashion_mnist_cnn_model.h5.

Run the Flask Application

bash
Copy
python app.py
Access the Application

Open your browser and visit:
http://127.0.0.1:5000

Note: This link works only on your local machine during development/testing.

🖼️ Testing the Model
Use the provided images inside the testing_images folder to test the AI model's predictions, or upload your own 28x28 grayscale images.

📦 Future Enhancements
Real-world color image predictions

Larger, real-world fashion datasets

Mobile-friendly, responsive UI

Display confidence scores and top-3 predictions

🤖 AI Model Details
Utilizes a Convolutional Neural Network (CNN) for image classification

Trained on the Fashion MNIST dataset

Demonstrates principles of Deep Learning and Computer Vision

Achieves over 90% accuracy on test data

📢 License
This project is for educational, learning, and demonstration purposes only.

🎯 Final Notes
This project serves as a beginner-friendly showcase of AI applied to image classification, combining Machine Learning concepts with practical web deployment using Flask.


