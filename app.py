from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = r"C:\Users\HP\Documents\_TB\cheast_model.h5"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(128, 128))  # Adjust size based on your model input
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            image_file = request.files["image"]
            if image_file:
                # Save the uploaded file temporarily
                image_path = os.path.join(r"C:\Users\HP\Pictures", image_file.filename)
                image_file.save(image_path)
                print(f"Image saved to {image_path}")

                # Preprocess the image
                processed_image = preprocess_image(image_path)
                if processed_image is not None:
                    prediction = model.predict(processed_image)
                    print(f"Prediction: {prediction}")

                    # Assuming the model output is a probability: 0 -> Normal, 1 -> TB Patient
                    result = "TB Patient" if prediction[0][0] > 0.5 else "Normal"
                else:
                    result = "Error in preprocessing image."

                # Remove the image file after processing
                try:
                    os.remove(image_path)  # Ensure the file is closed before removal
                    print(f"Image {image_path} deleted after processing.")
                except Exception as remove_error:
                    print(f"Error removing file: {remove_error}")

        except Exception as e:
            print(f"Error during prediction: {e}")
            result = "An error occurred during prediction."
        
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)