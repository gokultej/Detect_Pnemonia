from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
loaded_model = tf.keras.models.load_model("model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image from the user
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            # Load the image and preprocess it
            image = Image.open(uploaded_file)
            image = image.resize((224, 224))
            image = np.array(image)  # Convert PIL Image to NumPy array
            image = image / 255.0  # Normalize to the range [0, 1]

            # Make sure it has 3 channels
            if image.shape[-1] != 3:
                image = np.stack((image,) * 3, axis=-1)  # If not, duplicate the single channel to create 3 channels

            image = image.reshape(1, 224, 224, 3)  # Add batch dimension

            # Make a prediction
            prediction = loaded_model.predict(image)
            confidence =  prediction[0][0]

            if confidence >= 0.8:
                message = "Pneumonia chances are high"
            elif confidence > 0.5:
                message = "Pneumonia chances are low"
            else:
                message = "Pneumonia is not going to occur"

            return render_template("result.html", confidence=confidence, message=message)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=8080)
