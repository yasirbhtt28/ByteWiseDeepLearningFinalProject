from flask import Flask, render_template, request, redirect, url_for
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename


# Initialize Flask app
app = Flask(__name__)

# Load trained models
model_rmsprop = load_model(r"C:\Users\Yasir\CodeforLife\Twomodelproject\bone_fracture_model_rmspop.h5")
model_adam = load_model(r"C:\Users\Yasir\CodeforLife\Twomodelproject\bone_fracture_model_adams.h5")

# Define the class labels based on your training
class_labels = ["Avulsion fracture", "Comminuted fracture", "Fracture Dislocation", 
                "Greenstick fracture", "Hairline Fracture", "Impacted fracture", 
                "Longitudinal fracture", "Oblique fracture", "Pathological fracture", 
                "Spiral Fracture"]

# Define the folder to save uploaded images
UPLOAD_FOLDER = os.path.join(r"C:\Users\Yasir\CodeforLife\Twomodelproject\static",r"C:\Users\Yasir\CodeforLife\Twomodelproject\static\uploads" )
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit for uploads

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # Save the file

            # Image processing and prediction logic...
            img = image.load_img(filepath, target_size=(256, 256))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions_rmsprop = model_rmsprop.predict(img_array)
            predictions_adam = model_adam.predict(img_array)
            predicted_class_rmsprop = class_labels[np.argmax(predictions_rmsprop)]
            predicted_class_adam = class_labels[np.argmax(predictions_adam)]
            
            return render_template("result.html", image_path=filename,
                                   prediction_rmsprop=predicted_class_rmsprop,
                                   prediction_adam=predicted_class_adam)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=False)