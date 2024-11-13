from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(_name_)
model = load_model("C:\\Users\\Sandhya\\Downloads\\Sandhya leaf\\medicinal_plant_resnet_model.h5")
class_labels = os.listdir("C:\\Users\\Sandhya\\Downloads\\medicinal plants images")  # Assuming directory structure is maintained

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    return jsonify({'predicted_class': predicted_class})

if _name_ == '_main_':
    app.run(debug=True)