import numpy as np
import os
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

from tensorflow.keras.preprocessing.image import load_img
app = Flask(__name__)

model = tf.keras.models.load_model(r'C:\Users\jeeva\OneDrive\Desktop\major file\my_model.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict')
def index():
    return render_template('predict.html')


@app.route('/predict', methods=['GET', 'POST'])
def output():
    if request.method == "POST":
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        # Load the image and make a prediction
        img = load_img(filepath, target_size=(224, 224))
        image_array = np.array(img)
        image_array = np.expand_dims(image_array, axis=0)

        # Use the pre-trained model to make a prediction
        pred = np.argmax(model.predict(image_array), axis=1)
        index = ['Walk','Run']
        prediction = index[int(pred)]
        print("Prediction:")

        return render_template("predict.html", predict=prediction)

if __name__ == '__main__':
    app.run(debug=True)
