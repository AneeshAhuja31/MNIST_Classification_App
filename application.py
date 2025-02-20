from flask import Flask,render_template,request,jsonify
from PIL import Image
import numpy as np
from keras.api.models import load_model
import logging
import os

logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

model_path = os.path.join(os.getcwd(), 'modelv3.h5')
cnn = load_model(model_path)
app = Flask(__name__)

@app.route('/')
def index():
    print('')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pixel_data = request.get_json().get('pixels',[]) 
    img_arr = np.array(pixel_data,dtype=np.uint8).reshape(280,280)
    img_resized = Image.fromarray(img_arr).resize((28,28))
    img_arr = np.array(img_resized) / 255.0
    img_arr = np.expand_dims(img_arr, axis=-1)  # Ensure shape is (28, 28, 1)
    img_arr = np.expand_dims(img_arr, axis=0)   # Shape: (1, 28, 28, 1)
    prediction = cnn.predict(img_arr)
    prediction_list= prediction.tolist()
    predicted_digit = int(np.argmax(prediction,axis=1))

    return jsonify({'digit': predicted_digit,
                    'list':prediction_list})

if __name__ == '__main__':
    app.run(debug=True)





