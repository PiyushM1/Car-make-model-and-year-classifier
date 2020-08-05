# Monk
import os
import sys
sys.path.append("monk_v1/")

# For Keras backend
from monk.pytorch_prototype import prototype

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Load the trained models
make = None
model = None
year = None

def load_models():
    global make
    global model
    global year

    make = prototype(verbose=0)
    make.Prototype("Make_classifier", "inception-v3-pytorch", eval_infer=True)

    model = prototype(verbose=0)
    model.Prototype("Model_classifier", "inception-v3-pytorch", eval_infer=True)

    year = prototype(verbose=0)
    year.Prototype("Year_classifier", "inception-v3-pytorch", eval_infer=True)

def model_predict(img_path):
    make_preds = make.Infer(img_name=img_path, return_raw=False)
    make_class = make_preds['predicted_class']

    model_preds = model.Infer(img_name=img_path, return_raw=False)
    model_class = model_preds['predicted_class'].replace("_", " ")
    
    
    year_preds = year.Infer(img_name=img_path, return_raw=False)
    year_class = year_preds['predicted_class']

    return (make_class + ' ' + model_class + year_class)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.files.get('file'):
        # Get the file from post request
        f = request.files['file']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # print("./uploads/"+secure_filename(f.filename))
        # Make prediction 
        result = model_predict(img_path=file_path)

        return result

if __name__ == '__main__':
    print(("* Loading models and Flask starting server..."
        "please wait until server has fully started"))
    load_models()

    print('Models loaded')
    app.run(debug=True,host='0.0.0.0',port=5000,threaded=False)

