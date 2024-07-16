from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('dl_breast_cancer_prediction.keras')
sc = pickle.load(open('sc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    data = [float(data[x]) for x in data]
    scale = sc.transform(np.array(data).reshape(1, -1))
    pred = model.predict(scale)
    
    if pred[0][0] <= 0.5:
        result_text = "The tumor is Malignant"
    else:
        result_text = "The tumor is Benign"
    return render_template('home.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
