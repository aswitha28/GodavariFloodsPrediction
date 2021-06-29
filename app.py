from flask import Flask, request, render_template
import numpy as np
from joblib import load
app = Flask(__name__)
model= load('flood.save')
trans=load('transform')

@app.route('/')
def home():
    return render_template('template.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    x_test = np.array(x_test, dtype='float64')
    print(x_test)
    prediction = model.predict(x_test)
    print(prediction)
    pred=prediction[0]
    
    return render_template('template.html', prediction_text='Flood: {}'.format(pred))

'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True,port=5000)
