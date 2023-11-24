from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pickle', 'rb'))
scalar = pickle.load(open('scaler.pickle', 'rb'))

@app.route('/')  
def home():
    return render_template('home.html')

@app.route('/car_purchase_prediction')  
def car_purchase_prediction():
    return render_template('car_purchase_prediction.html')

@app.route('/educational_content')  
def educational_content():
    return render_template('educational_content.html')

@app.route('/feedback_form')  
def feedback_form():
    return render_template('feedback_form.html')

@app.route('/payment')  
def payment():
    return render_template('payment.html')

@app.route('/reviews')  
def reviews():
    return render_template('reviews.html')

@app.route('/pred', methods=['POST'])
def predict1():
    try:
        age = float(request.form["Age"])
        income = float(request.form["Income"])
        input_data = [[age, income]]
        scaled_input = scalar.transform(input_data)
        prediction = model.predict(scaled_input)
        result = np.round(prediction[0])
        return render_template("car_purchase_prediction.html", result=f"The predicted result is {result}")
    except Exception as e:
        return render_template("car_purchase_prediction.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
