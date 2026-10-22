from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import pickle

app = Flask(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

# Load trained MLP model and scaler
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        salary = float(data["salary"])
        experience = float(data["experience"])
        satisfaction = float(data["satisfaction"])
        overtime = int(data["overtime"])
        age = float(data["age"])

        input_data = np.array([[
            salary,
            experience,
            satisfaction,
            overtime,
            age
        ]])

        # Apply the same scaling used during training
        input_scaled = scaler.transform(input_data)

        prediction = int(model.predict(input_scaled)[0])

        # Probability of leaving
        probabilities = model.predict_proba(input_scaled)[0]
        leave_probability = float(probabilities[1]) * 100

        if prediction == 1:
            result = "Employee may LEAVE"
            message = "The MLP model predicts possible employee attrition."
        else:
            result = "Employee may STAY"
            message = "The MLP model predicts that the employee may stay."

        return jsonify({
            "prediction": prediction,
            "result": result,
            "message": message,
            "probability": round(leave_probability, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
