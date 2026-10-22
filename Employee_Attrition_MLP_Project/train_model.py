import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Demo employee dataset
data = {
    "Salary": [
        25000, 30000, 45000, 50000, 28000,
        60000, 35000, 70000, 27000, 55000,
        32000, 48000, 26000, 65000, 40000,
        29000, 52000, 37000, 75000, 31000,
        43000, 34000, 68000, 24000, 58000,
        39000, 72000, 33000, 47000, 36000
    ],
    "Experience": [
        1, 2, 5, 7, 1, 10, 4, 12, 2, 8,
        3, 6, 1, 11, 5, 2, 9, 4, 13, 2,
        6, 3, 10, 1, 8, 5, 12, 3, 7, 4
    ],
    "Satisfaction": [
        2, 3, 4, 5, 2, 5, 3, 5, 1, 4,
        2, 4, 1, 5, 3, 2, 5, 3, 5, 2,
        4, 2, 5, 1, 4, 3, 5, 2, 4, 3
    ],
    "Overtime": [
        1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ],
    "Age": [
        22, 25, 30, 35, 23, 42, 28, 45, 24, 38,
        26, 33, 21, 44, 31, 24, 40, 29, 46, 25,
        34, 27, 43, 22, 37, 30, 48, 26, 36, 28
    ],
    # 1 = Leave, 0 = Stay
    "Attrition": [
        1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

X = df[["Salary", "Experience", "Satisfaction", "Overtime", "Age"]]
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nmodel.pkl and scaler.pkl created successfully.")
