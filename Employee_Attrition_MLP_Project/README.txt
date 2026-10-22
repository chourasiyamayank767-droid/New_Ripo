EMPLOYEE ATTRITION PREDICTION USING MLP
=========================================

1. Open this folder in VS Code.

2. Open Terminal and install packages:
   pip install -r requirements.txt

3. Train the MLP model:
   python train_model.py

   This creates:
   - model.pkl
   - scaler.pkl

4. Start the website:
   python app.py

5. Open the browser at:
   http://127.0.0.1:5000

PROJECT INPUTS:
- Salary
- Experience
- Satisfaction
- Overtime
- Age

OUTPUT:
- Employee may STAY
- Employee may LEAVE
- Leave probability

IMPORTANT:
The included dataset is only a small demonstration dataset for a college project.
For meaningful real-world performance, train the model on a larger, representative HR dataset.
