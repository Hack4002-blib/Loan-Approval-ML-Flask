from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# --- Train model (same as your notebook code) ---
df = pd.read_csv("loan.csv")

# Encode categorical columns
le_emp = LabelEncoder()
le_prev = LabelEncoder()
df['Employment Type'] = le_emp.fit_transform(df['Employment Type'])
df['Previous Defaults'] = le_prev.fit_transform(df['Previous Defaults'])

# Features and target
x = df[['Income ($)', 'Credit Score', 'Loan Amount ($)',
        'Loan Term (Months)', 'Employment Type', 
        'Debt-to-Income Ratio (%)', 'Previous Defaults']]
y = df['Loan Approved (Yes=1, No=0)']

# Split data (not needed for prediction but for realism)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)


# --- Flask Routes ---
@app.route('/')
def home():
    # Employment options for dropdown
    emp_options = list(le_emp.classes_)
    return render_template('index.html', emp_options=emp_options)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from form
        income = float(request.form['income'])
        credit_score = float(request.form['credit_score'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = float(request.form['loan_term'])
        emp_type = request.form['employment_type']
        dti = float(request.form['dti'])
        prev_defaults = request.form['prev_defaults']

        # Encode categorical fields using same LabelEncoder
        emp_encoded = le_emp.transform([emp_type])[0]
        prev_encoded = le_prev.transform([prev_defaults])[0]

        # Create array in same order as training
        new_data = np.array([[income, credit_score, loan_amount,
                              loan_term, emp_encoded, dti, prev_encoded]])

        # Predict
        prediction = dt.predict(new_data)[0]

        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

        return render_template('index.html', 
                               result=result, 
                               emp_options=list(le_emp.classes_))

    except Exception as e:
        return render_template('index.html', 
                               result=f"Error: {str(e)}", 
                               emp_options=list(le_emp.classes_))


if __name__ == "__main__":
    app.run(debug=True)
