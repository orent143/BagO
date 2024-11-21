from flask import Flask, render_template, request, session, redirect, url_for, flash
import pandas as pd
from models.ml_models import (
    predict_salary, predict_churn, predict_performance,
    predict_suitability, predict_promotion, predict_success,
    train_linear_regression, train_naive_bayes, train_knn,
    train_svm, train_decision_tree, train_ann
)
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Add this to enable session functionality

# Load models
salary_model = train_linear_regression()
churn_model = train_naive_bayes()
performance_model = train_knn()
suitability_model = train_svm()
promotion_model = train_decision_tree()
success_model = train_ann()

# Helper function to validate required columns in CSV
def validate_columns(df, required_columns):
    return all(col in df.columns for col in required_columns)

@app.route('/')
def home():
    return render_template('index.html')  # Render the homepage

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        salary_data = []
        churn_data = []
        performance_data = []
        suitability_data = []
        promotion_data = []
        success_data = []

        # Handle Salary Prediction
        if 'salary_file' in request.files:
            file = request.files['salary_file']
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    required_columns = ['years_experience', 'performance_score']
                    if validate_columns(df, required_columns):
                        for _, row in df.iterrows():
                            predicted_salary = predict_salary(salary_model, row['years_experience'], row['performance_score'])
                            salary_data.append({
                                'years_experience': int(row['years_experience']),
                                'performance_score': float(row['performance_score']),
                                'predicted_salary': float(predicted_salary)
                            })
                    else:
                        flash('Invalid CSV for Salary Prediction. Missing required columns.', 'danger')
                except Exception as e:
                    flash(f'Error processing salary file: {e}', 'danger')

        # Handle Churn Prediction
        if 'churn_file' in request.files:
            file = request.files['churn_file']
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    required_columns = ['job_satisfaction', 'tenure_years']
                    if validate_columns(df, required_columns):
                        for _, row in df.iterrows():
                            churn_prediction = predict_churn(churn_model, row['job_satisfaction'], row['tenure_years'])
                            churn_data.append({
                                'job_satisfaction': float(row['job_satisfaction']),
                                'tenure_years': int(row['tenure_years']),
                                'churn_prediction': int(churn_prediction[0])
                            })
                    else:
                        flash('Invalid CSV for Churn Prediction. Missing required columns.', 'danger')
                except Exception as e:
                    flash(f'Error processing churn file: {e}', 'danger')

        # Handle Performance Prediction
        if 'performance_file' in request.files:
            file = request.files['performance_file']
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    required_columns = ['performance_score', 'years_experience']
                    if validate_columns(df, required_columns):
                        for _, row in df.iterrows():
                            performance_prediction = predict_performance(performance_model, row['performance_score'], row['years_experience'])
                            performance_data.append({
                                'performance': float(row['performance_score']),
                                'experience': int(row['years_experience']),
                                'performance_category': performance_prediction[0]
                            })
                    else:
                        flash('Invalid CSV for Performance Prediction. Missing required columns.', 'danger')
                except Exception as e:
                    flash(f'Error processing performance file: {e}', 'danger')

        # Handle Suitability Prediction
        if 'suitability_file' in request.files:
            file = request.files['suitability_file']
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    required_columns = ['years_of_experience', 'test_score']
                    if validate_columns(df, required_columns):
                        for _, row in df.iterrows():
                            suitability_prediction = predict_suitability(suitability_model, row['years_of_experience'], row['test_score'])
                            suitability_data.append({
                                'experience': int(row['years_of_experience']),
                                'score': float(row['test_score']),
                                'suitability': suitability_prediction[0]
                            })
                    else:
                        flash('Invalid CSV for Suitability Prediction. Missing required columns.', 'danger')
                except Exception as e:
                    flash(f'Error processing suitability file: {e}', 'danger')

        # Handle Promotion Prediction
        if 'promotion_file' in request.files:
            file = request.files['promotion_file']
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    required_columns = ['performance_score', 'years_with_company']
                    if validate_columns(df, required_columns):
                        for _, row in df.iterrows():
                            promotion_prediction = predict_promotion(promotion_model, row['performance_score'], row['years_with_company'])
                            promotion_data.append({
                                'performance': float(row['performance_score']),
                                'tenure': int(row['years_with_company']),
                                'promotion': promotion_prediction[0]
                            })
                    else:
                        flash('Invalid CSV for Promotion Prediction. Missing required columns.', 'danger')
                except Exception as e:
                    flash(f'Error processing promotion file: {e}', 'danger')

        # Handle Success Prediction
        if 'success_file' in request.files:
            file = request.files['success_file']
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    required_columns = ['leadership_score', 'years_of_experience']
                    if validate_columns(df, required_columns):
                        for _, row in df.iterrows():
                            success_prediction = predict_success(success_model, row['leadership_score'], row['years_of_experience'])
                            success_data.append({
                                'leadership_score': float(row['leadership_score']),
                                'experience': int(row['years_of_experience']),
                                'success': success_prediction[0]
                            })
                    else:
                        flash('Invalid CSV for Success Prediction. Missing required columns.', 'danger')
                except Exception as e:
                    flash(f'Error processing success file: {e}', 'danger')

        # Store data in session
        session['salary_data'] = salary_data
        session['churn_data'] = churn_data
        session['performance_data'] = performance_data
        session['suitability_data'] = suitability_data
        session['promotion_data'] = promotion_data
        session['success_data'] = success_data

        # Redirect to employees page to show results
        return redirect(url_for('employees'))

    return render_template('predict.html')  # For GET request, show the prediction form


@app.route('/employees')
def employees():
    # Retrieve prediction results from session
    salary_data = session.get('salary_data', [])
    churn_data = session.get('churn_data', [])
    performance_data = session.get('performance_data', [])
    suitability_data = session.get('suitability_data', [])
    promotion_data = session.get('promotion_data', [])
    success_data = session.get('success_data', [])

    return render_template('employees.html', 
                           salary_data=salary_data,
                           churn_data=churn_data,
                           performance_data=performance_data,
                           suitability_data=suitability_data,
                           promotion_data=promotion_data,
                           success_data=success_data)



if __name__ == '__main__':
    app.run(debug=True)
