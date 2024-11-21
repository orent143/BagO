# models/ml_models.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def train_linear_regression():
    data = {
        'years_experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'performance_score': [55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        'salary': [45000, 48000, 52000, 56000, 60000, 65000, 70000, 75000, 80000, 85000]
    }
    df = pd.DataFrame(data)
    X = df[['years_experience', 'performance_score']]
    y = df['salary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def predict_salary(model, years_experience, performance_score):
    input_data = pd.DataFrame({
        'years_experience': [years_experience],
        'performance_score': [performance_score]
    })
    
    predicted_salary = model.predict(input_data)
    
    return predicted_salary[0]  

def train_naive_bayes():
    data = {
        'job_satisfaction': [3, 2, 4, 5, 1, 3, 5, 2, 4, 1],
        'tenure_years': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)

    X = df[['job_satisfaction', 'tenure_years']]
    y = df['churn']

    # Train the Naive Bayes model
    model = GaussianNB()
    model.fit(X, y)
    return model

def predict_churn(model, job_satisfaction, tenure_years):
    # Predict churn based on the given features
    return model.predict([[job_satisfaction, tenure_years]])

def train_knn():
    data = {
        'performance_score': [90, 75, 60, 85, 55, 80, 70, 95, 50, 65],
        'years_experience': [5, 4, 3, 4, 2, 5, 3, 6, 2, 3],
        'performance_category': ['High', 'Average', 'Low', 'High', 'Low', 'High', 'Average', 'High', 'Low', 'Average']
    }
    df = pd.DataFrame(data)
    X = df[['performance_score', 'years_experience']]
    y = df['performance_category']
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    
    return model

def predict_performance(model, performance_score, years_experience):
    return model.predict(np.array([[performance_score, years_experience]]))

def train_svm():
    data = {
        'years_of_experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'test_score': [65, 70, 75, 80, 85, 90, 95, 90, 85, 100],
        'suitability': ['Unsuitable', 'Unsuitable', 'Suitable', 'Suitable', 'Suitable', 'Suitable', 'Suitable', 'Suitable', 'Unsuitable', 'Suitable']
    }
    df = pd.DataFrame(data)
    X = df[['years_of_experience', 'test_score']]
    y = df['suitability']
    
    model = SVC(kernel='linear')
    model.fit(X, y)
    
    return model

def predict_suitability(model, years_of_experience, test_score):
    return model.predict(np.array([[years_of_experience, test_score]]))

def train_decision_tree():
    data = {
        'performance_score': [95, 85, 70, 90, 60, 75, 80, 95, 65, 85],
        'years_with_company': [3, 4, 2, 5, 1, 3, 4, 6, 2, 4],
        'promotion_decision': ['Promoted', 'Promoted', 'Not Promoted', 'Promoted', 'Not Promoted', 'Not Promoted', 'Promoted', 'Promoted', 'Not Promoted', 'Promoted']
    }
    df = pd.DataFrame(data)
    X = df[['performance_score', 'years_with_company']]
    y = df['promotion_decision']
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    return model

def predict_promotion(model, performance_score, years_with_company):
    return model.predict(np.array([[performance_score, years_with_company]]))

def train_ann():
    data = {
        'leadership_score': [80, 85, 75, 90, 60, 70, 95, 80, 55, 85],
        'years_of_experience': [5, 6, 4, 7, 3, 5, 8, 6, 2, 7],
        'success_in_role': ['Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes']
    }
    df = pd.DataFrame(data)
    X = df[['leadership_score', 'years_of_experience']]
    y = df['success_in_role']
    
    model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
    model.fit(X, y)
    
    return model


# In models/ml_models.py
def predict_success(model, leadership_score, years_of_experience):
    return model.predict(np.array([[leadership_score, years_of_experience]]))
