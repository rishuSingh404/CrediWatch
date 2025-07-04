# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:15:47 2024

@author: Admin
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load model with error handling
try:
    model_data = joblib.load("model/model_data.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure the model file exists and is compatible with current XGBoost version")
    model_data = None


# Initialize model components with error handling
if model_data is not None:
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        columns_to_scale = model_data['cols_to_scale']
        print("✅ Model components initialized successfully")
    except KeyError as e:
        print(f"❌ Missing model component: {e}")
        model = None
        scaler = None
        features = None
        columns_to_scale = None
else:
    model = None
    scaler = None
    features = None
    columns_to_scale = None


def data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                     loan_amount, loan_tenure_months, total_loan_months, 
                     loan_purpose, loan_type, residence_type):
    """Prepare data for prediction with error handling"""
    if model is None or scaler is None or features is None or columns_to_scale is None:
        raise ValueError("Model not properly loaded. Please check model files.")
    
    data_input = {'age': age,
                  'avg_dpd_per_dm': avg_dpd_per_dm,
                  'credit_utilization_ratio': credit_utilization_ratio,
                  'dmtlm': dmtlm,
                  'income': income,
                  'loan_amount': loan_amount,
                  'lti': loan_amount / income if income > 0 else 0,
                  'total_loan_months': total_loan_months,
                  'loan_tenure_months': loan_tenure_months,
                  'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
                  'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
                  'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
                  'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
                  'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
                  'residence_type_Rented': 1 if residence_type == 'Rented' else 0}
    
    df = pd.DataFrame([data_input])
    
    try:
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        df = df[features]
        return df
    except Exception as e:
        raise ValueError(f"Error in data preprocessing: {e}")


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    """Calculate credit score with error handling"""
    if model is None:
        raise ValueError("Model not available for prediction")
    
    try:
        default_probability = model.predict_proba(input_df)[:, 1]  # Probability of default
        non_default_probability = 1 - default_probability

        # Calculate the credit score based on the probabilities
        credit_score = base_score + non_default_probability.flatten() * scale_length
        
        # Determine the rating category based on the credit score
        def get_rating(score):
            if 300 <= score < 500:
                return 'Poor'
            elif 500 <= score < 650:
                return 'Average'
            elif 650 <= score < 750:
                return 'Good'
            elif 750 <= score <= 900:
                return 'Excellent'
            else:
                return 'Undefined'  # in case of any unexpected score

        rating = get_rating(credit_score[0])

        return default_probability.flatten()[0], int(credit_score), rating
    except Exception as e:
        raise ValueError(f"Error in credit score calculation: {e}")


def predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                     loan_amount, loan_tenure_months, total_loan_months, 
                     loan_purpose, loan_type, residence_type):
    """Main prediction function with comprehensive error handling"""
    try:
        input_df = data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                             loan_amount, loan_tenure_months, total_loan_months, 
                             loan_purpose, loan_type, residence_type)

        probability, credit_score, rating = calculate_credit_score(input_df)

        return probability, credit_score, rating
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        # Return default values in case of error
        return 0.5, 500, 'Average'
