#pip install streamlit
#pip install pandas
#pip install sklearn

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
df = pd.read_csv(r'diabetes.csv')

# Page Configuration
st.set_page_config(page_title='Diabetes Prediction', page_icon=':hospital:', layout='wide')

# Sidebar Design
st.sidebar.image(r'diabetes-logo.png', use_column_width=True)
st.sidebar.title('Patient Data')
st.sidebar.subheader('Enter Your Information')

# Get the user input
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    return pd.DataFrame(user_data, index=[0])

user_data = get_user_input()

# Main Content Design
st.title('Diabetes Prediction')
st.write('Welcome to our Diabetes Prediction App!')
st.write('Please provide your information in the sidebar and click on the "Predict" button below.')

if st.button('Predict'):
    st.subheader('Patient Data')
    st.write(user_data)

    # Prepare data for modeling
    x = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Train the model
    rf = RandomForestClassifier()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    rf.fit(x_train, y_train)

    # Make prediction
    user_result = rf.predict(user_data)

    # Show prediction result
    st.subheader('Prediction Result')
    if user_result[0] == 0:
        st.write('Based on the provided information, you are predicted to be **Not Diabetic**.')
    else:
        st.write('Based on the provided information, you are predicted to be **Diabetic**.')

    # Show model accuracy
    st.subheader('Model Accuracy')
    fig, ax = plt.subplots()
    sns.barplot(x=['Model Accuracy'], y=[accuracy_score(y_test, rf.predict(x_test))], palette=['blue'], ax=ax)
    ax.set(ylabel='Accuracy')
    st.pyplot(fig)

# Footer Design
st.markdown('---')

