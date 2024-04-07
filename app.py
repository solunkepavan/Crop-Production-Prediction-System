from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import warnings
import streamlit as st

warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
st.set_page_config(page_title="Crop Production Prediction App")

st.title("Crop Production Prediction App")
st.write("This app predicts the production of crops in Maharashtra, India.")
df = pd.read_csv('./dataset/Maharastra Crop Production final.csv')
df = df.dropna()
st.write("Enter the following values for the crop you want to predict:")
district = st.selectbox('District', df['District '].unique())
crop = st.selectbox('Crop', df['Crop'].unique())
year = st.slider('Crop Year', 1997, 2030, 2006)
season = st.selectbox('Season', df['Season'].unique())
area = st.number_input('Area', min_value=0, max_value=1000000)

def load_model():
    # Load the trained model
    return pickle.load(open('model.pkl','rb'))


def predict_production(model, district, crop, year, season, area):
    # Create a sample input
    sample_input = pd.DataFrame({
        'District ': [district],
        'Crop': [crop],
        'Crop_Year': [year],
        'Season': [season],
        'Area ': [area]
    })

    # Use the trained pipeline to predict the production for the sample input
    predicted_production = model.predict(sample_input)


    # Return the predicted production
    return predicted_production[0]
model = load_model()
predicted_production = predict_production(model, district, crop, year, season, area)
# st.write("The predicted production for the crop in the given district, year, and season is:", predicted_production)

st.write('<p style="font-size:40px; color:white; font-family: Times New Roman;";>The predicted production for the crop in the given district, year, and season is: {}</p>'.format(predicted_production), unsafe_allow_html=True)
