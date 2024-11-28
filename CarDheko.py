
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from catboost import CatBoostError, Pool

st.set_page_config(page_title='Used Car Price Prediction',
                   layout="centered", page_icon='🚗')

# Load model function
@st.cache_resource
def model_loader(path):
    model = joblib.load(path)
    return model

# Load machine learning model
with st.spinner('🚕🛺🚙🚜🚚🚓🚗🚕 Hold on, the app is loading !! 🚕🛺🚙🚜🚚🚓🚗🚕'):
    ml_model = model_loader("xgb_model2.pkl")

# App title
st.markdown("<h2 style='text-align: center;'>🚗 Used Car Price Prediction™ 🚗</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.sidebar.markdown("# :rainbow[Select an option to filter:]")
    selected = st.selectbox("**Menu**", ("Home","prediction"))

if selected=="Home":   
    st.markdown('## :green[welcome to Home page:]')
    st.markdown('## :blue[Project Title:]')
    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;CarDekho Used Car Price Prediction ")
    st.markdown('## :blue[Skills takes away From This project:]')
    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Data Cleaning, Exploratory Data Analysis (EDA), Visualization and Machine Learning")
    st.markdown('## :blue[Domain:]')
    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Automobile")
    st.markdown('## :blue[Problem:]')
    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; The primary objective of is project is to create a data science solution for the car")
    st.subheader('for predicting used car prices accurately by analyzing a diverse dataset including car model,')
    st.subheader('no. of owners, age, mileage, fuel type, kilometers driven, features and location. The aim is ')
    st.subheader('to build a machine learning model that offers users to find current valuations for used cars.')

if selected=='prediction':
    st.markdown('## :green[welcome to Prediction values:]')
    
    column1, column2 = st.columns([2,2], gap='small')

    # Define brand and model mappings outside the column blocks
    brands = np.array(["Audi", "BMW", "Chevrolet", "Citroen", "Datsun", "Fiat", "Ford", "Hindustan", "Honda", "Hyundai",
                       "Isuzu", "Jaguar", "Jeep", "Kia", "Land", "Lexus", "MG", "Mahindra", "Maruti", "Mercedes-Benz",
                       "Mini", "Mitsubishi", "Nissan", "OpelCorsa", "Porsche", "Renault", "Skoda", "Tata", "Toyota",
                       "Volkswagen", "Volvo"])

    brand_to_models = {
        "Audi": ['A3', 'A4', 'A6', 'A8', 'Q3', 'Q5', 'Q7', 'Q2'],
        "BMW": ['X3', 'X5', 'X1', '3 Series', '7 Series', '5 Series', '6 Series', '2 Series'],
        "Chevrolet": ['Sail', 'Beat', 'Tavera', 'Aveo', 'Spark', 'Cruze'],
        "Citroen": ['C5 Aircross', 'C3'],
        "Datsun": ['GO', 'RediGO', 'GO Plus'],
        "Fiat": ['Punto', 'Linea', 'Abarth Avventura', 'Grande Punto', 'Palio', 'Punto EVO', 'Punto Pure'],
        "Ford": ['Ecosport', 'Endeavour', 'Figo', 'Freestyle', 'Mondeo', 'Fiesta', 'Ikon', 'Aspire'],
        "Hindustan": ['Contessa'],
        "Honda": ['Jazz', 'City', 'Amaze', 'Civic', 'BR-V', 'CR-V', 'Accord', 'Brio', 'Mobilio', 'WR-V'],
        "Hyundai": ['i10', 'i20', 'Creta', 'Verna', 'Venue', 'Grand i10', 'Santro', 'EON', 'Santro Xing', 'Xcent', 'Tucson'],
    }

    # Label Encoding for brand and model (defined outside the column block)
    label_encoder_brand = LabelEncoder()
    label_encoder_model = LabelEncoder()

    # Fit the encoders
    encoded_brands = label_encoder_brand.fit_transform(brands)
    encoded_models = label_encoder_model.fit_transform([model for models in brand_to_models.values() for model in models])

    with column1:
        # Brand selection
        selected_brand = st.selectbox('Select Brand', brands)
        models_for_selected_brand = brand_to_models.get(selected_brand, [])
        selected_model = st.selectbox('Select Model', models_for_selected_brand if models_for_selected_brand else ["No models available"])

        # Encode selected brand and model
        encoded_selected_brand = label_encoder_brand.transform([selected_brand])[0]
        encoded_selected_model = label_encoder_model.transform([selected_model])[0] if models_for_selected_brand else -1

        year = st.number_input('Year of Manufacture', min_value=1900, max_value=2024, value=2015)
        fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid'])
        fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'Electric': 3, 'Hybrid': 4}
        fuel_type_encoded = fuel_type_mapping[fuel_type]

        owner = st.selectbox('Select the No. of Owners', ['First Owner', 'Second Owner', 'Others'])
        owner_mapping = {'First Owner': 0, 'Second Owner': 1, 'Others': 2}
        owner_encoded = owner_mapping[owner]

    with column2:
        km_driven = st.number_input(label='Enter KM driven (enter number only)', help='how much the car is driven?')
        transmission = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
        transmission_encoded = 0 if transmission == 'Manual' else 1

        city = st.selectbox('City', ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad'])
        city_mapping = {'Delhi': 0, 'Mumbai': 1, 'Bangalore': 2, 'Chennai': 3, 'Kolkata': 4, 'Hyderabad': 5}
        city_encoded = city_mapping[city]

    # Prepare the feature array
    input_features = np.array([[encoded_selected_brand, encoded_selected_model, year, fuel_type_encoded, 
                                transmission_encoded, city_encoded, owner_encoded, km_driven]])

    if st.button('Predict'):
        try:
            # Create a Pool object with the input features
            input_pool = Pool(data=input_features)
            # Make the prediction
            prediction = ml_model.predict(input_pool)
            st.success(f"The predicted price of the car is ₹ {prediction[0]:,.2f}")
        except CatBoostError as e:
            st.error(f"Model encountered an error: {str(e)}")
