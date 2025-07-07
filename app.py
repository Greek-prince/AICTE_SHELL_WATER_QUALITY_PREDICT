import pandas as pd
import streamlit as st
import joblib
import numpy as np
import os

MODEL_PATH = "pollution_model.pkl"
MODEL_COLUMNS_PATH = "model_columns.pkl"
DATA_PATH = "PB_All_2000_2021.csv"


@st.cache_resource
def load_resources():
    """
    Loads the machine learning model, model columns, and the dataset.
    Uses st.cache_resource to avoid reloading on every rerun.
    """
    model = None
    model_columns = None
    data = None

    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory.")
        return None, None, None
    if not os.path.exists(MODEL_COLUMNS_PATH):
        st.error(f"Error: Model columns file '{MODEL_COLUMNS_PATH}' not found. Please ensure it's in the same directory.")
        return None, None, None
    if not os.path.exists(DATA_PATH):
        st.error(f"Error: Data file '{DATA_PATH}' not found. Please ensure it's in the same directory.")
        return None, None, None

    try:
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(MODEL_COLUMNS_PATH)
        st.success("Model and model columns loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model or model columns: {e}")
        return None, None, None

    try:
        data = pd.read_csv(DATA_PATH, sep=';')
        data.columns = data.columns.str.strip()
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data from '{DATA_PATH}': {e}")
        return None, None, None

    return model, model_columns, data

model, model_columns, data = load_resources()

if model is None or model_columns is None or data is None:
    st.stop()

if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['Year'] = data['date'].dt.year
    if 'id' in data.columns:
        data['Station_ID'] = data['id']
    else:
        st.warning("Column 'id' not found in data. Cannot create 'Station_ID'.")
        st.stop()
else:
    st.warning("Column 'date' not found in data. Cannot extract 'Year'.")
    st.stop()

years = sorted(data['Year'].dropna().astype(int).unique().tolist())
stations = sorted(data['Station_ID'].dropna().astype(str).unique().tolist()) # Keep as string for station ID

TARGET_POLLUTANTS = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# --- Streamlit Application Layout ---

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #f0f2f6;
        border-radius: 0.5rem;
        border: 1px solid #4f525e;
    }
    .stSelectbox>div>div>div {
        background-color: #262730;
        color: #f0f2f6;
        border-radius: 0.5rem;
        border: 1px solid #4f525e;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        font-size: 1rem;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stButton>button:active {
        background-color: #3e8e41;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transform: translateY(1px);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f0f2f6;
    }
    p {
        color: #c9d1d9;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

# --- User Input ---
col1, col2 = st.columns(2)

with col1:
    selected_year = st.selectbox("Enter Year", years, index=len(years)-1 if years else 0) # Default to latest year
with col2:
    selected_station_id = st.selectbox("Enter Station ID", stations)

# --- Prediction Button ---
if st.button("Predict"):
    st.subheader(f"Predicted pollutant levels for the station '{selected_station_id}' in {selected_year}:")

    try:
       
        input_df = pd.DataFrame(columns=model_columns)

        # Fill in the 'Year' and 'Station_ID'
        input_data = {'Year': [selected_year], 'Station_ID': [selected_station_id]}

        input_dict = {col: 0 for col in model_columns} # Initialize all with 0 or a default
        input_dict['Year'] = selected_year
        input_dict['Station_ID'] = selected_station_id

        
        input_df = pd.DataFrame([input_dict])[model_columns]

        
        predictions = model.predict(input_df)

        if predictions.ndim > 1:
            predicted_values = predictions[0]
        else:
            predicted_values = predictions # If model predicts a single value

        # Display predictions
        if len(predicted_values) == len(TARGET_POLLUTANTS):
            for i, pollutant in enumerate(TARGET_POLLUTANTS):
                st.write(f"**{pollutant}:** {predicted_values[i]:.2f}")
        else:
            st.warning("Number of predicted values does not match expected pollutant list.")
            st.write("Raw predictions:", predicted_values)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure your 'pollution_model.pkl' and 'model_columns.pkl' are correctly generated and match the expected input features.")

