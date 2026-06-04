import streamlit as st
import pickle
import json
import numpy as np

# Load the model
with open("../models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the columns
with open("../models/columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Get location names
locations = [col.replace("location_", "") for col in data_columns if col.startswith("location_")]

# Page config
st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏠", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .block-container { padding: 2rem 3rem; }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #1e2130;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2d3148;
    }
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1a3a2a, #1e2130);
        border: 1px solid #22c55e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-label {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-bottom: 0.5rem;
    }
    .result-price {
        font-size: 2.8rem;
        font-weight: 700;
        color: #22c55e;
    }
    .result-sub {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
    div.stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        transition: background 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🏠 Bengaluru House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get an instant estimate for any house in Bengaluru based on location, size, and configuration.</div>', unsafe_allow_html=True)

# Input card
st.markdown('<div class="section-title">Property Details</div>', unsafe_allow_html=True)

location = st.selectbox("📍 Location", sorted(locations))

col1, col2 = st.columns(2)
with col1:
    sqft = st.number_input("📐 Total Square Feet", min_value=300, max_value=10000, value=1000, step=50)
with col2:
    bhk = st.number_input("🛏 Number of BHK", min_value=1, max_value=10, value=2)

col3, col4 = st.columns(2)
with col3:
    bath = st.number_input("🚿 Number of Bathrooms", min_value=1, max_value=10, value=2)
with col4:
    st.write("")
    st.write("")

st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("🔍 Predict Price"):
    x = np.zeros(len(data_columns))
    x[data_columns.index("total_sqft")] = sqft
    x[data_columns.index("bath")] = bath
    x[data_columns.index("bhk")] = bhk
    x[data_columns.index("price_per_sqft")] = 6206

    loc_index = np.where(np.array(data_columns) == "location_" + location)[0]
    if len(loc_index) > 0:
        x[loc_index[0]] = 1

    prediction = round(model.predict([x])[0], 2)
    prediction_crore = round(prediction / 100, 2)

    st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Property Value</div>
            <div class="result-price">₹ {prediction} Lakhs</div>
            <div class="result-sub">≈ ₹ {prediction_crore} Crore &nbsp;|&nbsp; 📍 {location} &nbsp;|&nbsp; {bhk} BHK &nbsp;|&nbsp; {int(sqft)} sqft</div>
        </div>
    """, unsafe_allow_html=True)
