import streamlit as st
import numpy as np
import pickle

# Load models
model = pickle.load(open("best_diamond_model.pkl", "rb"))
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))

st.title("💎 Diamond Price Prediction & Segmentation")

# User Inputs
carat = st.number_input("Carat", min_value=0.0, value=0.5, step=0.01)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["IF","VVS1","VVS2","VS1","VS2","SI1","SI2","I1"])
length = st.number_input("Length (x)", min_value=0.0, value=5.0, step=0.1)
width = st.number_input("Width (y)", min_value=0.0, value=5.0, step=0.1)
height = st.number_input("Height (z)", min_value=0.0, value=3.0, step=0.1)

# Encoding maps
cut_map = {'Fair':1, 'Good':2, 'Very Good':3, 'Premium':4, 'Ideal':5}
color_map = {'D':7, 'E':6, 'F':5, 'G':4, 'H':3, 'I':2, 'J':1}
clarity_map = {'IF':8, 'VVS1':7, 'VVS2':6, 'VS1':5, 'VS2':4, 'SI1':3, 'SI2':2, 'I1':1}

# Convert input
cut_encoded = cut_map[cut]
color_encoded = color_map[color]
clarity_encoded = clarity_map[clarity]

# Feature Engineering
volume = length * width * height
dimension_ratio = (length + width) / (2 * height) if height > 0 else 0

# EXACTLY 10 FEATURES (matches model training)
input_data = np.array([[
    carat,             # 1
    cut_encoded,       # 2
    color_encoded,     # 3
    clarity_encoded,   # 4
    length,            # 5 (x)
    width,             # 6 (y)
    height,            # 7 (z)
    volume,            # 8
    0,                 # 9 (dummy - price_per_carat)
    dimension_ratio    # 10
]])

# Predict Price
if st.button("💰 Predict Price"):
    try:
        price = model.predict(input_data)
        st.success(f"💎 Predicted Price: ₹ {price[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

# Predict Cluster
if st.button("📊 Predict Cluster"):
    try:
        cluster = kmeans.predict(input_data)
        cluster_names = {
            0: "Affordable Small Diamonds",
            1: "Mid-range Balanced Diamonds",
            2: "Premium Heavy Diamonds"
        }
        st.success(f"📌 Cluster: {cluster[0]} - {cluster_names[cluster[0]]}")
    except Exception as e:
        st.error(f"Error: {e}")