# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and data
model = joblib.load("player_winrate_model.pkl")
features = joblib.load("player_features.pkl")
player_df = pd.read_csv("player_stats.csv")

st.title("🏏 IPL Player Winning Rate Predictor")

# Player selection
player_names = player_df['batsman'].sort_values().unique()
selected_player = st.selectbox("Select a Player", player_names)

# Display stats and prediction
if selected_player:
    player_data = player_df[player_df['batsman'] == selected_player][features].iloc[0:1]
    win_rate = model.predict(player_data)[0]

    st.subheader("📊 Player Stats")
    st.dataframe(player_data.T.rename(columns={player_data.index[0]: "Value"}))

    st.subheader("🎯 Predicted Win Rate")
    st.success(f"{win_rate:.4f}")
