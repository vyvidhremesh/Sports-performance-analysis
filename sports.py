import streamlit as st
import pandas as pd
import plotly.express as px

# Load datasets
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

st.set_page_config(layout="wide")
st.title("IPL Dashboard - Matches & Deliveries Analysis")

# --- First Row: Matches per Season and Top Teams ---
col1, col2 = st.columns(2)

with col1:
    season_count = matches['season'].value_counts().sort_index()
    fig1 = px.bar(x=season_count.index, y=season_count.values,
                  labels={'x': 'Season', 'y': 'Matches'},
                  title='Number of Matches per IPL Season')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    team_wins = matches['winner'].value_counts().reset_index()
    team_wins.columns = ['Team', 'Wins']
    fig2 = px.bar(team_wins.head(10), x='Wins', y='Team', orientation='h',
                  title='Top 10 Most Successful Teams', color='Wins')
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

# --- Second Row: Top Batsmen and Bowlers ---
col3, col4 = st.columns(2)

with col3:
    top_batsmen = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10).reset_index()
    fig3 = px.bar(top_batsmen, x='batsman', y='batsman_runs', color='batsman_runs',
                  title='Top 10 Run Scorers', labels={'batsman_runs': 'Runs', 'batsman': 'Batsman'})
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    wickets = deliveries[deliveries['player_dismissed'].notna()]
    top_bowlers = wickets['bowler'].value_counts().head(10).reset_index()
    top_bowlers.columns = ['Bowler', 'Wickets']
    fig4 = px.bar(top_bowlers, x='Bowler', y='Wickets', color='Wickets',
                  title='Top 10 Wicket Takers')
    st.plotly_chart(fig4, use_container_width=True)
