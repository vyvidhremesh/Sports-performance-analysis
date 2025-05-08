# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load data
matches = pd.read_csv('matches_stats.csv')
deliveries = pd.read_csv('ipl_stats.csv')
data = pd.merge(deliveries, matches[['id', 'winner']], left_on='match_id', right_on='id', how='left')

# Player match outcomes
player_match_outcomes = data[['match_id', 'batsman', 'batting_team', 'winner']].drop_duplicates()
player_match_outcomes['won'] = (player_match_outcomes['batting_team'] == player_match_outcomes['winner']).astype(int)

# Aggregate player-level win data
player_stats = player_match_outcomes.groupby('batsman').agg(
    total_matches=('match_id', 'nunique'),
    total_wins=('won', 'sum')
).reset_index()

# Smoothed win rate
prior = player_stats['total_wins'].sum() / player_stats['total_matches'].sum()
player_stats['win_rate_smooth'] = (
    (player_stats['total_wins'] + 3 * prior) / (player_stats['total_matches'] + 3)
)

# Batting stats
batting_perf = data.groupby('batsman').agg(
    total_runs=('batsman_runs', 'sum'),
    total_balls=('ball', 'count'),
    matches_played=('match_id', 'nunique')
).reset_index()

# Dismissals
dismissals = deliveries[~deliveries['player_dismissed'].isna()]
dismissed_count = dismissals.groupby('player_dismissed')['match_id'].count().reset_index()
dismissed_count.columns = ['batsman', 'times_out']

# Merge batting stats
player_df = pd.merge(player_stats, batting_perf, on='batsman', how='left')
player_df = pd.merge(player_df, dismissed_count, on='batsman', how='left')
player_df['times_out'] = player_df['times_out'].fillna(0)

# Batting metrics
player_df['strike_rate'] = 100 * player_df['total_runs'] / player_df['total_balls']
player_df['avg_runs_per_match'] = player_df['total_runs'] / player_df['matches_played']
player_df['runs_per_ball'] = player_df['total_runs'] / player_df['total_balls']
player_df['batting_average'] = player_df['total_runs'] / player_df['times_out'].replace(0, 1)

# Bowling stats
bowling_stats = data.groupby('bowler').agg(
    total_balls_bowled=('ball', 'count'),
    total_runs_conceded=('total_runs', 'sum')
).reset_index()

wickets = deliveries[~deliveries['player_dismissed'].isna()]
bowler_wickets = wickets.groupby('bowler')['player_dismissed'].count().reset_index()
bowler_wickets.columns = ['bowler', 'wickets']

bowling_stats = pd.merge(bowling_stats, bowler_wickets, on='bowler', how='left')
bowling_stats['wickets'] = bowling_stats['wickets'].fillna(0)
bowling_stats['economy'] = 6 * bowling_stats['total_runs_conceded'] / bowling_stats['total_balls_bowled']
bowling_stats.rename(columns={'bowler': 'batsman'}, inplace=True)

# Merge with main data
player_df = pd.merge(player_df, bowling_stats, on='batsman', how='left')
player_df[['total_balls_bowled', 'total_runs_conceded', 'wickets', 'economy']] = player_df[
    ['total_balls_bowled', 'total_runs_conceded', 'wickets', 'economy']
].fillna(0)

# Filter out low-match players
player_df = player_df[player_df['total_matches'] >= 5]

# Final features
features = [
    'total_matches', 'total_runs', 'total_balls', 'strike_rate',
    'avg_runs_per_match', 'runs_per_ball', 'batting_average',
    'total_balls_bowled', 'total_runs_conceded', 'wickets', 'economy'
]

X = player_df[features].fillna(0)
y = player_df['win_rate_smooth']
players = player_df['batsman']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model and metadata
joblib.dump(model, 'player_winrate_model.pkl')
joblib.dump(features, 'player_features.pkl')
player_df[['batsman'] + features + ['win_rate_smooth']].to_csv('player_stats.csv', index=False)
