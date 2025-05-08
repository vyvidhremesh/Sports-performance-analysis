import pandas as pd

class IPLSeasonSummaryModel:
    def __init__(self, matches_df):
        """
        Initialize with a DataFrame containing IPL match data.
        """
        self.matches_df = matches_df

    def get_season_summary(self, season_year):
        """
        Returns summary statistics for a given IPL season.
        """
        season_data = self.matches_df[self.matches_df['season'] == season_year]

        if season_data.empty:
            return f"No data found for season {season_year}."

        summary = {
            "Season": season_year,
            "Total Matches": len(season_data),
            "Total Teams": season_data['team1'].nunique(),
            "Most Wins Team": season_data['winner'].value_counts().idxmax(),
            "Team Wins Count": season_data['winner'].value_counts().to_dict(),
            "Player of the Match Count": season_data['player_of_match'].value_counts().to_dict(),
            "Venues Used": season_data['venue'].nunique()
        }

        return summary

    def get_team_performance(self, team_name):
        """
        Returns win/loss performance for a specific team.
        """
        matches = self.matches_df[(self.matches_df['team1'] == team_name) | (self.matches_df['team2'] == team_name)]

        wins = matches[matches['winner'] == team_name].shape[0]
        losses = matches.shape[0] - wins

        return {
            "Team": team_name,
            "Total Matches": matches.shape[0],
            "Wins": wins,
            "Losses": losses
        }
