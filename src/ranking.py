import pandas as pd 

class EmployeeScoring:

    SENT_MAP = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0,
            "Positive": 1, "Negative": -1, "Neutral": 0}
    """
    Task 3: Compute monthly sentiment scores per employee.
    Inputs required in df: ['employee_id','month','sentiment_Score'].
    Outputs: monthly scores table with counts.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.rankings_df = pd.DataFrame()
        
    def _compute_month_sentiment_scores(self, month_df) -> dict:
        
        employee_score = {}

        
        return employee_score
    
    """" Handel getting data per month and passes it onto the scoring function, then returns the final rankings dataframe."""
    def _get_monthly_data(self):    
        pass
    
    def _aggregate_monthly_scores(self):
        pass

    def get_rankings(self) -> pd.DataFrame:
        
        return self.rankings_df