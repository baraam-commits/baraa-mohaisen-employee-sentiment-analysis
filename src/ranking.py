import pandas as pd 

class EmployeeScoring:

    global SENT_MAP 
    SENT_MAP = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0,
            "Positive": 1, "Negative": -1, "Neutral": 0}
    """
    Task 3: Compute monthly sentiment scores per employee.
    Inputs required in df: ['employee_id','month','sentiment_Score'].
    Outputs: monthly scores table with counts per employee.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.df["sentiment_num"] = self.df["sentiment"].map(SENT_MAP)
        self.rankings = pd.DataFrame()
    def get_rankings(self) -> pd.DataFrame:
        
        self.rankings = self.df.groupby(["employee_id", "date"], as_index = False)["sentiment_num"].mean()
        self.rankings["date"] = self.rankings["date"].dt.to_period("M")
        self.rankings = self.rankings.groupby(["employee_id", "date"], as_index = False).sum()
        return self.rankings

        
         
    


  