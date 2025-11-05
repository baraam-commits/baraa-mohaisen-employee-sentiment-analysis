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
    def compute_scores(self) -> pd.DataFrame:
        
        self.rankings = self.df.groupby(["employee_id", "date"], as_index = False)["sentiment_num"].mean()
        self.rankings["date"] = self.rankings["date"].dt.to_period("M")
        self.rankings = self.rankings.groupby(["employee_id", "date"], as_index = False).sum()
        return self.rankings

class EmployeeRanking:
    """
    EmployeeRanking(df)
    Rank employees monthly based on computed sentiment scores.
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing raw employee sentiment data. Required columns depend
        on the preprocessing performed by EmployeeScoring.compute_scores, but the input
        must at minimum allow EmployeeScoring to produce the following columns on the
        returned dataframe:
          - 'employee_id'    : identifier for each employee
          - 'date' (or canonical monthly period) : grouping key for monthly rankings
          - 'sentiment_num'  : numeric sentiment score used for sorting and ranking
    Attributes
    ----------
    df : pandas.DataFrame
        The dataframe returned by EmployeeScoring(df).compute_scores(), and used for
        ranking operations.
    rankings : pandas.DataFrame
        The last computed rankings table (populated by any of the get_* methods).
    Methods
    -------
    get_positive_rankings() -> pandas.DataFrame
        Return the top 3 employees per month with the highest sentiment scores.
        - Sorts by ['date', 'sentiment_num', 'employee_id'] with sentiment_num in
          descending order (highest first) and employee_id ascending as a tiebreaker.
        - Groups by 'date' and selects the first 3 rows per group.
        - Updates self.rankings and returns the resulting DataFrame.
    get_negative_rankings() -> pandas.DataFrame
        Return the top 3 employees per month with the lowest sentiment scores.
        - Sorts by ['date', 'sentiment_num', 'employee_id'] with sentiment_num in
          ascending order (lowest first) and employee_id ascending as a tiebreaker.
        - Groups by 'date' and selects the first 3 rows per group.
        - Updates self.rankings and returns the resulting DataFrame.
    get_rankings() -> pandas.DataFrame
        Produce a combined rankings table containing both Negative and Positive top-3
        lists per month.
        - Calls get_negative_rankings() and get_positive_rankings(), adds a
          'Type' column with values "Negative" and "Positive" respectively.
        - Concatenates the two tables and sorts by ['date', 'Type', 'sentiment_num', 'employee_id'].
        - Returns the concatenated, sorted DataFrame. The returned table includes the
          'Type' column and the score columns produced by EmployeeScoring.
    Notes
    -----
    - This class depends on EmployeeScoring.compute_scores() to normalize/compute the
      numeric sentiment column ('sentiment_num') and the date/month grouping column.
    - Ties in sentiment_num are broken by employee_id ascending.
    - Each ranking method updates the instance attribute self.rankings as a side effect.
    - The methods select the top 3 rows per date; adjust grouping or selection logic if
      a different number of top employees is required.
    Example
    -------
    # After creating the object:
    ranker = EmployeeRanking(raw_df)
    positives = ranker.get_positive_rankings()   # top 3 per month by highest sentiment
    negatives = ranker.get_negative_rankings()   # top 3 per month by lowest sentiment
    combined = ranker.get_rankings()             # combined table with 'Type' column

    Task 4: Rank employees monthly based on sentiment scores.
    Inputs required in df: ['employee_id','month','sentiment_Score'].
    Outputs: monthly rankings table with top employees.
        
        """
    
    def __init__(self, df):
        self.df = EmployeeScoring(df).compute_scores()
        self.rankings = pd.DataFrame()
        
    def get_positive_rankings(self) -> pd.DataFrame:
        self.rankings = self.df.sort_values(by=["date","sentiment_num", "employee_id"], ascending=[True, False, True])
        self.rankings = self.rankings.groupby("date", as_index= False).head(3)
        return self.rankings
    
    def get_negative_rankings(self) -> pd.DataFrame:
        self.rankings = self.df.sort_values(by=["date","sentiment_num", "employee_id"], ascending=[True, True, True])
        self.rankings = self.rankings.groupby("date", as_index= False).head(3)
        return self.rankings

    def get_rankings(self) -> pd.DataFrame:
        negative_rankings = self.get_negative_rankings()
        negative_rankings["Type"] = "Negative"
        positive_rankings = self.get_positive_rankings()
        positive_rankings["Type"] = "Positive"
        
        grouped_rankings = pd.concat([negative_rankings, positive_rankings], ignore_index=True)
        
        grouped_rankings.sort_values(by=["date","Type","sentiment_num", "employee_id"], ascending=[True, False, False, True], inplace=True)
        # grouped_rankings.drop(columns=["Type"], inplace=True)
        return grouped_rankings