import numpy as np
import pandas as pd 
from ranking import EmployeeScoring
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


class TrainRegressionModel:
    
    def __init__(self, sentiment_scores_df: pd.DataFrame, raw_df: pd.DataFrame):
        
        self.raw_df = raw_df.copy().groupby(["date", "employee_id"], as_index= False)
        self.raw_df['month'] = self.raw_df['date'].dt.month
        self.raw_df['year'] = self.raw_df['date'].dt.year
        self.sentiment_scores_df = sentiment_scores_df.copy()
        
        if "sentiment_num" not in sentiment_scores_df: self.sentiment_scores_df["sentiment_num"] = EmployeeScoring(raw_df).compute_scores()["sentiment_num"]
        
        
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    def _prepare_features(self):
        
        self.raw_df
        
        def _compute_total_len_per_month(df: pd.DataFrame):
            
            result = (
                df.groupby(['month', 'employee_id'], as_index=False)['text_len']
                .sum()
            )

            return result[['employee_id', 'month', 'text_len']]
        
        def _compute_avg_len_per_month(df: pd.DataFrame):
            
            df = _compute_total_len_per_month(df)
            
            result = (
                df.groupby(['month', 'employee_id'], as_index=False)
                .agg(total_text_len=('text_len', 'sum'),
                    message_count=('text_len', 'count'))
            )

            result['avg_msg_len'] = result['total_text_len'] / result['message_count']
            return result[['employee_id', 'month', 'avg_msg_len']]
            
        def _compute_avg_punctuation_density(df: pd.DataFrame):
            pass
            
                        
            
            
        
        
        