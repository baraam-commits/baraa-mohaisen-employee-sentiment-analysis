# temporary file used for development. final code will be run in jupyter notebook in notebooks/main.ipynb
import pandas as pd
import os
from src.load_data import LoadData
from src.labeling import SentimentLabeler
from src.Plot_data import PlotData

global df
df = None

def load_data():
    if not os.path.exists("data\\labeld_sentiments.csv"):
        data_loader = LoadData("data\\test(in).csv")
        
        global df
        df = data_loader.load_pandas_dataframe()

        sl = SentimentLabeler(df)

        df = sl.get_sentiments()
        df.to_csv("data\\labeld_sentiments.csv", index=False)

    else:
        
        data_loader = LoadData("data\\labeld_sentiments.csv")
        
        df = data_loader.load_pandas_dataframe(clean=False)

    print(df)
def plot_preliminary_data():
    p = PlotData(df)
    p.plot_sentiment_distribution()
    p.plot_message_activity_over_time()
    p.message_length_distribution()
    p.plot_avg_sentiment_over_time()
    p.plot_top_employees()
    p.plot_sentiment_per_employee()
    p.plot_length_by_sentiment()
    p.plot_avg_message_length_per_employee()
    
load_data()
print(df)
plot_preliminary_data()