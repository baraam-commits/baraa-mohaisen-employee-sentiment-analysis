# temporary file used for development. final code will be run in jupyter notebook in notebooks/main.ipynb
import pandas as pd
import os
from src.load_data import LoadData
from src.labeling import SentimentLabeler
from src.Plot_data import PlotData

df = None
if not os.path.exists("data\\labeld_sentiments.csv"):
    data_loader = LoadData("data\\test(in).csv")
    df = data_loader.load_pandas_dataframe()

    sl = SentimentLabeler(df)

    df = sl.get_sentiments()

    print(df)
    df.to_csv("data\\labeld_sentiments.csv", index=False)

else:
    data_loader = LoadData("data\\labeld_sentiments.csv")
    df = data_loader.load_pandas_dataframe(clean=False)
    

plotter = PlotData(df)
plotter.plot_sentiment_distribution()
plotter.plot_message_activity_over_time()
plotter.message_length_distribution()

