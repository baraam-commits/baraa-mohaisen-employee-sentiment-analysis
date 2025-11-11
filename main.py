# temporary file used for development. final code will be run in jupyter notebook in notebooks/main.ipynb
import pandas as pd
import os
from src.load_data import LoadData
from src.labeling import SentimentLabeler
from src.plot_data import PlotData
from src.ranking import EmployeeScoring , EmployeeRanking
from src.regression import FeatureEngineer, TrainRegressionModel, PredictScore
from src.model_plotter import ModelPlotter



def load_data(file_path = "data\\labeld_sentiments.csv"):
    df = None
    if not os.path.exists(file_path):
        data_loader = LoadData()
        
        df = data_loader.load_pandas_dataframe("data\\test(in).csv")

        sl = SentimentLabeler(df)

        df = sl.get_sentiments()
        df.to_csv(file_path, index=False)

    else:
        
        data_loader = LoadData()
        
        df = data_loader.load_pandas_dataframe(clean=False, file_path= file_path)

    print(df)
    return df





labled_sent_df = load_data()
monthly_sent = load_data("data\\employee_monthly_sentiment_scores.csv")
print(labled_sent_df.head)
p = PlotData(labled_sent_df)
p.run_all_plots()

# teacher = TrainRegressionModel(monthly_sent,labled_sent_df)
# teacher.train()
# teacher.evaluate()
# teacher.save_model_artifacts()

# predictor = PredictScore.predict()
plotter = ModelPlotter(labled_sent_df,monthly_sent)
plotter.run_all_plots()
# # plot_preliminary_data()
# rank = EmployeeScoring(df)
# ranker = EmployeeRanking(df)
# temp = (rank.compute_scores())

# print(temp)

# print(ranker.get_positive_rankings())
# print(ranker.get_negative_rankings())
# print(ranker.get_rankings())

# print(rank.flight_risk_analysis(False))
# print(rank.flight_risk_analysis(True))

