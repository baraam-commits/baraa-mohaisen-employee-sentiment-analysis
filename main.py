# temporary file used for development. final code will be run in jupyter notebook in notebooks/main.ipynb
import pandas as pd
import os
from src.load_data import LoadData
from src.labeling import SentimentLabeler
from src.plot_data import PlotData, ModelPlotter, FlightRiskPlots
from src.ranking import EmployeeScoring , EmployeeRanking
from src.regression import FeatureEngineer, TrainRegressionModel, PredictScore




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

rank = EmployeeScoring(labled_sent_df)
rank.flight_risk_analysis(False,8).to_csv('data\\flight_risks_int8.csv')
rank.flight_risk_analysis(False,6).to_csv('data\\flight_risks_int6.csv')
rank.flight_risk_analysis(False,4).to_csv('data\\flight_risks_int4.csv')

# p = PlotData(labled_sent_df)
# p.run_all_plots()
# p.plot_avg_polarity_over_time()

# p.plot_avg_sentiment_over_time()

# teacher = TrainRegressionModel(monthly_sent,labled_sent_df)
# teacher.train()
# teacher.evaluate()
# teacher.save_model_artifacts()

# predictor = PredictScore.predict()
# plotter = ModelPlotter(labled_sent_df,monthly_sent)
# plotter.run_all_plots()

# # plot_preliminary_data()

ranker = EmployeeRanking(monthly_sent)
# temp = (rank.compute_scores())

# print(temp)

# print(ranker.get_positive_rankings().head())
# print(ranker.get_negative_rankings().head())
# ranker.get_rankings().head().to_csv('data\\temp1.csv')
# ranker.get_rankings(True).head().to_csv('data\\temp2.csv')

# print(rank.flight_risk_analysis(False))
# print(rank.flight_risk_analysis(True))

