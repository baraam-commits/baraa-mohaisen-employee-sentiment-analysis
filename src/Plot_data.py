import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

class PlotData:
    def __init__(self, df):
        self.df = df

    def plot_sentiment_distribution(self, sentiment_column='sentiment'):
        sentiment_counts = self.df[sentiment_column].value_counts()
        sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()
        plt.savefig("visualizations/sentiment_distribution.png")
    
    def plot_message_activity_over_time(self, date_column='dt'):
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        message_counts = self.df.set_index(date_column).resample('M').size()
        message_counts.plot(kind='bar')
        plt.title('Message Activity Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Messages')
        plt.show()
        plt.savefig("visualizations/message_activity_over_time.png")
    
    def message_length_distribution(self, text_column='body'):
        self.df['message_length'] = self.df[text_column].apply(lambda x: len(str(x)))
        self.df['message_length'].plot(kind='hist', bins=30)
        plt.title('Message Length Distribution')
        plt.xlabel('Message Length')
        plt.ylabel('Frequency')
        plt.show()
        plt.savefig("visualizations/message_length_distribution.png")