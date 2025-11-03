import matplotlib.pyplot as plt
import pandas as pd
import os

class PlotData:
    def __init__(self, df):
        """
        Initialize with a DataFrame containing sentiment and communication data.
        Creates an output directory for visualizations.
        """
        self.df = df
        os.makedirs("visualizations", exist_ok=True)

    # ------------------------------------------------------------
    # BASIC PLOTS
    # ------------------------------------------------------------
    def plot_sentiment_distribution(self, sentiment_column="sentiment"):
        """
        Plot the overall distribution of sentiment labels in the dataset.
        Helps visualize dataset balance (positive vs negative vs neutral).
        """
        plt.figure(figsize=(8, 5))
        # Count each sentiment label and order them consistently
        sentiment_counts = self.df[sentiment_column].value_counts()
        sentiment_counts = sentiment_counts.reindex(["POSITIVE", "NEGATIVE", "NEUTRAL"]).dropna()
        # Plot bar chart
        plt.bar(sentiment_counts.index, sentiment_counts.values,
                color=["green", "red", "blue"])
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("visualizations/sentiment_distribution.png")
        plt.close()

    def plot_message_activity_over_time(self, date_column="date"):
        """
        Plot number of messages per month.
        Shows communication frequency and trends over time.
        """
        plt.figure(figsize=(12, 5))
        # Convert to datetime and resample monthly
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors="coerce")
        msg_counts = (
            self.df.dropna(subset=[date_column])
            .set_index(date_column)
            .resample("M")
            .size()
            .sort_index()
        )
        # Bar plot of monthly message counts
        plt.bar(msg_counts.index.strftime("%Y-%m"), msg_counts.values)
        plt.title("Message Activity Over Time")
        plt.xlabel("Month")
        plt.ylabel("Number of Messages")
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.savefig("visualizations/message_activity_over_time.png")
        plt.close()

    def message_length_distribution(self, text_column="body"):
        """
        Plot histogram of message lengths.
        Highlights communication style (short vs long messages).
        """
        plt.figure(figsize=(10, 5))
        # Compute text length for each message
        self.df["message_length"] = self.df[text_column].astype(str).apply(len)
        # Plot histogram
        plt.hist(self.df["message_length"], bins=30)
        plt.title("Message Length Distribution")
        plt.xlabel("Message Length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("visualizations/message_length_distribution.png")
        plt.close()

    # ------------------------------------------------------------
    # ADDITIONAL EDA PLOTS
    # ------------------------------------------------------------
    def plot_avg_sentiment_over_time(self, date_column="date"):
        """
        Plot the monthly average sentiment score.
        Shows if employee sentiment improves or declines over time.
        """
        plt.figure(figsize=(12, 5))
        # Map sentiment to numeric values
        sent_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        self.df["sentiment_num"] = self.df["sentiment"].map(sent_map)
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors="coerce")
        # Compute monthly mean sentiment
        monthly_mean = (
            self.df.dropna(subset=[date_column])
            .set_index(date_column)
            .resample("M")["sentiment_num"]
            .mean()
            .sort_index()
        )
        # Line plot of average sentiment over time
        plt.plot(monthly_mean.index, monthly_mean.values, marker="o")
        plt.title("Average Sentiment Over Time")
        plt.xlabel("Month")
        plt.ylabel("Mean Sentiment Score")
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.savefig("visualizations/avg_sentiment_over_time.png")
        plt.close()

    def plot_top_employees(self, employee_column="employee_id"):
        """
        Plot top 10 employees by total number of sent messages.
        Identifies communication hubs within the dataset.
        """
        plt.figure(figsize=(8, 6))
        top_emps = self.df[employee_column].value_counts().head(10)
        plt.barh(top_emps.index, top_emps.values, color="skyblue")
        plt.title("Top 10 Most Active Employees")
        plt.xlabel("Message Count")
        plt.tight_layout()
        plt.savefig("visualizations/top_employees.png")
        plt.close()

    def plot_sentiment_per_employee(self, employee_column="employee_id"):
        """
        Create a stacked bar chart of sentiment counts per employee.
        Useful for identifying employees with consistently positive or negative tone.
        """
        plt.figure(figsize=(12, 6))
        grouped = self.df.groupby([employee_column, "sentiment"]).size().unstack(fill_value=0)
        grouped.plot(kind="bar", stacked=True, figsize=(12, 6))
        plt.title("Sentiment per Employee")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("visualizations/sentiment_per_employee.png")
        plt.close()

    def plot_length_by_sentiment(self, text_column="body"):
        """
        Plot boxplots of message lengths grouped by sentiment.
        Shows if longer or shorter messages tend to be more negative or positive.
        """
        plt.figure(figsize=(8, 6))
        self.df["message_length"] = self.df[text_column].astype(str).apply(len)
        categories = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        # Prepare message lengths per sentiment category
        data = [self.df[self.df["sentiment"] == cat]["message_length"] for cat in categories]
        plt.boxplot(data, labels=categories)
        plt.title("Message Length by Sentiment")
        plt.xlabel("Sentiment")
        plt.ylabel("Message Length")
        plt.tight_layout()
        plt.savefig("visualizations/message_length_by_sentiment.png")
        plt.close()
        

    def plot_avg_message_length_per_employee(self, text_column="body", employee_column="employee_id"):
        """
        Compute and visualize the average message length per employee.

        Output:
            - 'visualizations/avg_message_length_per_employee.png' (bar chart)
            - 'visualizations/avg_message_length_per_employee.csv' (data table)

        Purpose:
            Shows which employees tend to write longer or shorter messages,
            which can help interpret engagement style and communication behavior.
        """
        # Compute message length per record
        self.df["message_length"] = self.df[text_column].astype(str).apply(len)

        # Aggregate by employee
        avg_length = (
            self.df.groupby(employee_column)["message_length"]
            .mean()
            .sort_values(ascending=False)
        )


        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(avg_length.index, avg_length.values, color="teal")
        plt.title("Average Message Length per Employee")
        plt.xlabel("Employee")
        plt.ylabel("Average Message Length (characters)")
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.savefig("visualizations/avg_message_length_per_employee.png")
        plt.close()