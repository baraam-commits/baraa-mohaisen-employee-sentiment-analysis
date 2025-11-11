import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from typing import Optional

class PlotData:
    """
    Task 2 — Exploratory Data Analysis (EDA) visualizations.

    Generates static PNG charts from a DataFrame of employee messages and
    sentiment labels. All figures are saved into a local 'visualizations/'
    directory created at initialization.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing at least:
        - 'date' (parseable to datetime for time-series plots)
        - 'sentiment' (categorical: {'POSITIVE','NEGATIVE','NEUTRAL'})
        - 'employee_id' (for per-employee plots)
        - 'body' or 'text' (for message length)
        - 'sentiment_num' (Optional numarical represention of sentiments)
        - 'polarity' (the skew towards the given classification)

    Attributes
    ----------
    df : pandas.DataFrame
        Reference to the provided DataFrame (mutated by some methods that
        add helper columns like 'message_length' or 'sentiment_num').

    Notes
    -----
    - All plotting functions save a PNG to 'visualizations/' and close the figure.
    - Several methods compute and store 'message_length' in-place. If you need
      immutability, pass a copy of the DataFrame to the constructor.
    """

    def __init__(self, df):
        """
        Initialize plotting helper and ensure output directory exists.

        Parameters
        ----------
        df : pandas.DataFrame
            Source data for all plots.

        Side Effects
        ------------
        - Creates 'visualizations/' directory if missing.
        """
        self.df = df.copy()
        sent_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    
        if "sentiment_num" not in self.df.columns:
            self.df["sentiment_num"] = self.df["sentiment"].map(sent_map)
            
        if "message_length" not in self.df.columns:
            self.df["message_length"] = self.df["body"].astype(str).apply(len)
        
        os.makedirs("visualizations", exist_ok=True)

    # ------------------------------------------------------------
    # BASIC PLOTS
    # ------------------------------------------------------------
    def plot_sentiment_distribution(self, sentiment_column: str = "sentiment") -> None:
        """
        Plot overall frequency of sentiment labels.

        Parameters
        ----------
        sentiment_column : str, default "sentiment"
            Column containing sentiment class labels.

        Output
        ------
        visualizations/sentiment_distribution.png

        Returns
        -------
        None

        Notes
        -----
        - Bars are ordered as POSITIVE, NEGATIVE, NEUTRAL for consistency.
        - Drops categories that are absent in the data.
        """
        plt.figure(figsize=(8, 5))
        sentiment_counts = self.df[sentiment_column].value_counts()
        sentiment_counts = sentiment_counts.reindex(["POSITIVE", "NEGATIVE", "NEUTRAL"]).dropna()
        plt.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "red", "blue"])
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("visualizations/sentiment_distribution.png")
        plt.close()

    def plot_message_activity_over_time(self, date_column: str = "date") -> None:
        """
        Plot monthly message counts to show communication volume trends.

        Parameters
        ----------
        date_column : str, default "date"
            Timestamp column to resample by month.

        Output
        ------
        visualizations/message_activity_over_time.png

        Returns
        -------
        None

        Notes
        -----
        - Coerces `date_column` to datetime and drops NaT before resampling.
        - Uses month-end ('M') frequency for consistent binning.
        """
        plt.figure(figsize=(12, 5))
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors="coerce")
        msg_counts = (
            self.df.dropna(subset=[date_column])
            .set_index(date_column)
            .resample("M")
            .size()
            .sort_index()
        )
        plt.bar(msg_counts.index.strftime("%Y-%m"), msg_counts.values)
        plt.title("Message Activity Over Time")
        plt.xlabel("Month")
        plt.ylabel("Number of Messages")
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.savefig("visualizations/message_activity_over_time.png")
        plt.close()

    def message_length_distribution(self, text_column: str = "body") -> None:
        """
        Plot a histogram of message lengths (character count).

        Parameters
        ----------
        text_column : str, default "body"
            Column containing the raw message text.

        Output
        ------
        visualizations/message_length_distribution.png

        Returns
        -------
        None

        Side Effects
        ------------
        - Adds/overwrites 'message_length' column in `self.df`.

        Notes
        -----
        - Length is computed as `len(str(text))`. For token counts,
          precompute a separate column.
        """
        plt.figure(figsize=(10, 5))
        self.df["message_length"] = self.df[text_column].astype(str).apply(len)
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
    def plot_avg_sentiment_over_time(self, date_column: str = "date") -> None:
        """
        Plot monthly average sentiment score with a linear trend line.

        Parameters
        ----------
        date_column : str, default "date"
            Timestamp column to resample by month.

        Output
        ------
        visualizations/avg_sentiment_over_time.png

        Returns
        -------
        None

        Side Effects
        ------------
        - Adds/overwrites 'sentiment_num' column in `self.df`.

        Method
        ------
        - Maps {'POSITIVE':1, 'NEUTRAL':0, 'NEGATIVE':-1}.
        - Resamples mean sentiment by calendar month.
        - Fits a degree-1 polynomial (np.polyfit) for trend.

        Notes
        -----
        - Requires non-empty monthly series for regression.
        - X-axis uses month index; regression is on index order, not dates.
        """
        plt.figure(figsize=(12, 5))
        sent_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        self.df["sentiment_num"] = self.df["sentiment"].map(sent_map)
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors="coerce")

        monthly_mean = (
            self.df.dropna(subset=[date_column])
            .set_index(date_column)
            .resample("M")["sentiment_num"]
            .mean()
            .sort_index()
        )

        plt.plot(monthly_mean.index, monthly_mean.values, marker="o", label="Monthly Avg")

        x = np.arange(len(monthly_mean))
        y = monthly_mean.values
        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)
        plt.plot(monthly_mean.index, trend(x), color="red", linestyle="--", label="Line of Best Fit")

        plt.title("Average Sentiment Over Time")
        plt.xlabel("Month")
        plt.ylabel("Mean Sentiment Score")
        plt.xticks(rotation=75)
        plt.legend()
        plt.tight_layout()
        plt.savefig("visualizations/avg_sentiment_over_time.png")
        plt.close()

    def plot_top_employees(self, employee_column: str = "employee_id") -> None:
        """
        Plot the top 10 most active employees by message count.

        Parameters
        ----------
        employee_column : str, default "employee_id"
            Column used to count messages per employee.

        Output
        ------
        visualizations/top_employees.png

        Returns
        -------
        None

        Notes
        -----
        - Uses horizontal bars for label readability.
        """
        plt.figure(figsize=(8, 6))
        top_emps = self.df[employee_column].value_counts().head(10)
        plt.barh(top_emps.index, top_emps.values, color="skyblue")
        plt.title("Top 10 Most Active Employees")
        plt.xlabel("Message Count")
        plt.tight_layout()
        plt.savefig("visualizations/top_employees.png")
        plt.close()

    def plot_sentiment_per_employee(self, employee_column: str = "employee_id") -> None:
        """
        Stacked bar chart of sentiment counts per employee.

        Parameters
        ----------
        employee_column : str, default "employee_id"
            Column that identifies employees.

        Output
        ------
        visualizations/sentiment_per_employee.png

        Returns
        -------
        None

        Notes
        -----
        - Uses `groupby(...).size().unstack(fill_value=0)` to build counts.
        - If using an existing Axes, pass `ax=plt.gca()` to `grouped.plot(...)`
          to avoid creating a new figure implicitly.
        """
        plt.figure(figsize=(12, 6))
        grouped = self.df.groupby([employee_column, "sentiment"]).size().unstack(fill_value=0)
        grouped.plot(kind="bar", stacked=True, figsize=(12, 6))
        plt.title("Sentiment per Employee")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("visualizations/sentiment_per_employee.png")
        plt.close()

    def plot_length_by_sentiment(self, text_column: str = "body") -> None:
        """
        Boxplots of message length grouped by sentiment.

        Parameters
        ----------
        text_column : str, default "body"
            Column containing raw message text.

        Output
        ------
        visualizations/message_length_by_sentiment.png

        Returns
        -------
        None

        Side Effects
        ------------
        - Adds/overwrites 'message_length' column in `self.df`.

        Notes
        -----
        - Category order: POSITIVE, NEGATIVE, NEUTRAL.
        - For extreme outliers, consider log scale or clipping before plotting.
        """
        plt.figure(figsize=(8, 6))
        self.df["message_length"] = self.df[text_column].astype(str).apply(len)
        categories = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        data = [self.df[self.df["sentiment"] == cat]["message_length"] for cat in categories]
        plt.boxplot(data, labels=categories)
        plt.title("Message Length by Sentiment")
        plt.xlabel("Sentiment")
        plt.ylabel("Message Length")
        plt.tight_layout()
        plt.savefig("visualizations/message_length_by_sentiment.png")
        plt.close()

    def plot_avg_message_length_per_employee(
        self,
        text_column: str = "body",
        employee_column: str = "employee_id"
    ) -> None:
        """
        Bar chart of average message length per employee.

        Parameters
        ----------
        text_column : str, default "body"
            Column containing raw message text.
        employee_column : str, default "employee_id"
            Column identifying employees.

        Output
        ------
        visualizations/avg_message_length_per_employee.png

        Returns
        -------
        None

        Side Effects
        ------------
        - Adds/overwrites 'message_length' column in `self.df`.

        Method
        ------
        - Computes per-record length then mean by employee.
        - Sorts descending so longest-average writers appear first.

        Notes
        -----
        - If names overlap, rotate X labels for readability.
        """
        self.df["message_length"] = self.df[text_column].astype(str).apply(len)
        avg_length = (
            self.df.groupby(employee_column)["message_length"]
            .mean()
            .sort_values(ascending=False)
        )
        plt.figure(figsize=(10, 6))
        plt.bar(avg_length.index, avg_length.values, color="teal")
        plt.title("Average Message Length per Employee")
        plt.xlabel("Employee")
        plt.ylabel("Average Message Length (characters)")
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.savefig("visualizations/avg_message_length_per_employee.png")
        plt.close()

    def plot_polarity_distribution(self):
        """Histogram of raw polarity across all messages."""
        plt.figure(figsize=(10,5))
        plt.hist(self.df["polarity"].dropna().values, bins=40)
        plt.title("Raw Polarity Distribution")
        plt.xlabel("Polarity (-1 .. 1)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("visualizations/polarity_distribution.png")
        plt.close()

    def plot_avg_polarity_over_time(self, date_column="dt"):
        """Monthly mean polarity trend."""
         
        d = self.df.dropna(subset=[date_column, "polarity"]).copy()
        if d.empty:
            return
        s = (d.set_index(date_column)
               .resample("M")["polarity"]
               .mean()
               .sort_index())
        if s.empty:
            return
        plt.figure(figsize=(12,5))
        plt.plot(s.index, s.values, marker="o")
        plt.title("Average Polarity Over Time")
        plt.xlabel("Month")
        plt.ylabel("Mean Polarity")
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.savefig("visualizations/avg_polarity_over_time.png")
        plt.close()

    def plot_avg_polarity_per_employee(self, employee_column="employee_id"):
        """Bar chart of mean polarity per employee."""
         
        if employee_column not in self.df.columns:
            return
        g = (self.df.dropna(subset=["polarity"])
                    .groupby(employee_column)["polarity"]
                    .mean()
                    .sort_values(ascending=False))
        if g.empty:
            return
        plt.figure(figsize=(12,6))
        plt.barh(g.index, g.values)
        plt.title("Average Polarity per Employee")
        plt.xlabel("Mean Polarity")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("visualizations/avg_polarity_per_employee.png")
        plt.close()

    def plot_length_vs_polarity(self, text_column="body"):
        """Scatter: message length vs polarity, with linear fit."""
         
        d = self.df.dropna(subset=["polarity"]).copy()
        d["message_length"] = d[text_column].astype(str).apply(len)
        x = d["message_length"].astype(float).values
        y = d["polarity"].astype(float).values
        if len(d) == 0:
            return
        plt.figure(figsize=(10,6))
        plt.scatter(x, y, s=8, alpha=0.4)
        # simple least-squares line
        try:
            coeff = np.polyfit(x, y, deg=1)
            xline = np.linspace(x.min(), x.max(), 200)
            yline = coeff[0]*xline + coeff[1]
            plt.plot(xline, yline)
        except Exception:
            pass
        plt.title("Message Length vs Polarity")
        plt.xlabel("Message Length (chars)")
        plt.ylabel("Polarity")
        plt.tight_layout()
        plt.savefig("visualizations/length_vs_polarity.png")
        plt.close()

    def plot_polarity_heatmap(self, employee_column="employee_id"):
        """Month x Employee heatmap of mean polarity (matplotlib imshow)."""
         
        if ("month" not in self.df.columns) or (employee_column not in self.df.columns):
            return
        piv = (self.df.dropna(subset=["polarity"])
                         .pivot_table(index="month",
                                      columns=employee_column,
                                      values="polarity",
                                      aggfunc="mean"))
        if piv.empty:
            return
        piv = piv.sort_index()  # chronological by month
        plt.figure(figsize=(min(16, 1+0.5*len(piv.columns)), 8))
        im = plt.imshow(piv.values, aspect="auto", origin="lower")
        plt.colorbar(im, label="Mean Polarity")
        plt.yticks(range(len(piv.index)), piv.index)
        plt.xticks(range(len(piv.columns)), piv.columns, rotation=75, ha="right")
        plt.title("Mean Polarity by Month and Employee")
        plt.tight_layout()
        plt.savefig("visualizations/polarity_heatmap.png")
        plt.close()

    def run_all_plots(
        self,
        *,
        date_column: str = "date",
        sentiment_column: str = "sentiment",
        employee_column: str = "employee_id",
        text_column: str = "body",
        polarity_date_column: Optional[str] = None,
    ) -> None:
        """
        Generate every available visualization and persist each PNG.

        Parameters
        ----------
        date_column : str, default "date"
            Timestamp column for time-series charts.
        sentiment_column : str, default "sentiment"
            Sentiment label column.
        employee_column : str, default "employee_id"
            Identifier used for per-employee aggregations.
        text_column : str, default "body"
            Raw message text column for length-based analysis.
        polarity_date_column : str or None, default None
            Optional override for polarity trend timestamps. Falls back to
            `date_column` when omitted.
        """
        polarity_dates = polarity_date_column or date_column
        self.plot_sentiment_distribution(sentiment_column=sentiment_column)
        self.plot_message_activity_over_time(date_column=date_column)
        self.message_length_distribution(text_column=text_column)
        self.plot_avg_sentiment_over_time(date_column=date_column)
        self.plot_top_employees(employee_column=employee_column)
        self.plot_sentiment_per_employee(employee_column=employee_column)
        self.plot_length_by_sentiment(text_column=text_column)
        self.plot_avg_message_length_per_employee(
            text_column=text_column,
            employee_column=employee_column,
        )
        self.plot_polarity_distribution()
        self.plot_avg_polarity_over_time(date_column=polarity_dates)
        self.plot_avg_polarity_per_employee(employee_column=employee_column)
        self.plot_length_vs_polarity(text_column=text_column)
        self.plot_polarity_heatmap(employee_column=employee_column)

