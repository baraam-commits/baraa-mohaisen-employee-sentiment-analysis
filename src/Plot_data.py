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

    def plot_avg_polarity_over_time(self, date_column="date"):
        """Monthly mean polarity trend."""
         
        plt.figure(figsize=(12, 5))
        d = self.df.dropna(subset=[date_column, "polarity"]).copy()
        if d.empty:
            return
        s = (d.set_index(date_column)
               .resample("M")["polarity"]
               .mean()
               .sort_index())
        if s.empty:
            return
        
        plt.plot(s.index, s.values, marker="o", label="Monthly Avg")

        x = np.arange(len(s))
        y = s.values
        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)
        plt.plot(s.index, trend(x), color="red", linestyle="--", label="Line of Best Fit")

        plt.title("Average polarity Over Time")
        plt.xlabel("Month")
        plt.ylabel("Mean polarity Score")
        plt.xticks(rotation=75)
        plt.legend()
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



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.regression import FeatureEngineer


class ModelPlotter:
    
    """
    Generates diagnostic plots for a trained sentiment regression model.

    This class is self-contained. It takes the *raw data* and a
    *saved model path*, then re-runs the feature engineering
    and train/test split internally to generate plots.
    """

    def __init__(
        self,  
        raw_df: pd.DataFrame, 
        sentiment_scores_df: pd.DataFrame,
        test_size: float = 0.2, 
        random_state: int = 42,
        pipeline_path: str = "data\\regression_model.joblib"
    ):
        """
        Initializes the plotter by loading the model and recreating the data.

        Args:
            pipeline_path (str): File path to the saved .joblib pipeline.
            raw_df (pd.DataFrame): The *full* raw email dataset.
            sentiment_scores_df (pd.DataFrame): The *full* aggregated sentiment scores.
            test_size (float): The *exact* test_size used during training.
            random_state (int): The *exact* random_state used during training.
        """
        print("Initializing ModelPlotter...")
        
        # --- 1. Load the Fitted Pipeline ---
        try:
            self.pipeline: Pipeline = joblib.load(pipeline_path)
            print(f"Successfully loaded model from {pipeline_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {pipeline_path}")
            raise
            
        # --- 2. Re-create the Full Feature Set ---
        print("Re-engineering features for plotting...")
        engineer = FeatureEngineer()
        self.X, self.y = engineer.engineer_features(raw_df, sentiment_scores_df)
        
        # Combine for correlation and scatter plots
        self.df_features = self.X.join(self.y)
        
        # --- 3. Re-create the *Exact* Train/Test Split ---
        # This is CRITICAL. We must plot performance on the *same*
        # test set the model was evaluated on.
        print("Re-creating train/test split...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        self.output_dir = "visualizations/model"
        os.makedirs(self.output_dir, exist_ok=True)
        print("ModelPlotter ready.")

    def plot_residuals(self):
        """
        Plots a residual plot (Predicted vs. Errors).

        A good model shows residuals randomly scattered around y=0.
        Patterns (like a cone or a curve) indicate model problems.
        
        Output:
        ------
        visualizations/model/residual_plot.png
        """
        print("Plotting residuals...")
        y_pred = self.pipeline.predict(self.X_test)
        residuals = self.y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, s=50)
        plt.axhline(y=0, color='red', linestyle='--', lw=2)
        plt.title("Residual Plot (Predicted vs. Errors)", fontsize=16)
        plt.xlabel("Predicted Sentiment Score", fontsize=12)
        plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/residual_plot.png")
        plt.close()

    def plot_prediction_vs_actual(self):
        """
        Plots the model's predictions against the true (actual) values.

        For a perfect model (R-squared = 1.0), all points would
        fall on the 45-degree (y=x) "Perfect Prediction" line.
        
        Output:
        ------
        visualizations/model/prediction_vs_actual.png
        """
        print("Plotting prediction vs. actual...")
        y_pred = self.pipeline.predict(self.X_test)

        plt.figure(figsize=(8, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.5, s=50)
        
        lims = [
            min(self.y_test.min(), y_pred.min()),
            max(self.y_test.max(), y_pred.max()),
        ]
        # Add a 10% buffer
        lims = [lims[0] * 0.9, lims[1] * 1.1]

        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Prediction")
        
        plt.title("Prediction vs. Actual Values (Test Set)", fontsize=16)
        plt.xlabel("Actual Sentiment Score", fontsize=12)
        plt.ylabel("Predicted Sentiment Score", fontsize=12)
        plt.legend()
        plt.xlim(lims)
        plt.ylim(lims)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/prediction_vs_actual.png")
        plt.close()

    def plot_top_coefficients(self, top_n: int = 20):
        """
        Plots the Top N most important positive and negative features.
        
        This is the *explanation* of your model. It shows exactly
        which features the Lasso model decided to keep and how
        much they influence the final score.

        Args:
            top_n (int): The number of features to show.
        
        Output:
        ------
        visualizations/model/top_coefficients.png
        """
        print("Plotting top coefficients...")
        # 1. Get coefficients from the 'lassocv' step
        model = self.pipeline.named_steps['lassocv']
        
        # 2. Get feature names from the 'polynomialfeatures' step
        poly_features = self.pipeline.named_steps['polynomialfeatures']
        original_names = self.X.columns 
        all_poly_names = poly_features.get_feature_names_out(original_names)

        # 3. Create the DataFrame
        coeff_df = pd.DataFrame({
            'Feature': all_poly_names,
            'Coefficient': model.coef_
        })
        
        # 4. Filter to non-zero and get top/bottom
        coeff_df = coeff_df[coeff_df['Coefficient'] != 0].sort_values(by='Coefficient')
        top_features = pd.concat([coeff_df.head(top_n), coeff_df.tail(top_n)])
        
        # 5. Plot
        plt.figure(figsize=(10, 12))
        colors = ['red' if c < 0 else 'green' for c in top_features['Coefficient']]
        sns.barplot(x='Coefficient', y='Feature', data=top_features, palette=colors)
        plt.title(f"Top {top_n*2} Model Coefficients", fontsize=16)
        plt.xlabel("Coefficient Value (Impact on Score)", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/top_coefficients.png")
        plt.close()

    def plot_correlation_heatmap(self):
        """
        Plots a heatmap of the correlation between all *base* features
        and the target variable.
        
        Output:
        ------
        visualizations/model/feature_correlation_heatmap.png
        """
        print("Plotting correlation heatmap...")
        plt.figure(figsize=(24, 20))
        # Use self.df_features, which is X + y
        corr = self.df_features.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 9})
        plt.title("Feature Correlation Heatmap", fontsize=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_correlation_heatmap.png")
        plt.close()

    def run_all_plots(self):
        """
        A helper method to run all plotting functions.
        """
        print("--- Running All Model Plots ---")
        self.plot_residuals()
        self.plot_prediction_vs_actual()
        self.plot_top_coefficients()
        self.plot_correlation_heatmap()
        print("--- All Model Plots Saved to 'visualizations/model/' ---")
        

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List

class FlightRiskPlots:
    """
    Task 5 — Flight risk visualizations.

    Inputs
    ------
    df : pandas.DataFrame
        Must include at least:
        - 'employee_id' (hashable)
        - 'date' (datetime64[ns])
        - 'sentiment_num' in {+1, 0, -1}
    outdir : str
        Output directory for PNGs. Default 'visualizations/flight_risk'.
    min_neg_in_30d : int
        Rolling 30-day NEGATIVE message count threshold that defines "flight risk".

    What it draws
    -------------
    1) dist_neg30d.png
       Histogram of rolling 30-day negative counts across all employees/dates.

    2) bar_top_max_neg30d.png
       Top-N employees by their peak 30-day negative count.

    3) heatmap_neg_by_month_employee.png
       Employee x Month table of raw NEGATIVE counts (not the rolling value).

    4) gantt_flight_risk_periods.png
       Timeline bars showing intervals where each employee is flagged as flight risk.

    5) spark_grid_[K].png
       Small multiples: per-employee line of neg_30d with threshold line.

    6) line_neg30d_<employee>.png
       Single-employee time series of neg_30d with threshold line.

    7) cohort_days_to_first_risk.png
       Distribution of days from each employee's first message to first flight-risk flag.

    Usage
    -----
    fp = FlightRiskPlots(df)
    fp.run_all()                       # generate a standard set
    fp.line_neg30d('EMP001')           # deep dive for one employee
    """

    def __init__(self,
                 df: pd.DataFrame,
                 outdir: str = "visualizations/flight_risk",
                 min_neg_in_30d: int = 4) -> None:
        self.min_neg_in_30d = int(min_neg_in_30d)
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        # Work copy
        self.df = df.copy()
        # Basic checks
        required = {"employee_id", "date", "sentiment_num"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Ensure datetime
        if not np.issubdtype(self.df["date"].dtype, np.datetime64):
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.dropna(subset=["date"])

        # Prepare rolling features
        self._prepare()

    # ---------- data prep ----------
    def _prepare(self) -> None:
        self.df["neg"] = (self.df["sentiment_num"] == -1).astype(int)

        def _within_emp(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date").set_index("date")
            # time-based rolling window
            g["neg_30d"] = g["neg"].rolling("30D", closed="both").sum()
            g["flight_risk"] = g["neg_30d"] >= self.min_neg_in_30d
            return g.reset_index()

        self.df = (
            self.df.groupby("employee_id", group_keys=False)
            .apply(_within_emp)
            .sort_values(["employee_id", "date"])
            .reset_index(drop=True)
        )

        # Per-employee summaries
        risk_rows = self.df[self.df["flight_risk"]].copy()
        if not risk_rows.empty:
            self.emp_summary = (
                risk_rows.groupby("employee_id", as_index=False)
                .agg(first_flight_risk=("date", "min"),
                     max_neg_sent_in_30d=("neg_30d", "max"))
            )
        else:
            self.emp_summary = pd.DataFrame(columns=["employee_id", "first_flight_risk", "max_neg_sent_in_30d"])

        # Month table of raw negative counts (not rolling)
        self.df["month"] = self.df["date"].dt.to_period("M")
        neg_only = self.df[self.df["neg"] == 1]
        self.neg_by_emp_month = (
            neg_only.groupby(["employee_id", "month"], as_index=False)
            .size()
            .rename(columns={"size": "neg_count"})
        )

    # ---------- plot helpers ----------
    def _savefig(self, path: str) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    # 1) distribution of neg_30d across all points
    def dist_neg30d(self, bins: int = 15) -> str:
        s = self.df["neg_30d"].astype(float)
        plt.figure(figsize=(6, 4))
        plt.hist(s, bins=bins)
        plt.axvline(self.min_neg_in_30d, linestyle="--")
        plt.title("Rolling 30-day negatives (all employees, all dates)")
        plt.xlabel("neg_30d")
        plt.ylabel("count")
        out = os.path.join(self.outdir, "dist_neg30d.png")
        self._savefig(out)
        return out

    # 2) top-N by peak neg_30d
    def bar_top_max_neg30d(self, top_n: int = 10) -> str:
        if self.emp_summary.empty:
            # still show top by rolling max even if no one crossed threshold
            tmp = (self.df.groupby("employee_id", as_index=False)["neg_30d"].max()
                   .rename(columns={"neg_30d": "max_neg_sent_in_30d"}))
        else:
            tmp = self.emp_summary.copy()
        tmp = tmp.sort_values("max_neg_sent_in_30d", ascending=False).head(top_n)

        plt.figure(figsize=(8, 5))
        plt.barh(tmp["employee_id"].astype(str), tmp["max_neg_sent_in_30d"].astype(float))
        plt.axvline(self.min_neg_in_30d, linestyle="--")
        plt.gca().invert_yaxis()
        plt.title(f"Top {len(tmp)} employees by peak 30-day negatives")
        plt.xlabel("max_neg_sent_in_30d")
        out = os.path.join(self.outdir, "bar_top_max_neg30d.png")
        self._savefig(out)
        return out

    # 3) employee x month heatmap of raw negative counts
    def heatmap_neg_by_month_employee(self, max_employees: int = 30) -> str:
        # focus on employees with most negatives to keep plot readable
        totals = (self.neg_by_emp_month.groupby("employee_id", as_index=False)["neg_count"].sum()
                  .sort_values("neg_count", ascending=False)
                  .head(max_employees))
        focus_ids = set(totals["employee_id"])
        hm = self.neg_by_emp_month[self.neg_by_emp_month["employee_id"].isin(focus_ids)].copy()
        if hm.empty:
            # build an empty placeholder to avoid breaking
            plt.figure(figsize=(6, 3))
            plt.title("No negative messages recorded")
            out = os.path.join(self.outdir, "heatmap_neg_by_month_employee.png")
            self._savefig(out)
            return out

        # pivot to wide
        pivot = hm.pivot_table(index="employee_id",
                               columns="month",
                               values="neg_count",
                               fill_value=0,
                               aggfunc="sum")
        pivot = pivot.sort_index(axis=1)  # chronological months

        plt.figure(figsize=(max(6, 0.5 * pivot.shape[1]), max(6, 0.4 * pivot.shape[0])))
        # draw as an image; annotate a few large cells
        plt.imshow(pivot.values, aspect="auto")
        plt.xticks(np.arange(pivot.shape[1]), [str(m) for m in pivot.columns], rotation=90)
        plt.yticks(np.arange(pivot.shape[0]), pivot.index.astype(str))
        plt.colorbar(label="negative count")
        plt.title("Negative messages per employee per month")
        out = os.path.join(self.outdir, "heatmap_neg_by_month_employee.png")
        self._savefig(out)
        return out

    # 4) Gantt-like bars for risk intervals
    def gantt_flight_risk_periods(self, max_employees: int = 25) -> str:
        # Build contiguous intervals of flight_risk==True per employee
        risk = self.df[self.df["flight_risk"]].copy()
        if risk.empty:
            plt.figure(figsize=(6, 3))
            plt.title("No flight-risk intervals to show")
            out = os.path.join(self.outdir, "gantt_flight_risk_periods.png")
            self._savefig(out)
            return out

        # limit to employees with more risk days
        counts = (risk.groupby("employee_id", as_index=False).size()
                  .sort_values("size", ascending=False)
                  .head(max_employees))
        keep = set(counts["employee_id"])
        risk = risk[risk["employee_id"].isin(keep)]

        rows = []
        for emp_id, g in risk.groupby("employee_id"):
            g = g.sort_values("date")
            # identify contiguous runs
            run_start = None
            prev_date = None
            for _, r in g.iterrows():
                d = pd.to_datetime(r["date"]).normalize()
                if run_start is None:
                    run_start = d
                    prev_date = d
                    continue
                # if gap > 1 day, close previous run
                if (d - prev_date).days > 1:
                    rows.append((emp_id, run_start, prev_date))
                    run_start = d
                prev_date = d
            # close last
            if run_start is not None:
                rows.append((emp_id, run_start, prev_date))

        if not rows:
            plt.figure(figsize=(6, 3))
            plt.title("No contiguous flight-risk intervals")
            out = os.path.join(self.outdir, "gantt_flight_risk_periods.png")
            self._savefig(out)
            return out

        # Plot
        emp_order = sorted({r[0] for r in rows})
        ymap = {e: i for i, e in enumerate(emp_order)}
        plt.figure(figsize=(10, max(4, 0.4 * len(emp_order))))
        for emp_id, start, end in rows:
            y = ymap[emp_id]
            plt.plot([start, end], [y, y])
            plt.scatter([start, end], [y, y])

        plt.yticks(range(len(emp_order)), [str(e) for e in emp_order])
        plt.xlabel("date")
        plt.title("Flight-risk intervals (rolling 30-day rule)")
        out = os.path.join(self.outdir, "gantt_flight_risk_periods.png")
        self._savefig(out)
        return out

    # 5) small-multiples of neg_30d by employee
    def spark_grid(self, max_employees: int = 20, ncols: int = 4) -> str:
        # choose employees with highest max neg_30d first
        peak = (self.df.groupby("employee_id", as_index=False)["neg_30d"].max()
                .sort_values("neg_30d", ascending=False)
                .head(max_employees))
        emp_ids = list(peak["employee_id"])
        if not emp_ids:
            plt.figure(figsize=(6, 3))
            plt.title("No data for spark grid")
            out = os.path.join(self.outdir, "spark_grid.png")
            self._savefig(out)
            return out

        n = len(emp_ids)
        nrows = int(math.ceil(n / ncols))
        plt.figure(figsize=(3.0 * ncols, 2.0 * nrows))

        for i, emp in enumerate(emp_ids):
            ax = plt.subplot(nrows, ncols, i + 1)
            g = self.df[self.df["employee_id"] == emp].sort_values("date")
            ax.plot(g["date"], g["neg_30d"])
            ax.axhline(self.min_neg_in_30d, linestyle="--")
            ax.set_title(str(emp), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle("neg_30d by employee")
        out = os.path.join(self.outdir, "spark_grid.png")
        self._savefig(out)
        return out

    # 6) single employee time series
    def line_neg30d(self, employee_id) -> str:
        g = self.df[self.df["employee_id"] == employee_id].sort_values("date")
        if g.empty:
            plt.figure(figsize=(6, 3))
            plt.title(f"No data for {employee_id}")
            out = os.path.join(self.outdir, f"line_neg30d_{employee_id}.png")
            self._savefig(out)
            return out

        plt.figure(figsize=(8, 4))
        plt.plot(g["date"], g["neg_30d"])
        plt.axhline(self.min_neg_in_30d, linestyle="--")
        plt.title(f"{employee_id} — rolling 30-day negatives")
        plt.xlabel("date")
        plt.ylabel("neg_30d")
        out = os.path.join(self.outdir, f"line_neg30d_{employee_id}.png")
        self._savefig(out)
        return out

    # 7) cohort: days from first message to first flight risk
    def cohort_days_to_first_risk(self, bins: int = 12) -> str:
        # first message per employee
        first_msg = (self.df.groupby("employee_id", as_index=False)["date"].min()
                     .rename(columns={"date": "first_msg"}))
        # first risk per employee
        first_risk = (self.df[self.df["flight_risk"]]
                      .groupby("employee_id", as_index=False)["date"].min()
                      .rename(columns={"date": "first_risk"}))

        coh = pd.merge(first_msg, first_risk, on="employee_id", how="inner")
        if coh.empty:
            plt.figure(figsize=(6, 3))
            plt.title("No employees reached flight-risk")
            out = os.path.join(self.outdir, "cohort_days_to_first_risk.png")
            self._savefig(out)
            return out

        coh["days_to_first_risk"] = (coh["first_risk"].dt.normalize() - coh["first_msg"].dt.normalize()).dt.days
        coh = coh[coh["days_to_first_risk"].ge(0)]

        plt.figure(figsize=(6, 4))
        plt.hist(coh["days_to_first_risk"], bins=bins)
        plt.title("Days from first message to first flight-risk")
        plt.xlabel("days")
        plt.ylabel("employees")
        out = os.path.join(self.outdir, "cohort_days_to_first_risk.png")
        self._savefig(out)
        return out

    # ---------- convenience ----------
    def run_all(self,
                spark_max_employees: int = 20,
                top_n_bar: int = 10,
                heatmap_max_employees: int = 30) -> List[str]:
        outputs = []
        outputs.append(self.dist_neg30d())
        outputs.append(self.bar_top_max_neg30d(top_n=top_n_bar))
        outputs.append(self.heatmap_neg_by_month_employee(max_employees=heatmap_max_employees))
        outputs.append(self.gantt_flight_risk_periods())
        outputs.append(self.spark_grid(max_employees=spark_max_employees))
        outputs.append(self.cohort_days_to_first_risk())
        return outputs
