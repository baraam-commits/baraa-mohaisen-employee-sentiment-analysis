import numpy as np
import pandas as pd 
import re
import joblib
from src.ranking import EmployeeScoring
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score, mean_squared_error

class FeatureEngineer:
    """
    Handles all feature extraction and transformation logic.

    This class is responsible for taking raw email data and aggregated sentiment
    scores and transforming them into a complete feature matrix (X) and
    target vector (y) ready for machine learning.
    """
    
    # Class attribute for sentiment mapping
    SENT_MAP = {
        "POSITIVE": 1, "Positive": 1,
        "NEGATIVE": -1, "Negative": -1,
        "NEUTRAL": 0,  "Neutral": 0
    }
    
    def __init__(self):
        """Initializes the FeatureEngineer, setting state attributes to None."""
        self.raw_df = None
        self.sentiment_scores_df = None
        self.X = None
        self.y = None

    def _compute_total_len_per_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the sum of 'text_len' for each employee per month.

        Args:
            df (pd.DataFrame): The raw email DataFrame.

        Returns:
            pd.DataFrame: DataFrame with [employee_id, month, text_len].
        """
        result = (
            df.groupby(['month', 'employee_id'], as_index=False)['text_len']
            .sum()
        )
        return result[['employee_id', 'month', 'text_len']]
    
    def _compute_avg_len_per_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the average message length for each employee per month.

        Args:
            df (pd.DataFrame): The raw email DataFrame.

        Returns:
            pd.DataFrame: DataFrame with [employee_id, month, avg_msg_len].
        """
        df_len = self._compute_total_len_per_month(df)
        
        result = (
            df_len.groupby(['month', 'employee_id'], as_index=False)
            .agg(total_text_len=('text_len', 'sum'),
                 message_count=('text_len', 'count'))
        )

        result['avg_msg_len'] = result['total_text_len'] / result['message_count'].replace(0, np.nan)
        return result[['employee_id', 'month', 'avg_msg_len']]
        
    def _compute_avg_punctuation_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates punctuation density for subject and body per employee-month.
        
        Density is calculated as:
        (total '!' or '?' chars across all messages) / (total chars across all messages)

        Args:
            df (pd.DataFrame): The raw email DataFrame.

        Returns:
            pd.DataFrame: DataFrame with [employee_id, month,
                          avg_subject_punctuation_density, avg_body_punctuation_density].
        """

        def _calc_punctuation_density(
            subj_series: pd.Series,
            include_fullwidth: bool = False,
            exclude_whitespace_from_length: bool = True
        ) -> float:
            """Helper to calculate punctuation density on a series of texts."""
            s = subj_series.astype(str)
            pattern = r"[!?]" if not include_fullwidth else r"[!?\uFF01\uFF1F]"
            punct_total = s.str.count(pattern, flags=re.UNICODE).sum()

            if exclude_whitespace_from_length:
                char_total = s.str.replace(r"\s+", "", regex=True).str.len().sum()
            else:
                char_total = s.str.len().sum()

            if char_total == 0:
                return 0.0
            return float(punct_total / char_total)

        df = df.copy()
        if "month" not in df.columns or not pd.api.types.is_period_dtype(df['month']):
             df["date"] = pd.to_datetime(df["date"], errors="coerce")
             df["month"] = df["date"].dt.to_period("M")

        monthly = (
            df.groupby(["month", "employee_id"], as_index=False)
            .agg(
                avg_body_punctuation_density=("body", _calc_punctuation_density),
                avg_subject_punctuation_density=("Subject", _calc_punctuation_density),
            )
        )

        return monthly[["employee_id", "month",
                        "avg_subject_punctuation_density",
                        "avg_body_punctuation_density"]]
            
    def _compute_avg_caps_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates ALL CAPS density for subject and body per employee-month.

        Density is calculated as:
        (total ALL CAPS words) / (total words) across all messages in the group.

        Args:
            df (pd.DataFrame): The raw email DataFrame.

        Returns:
            pd.DataFrame: DataFrame with [employee_id, month,
                          avg_subject_caps_density, avg_body_caps_density].
        """
        df = df.copy()
        
        if "month" not in df.columns or not pd.api.types.is_period_dtype(df['month']):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["month"] = df["date"].dt.to_period("M")

        def _caps_density(
            series: pd.Series,
            min_caps_len: int = 2,
            word_pattern: str = r"\b[^\W_]+\b"
        ) -> float:
            """Helper to calculate ALL CAPS density on a series of texts."""
            s = series.fillna("").astype(str)
            total_words = s.str.findall(word_pattern, flags=re.UNICODE).str.len().sum()
            caps_regex = rf"\b[A-Z]{{{min_caps_len},}}\b"
            all_caps = s.str.findall(caps_regex).str.len().sum()

            if total_words == 0:
                return 0.0
            return float(all_caps / total_words)

        out = (
            df.groupby(["employee_id", "month"], as_index=False)
            .agg(
                avg_subject_caps_density=("Subject", _caps_density),
                avg_body_caps_density=("body", _caps_density),
            )
        )

        return out[["employee_id", "month",
                    "avg_subject_caps_density", "avg_body_caps_density"]]
        
    def _compute_avg_raw_polarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the mean 'polarity' score for each employee per month.

        Args:
            df (pd.DataFrame): The raw email DataFrame.

        Returns:
            pd.DataFrame: DataFrame with [employee_id, month, avg_raw_polarity].
        """
        result = (
            df.groupby(['month', 'employee_id'], as_index=False)["polarity"]
            .mean()
            .rename(columns={"polarity": "avg_raw_polarity"})
        )
        return result[['employee_id', 'month', 'avg_raw_polarity']]
    
    def _compute_sentiment_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the standard deviation (volatility) of 'sentiment_num'
        for each employee per month.

        Args:
            df (pd.DataFrame): The raw email DataFrame.

        Returns:
            pd.DataFrame: DataFrame with [employee_id, month, sentiment_volatility].
        """
        result = (
            df.groupby(['month', 'employee_id'], as_index=False)["sentiment_num"]
            .std()
            .rename(columns={"sentiment_num": "sentiment_volatility"})
        )
        result['sentiment_volatility'] = result['sentiment_volatility'].fillna(0)
        return result[['employee_id', 'month', 'sentiment_volatility']]
    
    def _is_flight_risk_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a binary flag if an employee had >= 'threshold' negative
        emails in a month.

        Args:
            df (pd.DataFrame): The raw email DataFrame.

        Returns:
            pd.DataFrame: DataFrame with [employee_id, month, is_flight_risk_month].
        """
        threshold: int = 4
        result = (
            df.groupby(['month', 'employee_id'], as_index=False)
            .agg(total_negative_risk=('sentiment_num', lambda x: (x < 0).sum()))
        )
        result['is_flight_risk_month'] = np.where(abs(result['total_negative_risk']) >= threshold, 1, 0) 
        return result[['employee_id', 'month', 'is_flight_risk_month']]
    
    def _compute_previous_month_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the lagged sentiment score from the previous month.

        Args:
            df (pd.DataFrame): The *aggregated* sentiment_scores_df.

        Returns:
            pd.DataFrame: DataFrame with [employee_id, month, 
                          previous_month_sentiment_score].
        """
        results = df.copy(deep=False)
        # Sort by employee and time to ensure .shift() is correct
        results = results.sort_values(by=['employee_id', 'month'])
        results['previous_month_sentiment_score'] = results.groupby('employee_id')['sentiment_num'].shift(1, fill_value=0)
        return results[['employee_id', 'month', 'previous_month_sentiment_score']]
              
    def _prepare_features(self):
        """
        Orchestrates running all feature computation functions and merging
        them into the main feature matrix `self.X`.
        """
        features_func = [
            self._compute_total_len_per_month,
            self._compute_avg_len_per_month,
            self._compute_avg_punctuation_density,
            self._compute_avg_caps_density,
            self._compute_avg_raw_polarity,
            self._compute_sentiment_volatility,
            self._is_flight_risk_month
        ]
        
        for func in features_func:
            feature_df = func(self.raw_df)
            self.X = self.X.merge(feature_df, on=['month', 'employee_id'], how='left')
            
        # Handle the previous_month_score feature, which supports two modes:
        # 1. A single int/float for live prediction.
        # 2. A full DataFrame for batch training/prediction.
        if isinstance(self.sentiment_scores_df, (int, float)):
            self.X["previous_month_sentiment_score"] = self.sentiment_scores_df
        else:
            if "sentiment_num" not in self.sentiment_scores_df: 
                self.sentiment_scores_df["sentiment_num"] = EmployeeScoring(self.raw_df).compute_scores()["sentiment_num"]
            self.X = self.X.merge(self._compute_previous_month_score(self.sentiment_scores_df), on=['employee_id', 'month'], how='left')

    def _kitchen_sink_feature_tranform(self):
        """
        Applies non-linear transforms (log, sqrt) to engineered features.
        
        This method also handles the final separation of X and y, and
        drops all non-numeric ID columns from X.
        """
        
        # Define columns to exclude from transformation
        id_cols = ['employee_id', 'month', 'sentiment_num']
        binary_cols = ['is_flight_risk_month']
        negative_cols = ['avg_raw_polarity', 'previous_month_sentiment_score']
        exclude_cols = set(id_cols + binary_cols + negative_cols)
        
        cols_to_transform = [col for col in self.X.columns if col not in exclude_cols]

        for col in cols_to_transform:
            if (self.X[col] >= 0).all():
                self.X[f'log_{col}'] = np.log1p(self.X[col]) # log1p handles log(0)
                self.X[f'sqrt_{col}'] = np.sqrt(self.X[col])
        
        self.X.fillna(0, inplace=True)
        self.X.sort_values(by=['employee_id', 'month'], inplace=True)
        
        # Separate target variable y
        self.y = self.X["sentiment_num"]
        
        # Drop all non-numeric and target columns from X
        self.X.drop(
            columns=["sentiment_num", "employee_id", "month"], 
            inplace=True, 
            errors='ignore'
        )
        
    def engineer_features(self, raw_df: pd.DataFrame, 
                        sentiment_scores_df: (pd.DataFrame | int | float)
                        ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Main public method to run the full feature engineering pipeline.

        Args:
            raw_df (pd.DataFrame): The raw email data.
            sentiment_scores_df (pd.DataFrame | int | float): 
                - For training: The full aggregated sentiment score DataFrame.
                - For prediction: A single (int or float) previous month's score.

        Returns:
            tuple[pd.DataFrame, pd.Series]: 
                - The final feature matrix (X).
                - The final target vector (y).
        """
        # --- 1. Prepare Raw Data ---
        self.raw_df = raw_df.copy().sort_values(by=["date", "employee_id"])
        self.raw_df['date'] = pd.to_datetime(self.raw_df['date'])
        self.raw_df['month'] = self.raw_df['date'].dt.to_period('M')
        if "sentiment" in self.raw_df.columns:
            self.raw_df["sentiment_num"] = self.raw_df["sentiment"].map(self.SENT_MAP)
        if "polarity" not in self.raw_df.columns:
            self.raw_df["polarity"] = 0.0 # Placeholder if not present
        if "text_len" not in self.raw_df.columns:
            self.raw_df["text_len"] = self.raw_df["body"].str.len().fillna(0)

        # --- 2. Prepare Base X and Scores ---
        self.sentiment_scores_df = sentiment_scores_df
        
        if isinstance(self.sentiment_scores_df, (int, float)):
            # Prediction case for a single employee
            # We must get the employee_id and month from the raw_df
            if self.raw_df.empty:
                raise ValueError("raw_df cannot be empty for single prediction")
            self.X = self.raw_df[['employee_id', 'month']].drop_duplicates().reset_index(drop=True)
            # Manually add sentiment_num as 0.0, it will be dropped later
            self.X['sentiment_num'] = 0.0 
        else:
            # Training or batch prediction case
            self.X = self.sentiment_scores_df[["employee_id", "month","sentiment_num"]].copy()
            self.X['month'] = pd.to_datetime(self.X['month'].astype(str)).dt.to_period('M')

        # --- 3. Run Engineering Pipeline ---
        self._prepare_features()
        self._kitchen_sink_feature_tranform()
        
        return self.X, self.y
                

class TrainRegressionModel:
    """
    Manages the training, evaluation, and saving of the regression model.
    
    This class uses the FeatureEngineer to build the dataset, then
    trains a LassoCV pipeline to predict sentiment scores.
    """

    def __init__(self, sentiment_scores_df: pd.DataFrame, raw_df: pd.DataFrame):
        """
        Initializes the trainer by engineering features.

        Args:
            sentiment_scores_df (pd.DataFrame): Aggregated sentiment scores.
            raw_df (pd.DataFrame): Raw email data.
        """
        print("Preparing features...")
        self.X, self.y = FeatureEngineer().engineer_features(raw_df, sentiment_scores_df)
        
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def train(self, test_size: float = 0.2, random_state: int = 42):
        """
        Splits data, defines, and trains the LassoCV pipeline.

        Args:
            test_size (float, optional): Proportion of data for the test set. 
                                         Defaults to 0.2.
            random_state (int, optional): Seed for reproducibility. 
                                          Defaults to 42.
        """
        print("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Define the pipeline: PolyFeatures -> Scaler -> LassoCV
        self.pipeline = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False), 
            StandardScaler(),
            LassoCV(cv=5, random_state=random_state, max_iter=2000) 
        )
        
        print("Training model...")
        self.pipeline.fit(self.X_train, self.y_train)
        print("Training complete.")

    def evaluate(self) -> float:
        """
        Evaluates the trained model on the test set.

        Prints the R-squared score and the top 10 features selected
        by the Lasso model.

        Returns:
            float: The R-squared score on the test set.
        """
        if self.pipeline is None:
            print("Model not trained yet. Please call .train() first.")
            return None
            
        print("Evaluating model...")
        score = self.pipeline.score(self.X_test, self.y_test)
        print(f"Model R-squared on test set: {score}")

        # --- Extract and display top features ---
        model = self.pipeline.named_steps['lassocv']
        poly_features = self.pipeline.named_steps['polynomialfeatures']
        
        # Get original feature names from the engineered X matrix
        original_names = self.X.columns 
        all_poly_feature_names = poly_features.get_feature_names_out(original_names)

        coeff_df = pd.DataFrame({
            'Feature': all_poly_feature_names,
            'Coefficient': model.coef_
        })
        
        selected_features = coeff_df[coeff_df['Coefficient'] != 0].sort_values(
            by='Coefficient', ascending=False
        )
        
        print("\n--- Top Model Features ---")
        print(selected_features.head(10))
        
        return score

    def save_model_artifacts(
        self, 
        model_path: str = "data/regression_model.joblib", 
        coeff_path: str = "data/regression_model_coefficients.csv"
    ):
        """
        Saves the trained pipeline and its coefficients.

        - Saves the entire pipeline object to a .joblib file.
        - Saves the non-zero feature coefficients to a .csv file.

        Args:
            model_path (str, optional): Path to save the .joblib model.
            coeff_path (str, optional): Path to save the .csv coefficients.
        """
        if self.pipeline is None:
            print("Model not trained yet. Cannot save artifacts.")
            return

        print(f"Saving model to {model_path}...")
        joblib.dump(self.pipeline, model_path)
        
        print(f"Saving coefficients to {coeff_path}...")
        
        model = self.pipeline.named_steps['lassocv']
        poly_features = self.pipeline.named_steps['polynomialfeatures']
        
        all_feature_names = poly_features.get_feature_names_out(self.X.columns)
        
        coeff_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Coefficient': model.coef_
        })
        
        # Save only the features Lasso actually used
        selected_features_df = coeff_df[coeff_df['Coefficient'] != 0].sort_values(
            by='Coefficient', ascending=False
        )
        
        selected_features_df.to_csv(coeff_path, index=False)
        print("Artifacts saved successfully.")
        

class PredictScore:
    """
    Loads a pre-trained model to make new sentiment predictions.
    
    Uses the same FeatureEngineer class from training to ensure
    identical feature creation for new data.
    """
    
    def __init__(self, model_path: str = "data/regression_model.joblib"):
        """
        Initializes the service by loading the model and feature engineer.

        Args:
            model_path (str, optional): Path to the saved .joblib model.
        """
        print(f"Loading model from {model_path}...")
        try:
            self.pipeline = joblib.load(model_path)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            self.pipeline = None
            
        self.feature_engineer = FeatureEngineer()

    def predict(
        self, 
        new_month_raw_data: pd.DataFrame, 
        historical_or_prior_score: (pd.DataFrame | int | float)
    ) -> (float | np.ndarray):
        """
        Makes a sentiment prediction for new data.

        This method supports both single predictions (one employee-month)
        and batch predictions (multiple).

        Args:
            new_month_raw_data (pd.DataFrame): Raw email data for the 
                                               employee-month(s).
            historical_or_prior_score (pd.DataFrame | int | float):
                - For a single prediction: The previous month's score (int/float).
                - For batch predictions: The full historical score DataFrame.

        Returns:
            (float | np.ndarray): 
                - A single predicted score (float) for a single prediction.
                - A NumPy array of scores for a batch prediction.
        """
        if self.pipeline is None:
            print("Error: Model is not loaded. Cannot make prediction.")
            return None
            
        print("Engineering features for new data...")
        # 1. Create the feature-engineered matrix (X)
        #    'y' will be None or dummy, so we ignore it with '_'
        X_pred, _ = self.feature_engineer.engineer_features(
            new_month_raw_data,
            historical_or_prior_score
        )
        
        if X_pred.empty:
            print("Warning: Feature engineering resulted in an empty DataFrame. No prediction.")
            return None
            
        print("Making prediction...")
        # 2. Predict using the loaded pipeline
        #    This automatically applies PolynomialFeatures and StandardScaler
        prediction_array = self.pipeline.predict(X_pred)
        
        # 3. Return a single number or the full array
        if len(prediction_array) == 1:
            return prediction_array[0]  
        else:
            return prediction_array