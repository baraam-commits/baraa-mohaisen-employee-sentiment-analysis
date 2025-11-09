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
    
    global SENT_MAP
    SENT_MAP = {
        "POSITIVE": 1, "Positive": 1,
        "NEGATIVE": -1, "Negative": -1,
        "NEUTRAL": 0,  "Neutral": 0
    }
    def __init__(self):
        self.raw_df = None
        self.sentiment_scores_df = None
        self.X = None
        self.y = None

    def _compute_total_len_per_month(self,df: pd.DataFrame):
        
        result = (
            df.groupby(['month', 'employee_id'], as_index=False)['text_len']
            .sum()
        )

        return result[['employee_id', 'month', 'text_len']]
    
    def _compute_avg_len_per_month(self,df: pd.DataFrame):
        
        df = self._compute_total_len_per_month(df)
        
        result = (
            df.groupby(['month', 'employee_id'], as_index=False)
            .agg(total_text_len=('text_len', 'sum'),
                message_count=('text_len', 'count'))
        )

        result['avg_msg_len'] = result['total_text_len'] / result['message_count']
        return result[['employee_id', 'month', 'avg_msg_len']]
        
    def _compute_avg_punctuation_density(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        avg_punct_density = mean( (count('!','?') per msg) / (char_length per msg) )
        grouped by (employee_id, month).
        """
        

        def _calc_punctuation_density(subj_series: pd.Series,
                                    include_fullwidth: bool = False,
                                    exclude_whitespace_from_length: bool = True) -> float:
            # Ensure strings
            s = subj_series.astype(str)

            # Pattern
            pattern = r"[!?]" if not include_fullwidth else r"[!?\uFF01\uFF1F]"

            # Count punctuation across all subjects
            punct_total = s.str.count(pattern, flags=re.UNICODE).sum()

            # Count characters across all subjects
            if exclude_whitespace_from_length:
                char_total = s.str.replace(r"\s+", "", regex=True).str.len().sum()
            else:
                char_total = s.str.len().sum()

            if char_total == 0:
                return 0.0

            return float(punct_total / char_total)

        # ensure datetime and month
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.to_period("M")

        monthly = (
            df.groupby(["month", "employee_id"], as_index=False)
            .agg(
                avg_body_punctuation_density    = ("body", _calc_punctuation_density),
                avg_subject_punctuation_density = ("Subject", _calc_punctuation_density),
            )
        )

        return monthly[["employee_id","month",
                        "avg_subject_punctuation_density",
                        "avg_body_punctuation_density"]]
            
    def _compute_avg_caps_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns one row per (employee_id, month) with:
        - avg_subject_caps_density
        - avg_body_caps_density

        ALL_CAPS words = tokens A–Z of length >= min_caps_len.
        total_words    = tokens matched by word_pattern.
        """
        df = df.copy()

        # Ensure month exists
        if "month" not in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["month"] = df["date"].dt.to_period("M")

        # Internal scalar helper that works on a Series of texts (one group)
        def _caps_density(series: pd.Series,
                        min_caps_len: int = 2,
                        word_pattern: str = r"\b[^\W_]+\b") -> float:
            s = series.fillna("").astype(str)

            # total words across the series
            total_words = s.str.findall(word_pattern, flags=re.UNICODE).str.len().sum()

            # ALL CAPS tokens across the series
            caps_regex = rf"\b[A-Z]{{{min_caps_len},}}\b"
            all_caps = s.str.findall(caps_regex).str.len().sum()

            if total_words == 0:
                return 0.0
            return float(all_caps / total_words)

        # Aggregate directly per employee-month
        out = (
            df.groupby(["employee_id", "month"], as_index=False)
            .agg(
                avg_subject_caps_density=("Subject", _caps_density),
                avg_body_caps_density=("body", _caps_density),
            )
        )

        return out[["employee_id", "month",
                    "avg_subject_caps_density", "avg_body_caps_density"]]
        
    def _compute_avg_raw_polarity(self,df: pd.DataFrame):
        result = (
            df.groupby(['month', 'employee_id'], as_index=False)["polarity"]
            .mean()
            .rename(columns={"polarity": "avg_raw_polarity"})
        )
        return result[['employee_id', 'month', 'avg_raw_polarity']]
    
    def _compute_sentiment_volatility(self,df: pd.DataFrame):
        result = (
            df.groupby(['month', 'employee_id'], as_index=False)["sentiment_num"]
            .std()
            .rename(columns={"sentiment_num": "sentiment_volatility"})
        )
        result['sentiment_volatility'] = result['sentiment_volatility'].fillna(0)
        return result[['employee_id', 'month', 'sentiment_volatility']]
    
    def _is_flight_risk_month(self, df: pd.DataFrame):
        threshold: int = 4
        result = (
            df.groupby(['month', 'employee_id'], as_index=False)
            .agg(total_negative_risk=('sentiment_num', lambda x: (x < 0).sum()))
        )
        result['is_flight_risk_month'] = np.where(abs(result['total_negative_risk']) >= threshold, 1, 0) 
        return result[['employee_id', 'month', 'is_flight_risk_month']]
    
    def _compute_previous_month_score(self,df: pd.DataFrame):
            results = df.copy(deep=False)
            results.set_index("month", inplace=True)
            results['previous_month_sentiment_score'] = results.groupby('employee_id')['sentiment_num'].shift(1, fill_value=0)
            return results.reset_index()[['employee_id', 'month', 'previous_month_sentiment_score']]
                
    def _prepare_features(self):
        
        
  
        features_func = [self._compute_total_len_per_month,
                        self._compute_avg_len_per_month,
                        self._compute_avg_punctuation_density,
                        self._compute_avg_caps_density,
                        self._compute_avg_raw_polarity,
                        self._compute_sentiment_volatility,
                        self._is_flight_risk_month]
        
        # this one takes self.sentiment_scores_df all the other ones take in raw_df
        print(self.raw_df.columns)
        print(self.raw_df.head)
        for func in features_func:
            print('calling')
            print(func)
            feature_df = func(self.raw_df)
            print("returned")
            print(feature_df)

            self.X = self.X.merge(feature_df, on=['month','employee_id'], how='left')
            
            print("final_df now has ")
            print(self.X.columns)
        if isinstance(self.sentiment_scores_df, (int,float)):
            self.X["previous_month_sentiment_score"] = self.sentiment_scores_df
        else:
            if "sentiment_num" not in self.sentiment_scores_df: self.sentiment_scores_df["sentiment_num"] = EmployeeScoring(self.raw_df).compute_scores()["sentiment_num"]
            self.X = self.X.merge(self._compute_previous_month_score(self.sentiment_scores_df), on=['employee_id', 'month'], how='left')
            
        print("final_df now has ")
        print(self.X.columns)
        
        print("Done\n")        
            

    def _kitchen_sink_feature_tranform(self):
        
        for col in self.X.columns:
            if col not in ['employee_id', 'month', 'sentiment_num', 'is_flight_risk_month',"previous_month_sentiment_score","avg_raw_polarity"]:
                if (self.X[col] >= 0).all():
                    self.X[f'log_{col}'] = np.log1p(self.X[col])
                    self.X[f'sqrt_{col}'] = np.sqrt(self.X[col])
        
        self.X.fillna(0, inplace=True)
        
        self.X.sort_values(by=['employee_id', 'month'], inplace=True)
        self.y = self.X["sentiment_num"]
        
        # 2. Drop all non-numeric columns from 'X'
        #    This is the fix!
        self.X.drop(
            columns=["sentiment_num", "employee_id", "month"], 
            inplace=True, 
            errors='ignore' # 'errors=ignore' is safer in case 'year' or others are in there
        )
        
    def engineer_features(self, raw_df: pd.DataFrame, sentiment_scores_df: (pd.DataFrame | int | float)):
        
        
        # raw dataset
        self.raw_df = raw_df.copy().sort_values(by=["date", "employee_id"])
        self.raw_df['month'] = self.raw_df['date'].dt.to_period('M')
        # self.raw_df['month'] = self.raw_df['date'].dt.year
        self.raw_df["sentiment_num"] = self.raw_df["sentiment"].map(SENT_MAP)
        # test dataset
        self.sentiment_scores_df = sentiment_scores_df.copy()
        
        self.X = self.sentiment_scores_df[["employee_id", "month","sentiment_num"]]
        self.y = pd.DataFrame()
        # add everything to x coppy that to y and then drop everything but the sentiment_num col 
        
        
        
        
        self._prepare_features()
        
        self._kitchen_sink_feature_tranform()
        
        print("final_df now has ")
        print(self.X.columns)
        return self.X, self.y
                

class TrainRegressionModel:

    def __init__(self, sentiment_scores_df: pd.DataFrame, raw_df: pd.DataFrame):
        
        self.X, self.y = FeatureEngineer().engineer_features(raw_df,sentiment_scores_df)
        print("Preparing features...")
        self.pipeline = None
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    

        
    def train(self, test_size: float = 0.2, random_state: int = 42):
        """
        Prepares features, splits data, and trains the LassoCV pipeline.
        """

        # This method should modify self.X in place
        
        print("Splitting data...")
        # Store the splits as class attributes
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Define the pipeline
        self.pipeline = make_pipeline(
            # include_bias=False is recommended
            PolynomialFeatures(degree=2, include_bias=False), 
            StandardScaler(),
            # Add random_state for reproducibility and max_iter to help convergence
            LassoCV(cv=5, random_state=random_state, max_iter=2000) 
        )
        
        print("Training model...")
        self.pipeline.fit(self.X_train, self.y_train)
        print("Training complete.")

    # --- 3. A Dedicated 'evaluate' Method ---
        
    def evaluate(self):
            """
            Scores the trained pipeline on the test set and prints the R-squared.
            Also prints the top features selected by the model.
            """
            if self.pipeline is None:
                print("Model not trained yet. Please call .train() first.")
                return None
                
            print("Evaluating model...")
            score = self.pipeline.score(self.X_test, self.y_test)
            print(f"Model R-squared on test set: {score}")

            # --- HERE IS THE CORRECTED LOGIC ---
            
            # 1. Get the fitted LassoCV model
            model = self.pipeline.named_steps['lassocv']
            
            # 2. Get the PolynomialFeatures step
            poly_features = self.pipeline.named_steps['polynomialfeatures']
            
            # 3. Get the *original* column names from self.X
            #    (self.X is the full feature set before train/test split)
            original_names = self.X.columns 
            
            # 4. Get the *full list* of generated polynomial feature names
            all_poly_feature_names = poly_features.get_feature_names_out(original_names)

            # 5. Now you can create your DataFrame of features and coefficients
            coeff_df = pd.DataFrame({
                'Feature': all_poly_feature_names,
                'Coefficient': model.coef_
            })
            
            # 6. Print the top features
            selected_features = coeff_df[coeff_df['Coefficient'] != 0].sort_values(
                by='Coefficient', ascending=False
            )
            
            print("\n--- Top Model Features ---")
            print(selected_features.head(10))
            # --- END CORRECTED LOGIC ---
            
            return score

    # --- 4. Your Fixed 'save' Method ---
    
    def save_model_artifacts(self, 
                             model_path: str = "data/regression_model.joblib", 
                             coeff_path: str = "data/regression_model_coefficients.csv"):
        """
        Saves the trained pipeline and a CSV of its selected features.
        """
        if self.pipeline is None:
            print("Model not trained yet. Cannot save artifacts.")
            return

        print(f"Saving model to {model_path}...")
        joblib.dump(self.pipeline, model_path)
        
        print(f"Saving coefficients to {coeff_path}...")
        
        # Extract components from pipeline
        model = self.pipeline.named_steps['lassocv']
        poly_features = self.pipeline.named_steps['polynomialfeatures']
        
        # --- FIX ---
        # Get all feature names generated by PolynomialFeatures.
        # It needs the *original* column names from self.X, *before* the split.
        all_feature_names = poly_features.get_feature_names_out(self.X.columns)
        
        # Create the DataFrame
        coeff_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Coefficient': model.coef_
        })
        
        # --- IMPROVEMENT ---
        # Save a CSV of only the features the model *kept*
        selected_features_df = coeff_df[coeff_df['Coefficient'] != 0].sort_values(
            by='Coefficient', ascending=False
        )
        
        selected_features_df.to_csv(coeff_path, index=False)
        print("Artifacts saved successfully.")
        

class PredictScore:
    
    def __init__(self, model_path: str = "data/regression_model.joblib"):
        
        print(f"Loading model from {model_path}...")
        try:
            # 1. Load the entire fitted pipeline
            self.pipeline = joblib.load(model_path)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            self.pipeline = None
            
        # 2. Create an instance of your feature engineer
        self.feature_engineer = FeatureEngineer()

    def predict(self, 
                new_month_raw_data: pd.DataFrame, 
                historical_or_prior_score: (pd.DataFrame | int | float)):
        
        if self.pipeline is None:
            print("Error: Model is not loaded. Cannot make prediction.")
            return None
            
        print("Engineering features for new data...")
        # 1. Create the feature-engineered row (X)
        # We don't need 'y' (the second return value), so we use _
        # Your engineer_features handles all the logic, including the
        # log/sqrt transforms and dropping the ID columns.
        X_pred, _ = self.feature_engineer.engineer_features(
            new_month_raw_data,
            historical_or_prior_score
        )
        
        if X_pred.empty:
            print("Warning: Feature engineering resulted in an empty DataFrame. No prediction.")
            return None
            
        print("Making prediction...")
        # 2. Predict using the loaded pipeline
        # The pipeline automatically handles PolynomialFeatures and StandardScaler
        prediction_array = self.pipeline.predict(X_pred)
        
        if len(prediction_array) == 1:
            return prediction_array[0]  # Return a single float
        else:
            return prediction_array