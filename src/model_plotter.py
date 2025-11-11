import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# --- The "Caveat" in action ---
# We import your FeatureEngineer to use it internally.
try:
    from src.regression import FeatureEngineer
except ImportError:
    # Fallback if the file structure is different
    from regression import FeatureEngineer

class ModelPlotter:
    """
    Generates diagnostic plots for a trained sentiment regression model.

    This class is self-contained. It takes the *raw data* and a
    *saved model path*, then re-runs the feature engineering
    and train/test split internally to generate plots.
    """

    def __init__(
        self, 
        pipeline_path: str, 
        raw_df: pd.DataFrame, 
        sentiment_scores_df: pd.DataFrame,
        test_size: float = 0.2, 
        random_state: int = 42
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