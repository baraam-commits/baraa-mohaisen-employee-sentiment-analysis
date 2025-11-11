# baraa-mohaisen-employee-sentiment-analysis

This project analyzes an unlabeled dataset of employee email messages to assess sentiment, identify trends, and build a predictive model for employee engagement. The entire pipeline, from raw data ingestion to predictive modeling, is containerized in a modular Python library and executed in the main Jupyter Notebook, as required by the project specification.

## Key Features & Methodology

The project is divided into six main tasks, following the project specification:

1.  **Sentiment Labeling (Task 1):** A sophisticated ensemble model is used to label each message as `POSITIVE`, `NEGATIVE`, or `NEUTRAL`. This approach leverages two models to balance polarity detection with contextual nuance:
    * **Primary Model:** `distilbert-base-uncased-finetuned-sst-2-english` (strong binary polarity).
    * **Fallback Model:** `j-hartmann/sentiment-roberta-large-english-3-classes` (handles neutrality).

2.  **Exploratory Data Analysis (Task 2):** A series of visualizations are generated to validate the labeled data and ensure it reflects "normative" behavior. This includes analyzing sentiment distribution, temporal trends, and correlations between message length and sentiment.

3.  **Employee Score Calculation (Task 3):** A monthly sentiment score is calculated for each employee by aggregating their message scores (Positive: `+1`, Negative: `-1`, Neutral: `0`).

4.  **Employee Ranking (Task 4):** Based on the monthly scores, the top three positive and top three negative employees are identified for each month.

5.  **Flight Risk Identification (Task 5):** A critical deliverable, this task identifies employees at "flight risk," defined as any employee who sends **4 or more negative messages within a 30-day rolling window**.

6.  **Predictive Modeling (Task 6):** A "clever" predictive model was built to forecast an employee's next monthly sentiment score. This solution directly addresses the project's "cleverness test" by:
    * **Engineering advanced features** like `sentiment_volatility` and `previous_month_score`.
    * **Solving the multicollinearity trap** (e.g., `message_length` vs. `word_count`) using an automated `scikit-learn` pipeline.
    * **Using `LassoCV`** to perform automatic feature selection and regularization, identifying the most significant predictors while shrinking redundant features to zero.

## Repository Structure

The project is organized in a modular and reproducible structure:

. ├── notebooks/ │ └── main.ipynb # Main executable notebook ├── src/ │ ├── load_data.py # Handles data ingestion and preprocessing │ ├── labeling.py # Contains the ensemble sentiment labeling logic │ ├── Plot_data.py # Generates all EDA visualizations │ ├── ranking.py # Calculates monthly scores and flight risks │ ├── regression.py # Builds and trains the LassoCV pipeline │ └── model_plotter.py # Generates model diagnostic plots ├── data/ │ ├── test(in).csv # The raw, unlabeled input dataset │ ├── labeld_sentiments.csv # Cached data after Task 1 labeling │ ├── ... # Other data outputs ├── visualizations/ │ ├── ... # All output plots from EDA and modeling ├── requirements.txt # All Python dependencies └── README.md # This file

## Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd employee-sentiment-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## How to Run

The entire project pipeline is executed from the main Jupyter Notebook, which is designed to be read like a report.

1.  **Start Jupyter:**
    ```sh
    jupyter lab
    ```
2.  **Open and run the notebook:**
    Navigate to `notebooks/main.ipynb` and run the cells in order.

**Note on Caching:** The sentiment labeling process (Task 1) is computationally expensive. The pipeline automatically saves the labeled results to `data/labeld_sentiments.csv`. On subsequent runs, it will load this cached file, skipping the slow labeling step. To re-run the labeling, delete this file.

---

## Summary of Findings

This section fulfills the summary requirement of the project deliverables.

### Key Insights & Recommendations

* **Overall sentiment is improving:** The "Average Sentiment Over Time" plot shows a gradual upward trend in employee sentiment, validating that the model is capturing meaningful dynamic patterns, not just noise.
* **Negative messages are more detailed:** The EDA shows that negative messages are, on average, longer and have a wider variance in length. This suggests employees are more verbose and elaborate when discussing issues. **Recommendation:** This provides a quantitative signal for managers to pay closer attention to uncharacteristically long messages.
* **High-volume communication is often neutral:** Highly active employees (e.g., Lydia Delgado, John Arnold) show a larger proportion of neutral messages. This is consistent with managerial or administrative roles involving high volumes of factual, non-emotional communication.
* **Sentiment is predictable:** The final regression model achieved an **R-squared of 0.795** on the held-out test set, indicating it can effectively predict future sentiment scores based on past behavior.
* **Key predictors were identified:** The `LassoCV` model automatically identified statistically significant predictors of monthly sentiment, such as `previous_month_sentiment_score` and `avg_raw_polarity^2`, while successfully ignoring redundant, collinear features.

### Final Deliverables

* **Top 3 Positive/Negative Employees:**
    The monthly rankings are generated by **Task 4** in the `notebooks/main.ipynb` notebook and saved to `data/employee_monthly_rankings.csv`.

* **Flight Risk Employees:**
    The full list of employees flagged as flight risks is generated by **Task 5** in the `notebooks/main.ipynb` notebook and saved to `data/flight_risk_employees.csv`.