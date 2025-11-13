# baraa-mohaisen-employee-sentiment-analysis

This project analyzes an unlabeled dataset of employee email messages to assess sentiment, identify trends, and build a predictive model for employee engagement. The entire pipeline, from raw data ingestion to predictive modeling, is containerized in a modular Python library and executed in the main Jupyter Notebook, as required by the project specification.

## Key Features & Methodology

The project is divided into six main tasks, following the project specification:

---

### **1. Sentiment Labeling (Task 1)**  
A sophisticated ensemble model labels each message as `POSITIVE`, `NEGATIVE`, or `NEUTRAL`.  
This balances polarity detection with contextual nuance.

**Models used:**
- **Primary:** `distilbert-base-uncased-finetuned-sst-2-english`  
  - Strong performance in binary sentiment classification.
- **Fallback:** `j-hartmann/sentiment-roberta-large-english-3-classes`  
  - Enables explicit “NEUTRAL” detection.

---

### **2. Exploratory Data Analysis (Task 2)**  
Generated a suite of visualizations to understand distributions, trends, and message behaviors:
- Sentiment over time  
- Message length distributions  
- Per-employee sentiment profiles  
- Polarity and classification-confidence patterns

Visualizations are produced through a dedicated `PlotData` class and saved as local PNG files for notebook/report inclusion.

---

### **3. Monthly Sentiment Scoring (Task 3)**  
Each employee receives a monthly sentiment score computed from aggregated labeled messages.

Metrics include:
- Count of positive, negative, and neutral messages
- Weighted sentiment score
- Additional **polarity metric** computed from model weights:
  - `+weight` for positive labels  
  - `−weight` for negative labels

These scores form the basis for downstream modeling and risk detection.

---

### **4. Predictive Modeling (Task 4)**  
A linear regression model is trained to predict next-month sentiment score for each employee.

Pipeline:
1. Feature engineering via `FeatureEngineer`
2. Training via `TrainRegressionModel`
3. Prediction through standardized `.predict()` API

This model forecasts expected engagement trends based on historical sentiment data.

---

### **5. Employee Ranking (Task 5)**  
Employees are ranked monthly based on:
- Highest sentiment scores  
- Lowest sentiment scores  
- Sentiment change across months  

Useful for identifying standout performers or early signs of morale issues.

---

### **6. Flight-Risk Detection (Task 6)**  
Employees are flagged as **flight risks** when they show sustained negativity:
- Rolling 30-day window  
- `>= 4` negative messages triggers risk flag  

The `flight_risk_analysis()` method supports:  
- Returning only employee IDs  
- Returning full event detail (`first_flight_risk`, `max_neg_sent_in_30d`)  

Additional plotting methods visualize:
- Negative-message rolling windows  
- Flight-risk timelines  
- Per-employee negativity patterns  
- Aggregate organization-level risk overview

---

## Project Architecture

.
├── src/
│   ├── load_data.py             # Data ingestion & preprocessing
│   ├── labeling.py              # Ensemble sentiment labeling
│   ├── plot_data.py             # All EDA visualization functions
│   ├── ranking.py               # Monthly scoring, rankings, flight risk
│   ├── regression.py            # LassoCV feature engineering + model
│   └── model_plotter.py         # Diagnostic plots for regression
├── data/
│   ├── test(in).csv             # Raw dataset
│   ├── labeled_sentiments.csv   # Cached output from Task 1
│   ├── employee_monthly_scores.csv
│   ├── employee_monthly_rankings.csv
│   └── flight_risk_employees.csv
├── visualizations/              # All EDA + model plots
├── requirements.txt
├── README.md
└── main.ipynb               # Main executable notebook


Each major task is encapsulated in a class-based module.

---

## Jupyter Notebook Workflow

The notebook serves as the main interface for executing the full pipeline:
- Load and inspect data
- Run sentiment labeling
- Generate EDA visuals
- Aggregate monthly scores
- Train regression model
- Generate predictions
- Run ranking and flight-risk analysis
- Display stored PNG plots for reporting

---

## Polarity Metric

During classification, each model output `(label, confidence)` contributes to a raw polarity score:

if label == "POSITIVE": polarity += weight
if label == "NEGATIVE": polarity -= weight


This yields a continuous polarity scale reflecting classification confidence and strength of sentiment.

Plots generated for polarity include:
- Monthly polarity distributions  
- Per-employee polarity trends  
- Polarity vs. sentiment class scatterplots  
- High-confidence polarity outlier charts  

---



## Overall Summary

This project delivers a complete, modular sentiment-analysis pipeline designed for:
- Efficient labeling of raw messages  
- Clear monthly scoring systems  
- Predictive modeling of employee morale  
- Ranking insights  
- Early detection of flight risks  
- Robust visual analytics  

The design emphasizes clarity, modularity, reproducibility, and extensibility for future analysis or integration into enterprise HR analytics systems.

# Top 3 and Bottom 3 Employees (Overall Totals)

Total sentiment scores summed across all months:

### **Top 3 Employees (Best Overall Sentiment)**

| Employee         | Total Score |
|------------------|-------------|
| **lydia.delgado** | **34.83** |
| **sally.beck**     | **28.50** |
| **patti.thompson** | **27.33** |

### **Bottom 3 Employees (Worst Overall Sentiment)**

| Employee         | Total Score |
|------------------|-------------|
| **kayne.coulter** | **18.17** |
| **bobette.riner** | **11.50** |
| **rhonda.denton** | **2.83** |

These appear consistent with communication volume, tone trends, and raw polarity scores.

---

# Flight Risk Employees

Detected using the 30-day rolling window with threshold ≥ 4 negative messages:

- bobette.riner
- don.baughman
- eric.bass
- john.arnold
- johnny.palmer
- kayne.coulter
- lydia.delgado
- patti.thompson
- rhonda.denton
- sally.beck
