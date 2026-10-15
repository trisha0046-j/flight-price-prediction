# Flight Price Prediction (PRCP-1025)

A supervised machine learning (regression) project that predicts the price of a domestic Indian flight ticket from details such as airline, route, stops, duration, journey date and timings.

## Problem Statement

Flight prices change frequently, so travellers find it hard to judge whether a fare is reasonable, and airlines and travel sites want to know which factors drive price. This project:

- performs a full **EDA** on the flight fare data
- builds a **regression model** that estimates ticket price
- compares several models and selects one for production
- documents data challenges and how they were handled

## Dataset

- File: `Flight_Fare.xlsx` — 10,683 records, 11 columns, journeys between **March and June 2019**
- Columns: Airline, Date_of_Journey, Source, Destination, Route, Dep_Time, Arrival_Time, Duration, Total_Stops, Additional_Info, Price (target, in Rs.)
- The dataset is **not included** in this repository. Place `Flight_Fare.xlsx` in the project folder before running the notebook.

## Workflow

1. **Data understanding** — shape, types, missing values (1 row), duplicates (220 rows)
2. **EDA** — price distribution (right-skewed), airline, month/day, source/destination, route, departure/arrival hour, duration, stops, additional info, correlation heatmap
3. **Cleaning and feature engineering**
   - dropped the single row with missing values and removed 220 duplicates (10,683 → 10,462 rows)
   - merged inconsistent labels (`New Delhi` → `Delhi`, `No Info` → `No info`)
   - extracted `Journey_Day`, `Journey_Month`, departure/arrival hour and minute, and `Duration_Mins`
   - converted `Total_Stops` to numbers (0–4)
   - checked price outliers with the IQR rule (0.9% of rows) and kept them, since they are mostly genuine business-class and premium fares
4. **Encoding** — one-hot encoding of Airline, Source, Destination and Additional_Info, **fitted on the training set only** to avoid data leakage (12 features → 39 columns)
5. **Train/test split** — 80/20 (8,369 / 2,093 rows)
6. **Models** — Linear Regression, Decision Tree, Random Forest, Gradient Boosting
7. **Evaluation** — R², MAE, RMSE, 5-fold cross-validation, overfitting gap
8. **Tuning** — RandomizedSearchCV (20 combinations × 5 folds) on the Random Forest
9. **Prediction function** — takes raw flight details and returns an estimated price

## Results

| Model | Test R² | Test MAE (Rs.) | Test RMSE (Rs.) | CV R² (mean) |
|---|---:|---:|---:|---:|
| **Random Forest** | **0.895** | **673** | **1,481** | 0.891 |
| Decision Tree | 0.856 | 753 | 1,735 | 0.834 |
| Gradient Boosting | 0.802 | 1,262 | 2,031 | 0.819 |
| Linear Regression | 0.649 | 1,795 | 2,705 | 0.680 |

**After tuning** (best: `n_estimators=300`, `max_depth=30`, `max_features=0.5`, `min_samples_split=5`, `min_samples_leaf=1`):

| Metric | Default RF | Tuned RF |
|---|---:|---:|
| 5-fold CV RMSE | 1,528 | 1,484 |
| Test R² | 0.895 | 0.896 |
| Test RMSE | 1,481 | 1,476 |
| Test MAE | 673 | 692 |

**Key takeaways**

- Random Forest is the best model; the average prediction is roughly Rs. 700 away from the actual fare and the model explains about 90% of the variation in price.
- Tuning gave only a small improvement (about 2.9% lower cross-validated RMSE) and reduced the train-test gap from 0.088 to about 0.070, so the default model was already close to the best this feature set allows.
- Linear Regression underfits; the Decision Tree overfits (train R² 0.996 vs test 0.856).
- Most important features: `Duration_Mins`, `Total_Stops`, `Journey_Day`, Jet Airways (and Jet Airways Business) and `Additional_Info`.
- Flights with more stops and longer duration cost more; Jet Airways and business class are the most expensive; March had the highest average fares and April the lowest.
- The model tends to under-predict very expensive tickets.

## Limitations & Future Improvements

- Data covers only March–June 2019, so seasonality and festival pricing are not captured
- The train-test split is random, not time-based; real use would need booking-date / days-before-departure features
- Try Gradient Boosting with tuning, XGBoost or LightGBM, and a log-transform of price to handle the skew
- Wrap the preprocessing and model in a scikit-learn `Pipeline` and deploy as a Streamlit or FastAPI app

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter

## How to Run

```bash
git clone https://github.com/<your-username>/flight-price-prediction.git
cd flight-price-prediction

python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

# place Flight_Fare.xlsx in this folder, then:
jupyter notebook Flight_Fare_Prediction_Project.ipynb
```

## Project Structure

```
flight-price-prediction/
├── Flight_Fare_Prediction_Project.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Author

**Trisha** — AI Engineer Trainee, DataMites
B.Tech CSE · Python · Machine Learning · Data Visualization

[GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-handle>)
