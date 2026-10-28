
# Cloud Workload Forecasting & Analysis

A machine learning project that analyzes Azure virtual machine workloads and forecasts active VM count and aggregate CPU utilization using time-series forecasting techniques.

## Project Overview

This project uses CPU readings from 10,000 Azure virtual machines to understand workload patterns and predict future resource demand. It includes exploratory data analysis, preprocessing, baseline forecasting, machine learning, and model evaluation.

## Features

- Exploratory Data Analysis (EDA)
- CPU utilization and workload trend analysis
- Data preprocessing and timestamp-based aggregation
- Time-aware training and testing split
- Baseline forecasting comparison
- Machine learning model training
- Recursive forecasting for future timestamps
- Model evaluation using MAE and RMSE
- Export predictions to `predictions.csv`

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook

---

## Project Structure

```text
forecasting-challenge/
│
├── data/
│   ├── DATA_CARD.md
│   ├── vm_cpu_readings.csv
│   └── vm_metadata.csv
│
├── models/
│   └── gradient_boosting.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── baseline_model.ipynb
│   └── test_models.ipynb
│
├── reports/
│   └── final_report.md
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── predictions.csv
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset is based on a curated subset of the Microsoft Azure Public Dataset V2.

- **Virtual machines:** 10,000
- **CPU readings per VM:** 134
- **Total VM-level observations:** 1,340,000
- **Sampling interval:** 5 minutes
- **Observation window:** Approximately 11.1 hours

The dataset includes CPU utilization readings and virtual machine metadata.

## Forecasting Targets

The project forecasts two workload metrics:

- **`active_vm_count`** — The number of unique virtual machines observed at each timestamp.
- **`aggregate_cpu_utilization`** — The mean of the VMs' average CPU utilization at each timestamp, expressed as a percentage.

---

## Exploratory Data Analysis

EDA is used to understand CPU utilization distributions, workload changes, and VM activity over time.

### Average CPU Utilization Distribution

![Average CPU](images/img1.png)

### Aggregate CPU Utilization Over Time

![Aggregate CPU Utilization Over Time](images/img2.png)

### Active VM Count Over Time

![Active VM](images/img3.png)
---

## Installation

Clone the repository:

```bash
git clone https://github.com/NisarAhmad7/forecasting-challenge.git
cd forecasting-challenge
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Training

Run the training script:

```bash
python src/train.py
```

## Generate Predictions

Run the prediction script:

```bash
python src/predict.py
```

**Note:** These commands must match the actual interfaces implemented in the source files before submission.

---

## Forecasting Approach

The forecasting workflow includes:

1. Aggregate VM-level readings into timestamp-level workload targets.
2. Sort observations chronologically.
3. Use the first 110 timestamps as the historical training period.
4. Reserve the final 24 timestamps as the holdout forecasting period.
5. Create lag features using historical workload observations.
6. Train and compare forecasting models against a simple baseline.
7. Generate the 24 future predictions without using actual holdout targets as input features.
8. Evaluate predictions against the held-out actual values.

A recursive forecasting approach feeds each predicted step into the next step's lag features.

---

## Model Evaluation

The models are evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** Gives greater weight to larger prediction errors.

The machine learning model should be compared with a baseline forecast using the same time-based evaluation period.

Results should be reported separately for active VM count and aggregate CPU utilization because the targets use different scales.

---

## Prediction Output

The final output file is `predictions.csv`.

It must contain exactly these columns, in this order:

```csv
timestamp,active_vm_count,aggregate_cpu_utilization
```

The submission must contain:

- Exactly 24 forecast rows
- Chronological timestamps matching the final 24 dataset timestamps
- No duplicate or missing timestamps
- Finite numeric predictions
- Non-negative active VM counts
- CPU utilization values between 0 and 100

The CSV must not contain an extra index column.

---

## Reproducibility

Install the dependencies listed in `requirements.txt` and follow the documented training and prediction commands.

The code should use repository-relative file paths and keep the holdout targets out of model fitting, feature construction, and model selection.

---

## Limitations

The dataset covers approximately 11.1 hours of workload observations. This short window limits the ability to identify daily or weekly seasonality and long-term workload trends.

Forecast accuracy may also be affected by sudden workload changes and the limited amount of historical data available for recursive predictions.

---

## Author

**Nisar Ahmad Zamani**

- GitHub: [NisarAhmad7](https://github.com/NisarAhmad7)
- LinkedIn: [LinkedIn_Profile](https://www.linkedin.com/in/nisar-ahmad-zamani-440769436/)
