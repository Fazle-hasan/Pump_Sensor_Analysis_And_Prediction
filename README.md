# Pump Sensor Data Analysis

A complete predictive maintenance pipeline for the Pump Sensor dataset: Exploratory Data Analysis, Feature Selection, Multi-model Failure Classification, and LSTM Temporal Sequence Modeling.

## Dataset

You can download the dataset from Kaggle:
[Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)

> **Instructions**: Download `sensor.csv` and place it in the same directory as the notebook. The dataset is not included in this repository due to its size.

## Highlights & Key Findings
- **Data Cleaning**: `sensor_15` had 0 valid readings and was dropped. Missing values in other sensors were imputed using forward fill, avoiding data leakage, followed by backward fill to handle initial gaps.
- **Exploratory Data Analysis**: The data is highly imbalanced (`NORMAL` 93.4%, `RECOVERING` 6.6%, `BROKEN` <0.01%). Traditional accuracy is a misleading metric here. Instead, Macro-F1 and Precision-Recall curves are used.
- **Class-Imbalance Handling**: Applied a refined SMOTE strategy only to the training set after chronological split, oversampling the minority class (`BROKEN`) up to the count of the intermediate class (`RECOVERING`) to avoid overfitting on noisy synthetic samples.
- **Feature Selection**: Relying primarily on the raw 51 sensor readings provides an excellent signal. This makes the model more robust and less prone to overfitting on a tiny set of failures compared to an exploded feature space.
- **Model Comparison**: Three static classifiers were benchmarked. XGBoost and Random Forest captured the non-linear correlated features best, handling residual imbalance via class weights (or sample weights).
- **Advanced Temporal Modeling (LSTM)**: To capture the temporal evolution leading up to a breakdown, an LSTM network was implemented. It maps the machine status to a continuous "Operation Score" (NORMAL=1.0, RECOVERING=0.5, BROKEN=0.0). By framing the problem as time-series forecasting, the LSTM smoothly tracks the shift towards failure, providing a powerful continuous early warning system rather than just a static binary prediction.

## Setup

Install the required dependencies. It is recommended to use a virtual environment:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn tensorflow
```

## Running the Notebook

Open `Pump_Sensor_Analysis.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab, and run all cells top-to-bottom. If you open the notebook in Colab, the environment (including TensorFlow) is already pre-configured for you.

```bash
jupyter notebook Pump_Sensor_Analysis.ipynb
```

## Files in this Repository

| File | Description |
|---|---|
| `Pump_Sensor_Analysis.ipynb` | Main analysis, machine learning, and deep learning notebook |