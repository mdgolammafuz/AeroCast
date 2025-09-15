# scripts/compare_models.py
import pandas as pd

df_gru = pd.read_csv('logs/prediction_gru.csv')
df_lstm = pd.read_csv('logs/prediction_lstm.csv')
df_prophet = pd.read_csv('logs/prediction_prophet.csv')

# Compare MAE, MAPE
def score(df, name):
    mae = (df['forecast'] - df['actual']).abs().mean()
    print(f"{name} MAE: {mae:.2f}")

score(df_gru, "GRU")
score(df_lstm, "LSTM")
score(df_prophet, "Prophet")
