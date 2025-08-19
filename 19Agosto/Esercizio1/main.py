import pandas as pd

# Load pjm_hourly_est.csv dataset
pjm_hourly_est = pd.read_csv('pjm_hourly_est.csv')

# Extract only Datetime and PJM_load columns
pjm_hourly_load = pjm_hourly_est[['Datetime', 'PJM_load']]

