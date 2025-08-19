import pandas as pd

# Load pjm_hourly_est.csv dataset
pjm_hourly_est = pd.read_csv('pjm_hourly_est.csv')

# Extract only Datetime and PJM_Load columns
pjm_hourly_load = pjm_hourly_est[['Datetime', 'PJM_Load']]

# Extract date, hour, month, and week from Datetime
pjm_hourly_load['Date'] = pd.to_datetime(pjm_hourly_load['Datetime']).dt.date
pjm_hourly_load['Hour'] = pd.to_datetime(pjm_hourly_load['Datetime']).dt.hour
pjm_hourly_load['Month'] = pd.to_datetime(pjm_hourly_load['Datetime']).dt.month
pjm_hourly_load['Week'] = pd.to_datetime(pjm_hourly_load['Datetime']).dt.isocalendar().week

# Average PJM_Load by day, week, and month
pjm_daily_load = pjm_hourly_load.groupby('Date')['PJM_Load'].mean().reset_index()
pjm_weekly_load = pjm_hourly_load.groupby('Week')['PJM_Load'].mean().reset_index()
pjm_monthly_load = pjm_hourly_load.groupby('Month')['PJM_Load'].mean().reset_index()
