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

# Classify each hour of the day as "basso consumo" or "alto consumo" compared to the daily, weekly, and monthly average load
# Assuming "basso consumo" is below average and "alto consumo" is above average
# Calculate daily average load
pjm_hourly_load['Daily_classification'] = pjm_hourly_load.apply(
    lambda row: 'basso consumo' if row['PJM_Load'] < pjm_daily_load[pjm_daily_load['Date'] == row['Date']]['PJM_Load'].values[0] else 'alto consumo',
    axis=1
)
print("Daily classification completed.")

# Calculate weekly average load
pjm_hourly_load['Weekly_classification'] = pjm_hourly_load.apply(
    lambda row: 'basso consumo' if row['PJM_Load'] < pjm_weekly_load[pjm_weekly_load['Week'] == row['Week']]['PJM_Load'].values[0] else 'alto consumo',
    axis=1
)
print("Weekly classification completed.")

# Calculate monthly average load
pjm_hourly_load['Monthly_classification'] = pjm_hourly_load.apply(
    lambda row: 'basso consumo' if row['PJM_Load'] < pjm_monthly_load[pjm_monthly_load['Month'] == row['Month']]['PJM_Load'].values[0] else 'alto consumo',
    axis=1
)
print("Monthly classification completed.")

# Save the classified data to a new CSV file
pjm_hourly_load.to_csv('pjm_hourly_load_classified.csv', index=False)
print("Classified data saved to 'pjm_hourly_load_classified.csv'.")