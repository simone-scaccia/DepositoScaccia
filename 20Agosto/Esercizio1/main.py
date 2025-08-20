import pandas as pd

# Read Mall_Customers.csv
df = pd.read_csv('Mall_Customers.csv')

# Print the head of the DataFrame
print(df.head())

# Print the summary statistics of the DataFrame
print(df.describe())