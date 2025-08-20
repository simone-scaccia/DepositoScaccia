import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read Mall_Customers.csv
df = pd.read_csv('Mall_Customers.csv')

# Print the head of the DataFrame
print(df.head())

# Print the summary statistics of the DataFrame
print(df.describe())

# Show the points of "Annual Income (k$)" vs "Spending Score (1-100)"
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
