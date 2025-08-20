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

# Compute Kmeans to cluster "Annual Income (k$)" vs "Spending Score (1-100)"
kmeans = KMeans(n_clusters=5, random_state=42)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

# Show the points of "Annual Income (k$)" vs "Spending Score (1-100)" with clusters and centroids
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Annual Income vs Spending Score with Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()
