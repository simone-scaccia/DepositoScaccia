import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# Apply scaling to "Annual Income (k$)" and "Spending Score (1-100)"
scaler = StandardScaler()
df[['Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])

def compute_and_plot_kmeans_clusters(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans.fit(X)
    df['Cluster'] = kmeans.labels_

    # Show the points of "Annual Income (k$)" vs "Spending Score (1-100)" with clusters and centroids
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f'Annual Income vs Spending Score with {n_clusters} Clusters')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.colorbar(label='Cluster')
    plt.show()

    return kmeans, df

kmeans_5, df_5 = compute_and_plot_kmeans_clusters(5)
kmeans_7, df_7 = compute_and_plot_kmeans_clusters(7)

