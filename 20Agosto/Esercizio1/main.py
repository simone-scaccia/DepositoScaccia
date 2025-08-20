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

# Show the points of "Annual Income (k$)" vs "Spending Score (1-100)" and save the plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig('income_vs_spending.png')
plt.close()

# Apply scaling to "Annual Income (k$)" and "Spending Score (1-100)"
scaler = StandardScaler()
df[['Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])

def compute_and_plot_kmeans_clusters(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans.fit(X)
    df['Cluster'] = kmeans.labels_

    # Show the points of "Annual Income (k$)" vs "Spending Score (1-100)" with clusters and centroids and save the plots
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f'Annual Income vs Spending Score with {n_clusters} Clusters')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.colorbar(label='Cluster')
    plt.savefig(f'income_vs_spending_clusters_{n_clusters}.png')
    plt.close()

    return kmeans, df

kmeans_5, df_5 = compute_and_plot_kmeans_clusters(5)
kmeans_7, df_7 = compute_and_plot_kmeans_clusters(7)


# Compute the average of the distance between points and their cluster centroids
def compute_cluster_density(kmeans, df):
    distances = kmeans.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    avg_distances = distances.mean(axis=0)
    return avg_distances

density_5 = compute_cluster_density(kmeans_5, df_5)
density_7 = compute_cluster_density(kmeans_7, df_7)

print("Density for 5 clusters:")
print(density_5)
print("Density for 7 clusters:")
print(density_7)