import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

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

def compute_and_plot_kmeans_clusters(df, n_clusters):
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

kmeans_5, df_5 = compute_and_plot_kmeans_clusters(df.copy(), 5)
kmeans_7, df_7 = compute_and_plot_kmeans_clusters(df.copy(), 7)


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

# Compute the silhouette score
def compute_and_plot_silhouette(df, labels):
    # Compute the silhouette score
    silhouette_avg = silhouette_score(df[['Annual Income (k$)', 'Spending Score (1-100)']], labels)
    samples_silhouette = silhouette_samples(df[['Annual Income (k$)', 'Spending Score (1-100)']], labels)

    # Sort the silhouette scores for visualization
    sorted_labels = np.argsort(labels)
    sorted_scores = samples_silhouette[sorted_labels]
    sorted_clusters = labels[sorted_labels]

    # The number of clusters are the unique values in labels
    n_clusters = len(set(labels))

    # Cluster colors
    cluster_colors = {i: plt.cm.viridis(i / n_clusters) for i in range(n_clusters)} # select n_clusters colors
    bar_colors = [cluster_colors[c] for c in sorted_clusters]

    # Save silhouette plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_scores)), sorted_scores, color=bar_colors)
    plt.axhline(y=silhouette_avg, color='red', linestyle='--', label='Average Silhouette Score')
    plt.title(f'Silhouette Scores for {n_clusters} Clusters')
    plt.xlabel('Sample Index')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.savefig(f'silhouette_plot_{n_clusters}.png')
    plt.close()

compute_and_plot_silhouette(df_5, kmeans_5.labels_)
compute_and_plot_silhouette(df_7, kmeans_7.labels_)