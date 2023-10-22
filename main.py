# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load data
df = pd.read_csv('data/all_mtg_cards.csv', header = 0, dtype=np.dtype('unicode'))

# Convert the power and toughness columns to numeric
df['power'] = pd.to_numeric(df['power'], errors='coerce')
df['toughness'] = pd.to_numeric(df['toughness'], errors='coerce')

# Drop rows with missing values in power and toughness columns
df = df.dropna(subset=['power', 'toughness'])

# Select the features for K-means clustering
features = df[['power', 'toughness', 'cmc']]

# Apply the elbow method to choose an appropriate value for k
distortions = []
cluster_range = range(1, 11)
for k in cluster_range:
    # Fit the K-means clustering model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    # Compute the inertia value
    kmeans.fit(df[['power', 'toughness', 'cmc']])
    # Append the inertia value to the list of distortions
    distortions.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 4))
# Plot the distortions against values of k
plt.plot(cluster_range, distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for K-means Clustering')
plt.show()

# Fit and plot the K-means clustering results with k=3 in another separate figure
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df[['power', 'toughness', 'cmc']])

plt.figure(figsize=(8, 6))
plt.scatter(df['power'], df['toughness'], c=df['cluster'], cmap='viridis')
plt.xlabel('Power')
plt.ylabel('Toughness')
plt.title('K-means Clustering (k=3)')
plt.show()

plt.interactive(False)
plt.show(block=True)


# Select the features for EM clustering
X = df[['power', 'toughness', 'cmc']]

# Apply the Expectation-Maximization (EM) algorithm with k=3 clusters
n_clusters = 3
em = GaussianMixture(n_components=n_clusters, random_state=42)
cluster_em = em.fit_predict(X)  # Store the cluster labels

# Add the cluster labels to the dataframe
df['cluster_em'] = cluster_em

# Plot the clustering results
plt.figure(figsize=(8, 6))
plt.scatter(df['power'], df['toughness'], c=df['cluster_em'], cmap='viridis')
plt.xlabel('Power')
plt.ylabel('Toughness')
plt.title(f'EM Clustering (k={n_clusters})')
plt.show()
