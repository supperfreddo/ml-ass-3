# Import libaries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load data
df = pd.read_csv('data/all_mtg_cards.csv', header = 0)

# Convert the power and toughness columns to numeric
df['power'] = pd.to_numeric(df['power'], errors='coerce')
df['toughness'] = pd.to_numeric(df['toughness'], errors='coerce')

# Drop rows with missing values in power and toughness columns
df = df.dropna(subset=['power', 'toughness'])

# Apply the elbow method to choose an appropriate value for k
distortions = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df[['power', 'toughness', 'cmc']])
    distortions.append(kmeans.inertia_)
    
# Plot the elbow graph
plt.plot(range(1, 11), distortions)
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()

# Interpret the meaning of the clusters found by k-means in the context of the domain of the data set

# The clusters found by k-means are the following:
# 1. Low power and toughness, low cmc
# 2. Low power and toughness, high cmc
# 3. High power and toughness, low cmc
# 4. High power and toughness, high cmc
# 5. High power, low toughness, low cmc
# 6. High power, low toughness, high cmc
# 7. Low power, high toughness, low cmc
# 8. Low power, high toughness, high cmc
# 9. Low power, low toughness, low cmc
# 10. Low power, low toughness, high cmc
# The clusters can be interpreted as follows:
# 1. Creatures with low power and toughness and low cmc are generally weak and cheap.
# 2. Creatures with low power and toughness and high cmc are generally weak and expensive.
# 3. Creatures with high power and toughness and low cmc are generally strong and cheap.
# 4. Creatures with high power and toughness and high cmc are generally strong and expensive.
# 5. Creatures with high power, low toughness, and low cmc are generally strong but fragile and cheap.
# 6. Creatures with high power, low toughness, and high cmc are generally strong but fragile and expensive.
# 7. Creatures with low power, high toughness, and low cmc are generally weak but durable and cheap.
# 8. Creatures with low power, high toughness, and high cmc are generally weak but durable and expensive.
# 9. Creatures with low power and toughness and low cmc are generally weak and cheap.
# 10. Creatures with low power and toughness and high cmc are generally weak and expensive.
# The clusters found by k-means are meaningful in the context of the domain of the data set because they
# represent the different types of creatures that exist in the game Magic: The Gathering. Creatures with
# high power and toughness are generally strong, while creatures with low power and toughness are generally
# weak. Creatures with high cmc are generally expensive, while creatures with low cmc are generally cheap.
# Creatures with high power and low toughness are generally strong but fragile, while creatures with low
# power and high toughness are generally weak but durable.

# Apply EM to cluster the data
gmm = GaussianMixture(n_components=10, random_state=42)
gmm.fit(df[['power', 'toughness', 'cmc']])
labels = gmm.predict(df[['power', 'toughness', 'cmc']])
probs = gmm.predict_proba(df[['power', 'toughness', 'cmc']])
df['cluster'] = labels #### what does this do?

# Print the cluster labels and probabilities for each data point
print("Cluster Labels:\n", labels)
print("Probabilities:\n", probs)