from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load customer dataset
data_path = "c:/Desktop/Sem6_LAB/Q12/supermarket_sales_Sheet1.csv"
df = pd.read_csv(data_path)

# Preprocess data
df.dropna(inplace=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(scaled_data)

# Apply K-Medoids Clustering
kmedoids = KMedoids(n_clusters=3, random_state=42)
df['KMedoids_Cluster'] = kmedoids.fit_predict(scaled_data)

# Apply Fuzzy C-Means Clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(scaled_data.T, 3, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)
df['FCM_Cluster'] = cluster_membership

# Analyze clusters
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(df.groupby('KMeans_Cluster')[numeric_columns].mean())
print(df.groupby('KMedoids_Cluster')[numeric_columns].mean())
print(df.groupby('FCM_Cluster')[numeric_columns].mean())

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
sns.scatterplot(x='PCA1', y='PCA2', hue='KMeans_Cluster', data=df, palette='viridis')
plt.title('K-Means Clustering')

plt.subplot(1, 3, 2)
sns.scatterplot(x='PCA1', y='PCA2', hue='KMedoids_Cluster', data=df, palette='viridis')
plt.title('K-Medoids Clustering')

plt.subplot(1, 3, 3)
sns.scatterplot(x='PCA1', y='PCA2', hue='FCM_Cluster', data=df, palette='viridis')
plt.title('Fuzzy C-Means Clustering')

plt.show()
