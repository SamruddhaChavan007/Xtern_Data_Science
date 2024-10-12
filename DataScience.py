import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

file_path = "2025-VeloCityX-Expanded-Fan-Engagement-Data.csv"
data = pd.read_csv(file_path)

print("Dataset Preview:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDataset Summary:")
print(data.describe())

correlation_matrix = data.drop(columns=['User ID']).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Heatmap of User Engagement Metrics')
plt.show()

features = data[
    [
        "Fan Challenges Completed",
        "Predictive Accuracy (%)",
        "Virtual Merchandise Purchases",
        "Sponsorship Interactions (Ad Clicks)",
        "Time on Live 360 (mins)",
        "Real-Time Chat Activity (Messages Sent)",
    ]
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(8, 5))
data['Cluster'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Users Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Users')
plt.show()

cluster_profile = data.groupby('Cluster').mean()

print("\nCluster Profiles:")
print(cluster_profile)

cluster_profile.plot(kind='bar', figsize=(12, 6))
plt.title('Cluster Profiles - Average Metrics per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Value')
plt.legend(loc='upper right')
plt.show()

print("\nProposed Fan Challenge:")
print("""
'Prediction Mastery Bonus' Challenge:
- Reward users with virtual merchandise discounts or exclusive items for achieving 
  a predictive accuracy above 85% across multiple races.
- Expected Outcome: Increase engagement in predictive challenges and drive merchandise purchases.
""")
