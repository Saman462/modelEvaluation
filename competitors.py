import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_samples, silhouette_score

# Step 1: Data Preparation
data = "./cars_clus.csv"
df = pd.read_csv(data)
# 2 Data Cleaning
df = df.dropna()  # Remove rows with missing values
df = df.replace('$null$', np.nan)  # Replace '$null$' with NaN
df = df.drop(['manufact', 'model'], axis=1)  # Drop the 'manufact' and 'model' columns
df['type'] = df['type'].replace({'sedan': 1, 'sports': 2, 'wagon': 3, 'coupe': 4, 'hatchback': 5})  # Convert categorical variables to numeric representation if required
#Filling sales and resale null values
#Sales have 2 nulls
df['sales'] = df['sales'].fillna(df['sales'].astype(float).mean())
#Resales have 36 nulls - use regression model to predict
lr=LinearRegression()
test = df[df['resale'].isnull()==True]
train = df[df['resale'].isnull()==False]
y=train["resale"]
train.drop("resale",axis=1,inplace=True)
train.fillna(0,inplace=True)
lr.fit(train,y)
test.drop("resale",axis=1,inplace=True)
test.fillna(0,inplace=True)
pred = lr.predict(test)
test['resale'] = pred

#Now delete empty rows
df = df.dropna()

# Feature Scaling or Normalization (mean of all columns is 0 and sd is 1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 2: Data Analysis
# Descriptive statistics
print(df.describe())

# Correlation analysis
correlation_matrix = df.corr()
# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='magma')
plt.title('Correlation Matrix')
plt.show()

# Step 3: Data Modeling
# Select the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow curve, select the optimal number of clusters
n_clusters = 3

# Step 4: Model Selection and Building
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
kmeans.fit(df_scaled)
labels = kmeans.predict(df_scaled) #Finding lables

# Step 5: Model Evaluation
# Calculate the silhouette scores
silhouette_scores = silhouette_samples(df_scaled, labels)
silhouette_avg = silhouette_score(df_scaled, kmeans.labels_)
print(f"Average Silhouette Score: {silhouette_avg}")

# Create a subplot with 1 row and 1 column
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(8, 6)

# Set the y-axis limit to accommodate the silhouette plots
ax1.set_xlim([-0.1, 1])

# Initialize the y-axis position
y_lower = 10

# Iterate over each cluster
for i in range(n_clusters):
    # Collect the silhouette scores for samples in the current cluster
    cluster_silhouette_scores = silhouette_scores[labels == i]
    # Sort the scores in ascending order
    cluster_silhouette_scores.sort()
    # Calculate the number of samples in the current cluster
    cluster_size = cluster_silhouette_scores.shape[0]
    # Calculate the upper limit for the silhouette plot
    y_upper = y_lower + cluster_size
    # Generate a color map for the current cluster
    color = plt.cm.get_cmap("Spectral")(float(i) / n_clusters)
    # Plot the silhouette scores for the samples in the current cluster
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_scores,
                      facecolor=color, edgecolor=color, alpha=0.7)
    # Label the silhouette plot with the cluster number
    ax1.text(-0.05, y_lower + 0.5 * cluster_size, str(i))

    # Update the y-axis position for the next cluster
    y_lower = y_upper + 10

# Set labels and title for the plot
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")
ax1.set_title("Silhouette plot for K-means clustering")

# Draw a vertical line at the average silhouette score
ax1.axvline(x=silhouette_avg, color="blue", linestyle="--")

# Display the average silhouette score as text
ax1.text(silhouette_avg + 0.01, y_lower - 30,
         f"Average Score: {silhouette_avg:.2f}", color="red")

# Set the y-axis ticks and remove the tick labels
ax1.set_yticks([])
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

# Step 6: Visualization and Conclusion
# Visualize the clusters
df['cluster'] = kmeans.labels_
# Plot the clusters
plt.scatter(df['mpg'], df['price'], c=df['cluster'])
plt.xlabel('MPG')
plt.ylabel('Price')
plt.title('Vehicle Clusters')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust plot layout
plt.show()
# Summarize the clusters and provide recommendations
for cluster in range(n_clusters):
    cluster_vehicles = df[df['cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(cluster_vehicles)
    print()
