import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

plt.figure(figsize=(7,5))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customers")
plt.show()

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1,11):
    model = KMeans(n_clusters=i, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))

for i in range(5):
    cluster = df[df["Cluster"] == i]
    plt.scatter(
        cluster["Annual Income (k$)"],
        cluster["Spending Score (1-100)"],
        label=f"Cluster {i}"
    )

centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:,0],
    centers[:,1],
    s=250,
    c="black",
    marker="X",
    label="Centers"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.legend()
plt.show()

print("\nAverage Values Per Cluster\n")

print(
    df.groupby("Cluster")[["Annual Income (k$)","Spending Score (1-100)"]].mean()
)

dbscan = DBSCAN(eps=0.5, min_samples=5)

df["DBSCAN"] = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(8,6))

for label in sorted(df["DBSCAN"].unique()):

    cluster = df[df["DBSCAN"] == label]

    plt.scatter(
        cluster["Annual Income (k$)"],
        cluster["Spending Score (1-100)"],
        label=f"Cluster {label}"
    )

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("DBSCAN")
plt.legend()
plt.show()

print("\nCustomers in each K-Means Cluster\n")
print(df["Cluster"].value_counts().sort_index())

print("\nCustomers in each DBSCAN Cluster\n")
print(df["DBSCAN"].value_counts().sort_index())