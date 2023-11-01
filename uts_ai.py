import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Membaca data dari file CSV
df = pd.read_csv('data_properti.csv')

# Menyiapkan data untuk clustering
df_clustering = df[['usia', 'harga_properti', 'luas_tanah']]

# Melakukan standarisasi data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clustering)

# Menentukan jumlah cluster yang diinginkan
kmeans = KMeans(n_clusters=3, random_state=42)

# Melakukan clustering
clusters = kmeans.fit_predict(scaled_data)

# Menambahkan kolom cluster ke dataframe
df['cluster'] = clusters
df['usia_cluster'] = clusters

print("Range usia yang paling memungkinkan membeli properti:")
for i in range(3):
    cluster_mean = df[df['usia_cluster'] == i]['usia'].mean()
    print(f"Cluster {i+1}: {int(cluster_mean-5)} - {int(cluster_mean+5)} tahun")

# Visualisasi hasil clustering
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['usia'], df['harga_properti'], df['luas_tanah'], c=df['cluster'], cmap='viridis', edgecolor='k', s=50)
ax.set_xlabel('Usia')
ax.set_ylabel('Harga Properti')
ax.set_zlabel('Luas Tanah')
ax.set_title('Hasil Clustering Data Properti')
plt.show()
# Menghitung jumlah setiap jenis properti berdasarkan usia
property_counts = df.groupby(['usia', 'jenis_properti']).size().unstack()

# Visualisasi menggunakan grafik batang
property_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xlabel('Usia')
plt.ylabel('Jumlah Properti')
plt.title('Korelasi antara Usia dan Jenis Properti yang Dimiliki')
plt.show()
