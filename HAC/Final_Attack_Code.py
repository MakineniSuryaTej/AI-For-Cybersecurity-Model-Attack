import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


model_filename = "agglomerative_clustering_model.pkl"
loaded_model = joblib.load(model_filename)


file_path = "UCS-Satellite-Database-1-1-2023.xlsx"
data = pd.read_excel(file_path)
df = pd.DataFrame()
df1 = pd.DataFrame()
data['Inclination (degrees)'] = pd.to_numeric(data['Inclination (degrees)'], errors='coerce')
features = data[['Longitude of GEO (degrees)', 'Perigee (km)', 'Apogee (km)', 'Eccentricity', 'Inclination (degrees)']]
features = features.dropna()
features = features.reset_index()
features = features.drop('index',axis=1)
print(features.shape)
df = features.copy()
df1 = features.copy()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)



cluster_labels_original_data = loaded_model.fit_predict(scaled_features)
df['Original_Cluster'] = cluster_labels_original_data

print(df.head())


new_samples = pd.DataFrame({
    'Longitude of GEO (degrees)': [200,150,250],
    'Perigee (km)': [34566,12334,12334],
    'Apogee (km)': [12345,23456,45678],
    'Eccentricity': [0.002,0.0003,0.1],
    'Inclination (degrees)': [100,150,180]
})


extended_features = pd.concat([features, new_samples])
extended_features.reset_index(inplace=True)
scaled_extended_features = scaler.fit_transform(extended_features)
cluster_labels_Modified = loaded_model.fit_predict(scaled_extended_features)


extended_features['cluster_labels_Modified'] = cluster_labels_Modified
Original_cluster_counts = df['Original_Cluster'].value_counts()
Modified_cluster_counts = extended_features['cluster_labels_Modified'].value_counts()


fig, axes = plt.subplots(1, 2, figsize=(20, 8))
scatter1 = axes[0].scatter(df['Longitude of GEO (degrees)'], df['Inclination (degrees)'], c=cluster_labels_original_data, cmap='rainbow', label='Clusters')
axes[0].set_title('HAC Original Data')
axes[0].set_xlabel('Longitude of GEO (degrees)')
axes[0].set_ylabel('Inclination (degrees)')
legend_labels1 = [f'Cluster {i+1} ({Original_cluster_counts[i]} data points)' for i in range(len(Original_cluster_counts))]
legend1 = axes[0].legend(handles=scatter1.legend_elements()[0], title='Clusters', labels=legend_labels1)
axes[0].grid(True)


scatter2 = axes[1].scatter(extended_features['Longitude of GEO (degrees)'], extended_features['Inclination (degrees)'], c=cluster_labels_Modified, cmap='rainbow', label='Clusters')
axes[1].set_title('HAC Modified Data')
axes[1].set_xlabel('Longitude of GEO (degrees)')
axes[1].set_ylabel('Inclination (degrees)')
legend_labels2 = [f'Cluster {i+1} ({Modified_cluster_counts[i]} data points)' for i in range(len(Modified_cluster_counts))]
legend2 = axes[1].legend(handles=scatter2.legend_elements()[0], title='Clusters', labels=legend_labels2)
axes[1].grid(True)

plt.tight_layout()


plt.show()
c = 0 
for i in range(len(df)):
    if(df['Original_Cluster'][i] != extended_features['cluster_labels_Modified'][i]):
        c += 1
print("Percentage of misclassfication: ",int(c*100/len(df)),"%")
