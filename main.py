#Creado por Alexandro Gutierrez Serna

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Cargar solo los primeros 3000 registros desde el archivo CSV
data = pd.read_csv('diabetes_prediction_dataset.csv').head(3000)

# Separar características numéricas y categóricas
categorical_features = data[['gender', 'smoking_history']]
numeric_features = data.drop(columns=['gender', 'smoking_history', 'diabetes'])

# Codificar las características categóricas usando one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
categorical_features_encoded = encoder.fit_transform(categorical_features)

# Combinar las características codificadas con las características numéricas
X = np.concatenate([numeric_features.values, categorical_features_encoded], axis=1)

# Escalar las características para normalizarlas
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, data['diabetes'], test_size=0.2, random_state=42, shuffle=False)


# Restablecer los índices para y_train
y_train = y_train.reset_index(drop=True)

# ----------------------- k-NN -----------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def k_nearest_neighbors(X_train, y_train, X_test, k=3):
    predictions = []
    for x in X_test:
        distances = [euclidean_distance(x, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train.iloc[i] for i in k_indices]  # Utiliza iloc para acceder a elementos por ubicación entera
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)
    return predictions

k = 3
# Realizar predicciones para todo el conjunto de datos (entrenamiento + prueba)
y_pred = k_nearest_neighbors(np.concatenate((X_train, X_test)), pd.concat([y_train, y_test]), np.concatenate((X_train, X_test)), k)

# Calcular la precisión para el conjunto completo
accuracy_full = accuracy_score(np.concatenate((y_train, y_test)), y_pred)
print(f'Precisión para k-NN en el conjunto completo con k={k}: {accuracy_full}')


# Graficar
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores reales', marker='o', s=30)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones', marker='x', s=30)
plt.title('Valores reales vs Predicciones')
plt.xlabel('Índice de la muestra')
plt.ylabel('Valor')
plt.legend()
plt.show()


## ----------------------- k-means -----------------------
def k_means(X, k, max_iters=100):
    # Inicializar los centroides aleatoriamente
    centroids = X[np.random.choice(range(X.shape[0]), k, replace=False)]
    
    for _ in range(max_iters):
        # Asignar puntos al centroide más cercano
        labels = []
        for x in X:
            distances = [euclidean_distance(x, centroid) for centroid in centroids]
            label = np.argmin(distances)
            labels.append(label)
        
        # Actualizar los centroides
        new_centroids = []
        for i in range(k):
            cluster_points = X[np.array(labels) == i]
            if len(cluster_points) > 0:
                new_centroid = cluster_points.mean(axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[i])
        new_centroids = np.array(new_centroids)
        
        # Verificar si los centroides convergen
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Uso de k-Means
k = 2

labels, centroids = k_means(np.concatenate((X_train, X_test)), k)

# Imprimir etiquetas y centroides
#print(f'k-Means labels for k={k}:\n{labels}')
#print(f'k-Means centroids for k={k}:\n{centroids}')

accuracy_full = accuracy_score(np.concatenate((y_train, y_test)), labels)
print(f'Precisión para k-means en el conjunto completo con k={k}: {accuracy_full}')

accuracy_full = accuracy_score(y_pred, labels)
print(f'Precisión kmeans vs knn: {accuracy_full}')

# Graficar
plt.figure(figsize=(10, 6))
plt.scatter(range(len(labels)), labels, color='blue', label='Predicciones k-means', marker='o', s=30)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones knn', marker='x', s=30)
plt.title('Predicciones k-means vs Predicciones knn')
plt.xlabel('Índice de la muestra')
plt.ylabel('Valor')
plt.legend()
plt.show()
