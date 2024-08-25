import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import friedmanchisquare
from ucimlrepo import fetch_ucirepo

# Carregar a base de dados Iris
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets

# Definir os parâmetros para o experimento
k_values = [1, 3, 5]
distances = ['euclidean', 'manhattan', 'minkowski']
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Guardar os resultados dos experimentos
results = {}

for distance in distances:
    results[distance] = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric=distance)
        scores = cross_val_score(model, X, y, cv=kf)
        results[distance].append(np.mean(scores))

# Comparar os resultados usando teste de Friedman
_, p_value = friedmanchisquare(*[results[distance] for distance in distances])

print("Resultados dos Experimentos:")
for distance, scores in results.items():
    print(f"Distância: {distance}, Scores: {scores}")

print("\nP-valor do teste de Friedman:", p_value)


