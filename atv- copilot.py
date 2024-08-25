import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind
from scipy.stats import friedmanchisquare

# Carregar o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Definir os valores de K e as métricas de distância
k_values = [3, 5, 7]
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
# Configurar a validação cruzada 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Armazenar os resultados
results = []

for metric in distance_metrics:
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
        results.append((k, metric, scores.mean(), scores.std()))

# Exibir os resultados
for result in results:
    print(f"K: {result[0]}, Métrica: {result[1]}, Acurácia Média: {result[2]:.4f}, Desvio Padrão: {result[3]:.4f}")
    
# Comparar as acurácias médias usando o teste t de Student
for i in range(len(results)):
    for j in range(i+1, len(results)):
        t_stat, p_value = ttest_ind(results[i][2], results[j][2])
        print(f"Comparação entre K={results[i][0]}, Métrica={results[i][1]} e K={results[j][0]}, Métrica={results[j][1]}: p-valor={p_value:.4f}")
# Preparar os dados para o teste de Friedman
scores_matrix = np.array([result[2] for result in results])

# Executar o teste de Friedman
stat, p_value = friedmanchisquare(*scores_matrix)
print(f'Estatística de teste: {stat}, p-valor: {p_value}')
