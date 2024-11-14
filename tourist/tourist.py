import dimod
import networkx as nx

# Граф маршрутов
G = nx.Graph()
G.add_edge(0, 1, weight=5)
G.add_edge(1, 2, weight=3)
G.add_edge(2, 3, weight=7)
G.add_edge(3, 0, weight=4)
# Добавьте остальные ребра и их веса

# Построение QUBO
qubo = {}

# Целевая функция: минимизация стоимости маршрутов
for i, j in G.edges:
    qubo[(f"x_{i}_{j}", f"x_{i}_{j}")] = G[i][j]["weight"]

# Ограничения на посещение каждой точки ровно один раз
lambda1 = 10  # штраф за нарушение
nodes = list(G.nodes)

for node in nodes:
    involved_edges = [
        (f"x_{i}_{node}", f"x_{j}_{node}") for i, j in G.edges if node in (i, j)
    ]
    for u, v in involved_edges:
        qubo[(u, v)] = qubo.get((u, v), 0) - lambda1

# Запуск классического отжига
sampler = dimod.SimulatedAnnealingSampler()
response = sampler.sample_qubo(qubo, num_reads=100)

# Обработка результатов
best_solution = response.first.sample
selected_edges = [edge for edge in best_solution if best_solution[edge] == 1]

# Исправленный блок: обработка и вывод результатов
print("Выбранные маршруты:")
total_cost = 0

for edge in selected_edges:
    # Извлекаем номера узлов из имен переменных QUBO
    edge_nodes = edge.split("_")[1:]  # Получаем узлы, например ['0', '1']
    u, v = int(edge_nodes[0]), int(edge_nodes[1])  # Преобразуем в целые числа

    if G.has_edge(u, v):
        total_cost += G[u][v]["weight"]
        print(f"Маршрут: {u} -> {v}")

print(f"Итоговая стоимость: {total_cost}")
