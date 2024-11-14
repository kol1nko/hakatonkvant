import dimod
import networkx as nx
import pandas as pd
import time


def create_test_graph(num_nodes, num_edges):
    G = nx.gnm_random_graph(num_nodes, num_edges)
    for u, v in G.edges():
        G[u][v]["weight"] = abs(hash(f"{u}-{v}")) % 10 + 1  # случайные веса от 1 до 10
    return G


def run_benchmark(num_nodes, num_edges, num_reads=100, iterations=[50, 100, 200]):
    results = []
    G = create_test_graph(num_nodes, num_edges)

    for I in iterations:
        # Создание QUBO модели
        qubo = {}
        for i, j in G.edges:
            qubo[(f"x_{i}_{j}", f"x_{i}_{j}")] = G[i][j]["weight"]

        lambda1 = 10
        nodes = list(G.nodes)
        for node in nodes:
            involved_edges = [
                (f"x_{i}_{node}", f"x_{j}_{node}") for i, j in G.edges if node in (i, j)
            ]
            for u, v in involved_edges:
                qubo[(u, v)] = qubo.get((u, v), 0) - lambda1

        # Замер времени отжига
        start_time = time.time()
        sampler = dimod.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=I)
        end_time = time.time()

        # Анализ результатов
        best_solution = response.first.sample
        total_cost = 0

        for edge in best_solution:
            if best_solution[edge] == 1:
                # Разделение названия переменной, чтобы получить номера узлов
                _, u, v = edge.split("_")
                u, v = int(u), int(v)

                # Проверка, что такое ребро есть в графе, и добавление его веса
                if G.has_edge(u, v):
                    total_cost += G[u][v]["weight"]

        # Сохранение результатов
        results.append(
            {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "iterations": I,
                "num_reads": num_reads,
                "execution_time": end_time - start_time,
                "total_cost": total_cost,
            }
        )

    return pd.DataFrame(results)


# Запуск бенчмарков для различных графов
benchmark_results = []
for num_nodes, num_edges in [(10, 20), (20, 50), (50, 100), (100, 200)]:
    benchmark_results.append(run_benchmark(num_nodes, num_edges))

# Объединение всех результатов в одну таблицу
all_results = pd.concat(benchmark_results, ignore_index=True)

# Вывод таблицы с результатами
print(all_results)
