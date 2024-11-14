import numpy as np
import networkx as nx
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
import pandas as pd

# Загрузка и обработка матрицы смежности
adjacency_matrix = pd.read_csv(
    "/path/to/your/file/task-2-adjacency_matrix.csv", index_col=0
)
adjacency_matrix.replace("-", np.inf, inplace=True)  # Заменяем "-" на бесконечность
adjacency_matrix = adjacency_matrix.apply(pd.to_numeric, errors="coerce").fillna(np.inf)

# Создание графа
G = nx.Graph()
for i, row in adjacency_matrix.iterrows():
    for j, cost in row.items():
        if cost != np.inf:
            G.add_edge(i, j, weight=cost)

# Настройка задачи оптимизации маршрутов как квадратичной программы
quadratic_program = QuadraticProgram()

# Добавление переменных для каждой возможной дороги
for u, v in G.edges:
    quadratic_program.binary_var(name=f"x_{u}_{v}")

# Целевая функция: минимизация общей стоимости перемещений
quadratic_program.minimize(linear=[G[u][v]["weight"] for u, v in G.edges])

# Ограничения:
# 1. Каждый узел посещается ровно один раз (кроме вокзала).
# 2. Автобусы начинают и заканчивают маршруты на вокзале.
# 3. Ограничения по вместимости и времени маршрутов (добавляются в зависимости от группы и времени).
# Пример: добавим ограничения для узлов (по аналогии добавляются и другие)
for node in G.nodes:
    neighbors = list(G.neighbors(node))
    constraint_expr = {
        f"x_{min(node, neighbor)}_{max(node, neighbor)}": 1 for neighbor in neighbors
    }
    quadratic_program.linear_constraint(
        linear=constraint_expr, sense="==", rhs=1, name=f"visit_{node}"
    )

# Настройка QAOA и симулятора
backend = Aer.get_backend("qasm_simulator")
quantum_instance = QuantumInstance(backend, shots=1024)
qaoa = QAOA(optimizer=COBYLA(), quantum_instance=quantum_instance)
optimizer = MinimumEigenOptimizer(qaoa)

# Запуск оптимизации
result = optimizer.solve(quadratic_program)

# Интерпретация результата
routes = []
total_cost = 0
for u, v in G.edges:
    if result.x[f"x_{u}_{v}"] > 0.5:  # Если дорога выбрана
        routes.append((u, v))
        total_cost += G[u][v]["weight"]

# Проверка ограничений и вывод результата
print("Сформированные маршруты:", routes)
print("Общая стоимость перемещений:", total_cost)
