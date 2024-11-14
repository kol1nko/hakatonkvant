import numpy as np
import pandas as pd
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.optimizers import COBYLA

# Загрузка данных о доходности и ковариации
data = pd.read_csv("/task-1-stocks.csv")
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Параметры задачи
num_assets = 10  # Упрощенная задача с 10 активами
target_risk = 0.2
lambda_risk = 0.5  # Коэффициент для контроля риска
k = 5  # Ограничение на количество активов в портфеле

# Создание квадратичной программы для оптимизации
qp = QuadraticProgram()

# Добавление бинарных переменных для каждого актива
for i in range(num_assets):
    qp.binary_var(name=f"x{i}")

# Целевая функция: максимизация доходности с учетом риска
return_vector = mean_returns[:num_assets].values
for i in range(num_assets):
    for j in range(num_assets):
        if i == j:
            qp.objective.set_linear(
                i, -return_vector[i] + lambda_risk * cov_matrix.iloc[i, i]
            )
        else:
            qp.objective.set_quadratic(i, j, lambda_risk * cov_matrix.iloc[i, j])

# Ограничение на количество активов
qp.linear_constraint(
    linear={f"x{i}": 1 for i in range(num_assets)},
    sense="==",
    rhs=k,
    name="asset_constraint",
)

# Настройка квантового симулятора и QAOA
backend = Aer.get_backend("qasm_simulator")  # Задаем симулятор напрямую
qaoa = QAOA(reps=3, optimizer=COBYLA())  # Параметры QAOA
optimizer = MinimumEigenOptimizer(qaoa)  # Используем QAOA для оптимизации

# Запуск оптимизации
result = optimizer.solve(qp)

# Обработка и вывод результатов
selected_assets = [i for i in range(num_assets) if result.x[i] > 0.5]
optimized_return = sum(return_vector[i] for i in selected_assets)
optimized_risk = np.sqrt(
    sum(cov_matrix.iloc[i, j] for i in selected_assets for j in selected_assets)
)

print("Выбранные активы:", selected_assets)
print("Ожидаемая доходность портфеля:", optimized_return)
print("Ожидаемый риск портфеля:", optimized_risk)
