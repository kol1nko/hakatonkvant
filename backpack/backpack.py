import numpy as np
import pandas as pd
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.optimizers import COBYLA

# Загрузка и подготовка данных
data = pd.read_csv("/path/to/task-1-stocks.csv")
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

portfolio_value = 1_000_000
risk_tolerance = 0.2
num_assets = len(mean_returns)

# Конфигурация задачи оптимизации
quadratic_program = QuadraticProgram()

# Добавление переменных и их ограничений
for i in range(num_assets):
    quadratic_program.binary_var(name=f"x{i}")

# Матрица ковариации (риск) и ожидаемая доходность для целевой функции
mu = mean_returns.values
sigma = cov_matrix.values

# Целевая функция: максимизация доходности - коэффициент риска
# В Qiskit мы минимизируем - целевая функция будет `-доходность + риск`
risk_penalty = 100 * risk_tolerance
quadratic_program.maximize(
    linear=[mu[i] for i in range(num_assets)],
    quadratic={
        (i, j): -risk_penalty * sigma[i, j]
        for i in range(num_assets)
        for j in range(num_assets)
    },
)

# Ограничение на вес суммарных инвестиций
quadratic_program.linear_constraint(
    linear={f"x{i}": 1 for i in range(num_assets)}, sense="==", rhs=1
)

# Настройка квантового симулятора и алгоритма QAOA
backend = Aer.get_backend("qasm_simulator")
quantum_instance = QuantumInstance(backend, shots=1024)
qaoa = QAOA(optimizer=COBYLA(), quantum_instance=quantum_instance)
optimizer = MinimumEigenOptimizer(qaoa)

# Запуск оптимизации
result = optimizer.solve(quadratic_program)

# Интерпретация и отображение результатов
optimized_weights = np.array([result.x[i] for i in range(num_assets)])
portfolio_return = np.dot(optimized_weights, mean_returns)
portfolio_risk = np.sqrt(
    np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights))
)

# Показать результат
print("Оптимизированный портфель:")
for i, weight in enumerate(optimized_weights):
    if weight > 0:
        print(f"Акция {data.columns[i]}: Вес {weight:.2f}")

print(f"Доходность портфеля: {portfolio_return:.4f}")
print(f"Риск портфеля: {portfolio_risk:.4f}")
