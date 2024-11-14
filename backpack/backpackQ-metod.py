# Импортируем необходимые библиотеки
import numpy as np
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

# Описание задачи и обоснование выбранного метода
# Задача: Оптимизация инвестиционного портфеля из 100 акций для максимизации доходности при заданном уровне риска.
# Вводные данные включают исторические данные доходностей акций за 100 дней, на основе которых вычисляются доходности и ковариационная матрица.
# Метод: Используется квантовая оптимизация (Quantum Annealing) с преобразованием задачи в формат QUBO (Quadratic Unconstrained Binary Optimization).
# Обоснование метода: Квантовый отжиг позволяет эффективно решать задачи оптимизации большой размерности, особенно когда задача имеет
# квадратичную структуру, как в случае минимизации риска портфеля. Метод квантового отжига позволяет параллельно исследовать множество
# возможных комбинаций акций, что делает его подходящим для поиска оптимального решения в данной финансовой задаче.

# Чтение данных из CSV-файла
file_path = "/mnt/data/task-1-stocks (3).csv"
data = pd.read_csv(file_path)

# Вычисление доходностей и ковариационной матрицы
returns = data.pct_change().dropna()
cov_matrix = returns.cov()

# Параметры задачи
N = 100  # Количество акций
r = returns.mean().values  # Средняя доходность каждой акции
cov_matrix = cov_matrix.values  # Ковариационная матрица
risk_target = 0.2  # Уровень риска

# Гиперпараметры для построения QUBO
lambda_risk = 1.0  # Коэффициент для штрафа за риск
lambda_sum = 10.0  # Коэффициент для штрафа за сумму весов

# Построение QUBO
Q = {}
for i in range(N):
    for j in range(N):
        # Компоненты QUBO для минимизации риска
        if i == j:
            Q[(i, j)] = lambda_risk * cov_matrix[i, j] - r[i]
        else:
            Q[(i, j)] = lambda_risk * cov_matrix[i, j]

# Добавление штрафа за сумму весов (чтобы сумма весов была близка к 1)
for i in range(N):
    Q[(i, i)] += lambda_sum * (1 - 2 * risk_target)
    for j in range(i + 1, N):
        Q[(i, j)] += 2 * lambda_sum

# Создаем модель бинарной квадратичной оптимизации
bqm = BinaryQuadraticModel.from_qubo(Q)

# Настройка квантового самплера и отправка задачи на квантовый отжиг
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample(bqm, num_reads=100)

# Получение результатов
best_sample = response.first.sample
best_weights = np.array([best_sample[i] for i in range(N)])

# Нормализация весов, чтобы их сумма была равна 1
if np.sum(best_weights) > 0:
    best_weights = best_weights / np.sum(best_weights)

# Вывод оптимального портфеля
portfolio = pd.DataFrame({"Акция": returns.columns, "Доля": best_weights})

# Расчет доходности и риска портфеля
portfolio_return = np.dot(best_weights, r)
portfolio_risk = np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))

print(portfolio)
print(f"Доходность портфеля: {portfolio_return:.4f}")
print(f"Уровень риска портфеля: {portfolio_risk:.4f}")
