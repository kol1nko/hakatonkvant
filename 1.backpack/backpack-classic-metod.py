# Импортируем необходимые библиотеки
import numpy as np
import pandas as pd
import scipy.optimize as sco

# Описание задачи и обоснование выбранного метода
# Задача: Оптимизация инвестиционного портфеля из 100 акций для максимизации доходности при заданном уровне риска.
# Вводные данные включают исторические данные доходностей акций за 100 дней, на основе которых вычисляются доходности и ковариационная матрица.
# Метод: Используется классический подход к оптимизации, основанный на модели Марковица.
# Описание метода: для решения задачи используется квадратичное программирование, которое хорошо подходит для задач оптимизации портфеля,
# где необходимо найти оптимальное распределение весов акций с учетом доходности и риска.

# Чтение данных из CSV-файла
file_path = "/kolin/Downloads/task-1-stocks (3).csv"
data = pd.read_csv(file_path)

# Вычисление доходностей и ковариационной матрицы
returns = data.pct_change().dropna()
cov_matrix = returns.cov()

# Параметры задачи
N = 100  # Количество акций
r = returns.mean().values  # Средняя доходность каждой акции
cov_matrix = cov_matrix.values  # Ковариационная матрица
risk_target = 0.2  # Уровень риска

# Начальная равная доля для всех акций в портфеле
init_guess = N * [1.0 / N]

# Ограничения на сумму долей (все доли должны суммироваться до 1)
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

# Границы для долей акций (от 0 до 1)
bounds = tuple((0, 1) for _ in range(N))


# Целевая функция для минимизации - отрицательная доходность
def portfolio_return(weights, mean_returns):
    return -np.sum(mean_returns * weights)


# Функция для расчета риска портфеля
def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


# Оптимизация: минимизируем отрицательную доходность при заданном уровне риска
def objective_function(weights):
    return -np.sum(r * weights)


# Ограничение на риск портфеля
def risk_constraint(weights):
    return risk_target - portfolio_risk(weights, cov_matrix)


# Оптимизация портфеля при заданном уровне риска
constraints = (
    {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    {"type": "ineq", "fun": risk_constraint},
)

# Оптимизация
optimal_portfolio = sco.minimize(
    objective_function,
    init_guess,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

# Полученные доли акций
optimal_weights = optimal_portfolio.x

# Расчет доходности и риска оптимального портфеля
optimal_return = -objective_function(optimal_weights)
optimal_risk = portfolio_risk(optimal_weights, cov_matrix)

# Формируем портфель
portfolio = pd.DataFrame({"Акция": returns.columns, "Доля": optimal_weights})

print(portfolio)
print(f"Доходность портфеля: {optimal_return:.4f}")
print(f"Уровень риска портфеля: {optimal_risk:.4f}")
