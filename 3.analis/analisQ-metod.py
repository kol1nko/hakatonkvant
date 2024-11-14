import pandas as pd
import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# Загрузка данных из файла CSV
def load_data(filename):
    df = pd.read_csv(filename)

    # Убираем пробелы в именах столбцов
    df.columns = df.columns.str.strip()

    # Проверка на пропущенные значения и выводим их
    print("Columns in dataset:", df.columns)
    print("Missing values in dataset:", df.isnull().sum())

    # Удаляем строки с пропущенными метками
    df = df.dropna(subset=["label"])

    # Проверяем, что в метках больше нет NaN
    print("Unique labels in the dataset after dropping NaN:", np.unique(df["label"]))

    # Преобразуем текстовые данные в векторы
    word_dict = {}
    for review in df["review"]:  # Используем столбец 'review'
        for word in review.split():
            word_dict[word] = word_dict.get(word, 0) + 1

    X = []
    for review in df["review"]:  # Используем столбец 'review'
        vector = np.zeros(len(word_dict))
        for i, word in enumerate(word_dict):
            if word in review:
                vector[i] = 1  # Пример бинарной векторизации
        X.append(vector)

    X = np.array(X)

    # Преобразуем метки в 0 и 1 (положительные отзывы — 1, отрицательные — 0)
    y = df["label"].map({"+": 1, "-": 0}).values

    # Проверка целевых меток
    print("Unique labels in the dataset:", np.unique(y))

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Определение обёртки CustomCircuitQNN
class CustomCircuitQNN(nn.Module):
    def __init__(self, num_qubits, layers):
        super(CustomCircuitQNN, self).__init__()
        self.num_qubits = num_qubits
        self.layers = layers
        self.backend = AerSimulator()  # Используем AerSimulator для симуляции
        self.params = nn.Parameter(
            torch.rand(self.num_qubits * self.layers)
        )  # Параметры квантовой схемы

    def build_circuit(self, params):
        # Создание квантовой схемы
        circuit = QuantumCircuit(self.num_qubits)
        param_idx = 0
        for _ in range(self.layers):
            for qubit in range(self.num_qubits):
                circuit.ry(
                    params[param_idx].item(), qubit
                )  # Преобразуем torch.Tensor в float
                param_idx += 1
            # Запутанность с использованием CNOT
            for qubit in range(self.num_qubits - 1):
                circuit.cx(qubit, qubit + 1)

        # Добавление измерений в схему
        circuit.measure_all()  # Все кубиты измеряются

        return circuit

    def forward(self, x):
        # Компиляция и выполнение квантовой схемы
        circuit = self.build_circuit(self.params)
        compiled_circuit = transpile(circuit, self.backend)

        # Используем run для симуляции
        job = self.backend.run(compiled_circuit)  # Запуск на симуляторе
        result = job.result()  # Получение результата

        # Получение counts для статистического результата
        counts = result.get_counts(
            circuit
        )  # Получаем counts для статистического результата

        # Преобразуем количество измерений в torch.Tensor для использования в классификации
        probabilities = []
        for i in range(x.shape[0]):  # Для каждого примера в батче
            prob = counts.get("1", 0) / sum(counts.values())  # Вероятность класса 1
            probabilities.append(prob)

        return torch.tensor(probabilities).view(
            -1, 1
        )  # Возвращаем вектор вероятностей для каждого примера


# Модель на основе CustomCircuitQNN для гибридного подхода
class HybridVQC(nn.Module):
    def __init__(self, num_qubits, layers):
        super(HybridVQC, self).__init__()
        self.qnn = CustomCircuitQNN(num_qubits, layers)
        self.fc = nn.Linear(1, 1)  # Полносвязный слой для бинарной классификации

    def forward(self, x):
        x = self.qnn(x)
        x = self.fc(x)
        return torch.sigmoid(x)  # Сигмоид для получения вероятности бинарного класса
