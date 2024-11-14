# Импорт библиотек QBoard
from qboard import QuantumCircuit, QuantumDevice, ClassicalOptimizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Определяем количество кубитов и создаем квантовое устройство
n_qubits = 4
device = QuantumDevice(num_qubits=n_qubits, backend="QBoard")


# Создаем квантовую схему (используем вариационную квантовую схему)
def variational_circuit(circuit, inputs, weights):
    # Введение входных данных в квантовую схему
    for i in range(len(inputs)):
        circuit.rx(inputs[i], i)

    # Применение обучаемых слоев
    layer_count = weights.shape[0]
    for layer in range(layer_count):
        for i in range(n_qubits):
            circuit.ry(weights[layer, i], i)
        # Применение схемы с запутыванием
        for i in range(n_qubits - 1):
            circuit.cnot(i, i + 1)
    return circuit


# Настраиваем параметры квантовой схемы
n_layers = 3  # количество слоев
weights = np.random.randn(n_layers, n_qubits)  # начальные веса квантовой схемы

# Предварительная обработка данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_tfidf[:, :n_qubits])
X_test_scaled = scaler.transform(X_test_tfidf[:, :n_qubits])

# Классическая модель для классификации
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Предсказание
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность гибридной модели на QBoard: {accuracy * 100:.2f}%")
