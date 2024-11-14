import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Загрузка данных
file_path = "task-3-dataset.csv"  # Убедитесь, что указанный путь корректен
data = pd.read_csv(file_path)

# Определение признаков и меток
X = data["отзывы"]
y = data["разметка"].map({"+": 1, "-": 0})  # Преобразуем метки '+' в 1 и '-' в 0

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Векторизация текста
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Оценка модели и вывод отчета по метрикам
y_pred = model.predict(X_test_vec)
print("Точность:", accuracy_score(y_test, y_pred))
print("Отчет по метрикам классификации:")
print(classification_report(y_test, y_pred, target_names=["Негативные", "Позитивные"]))

# Построение кривых обучения
train_accuracies = []
test_accuracies = []

# Постепенное увеличение обучающей выборки для построения кривой
for i in range(1, len(X_train_vec.toarray())):
    model.fit(X_train_vec[:i], y_train[:i])
    train_accuracies.append(model.score(X_train_vec[:i], y_train[:i]))
    test_accuracies.append(model.score(X_test_vec, y_test))

# Построение графиков
plt.plot(train_accuracies, label="Точность на обучающей выборке")
plt.plot(test_accuracies, label="Точность на тестовой выборке")
plt.xlabel("Количество обучающих примеров")
plt.ylabel("Точность")
plt.legend()
plt.title("Кривые обучения")
plt.show()


# Функция для анализа новых отзывов
def analyze_reviews(new_reviews):
    new_reviews_vec = vectorizer.transform(new_reviews)
    predictions = model.predict(new_reviews_vec)
    return ["+" if pred == 1 else "-" for pred in predictions]


# Пример использования функции
new_reviews = [
    "Отличный телефон, все работает идеально!",
    "Батарея садится очень быстро",
]
print("Результаты анализа:", analyze_reviews(new_reviews))
