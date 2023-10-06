import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class PerceptronSigmoid:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = np.random.rand(num_inputs)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data):
        weighted_sum = np.dot(input_data, self.weights)
        return self.sigmoid(weighted_sum)

    def train(self, inputs, labels, num_epochs):
        for epoch in range(num_epochs):
            for i in range(len(inputs)):
                input_data = inputs[i]
                label = labels[i]

                weighted_sum = np.dot(input_data, self.weights)
                predicted = self.sigmoid(weighted_sum)
                
                error = label - predicted
                
                # Обновление весов с использованием градиентного спуска
                gradient = input_data * (predicted * (1 - predicted)) * error
                self.weights += self.learning_rate * gradient

    def loss(self, inputs, labels):
        total_loss = 0
        for i in range(len(inputs)):
            input_data = inputs[i]
            label = labels[i]

            weighted_sum = np.dot(input_data, self.weights)
            predicted = self.sigmoid(weighted_sum)

            error = label - predicted
            total_loss += 0.5 * (error ** 2)

        return total_loss / len(inputs)

if __name__ == "__main__":
    # Загрузка датасета Iris
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # Преобразование задачи в бинарную классификацию

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение перцептрона
    perceptron = PerceptronSigmoid(num_inputs=4, learning_rate=0.1)

    num_epochs = 1000
    for epoch in range(num_epochs):
        perceptron.train(X_train, y_train, num_epochs=1)  # Обучение на одной эпохе
        training_loss = perceptron.loss(X_train, y_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {training_loss:.4f}")

    # Оценка производительности модели на тестовых данных
    y_pred = []
    for test_input in X_test:
        prediction = perceptron.predict(test_input)
        y_pred.append(round(prediction))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")


