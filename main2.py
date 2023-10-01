import numpy as np
from sklearn.datasets import *


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

# Пример использования
if __name__ == "__main__":
    # Обучающие данные для операции "И" (AND)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 0, 1])

    # Создание и обучение перцептрона
    perceptron = PerceptronSigmoid(num_inputs=2, learning_rate=0.1)
    perceptron.train(inputs, labels, num_epochs=1000)

    # Предсказание результатов
    test_inputs = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    for test_input in test_inputs:
        prediction = perceptron.predict(test_input)
        print(f"Input: {test_input}, Prediction: {prediction:.4f}")



