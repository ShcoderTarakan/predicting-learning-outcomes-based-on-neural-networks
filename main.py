import pandas as pd
import numpy as np

# Загрузка данных из CSV файла
data = pd.read_csv('student_data.csv')

# Разделение данных на входные (X) и выходные (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Нормализация входных данных
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Определение функций активации и их производных
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Определение класса NeuralNetwork
class NeuralNetwork:
    def __init__(self, x, y, activation_function):
        # Инициализация входных данных, весов и функций активации
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 2)  # Веса между входными и скрытыми слоями
        self.weights2 = np.random.rand(2, 1)  # Веса между скрытыми и выходными слоями
        self.y = y
        self.output = np.zeros(y.shape)
        # Выбор функции активации и ее производной
        if activation_function == 'sigmoid':
            self.activation_function = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_function == 'relu':
            self.activation_function = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError('Invalid activation function')

    # Процесс прямого распространения сигнала через сеть
    def feedforward(self, x):
        self.layer1 = self.activation_function(np.dot(x, self.weights1))
        self.output = np.dot(self.layer1, self.weights2)
        return self.output

    # Процесс обратного распространения ошибки для обучения сети
    def backprop(self, x, y):
        self.layer1 = self.activation_function(np.dot(x, self.weights1))
        layer2_error = 2 * (y - self.output) * self.activation_derivative(self.output)
        layer2_error = np.reshape(layer2_error, (2, 1))
        d_weights2 = np.dot(self.layer1.T, layer2_error)
        x = np.reshape(x, (1, x.shape[0]))
        d_weights1 = np.dot(x.T, (np.dot(layer2_error, self.weights2.T) * self.activation_derivative(self.layer1)))
        d_weights2 = np.reshape(d_weights2, (2, 1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    # Метод для обучения сети
    def train(self, x, y):
        for x_i, y_i in zip(x, y):
            self.feedforward(x_i)
            self.backprop(x_i, y_i)


# Создание экземпляров нейронных сетей с функциями активации sigmoid и relu
network_sigmoid = NeuralNetwork(X, y, activation_function='sigmoid')
network_relu = NeuralNetwork(X, y, activation_function='relu')

# Обучение сетей
for i in range(1500):
    network_sigmoid.train(X, y)
    network_relu.train(X, y)

# Предсказание результатов с помощью обученных сетей
predictions_sigmoid = network_sigmoid.output
predictions_relu = network_relu.output

# Вычисление среднеквадратичной ошибки (MSE) для каждой сети
mse_sigmoid = np.mean((predictions_sigmoid - y) ** 2)
mse_relu = np.mean((predictions_relu - y) ** 2)

# Вывод среднеквадратичной ошибки для каждой сети
print(f"Mean Squared Error (sigmoid): {mse_sigmoid}")
print(f"Mean Squared Error (ReLU): {mse_relu}")
