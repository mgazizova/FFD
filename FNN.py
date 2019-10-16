# Import PyTorch
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.autograd import Variable

# Инициализация Гипер-параметров
input_size = 784       # Размеры изображения = 28 x 28 = 784
hidden_size = 500      # Количество узлов на скрытом слое
num_classes = 10       # Число классов на выходе. В этом случае от 0 до 9
num_epochs = 5         # Количество тренировок всего набора данных
batch_size = 100       # Размер входных данных для одной итерации
learning_rate = 0.001  # Скорость конвергенции


# Загрузка набора данных MNIST
train_dataset = dsets.MNIST(root='./data',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())
# features - двумерный тезор, где каждая строка представляет один образец обучения
# targets - тензор, может быть 1-D или 2-D, в зависимости от попытки предсказать скаляр или вектор
# train = data_utils.TensorDataset()

# Загрузка данных в код
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Feedforward Neural Network Model Structure
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()  # Наследуемый родительским классом nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
        self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x) ReLU - ФУНКЦИЯ АКТИВАЦИИ
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 2й связанный слой: 500 (скрытый узел) -> 10 (класс вывода)

    def forward(self, x):  # Передний пропуск: складывание каждого слоя вместе
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Инициализация FNN
net = Net(input_size, hidden_size, num_classes)


# Можно включить графический процессор GPU
#net.cuda()    # Закомментировать для отключения GPU


# Выбор функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Тестируем FNN модель
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  # Загрузка партии изображений с индексом, данными, классом
        images = Variable(images.view(-1, 28 * 28))  # Конвертация тензора в переменную: изменяем изображение с вектора, размером 784 на матрицу 28 x 28
        labels = Variable(labels)

        optimizer.zero_grad()  # Инициализация скрытых масс до нулей
        outputs = net(images)  # Передний пропуск: определение выходного класса, данного изображения
        loss = criterion(outputs, labels)  # Определение потерь: разница между выходным классом и предварительно заданной меткой
        loss.backward()  # Обратный проход: определение параметра weight
        optimizer.step()  # Оптимизатор: обновление параметров веса в скрытых узлах

        if (i + 1) % 100 == 0:  # Логирование
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data.item()))
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)  # Выбор лучшего класса из выходных данных: класс с лучшим счетом
    total += labels.size(0)  # Увеличиваем суммарный счет
    correct += (predicted == labels).sum()  # Увеличиваем корректный счет

print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))