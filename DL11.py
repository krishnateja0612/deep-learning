# DL11.py CS5173/6073 2020 cheng
# CNN for MNIST
# based on https://nextjournal.com/gkoehler/pytorch-mnist
# prints loss in training and accuracy in testing
# Usage: at command prompt, run "python DL11.py"

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size_train = 256
n_epochs = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/Users/8umelec/Downloads/Deep-Learning/data/', train=True, download=True,
        transform=torchvision.transforms.ToTensor()),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/Users/8umelec/Downloads/Deep-Learning/data/', train=False, download=True,
        transform=torchvision.transforms.ToTensor()))
test_size = 10000  # may be derived from test_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3,padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3,padding=1,stride=2)
#        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3136, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

model3 = Net()
optimizer = optim.SGD(model3.parameters(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        p = model3(data)
        train_loss = loss_fn(p, target)
        if batch_idx % 100 == 0:
            print('train', epoch, batch_idx, float(train_loss)) 
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    m = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if int(torch.argmax(model3(data))) == int(target[0]):
            m = m + 1
    print("test", epoch, m, "among", test_size, "correctly classified")

