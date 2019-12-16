import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from load_dataset import load_dataset
import torch.optim as optim

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(576, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 20)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

    # copied from torch example
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

deviceName = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(deviceName)
cnn = SimpleConvNet().to(device)
print("Device: " + deviceName)
print(cnn)

# data
# No test train split for now
dataset = load_dataset()
numpy_X = dataset.x
numpy_X = numpy_X.reshape((-1, 1, 32, 32))
print(numpy_X.shape)
numpy_y_labels = dataset.y

labels = ["abra", "alakazam", "articuno", "blastoise", "bulbasaur", "charizard", "charmander", "charmeleon", "gengar", "ivysaur", "magikarp", "meowth", "mew", "mewtwo", "moltres", "pikachu", "squirtle", "venusaur", "wartortle", "zapdos"]

numpy_y = []
for i in range(len(numpy_y_labels)):
    numpy_y.append(labels.index(numpy_y_labels[i]))

numpy_y = np.array(numpy_y)

X_train = torch.from_numpy(numpy_X).type(torch.LongTensor)
print(X_train.shape)
y_train = torch.from_numpy(numpy_y)
# y_train = torch.nn.functional.one_hot(y_train)

print(y_train)


BATCH_SIZE=1
train = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)

num_epochs = 5
optimizer = optim.SGD(cnn.parameters(), lr=0.001)
error = nn.CrossEntropyLoss()

# training
# Temp - adding .cuda() to all tensor. TODO: Move to cuda in the beginning 
epochs=30
for epoch in range(epochs):
    correct = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = cnn(X_batch.float().cuda())
        loss = error(output, y_batch.cuda())
        loss.backward()
        optimizer.step()
        
        predicted = torch.max(output.data, 1)[1]
        correct += (predicted == y_batch.cuda()).sum()
        if batch_idx % 2 == 0:
            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data.item(), float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))
