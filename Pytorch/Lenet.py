# reference https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://zhuanlan.zhihu.com/p/42692344
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim





class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5) # in_channels, num_kernels, kernel_size
        self.pool1 = nn.MaxPool2d(2, 2) #size = 2*2, stride = 2
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)  #input for fc layer should be one-dimensional
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
# Formula of the matrix size after conv: N = (W - F + 2P)/S + 1
# N = (32 - 5 + 2 * 0)/1 + 1 =28
    def forward(self, x):
        # originial x = [batch, channel, height, width]
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5), -1 means infered (not sure about the resulting shape)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


# pytorch transform.normalize: image = (image - mean) / std, where the first paramter is mean and the 2nd is std
# both mean and std = 5 will map the range(0,1) to (-1,1)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# if __name__ == '__main__': # this is to avoid RuntimeError: freeze_support()
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=0)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # outputs will have an grad_fn attribute, that references a function that created it, and ties it back to the model. 
        # Therefore, loss.backward() will have information about the model it is working with.
        # https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')