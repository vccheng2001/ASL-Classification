
'''
Neural Network for ASL Classification 

'''
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


classes = 'ABCDEFGHIKLMNOPQRSTUVWXY'

DATA_DIR = '../dataset/'
MODEL_DIR = '../models/'

# Dataloader 
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),])

train_set = datasets.ImageFolder(root=DATA_DIR +'Train/',transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)

test_set = datasets.ImageFolder(root=DATA_DIR +'Test/',transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 24)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x) 
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        
        return out

net = Net()

''' RETRAIN '''
# net = torch.load(MODEL_DIR + 'model.pt')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
num_epochs = 5

n_total_steps = len(train_loader)


# Train
print('Train loader size: ', len(train_loader))
for epoch in range(num_epochs):
    print(f'**** epoch {epoch} ******')
    net.train()

    for i, (x, y) in enumerate(train_loader):

        # print('x', x.shape, 'y', y.shape)
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}') 


correct = 0
total = 0

print('Test loader size: ', len(test_loader))
with torch.no_grad():
     n_correct = 0
     n_samples = 0
     for i, (x, y) in enumerate(test_loader):
         output = net(x)
         _, predicted = torch.max(output.data, 1)
         n_samples += y.size(0)
         n_correct += (predicted == y).sum().item() 


acc = 100.0 * n_correct / n_samples
print(f'Test accuracy: {acc} %')


''' Show image and predicted/ground truth class'''
plt.imshow(x.view(28,28,1))
plt.show()
pred_idx = torch.argmax(net(x)[0])
print('ground truth letter', classes[int(y)])
print('predicted letter', classes[int(pred_idx)])



torch.save(net.state_dict(), MODEL_DIR+f'cnn1.pt')
writer.close()

