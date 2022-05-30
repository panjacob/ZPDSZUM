import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from utilis.visualize_batch import visualize_batch
from utilis.statistics_pytorch import make_confusion_matrix, make_plots

RESULT_DESTINATION = "/files/results/pytorch_cnn"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print("Device:", device)

EPOCHS = 50
BATCH_SIZE = 12
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "images/Data_08/train"
TEST_DATA_PATH = "images/Data_08/test"
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    ])

train_data = ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
test_data = ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)


print("[INFO] training dataset contains {} samples...".format(
        len(train_data)))
print("[INFO] validation dataset contains {} samples...".format(
        len(test_data)))

print("[INFO] creating training and validation set dataloaders...")
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

trainBatch = next(iter(train_data_loader))
valBatch = next(iter(test_data_loader))

print("[INFO] visualizing training and validation batch...")
#visualize_batch(trainBatch, BATCH_SIZE, train_data.classes, "train")
#visualize_batch(valBatch, BATCH_SIZE, test_data.classes, "val")



# CNN model
# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output

model = Network()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the Model
def train_epoch(model,device,dataloader,criterion,optimizer):
    
    train_loss,train_correct=0.0,0
    
    model.train()

    for i, (images, labels) in enumerate(dataloader):
        
        images,labels = images.to(device),labels.to(device)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(outputs.data, 1)
        train_correct += (predictions == labels).sum().item()
    return train_loss,train_correct

def valid_epoch(model,device,dataloader,criterion):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=criterion(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct

def crossvalidation(model,device ,criterion, dataset):
    splits=KFold(n_splits=5,shuffle=True,random_state=42)
    foldperf = {}
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print('Fold {}'.format(fold + 1))
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx) 
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

        history = {'loss': [], 'val_loss': [],'accuracy':[],'val_accuracy':[]}

        for epoch in range(EPOCHS):
            train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            test_loss, test_correct=valid_epoch(model,device,val_loader,criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(val_loader.sampler)
            test_acc = test_correct / len(val_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                EPOCHS,
                                                                                                                train_loss,
                                                                                                                test_loss,
                                                                                                                train_acc,
                                                                                                                test_acc))
            history['loss'].append(train_loss)
            history['val_loss'].append(test_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(test_acc)

        foldperf['fold{}'.format(fold+1)] = history
    torch.save(model,'k_cross_CNN.pt')
    return foldperf   

foldprerf = crossvalidation(model, device, criterion, train_data)
make_plots(foldprerf['fold1'], train_data_loader,RESULT_DESTINATION)
make_confusion_matrix(test_data_loader, model, 4)
#Save the Trained Model
torch.save(model.state_dict(),'cnn.pkl')

