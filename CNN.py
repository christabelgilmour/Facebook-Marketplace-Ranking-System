
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import copy
from torchvision import transforms
from torch.autograd import Variable
from torchsummary import summary
from datetime import date
from torch.utils.data import DataLoader
from torch import manual_seed
from pytorch import pytorchdataset
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard.writer import SummaryWriter


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)

    def forward(self, X):
        return self.resnet50(X)        
            

def train(model, train_loader, val_loader, epochs=10):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    writer = SummaryWriter()

    batch_idx = 0
    model_name = f"model_{date.today()}"
    
    if not os.path.exists(f"model_evaluation/{model_name}"):
        os.makedirs(f"final_model/{model_name}")

    if not os.path.exists(f"model_evaluation/{model_name}/weights"):
        os.makedirs(f"final_model/{model_name}/weights")

    for epoch in range(epochs):
        model.train()
        print(f'Training Epoch {epoch}/{epochs - 1}')
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        total_examples = 0

        for batch in train_loader:
            features, labels, index = batch
            with torch.set_grad_enabled(True):

                optimiser.zero_grad()
                prediction = model(features)
                preds = torch.argmax(prediction, 1)
                loss = F.cross_entropy(prediction, labels)
                loss.backward()
                optimiser.step()

            writer.add_scalar("Training Batch Loss", loss.item(), batch_idx)

            batch_idx += 1

            running_loss += loss.item() * features.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_examples += len(labels)
            
        epoch_loss = running_loss / total_examples

        print(f'Train Loss: {epoch_loss:.4f}')
        writer.add_scalar("Training epoch loss", epoch_loss, epoch)
        
        val(model, val_loader)

        model_weights = copy.deepcopy(model.state_dict())
        torch.save(model_weights, f"final_model/{model_name}/weights/model_weights_{epoch}.pt")

def val(model, val_loader, epochs=10):
    
    writer = SummaryWriter()
    batch_idx = 0

    model.eval()

    validation_corrects = 0
    validation_loss = 0.0
    val_examples = 0
    with torch.no_grad():
        
        for batch in val_loader:
            features, labels, index = batch
            prediction = model(features)
            preds = torch.argmax(prediction, 1)

            val_loss = F.cross_entropy(prediction, labels)
            writer.add_scalar("Validation Batch Loss", val_loss.item(), batch_idx)
            batch_idx += 1

            validation_loss += val_loss.item() * features.size(0)
            validation_corrects += torch.sum(preds == labels.data)
            val_examples += len(labels)

        val_loss = validation_loss / val_examples
        val_acc = (validation_corrects / val_examples) * 100
        print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')


def finetune(model):
    for name, para in model.named_parameters():
        if "layers.3" in name:
            para.requires_grad = True
        elif "fc" in name:
            para.requires_grad = True
        else:
            para.requires_grad = False
    

def load_data():
    np.random.seed(2)
    random.seed(0)
    torch.manual_seed(0)
    dataset = pytorchdataset()
    train_data, test_data = random_split(dataset, [0.7, 0.3])
    test_data, val_data = random_split(test_data, [0.5, 0.5])
    train_loader = DataLoader(train_data, shuffle=True, batch_size=10)
    test_loader = DataLoader(test_data, batch_size=10)
    val_loader = DataLoader(val_data, batch_size=10)
    return train_loader, test_loader, val_loader

def run():
    finetune(model)
    train_loader, test_loader, val_loader = load_data()
    train(model, train_loader, val_loader)


if __name__ == "__main__":
    model = Classifier()
    run()
    
