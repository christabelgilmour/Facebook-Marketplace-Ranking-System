import torch
import numpy as np
import random
import torch.nn as nn
from pytorch import pytorchdataset
from torch.utils.data import random_split 
from torch.utils.data import DataLoader
import torch
import pickle


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        saved_weights = torch.load('final_model/model_2023-06-14/weights/model_weights_9.pt')
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)
        self.resnet50.load_state_dict(saved_weights, strict=False)
        self.new_model = nn.Sequential(*list(self.resnet50.children())[:-1])

    def forward(self, X):
        return self.new_model(X)

def get_embeddings(model, embedding_loader):
    model.eval()
    ids = []
    embeddings = []
    with torch.no_grad():

        for batch in embedding_loader:
            print(type(batch))
            features, labels, index = batch
            embedding = model(features)
            print(embedding.shape)
            embedding = embedding.squeeze(0)
            embedding = embedding.squeeze(-1)
            embedding = embedding.squeeze(-1).detach().numpy()
            index = str(index[0])
            ids.append(index)
            embeddings.append(embedding)
            
        image_embeddings = {str(ids[i]): embeddings[i] for i in range(len(ids))}

        f = open("embedding_ids.pkl", "wb")
        pickle.dump(ids, f)
        f.close()

        f = open("my_embeddings.pkl", "wb")
        pickle.dump(image_embeddings, f)
        f.close()


def load_data():
    np.random.seed(2)
    random.seed(0)
    dataset = pytorchdataset()
    train_data, test_data = random_split(dataset, [0.7, 0.3])
    test_data, val_data = random_split(test_data, [0.5, 0.5])
    embedding_loader = DataLoader(train_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=10)
    val_loader = DataLoader(val_data, batch_size=10)
    return embedding_loader, test_loader, val_loader


def run():
    embedding_loader, test_loader, val_loader = load_data()
    get_embeddings(model, embedding_loader)


if __name__ == "__main__":
    model = Classifier()

    run()