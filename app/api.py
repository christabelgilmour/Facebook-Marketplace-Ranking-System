import pickle
import uvicorn
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from PIL import Image
from image_processor import transform_image

with open('image_decoder.pkl', 'rb') as file:
    decoder = pickle.load(file)

with open('embedding_ids.pkl', 'rb') as file:
    embedding_ids = pickle.load(file)

with open("my_embeddings.pkl", "rb") as file:
            my_embeddings = pickle.load(file)

class ImageClassifier(nn.Module):

    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)

    def forward(self, image):
        x = self.resnet50(image)
        return x

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        saved_weights = torch.load('model_weights.pt')
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)
        self.resnet50.load_state_dict(saved_weights, strict=False)
        self.new_model = nn.Sequential(*list(self.resnet50.children())[:-1])

    def forward(self, X):
        return self.new_model(X)

class FaissSearchIndex():
    def __init__(self):
        embeddings = list(my_embeddings.values())
        d = embeddings[0].shape[0]
        n = len(embedding_ids)

        embeddings_matrix = np.empty((n, d), dtype=np.float32)
        embedding_index=0
        for id in embedding_ids:

            embedding = my_embeddings[id]

            embeddings_matrix[embedding_index] = embedding
            embedding_index += 1

        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings_matrix)
       
def get_embeddings(feature_extractor, transformed_image):
    feature_extractor.eval()
    with torch.no_grad():
        features = transformed_image
        embedding = feature_extractor(features)
        embedding = embedding.squeeze(0)
        embedding = embedding.squeeze(-1)
        embedding = embedding.squeeze(-1).detach().numpy()
        print(embedding)
    
    return embedding

def get_prediction(image_model, transformed_image):
    image_model.eval()
    with torch.no_grad():
        features = transformed_image
        prediction = image_model(features)
        preds = torch.argmax(prediction, 1)
        preds = preds.item()

    return preds

def search(search_index, query_embedding, k=5):

    query_embedding = query_embedding.reshape(1, -1)
    D, I = search_index.index.search(query_embedding, k)

    print(f"Distances: {D}")
    print(f"Indices: {I}")

    return D, I


image_model = ImageClassifier()
image_model.load_state_dict(torch.load('model_weights.pt'))
feature_extractor = FeatureExtractor()
search_index = FaissSearchIndex()

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}
  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):

    # pil_image = Image.open(io.BytesIO(image.file.read()))
    pil_image = Image.open(image.file)
    transformed_image = transform_image(pil_image)
    preds = get_prediction(image_model, transformed_image)
    category = decoder[preds]

    return JSONResponse(content={
        "Predicted category": category
        })    


@app.post('/predict/embedding')
def predict_embedding(image: UploadFile = File(...)):

    pil_image = Image.open(image.file)
    transformed_image = transform_image(pil_image)
    embedding = get_embeddings(feature_extractor, transformed_image)
    D, I = search(search_index, embedding)
    ids = [embedding_ids[index] for index in I[0]]
    return JSONResponse(content={
        "K nearest neighbour image ids": ids
        })    

    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)