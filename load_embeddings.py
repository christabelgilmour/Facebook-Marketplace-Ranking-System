import pickle

with open('my_embeddings.pkl', 'rb') as file:
    embeddings = pickle.load(file)

with open('embedding_ids.pkl', 'rb') as file:
    ids = pickle.load(file)

# Use the loaded data
print(type(embeddings['b800e353-2231-4f8f-8dcc-35cfa0b1b209']))
print(type(embeddings))
# print(type(ids))
