import faiss
import pickle
import numpy as np


class FaissSearchIndex():

    def __init__(self):
        
        with open("my_embeddings.pkl", "rb") as file:
            my_embeddings = pickle.load(file)

        with open("embedding_ids.pkl", "rb") as file:
            embedding_ids = pickle.load(file)

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


def search(search_index, query_embedding, k=5):
    query_embedding = query_embedding.reshape(1, -1)
    D, I = search_index.index.search(query_embedding, k)

    print(f"Distances: {D}")
    print(f"Indices: {I}")

    return D, I


if __name__ == "__main__":
    search_index = FaissSearchIndex()

    with open("my_embeddings.pkl", "rb") as file:
            my_embeddings = pickle.load(file)

    with open("embedding_ids.pkl", "rb") as file:
            embedding_ids = pickle.load(file)

    query_embedding = my_embeddings[embedding_ids[0]]
    print(query_embedding)
    print(query_embedding.shape)

    search(search_index, query_embedding)

