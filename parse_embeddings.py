import pickle, faiss, numpy as np

with open('outputs/ROC-spring-embeddings-all-mpnet-base-v2-size-100000.pkl', 'rb') as f:
    data = pickle.load(f)
corpus_embeddings = data['embeddings'].astype('float32')
corpus_embeddings /= np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

embedding_size = 768
n_clusters = max(1, len(corpus_embeddings) // 40)  # roughly 4–16×√N rule simplified
print(f"Using {n_clusters} clusters")

quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
index.nprobe = 3
index.train(corpus_embeddings)
index.add(corpus_embeddings)
faiss.write_index(index, 'outputs/blog_index.faiss')
