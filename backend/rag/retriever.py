from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

client = PersistentClient(
    path="./backend/chroma_db"
)

collection = client.get_collection(
    "products"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def retrieve_products(query, top_k=5):

    query_embedding = model.encode(query).tolist()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )