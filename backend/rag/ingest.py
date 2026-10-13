import pandas as pd
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Load documents
df = pd.read_csv("backend/data/product_documents.csv")

# ChromaDB persistent storage
client = PersistentClient(path="./backend/chroma_db")

collection = client.get_or_create_collection(
    name="products"
)

# Embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

batch_size = 100

for i in range(0, len(df), batch_size):

    batch = df.iloc[i:i + batch_size]

    documents = batch["document"].tolist()

    ids = [
        str(pid)
        for pid in batch["product_id"]
    ]

    embeddings = model.encode(
        documents,
        show_progress_bar=False
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print(
        f"Inserted {min(i + batch_size, len(df))}/{len(df)}"
    )

print("ChromaDB ingestion completed!")