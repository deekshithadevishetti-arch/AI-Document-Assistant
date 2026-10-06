import chromadb
from pathlib import Path
import uuid


# Project folder
project_folder = Path(__file__).resolve().parent.parent

# ChromaDB folder
chroma_path = project_folder / "chroma_db"


# Create persistent ChromaDB client
client = chromadb.PersistentClient(
    path=str(chroma_path)
)


# Create or load the collection
collection = client.get_or_create_collection(
    name="documents"
)


def add_documents(chunks, embeddings, source):
    """
    Store document chunks and embeddings in ChromaDB.
    """

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    metadatas = [
        {"source": source}
        for _ in chunks
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Documents stored in ChromaDB successfully!")


def search_vectors(query_embedding, top_k=3):
    """
    Search ChromaDB using a query embedding.
    """

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results