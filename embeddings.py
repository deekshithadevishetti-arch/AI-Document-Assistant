from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


def generate_embeddings(texts):
    """
    Convert text chunks into embeddings.
    """

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()


def generate_embedding(text):
    """
    Convert one text/query into an embedding.
    """

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()