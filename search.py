from sentence_transformers import SentenceTransformer

from rag.vector_store import search_vectors


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def search_documents(query, top_k=3):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = search_vectors(
        query_embedding,
        top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    search_results = []

    for i in range(len(documents)):
        search_results.append({
            "text": documents[i],
            "source": metadatas[i]["source"],
            "distance": float(distances[i])
        })

    return search_results