from embeddings import generate_embedding
from vector_store import search_vectors


def search_documents(query, top_k=3):

    query_embedding = generate_embedding(query)

    results = search_vectors(
        query_embedding,
        top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    search_results = []

    for i in range(len(documents)):

        source = "Unknown"

        if i < len(metadatas):
            source = metadatas[i].get(
                "source",
                "Unknown"
            )

        distance = 0.0

        if i < len(distances):
            distance = float(distances[i])

        search_results.append({
            "text": documents[i],
            "source": source,
            "distance": distance
        })

    return search_results
