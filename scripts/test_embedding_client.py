from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings


embedding_client = EmbeddingClient(
    api_key=settings.pinecone_api_key,
    model_name=settings.embedding_model,
    dimension=settings.embedding_dimension,
)

passages = [
    "Contractors receive 12 paid leave days after six months of service.",
    "The company provides health insurance to full-time employees.",
]

vectors = embedding_client.embed_passages(passages)
query_vector = embedding_client.embed_query(
    "What paid leave do contractors receive?"
)

print("Passage vectors:", len(vectors))
print("Passage dimension:", len(vectors[0]))
print("Query dimension:", len(query_vector))