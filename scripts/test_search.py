from app.clients.embedding_client import EmbeddingClient
from app.clients.pinecone_client import PineconeClient
from app.core.config import settings


embedding_client = EmbeddingClient(
    api_key=settings.pinecone_api_key,
    model_name=settings.embedding_model,
    dimension=settings.embedding_dimension,
)

pinecone_client = PineconeClient(
    api_key=settings.pinecone_api_key,
    index_host=settings.pinecone_index_host,
    namespace=settings.pinecone_namespace,
)

query = "What paid leave do contractors receive?"

query_vector = embedding_client.embed_query(query)

results = pinecone_client.search(
    query_vector=query_vector,
    top_k=3,
)

print(f"\nQuery: {query}\n")

for match in results.matches:
    print(f"ID: {match.id}")
    print(f"Score: {match.score:.4f}")
    print(f"Text: {match.metadata['chunk_text']}")
    print("-" * 50)