import time

from app.clients.embedding_client import EmbeddingClient
from app.clients.pinecone_client import PineconeClient
from app.core.config import settings


chunks = [
    "Contractors receive 12 paid leave days after six months of service.",
    "Full-time employees receive health insurance coverage.",
]

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

embeddings = embedding_client.embed_passages(chunks)

vectors = [
    {
        "id": f"demo_chunk_{index}",
        "values": embedding,
        "metadata": {
            "chunk_text": chunk,
            "document_id": "demo_document",
            "chunk_index": index,
        },
    }
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]

pinecone_client.upsert_vectors(vectors)

# Pinecone stats can update asynchronously.
time.sleep(2)

stats = pinecone_client.index.describe_index_stats()
print(stats)