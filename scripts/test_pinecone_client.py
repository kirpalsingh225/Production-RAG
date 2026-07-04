from app.clients.pinecone_client import PineconeClient
from app.core.config import settings


client = PineconeClient(
    api_key=settings.pinecone_api_key,
    index_host=settings.pinecone_index_host,
    namespace=settings.pinecone_namespace,
)

stats = client.index.describe_index_stats()
print(stats)