from pinecone import Pinecone


class PineconeClient:
    def __init__(
        self,
        api_key: str,
        index_host: str,
        namespace: str = "default",
    ):
        self.namespace = namespace

        try:
            self.client = Pinecone(api_key=api_key)
            self.index = self.client.Index(host=index_host)

        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Pinecone client"
            ) from exc
        

    def upsert_vectors(self, vectors: list[dict]) -> None:
        try:
            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace,
            )
        except Exception as exc:
            raise RuntimeError("Failed to upsert vectors to Pinecone") from exc
        
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ):
        try:
            query_args = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "namespace": self.namespace,
            }

            if metadata_filter:
                query_args["filter"] = metadata_filter

            return self.index.query(**query_args)

        except Exception as exc:
            raise RuntimeError("Failed to search Pinecone index") from exc