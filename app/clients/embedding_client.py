from pinecone import Pinecone


class EmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-text-embed-v2",
        dimension: int = 1024,
    ):
        self.client = Pinecone(api_key=api_key)
        self.model_name = model_name
        self.dimension = dimension

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        response = self.client.inference.embed(
            model=self.model_name,
            inputs=texts,
            parameters={
                "input_type": "passage",
                "truncate": "END",
                "dimension": self.dimension,
            },
        )

        return [item["values"] for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        response = self.client.inference.embed(
            model=self.model_name,
            inputs=[query],
            parameters={
                "input_type": "query",
                "truncate": "END",
                "dimension": self.dimension,
            },
        )

        return response.data[0]["values"]