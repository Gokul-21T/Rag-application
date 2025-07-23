from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import torch

class SimpleRAG:
    def __init__(self, collection_name="rag_chunks", qdrant_host="localhost", qdrant_port=6333):
        self.collection_name = collection_name
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        local_model_path = os.path.join(os.path.dirname(__file__), "local_models", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(local_model_path)
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to('cuda')
            print("Embedding model device:", next(self.embedding_model.parameters()).device)
        else:
            print("Embedding model is using CPU.")
        self._init_collection()

    def _init_collection(self):
        # Create collection if it doesn't exist
        if self.collection_name not in [c.name for c in self.qdrant.get_collections().collections]:
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE)
            )

    def embed_chunks(self, chunks, filename=None):
        # Improved chunking: add filename, page number, and chunk_id to each chunk
        points = []
        for i, chunk in enumerate(chunks):
            text = chunk['text']
            page_number = chunk.get('page_number', None)
            chunk_id = f"{filename or 'document'}_page{page_number}_chunk{i}"
            embedding = self.embedding_model.encode([text])[0]
            payload = {
                'text': text,
                'filename': filename or 'document',
                'page_number': page_number,
                'chunk_id': chunk_id
            }
            points.append(
                qdrant_models.PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )
        self.qdrant.upsert(collection_name=self.collection_name, points=points)

    def query(self, question, top_k=3):
        # Embed the question
        query_vec = self.embedding_model.encode([question])[0]
        # Search Qdrant
        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vec.tolist(),
            limit=top_k
        )
        # Return all metadata for each relevant chunk
        return [hit.payload for hit in search_result] 