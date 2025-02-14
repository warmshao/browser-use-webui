import logging
import json
from typing import Any, Dict, List, Optional, Union

from pymilvus import DataType, MilvusClient, Function, FunctionType
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MilvusStorage:
    def __init__(self, uri: str = "http://localhost:19530", collection_name: str = "web_data"):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self._recreate_collection()

    def _recreate_collection(self):
        """Drop existing collection and create a new one with the current schema."""
        try:
            # Drop existing collection if it exists
            if self.client.has_collection(self.collection_name):
                logger.info(f"Dropping existing collection '{self.collection_name}'")
                self.client.drop_collection(self.collection_name)

            # Create schema with dynamic fields enabled
            schema = self.client.create_schema(enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=384)
            schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
            
            # Add BM25 function for full-text search
            bm25_function = Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)
            
            # Prepare index parameters
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="COSINE"
            )
            index_params.add_index(
                field_name="sparse",
                index_type="AUTOINDEX",
                metric_type="BM25"
            )
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            
            # Load collection for searching
            self.client.load_collection(self.collection_name)
            logger.info(f"Successfully recreated Milvus collection '{self.collection_name}' with dynamic fields enabled")
        except Exception as e:
            logger.error(f"Error recreating Milvus collection: {str(e)}")
            raise

    def store(self, content: Union[str, Dict[str, Any]], url: str = "", content_type: str = "generic", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in Milvus with its embedding."""
        try:
            # Convert content to string if it's a dict
            if isinstance(content, dict):
                content_str = json.dumps(content)
            else:
                content_str = str(content)

            # Create embedding
            embedding = self.encoder.encode(content_str).tolist()
            
            # Prepare data with dynamic fields
            data = {
                'text': content_str,
                'vector': embedding,
                'url': url,
                'type': content_type,
                'metadata': json.dumps(metadata or {})
            }

            # Insert into Milvus
            self.client.insert(
                collection_name=self.collection_name,
                data=[data]
            )
            return f"Successfully stored content of type '{content_type}'"
        except Exception as e:
            return f"Error storing in Milvus: {str(e)}"

    def search(self, query: str, limit: int = 3, content_type: Optional[str] = None, use_vector: bool = True) -> List[Dict[str, Any]]:
        """Search for content in Milvus."""
        try:
            if use_vector:
                query_embedding = self.encoder.encode(query).tolist()
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    anns_field="vector",
                    params={
                        "metric_type": "COSINE",
                        "params": {"nprobe": 10}
                    },
                    limit=limit,
                    output_fields=["text", "url", "type", "metadata"]
                )
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query],
                    anns_field="sparse",
                    params={
                        "drop_ratio_search": 0.2
                    },
                    limit=limit,
                    output_fields=["text", "url", "type", "metadata"]
                )

            processed_results = []
            for hits in results:
                for hit in hits:
                    entity = hit['entity']
                    content = entity.get('text', '')
                    url = entity.get('url', '')
                    content_type = entity.get('type', 'generic')
                    metadata = json.loads(entity.get('metadata', '{}'))
                    score = 1 - hit['distance'] if use_vector else hit['score']
                    
                    # Try to parse content as JSON if possible
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass  # Keep as string if not valid JSON
                    
                    processed_results.append({
                        "content": content,
                        "url": url,
                        "type": content_type,
                        "metadata": metadata,
                        "score": round(score, 4)
                    })
            return processed_results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# Create a global instance
storage = MilvusStorage() 