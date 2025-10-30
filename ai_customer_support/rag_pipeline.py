from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

documents = [
    "Our refund policy allows returns within 30 days of purchase.",
    "Customers can contact support via email at support@company.com.",
    "We offer integrations with Salesforce and HubSpot for CRM management.",
    "Orders above $100 qualify for free shipping worldwide.",
    "Technical support is available 24/7 for premium customers."
]

vectors = model.encode(documents).tolist()
vector_size = len(vectors[0])
collection_name = "support_docs"

if collection_name not in [col.name for col in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

qdrant.upsert(
    collection_name=collection_name,
    points=[PointStruct(id=i, vector=vectors[i], payload={"text": documents[i]}) for i in range(len(documents))]
)

def get_context(query, top_k=3):
    query_vector = model.encode(query).tolist()
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return " ".join([r.payload["text"] for r in results])
