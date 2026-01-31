import chromadb
from chromadb.utils import embedding_functions
from app.core.settings import settings

class VectorService:
    def __init__(self):
        self.client = None
        self.collection = None
        self._connect()

    def _connect(self):
        try:
            # Initialize Embedding Function
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL
            )
            # Initialize Client
            self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
            # Get Collection
            self.collection = self.client.get_collection(
                name="vachanamrut_rag", 
                embedding_function=self.ef
            )
            print("🧠 Vector DB: Connected successfully.")
        except Exception as e:
            print(f"❌ Vector DB Error: {e}")
            self.client = None
            self.collection = None

    def search(self, query: str, filters: dict = None, n_results: int = 5):
        if not self.collection:
            return None
            
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filters
        )

vector_service = VectorService()
