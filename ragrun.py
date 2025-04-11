import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# initialize ChromaDB client
db = chromadb.PersistentClient(path="./chroma_db")

# get or create a collection
chroma_collection = db.get_or_create_collection("coalindia")

llm = None
Settings.llm=Ollama(model="llama3.2", request_timeout=360.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)


query_engine = index.as_query_engine(llm=Settings.llm, similarity_top_k=5)
# response = query_engine.query("How did the coal india perform in 2024?")
# print(response)
user_input = ''
while user_input != 'bye':
    user_input = input("Ask something about Coal India company: ")
    response = query_engine.query(user_input)
    print(response)

