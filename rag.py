import chromadb
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader,StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient

#llm = Ollama(model="llama3.2")
llm = None
Settings.llm=Ollama(model="llama3.2", request_timeout=360.0)
# response = llm.complete("What is the university at buffalo?")
# chroma_client = chromadb.EphemeralClient()
# chroma_collection = chroma_client.create_collection("mgs636test")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("./data/").load_data()

db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("coalindia")

# persist_dir = "./storage"
documents = SimpleDirectoryReader("./data/").load_data()
# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
print('Learned')

# query_engine = index.as_query_engine(llm=Settings.llm)
# response = query_engine.query("How did the coal india perform in 2024")
# print(response)





# llm2 = Ollama(model="qwen2.5")
# response2 = llm2.complete("What is the university at buffalo?")
# print('Q1) Hows the stock market today?')
# response1 = llm.complete("Hows the stock market today?")
# print(response1)
# print('Q2) Whats special about US?')
# response2 = llm.complete("Whats special about US?")
# print(response2)
# print('Q3) Whats the latest marvel movie?')
# response3 = llm.complete("Whats the latest marvel movie?")
# print(response3)
