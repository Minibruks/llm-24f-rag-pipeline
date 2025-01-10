from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader

def load_vector_store(knowledge_base_path):
    loader = TextLoader(knowledge_base_path)
    documents = loader.load()

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings)

    return vectorstore.as_retriever()
