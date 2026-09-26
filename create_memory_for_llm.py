from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


#.env file ko read karke uske variables Python program ke environment mein load kare
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


#step 1: load raw pdfs


DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
#print("length of documents: ", len(documents))


#step 2: create chunks


def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data = documents)               #function call to create chunks step 2 ko call kiya 
#print("length of text chunks: ", len(text_chunks))


#step 3: create vector embeddings


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")    #for semantic search --text embedding model text chucks se numerical represen ke liye
    return embedding_model

get_embedding_model = get_embedding_model()     #function call to get embedding model


#step 4: store embeddings in a faiss


DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, get_embedding_model)    #function call to store embeddings in a faiss
db.save_local(DB_FAISS_PATH)                                   #save the faiss vector store to local path

