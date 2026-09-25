from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#step 1: load raw pdfs
DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print("length of documents: ", len(documents))
#step 2: create chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data = documents)               #function call to create chunks step 2 ko call kiya 
print("length of text chunks: ", len(text_chunks))


#step 3: create vector embeddings
#step 4: store embeddings in a faiss
