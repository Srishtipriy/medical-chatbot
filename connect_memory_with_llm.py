import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain_huggingface import HuggingFaceEndpoint       #used in step 1
# from langchain_core.prompts import PromptTemplate           #used in line 36
# from langchain_classic.chains import RetrievalQA  

from dotenv import load_dotenv              #import .env file
load_dotenv()


#step 1: set up GROQ LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")                    # abh is variable ke andr groq token store h
GROQ_MODEL_NAME = "openai/gpt-oss-20b"                       #groq model


llm = ChatGroq(
    model = GROQ_MODEL_NAME,
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY
)

#step 2: connect llm with faiss create chain

#load database

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")    #for semantic search
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)    #load the faiss vector store from local path and kyuki kudh ka db h deserial true kar do

#step 3 build rag chain
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

#document combiner chain
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

#retrieval chain
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs= {'k': 3}), combine_docs_chain)


#invoke with a single query kyuki chain abh ban gai h step 2 ki


user_query = input("Write query here: ")
response = rag_chain.invoke({'input': user_query})
print("Answer: ", response["answer"])


for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")