import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate 

from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource          #decorator function
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")    #for semantic search
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)    #load the faiss vector store from local path and kyuki kudh ka db h deserial true kar do
    return db

def set_custom_prompt(custom_prompt_template):                                  #func to return prompt
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])      #prompt template ko set kiya
    return prompt




def main():
    st.title("ask chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Enter your prompt here:")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})


        custom_prompt_template = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer.
            Dont provide anything out of the given context

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
            """


        try:
            vectorstore = get_vectorstore()             #humara database db vectorstore variable ke andr store h yaha

            if vectorstore is None:
                st.error("failed to load vectorstore")
                return

            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")                    # abh is variable ke andr groq token store h
            GROQ_MODEL_NAME = "openai/gpt-oss-20b"                       #groq model
            
            llm = ChatGroq(
                        model = GROQ_MODEL_NAME,
                        temperature=0.5,
                        max_tokens=512,
                        api_key=GROQ_API_KEY
            )
            

            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

            #document combiner chain
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

            #retrieval chain
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs= {'k': 3}), combine_docs_chain)


                        
            #user_query = input("Write query here: ")
            response = rag_chain.invoke({'input': prompt})

            result = response["answer"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"Error : {str(e)}")


if __name__ == "__main__":
    main()