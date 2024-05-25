from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
from hugging_face import huggingface_llm, huggingface_embedding
import time

st.set_page_config(layout='wide', page_title="Hugging Face for RAG")

# setup prompt_template
prompt_template="""
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{question}
 """

# setup prompt
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])


# function for vector embedding
def vector_embedding():
    if 'vector' not in st.session_state:
        st.session_state.embeddings = huggingface_embedding() # embedding
        st.session_state.loader = PyPDFDirectoryLoader('End-to-End-Gen-AI-Powered-App/us-census-data') # data ingestion
        st.session_state.docs = st.session_state.loader.load() # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title('Hugging Face with Mistral-7B-Instruct-v0.2')

if st.button('Embedd Documents'):
    vector_embedding()
    st.write('Vector Store DB is Ready')

user_input = st.text_input('Enter Your Question From Documents')





if user_input:
    retriever = st.session_state.vectors.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    retrievalQA = RetrievalQA.from_chain_type(
        llm=huggingface_llm(),
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    start = time.process_time()
    response = retrievalQA.invoke(user_input)
    st.write(response['result'])
    st.write(f'response time:  {(time.process_time() - start):.2f} sec')


    # streamlit expander
    with st.expander('Document Similarity Search'):
        # find relevant chunks
        for i, doc in enumerate(response['source_documents']):
            st.write(doc.page_content)
            st.write('---------------------------------')


