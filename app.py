import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document import Document
from langchain.loader import UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
groq=os.environ.get('groq')


st.title("Question Answering App")

# Create a file uploader
uploaded_file = st.file_uploader("Choose a file:", type=["txt", "pdf", "docx"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Create a file loader
    loader = UnstructuredFileLoader()
    # Load the uploaded file
    doc = loader.load(uploaded_file)
if doc:
    embedding = HuggingFaceEmbeddings()
    llm = ChatGroq(temperature=0, api_key=groq, model_name="llama-3.1-70b-versatile")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents([doc])

    db = FAISS.from_documents(splits, embedding)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer and don't find it in the given context, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    query = st.text_input("Enter your question:", "")

    if query:
        result = qa_chain({"query": query})
        st.write(result['result'])
        st.write("Source documents:")
        for doc in result['source_documents']:
            st.write(doc.page_content)