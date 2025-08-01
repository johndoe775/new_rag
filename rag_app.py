import os
from fastapi import FastAPI, File, UploadFile, Form
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq = os.environ.get("groq")

app = FastAPI()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Read the content of the uploaded file
        file_content = await file.read()

        # Create a file loader and load the uploaded file content
        loader = UnstructuredFileLoader(file_path=None)
        doc = loader.load_from_string(file_content.decode("utf-8"))

        if doc:
            embedding = HuggingFaceEmbeddings()
            llm = ChatGroq(
                temperature=0, api_key=groq, model_name="llama-3.1-70b-versatile"
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=150, separators=["\n\n", "\n", ".", " "]
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
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )

            result = qa_chain({"query": question})
            answer = result["result"]
            source_documents = [doc.page_content for doc in result["source_documents"]]

            return {"answer": answer, "source_documents": source_documents}

        return {"error": "No document could be processed."}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
