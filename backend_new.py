import os
import tempfile
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import requests
import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
# Corrected import and instantiation
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from base64 import b64decode

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.email import partition_email
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY") or "changeme"
HF_API_TOKEN = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = "AIzaSyB2rGBTsrz9w9D2owsnz2d-X2__bYxtxIg"

app = FastAPI(title="HackRX Multimodal RAG Backend")




# Auth utility
def check_auth(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail='No or invalid Authorization header')
    _api_key = authorization[7:]
    if _api_key != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid API key')

# Utility: processing file type
def _get_unstructured_partition(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == "pdf":
        print("PDF")
        chunk_list = partition_pdf(
            filename=file_path,
            infer_table_structure=True,         # extract tables
            strategy="fast",                  # mandatory to infer tables, fast or highres

            # extract_image_block_types=["Image"],  # Add 'Table' to list to extract image of tables
            # image_output_dir_path=file_path+"assets",  # if None, images and tables will be saved in base64
            # extract_image_block_to_payload=True,  # if true, will extract base64 for API usage

            chunking_strategy="by_title",       # or 'basic'
            max_characters=10000,               # defaults to 500
            combine_text_under_n_chars=2000,    # defaults to 0
            new_after_n_chars=6000,             #
            # language="en",
        )
        print("CHUNKING SUCCESSFUL. TOTAL CHUNKS: ", len(chunk_list))
        return chunk_list
    elif ext in ["docx", "doc"]:
        print("WORD")
        return partition_docx(file_path)
    elif ext in ["eml", "msg"]:
        print("EMAIL")
        return partition_email(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF, Word, or Email.")



# Extract chunks, keeping tables/images separately
def process_document(file_path, questions):
    try:
        # > MODEL
        print("[DEBUG] Loading model...")
        # model  = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
        key = os.getenv("GEMINI_API_KEY1")
        model  = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=key)
        # from langchain_together import Together
        # model = Together(
        #     model="meta-llama/Llama-3-70b-chat-hf",
        #     together_api_key=os.getenv("TOGETHER_API_KEY"),
        #     max_tokens=1000
        # )

  
        #> CHUNKING
        print("[DEBUG] Chunking...")
        chunk_list = _get_unstructured_partition(file_path)
        print("chunk_list of lenghth ", len(chunk_list),"obtained")
        texts, tables, images = [], [], []

        for chunk in chunk_list:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            if "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
                if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
                    for el in chunk.metadata.orig_elements:
                        if "Table" in str(type(el)):
                            tables.append(el)
                        if "Image" in str(type(el)) and hasattr(el, "metadata") and hasattr(el.metadata, "image_base64"):
                            images.append(el.metadata.image_base64)

        print(len(tables), "tables found")
        print(len(texts), "texts found")
        print(len(images), "images found")

        
        # Create LangChain Documents
        docs_for_embedding = []

        for chunk in chunk_list:
            docs_for_embedding.append(Document(page_content=chunk.text, metadata={"page": chunk.metadata.page_number}))
        #> EMBEDDING
        print("[DEBUG] Embedding...")
        # Use the HuggingFaceEmbeddings wrapper
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vectorstore = Chroma.from_documents(docs_for_embedding, embedding_model, persist_directory="vectorstore")
        print("[DEBUG] Embedding DONE, Vectorstore loaded...")


        #> RETRIEVAL
        print("[DEBUG] Retrieval AND chaining...")
        prompt_template = """
            You are an assistant that answers questions using the provided context.
            If the answer is not in the context, say you don't know.

            Context:
            {context}

            Question:
            {question}

            Answer:
        """

        prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
        
        #> QA
        print("[DEBUG] QA...")
        answers = []
        for q in questions:
            # print("Q:", q)
            answers.append(rag_chain.invoke(q)["result"])
            # print("A:", answers[-1])
        print("[DEBUG] QA DONE")
        print("ANSWERS:", answers)
        return answers
        
    except Exception as e:
        print("ERROR FROM process_document:", e)
        return None




### API Definitions ###

class HackRxRunRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxRunResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=HackRxRunResponse) 
def hackrx_run(req: HackRxRunRequest):
    tmp_dir = tempfile.mkdtemp(dir=os.getcwd())
    try:
        file_url = req.documents
        local_filename = os.path.join(tmp_dir, os.path.basename(file_url.split("?")[0]))
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("THE FILE WRITTEN IN THE TEMP DIR IS", local_filename)
        answers = process_document(local_filename, req.questions)
    
        # answers = get_answers(retreiver, req.questions)
        print("ANSWERS:", {"answers": answers })
        return {"answers": answers }
    except Exception as e:
        print("ERROR FROM hackrx_run:", e)
        return None
    finally:
        import gc
        gc.collect()  # Force close lingering file handles
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"[WARN] Cleanup failed: {e}")
