import streamlit as st
import PyPDF2
import os
import time
from dotenv import load_dotenv
import uuid

from pinecone import Pinecone,ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="ChatBot")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# --- File upload ---
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file and not st.session_state.file_uploaded:
    file_name = uploaded_file.name
    file_text = ""
    
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        file_text = "\n".join(page.extract_text() for page in pdf_reader.pages)
    else:
        file_text = uploaded_file.read().decode("utf-8")
    
    st.session_state.file_text = file_text
    st.session_state.file_name = file_name
    st.session_state.file_uploaded = True
    st.success("✅ File uploaded successfully!")

# --- Pinecone setup ---
load_dotenv()
pc = Pinecone(os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("Index_name")

if "index_created" not in st.session_state:
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    st.session_state.index_created = True

index = pc.Index(index_name)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# --- Ingest file once ---
if st.session_state.file_uploaded and "chunks_added" not in st.session_state:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(st.session_state.file_text)
    chunk_documents = [
        Document(page_content=chunk, metadata={"source": st.session_state.file_name, "chunk_id": str(i+1)})
        for i, chunk in enumerate(chunks)
    ]
    chunk_ids = [str(uuid.uuid4()) for _ in chunk_documents]
    vector_store.add_documents(documents=chunk_documents, ids=chunk_ids)
    st.session_state.chunks_added = True
    st.success(f"✅ Added {len(chunk_documents)} chunks to Pinecone index!")

# --- Retriever (once) ---
if "retriever" not in st.session_state:
    st.session_state.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

retriever = st.session_state.retriever

# --- Chat UI ---
st.subheader("💬 Chat")
user_input = st.text_input("You:")

if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Retrieval
    results = retriever.get_relevant_documents(user_input)
    retrieved_text = "\n".join(
        f"- {res.page_content} [source: {res.metadata.get('source')}, chunk: {res.metadata.get('chunk_id')}]"
        for res in results
    )
    
    response = f"I found the following relevant parts from your file:\n{retrieved_text}"
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Display chat messages ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
