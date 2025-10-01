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

load_dotenv()

# -------------------- Pinecone setup --------------------
pine_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(pine_key)

index_name =os.environ.get("Index_name")

# If index exists, delete it to avoid dimension mismatch
if pc.has_index(index_name):
    print(f"Deleting existing index '{index_name}'...")
    pc.delete_index(index_name)
    time.sleep(5)  # wait for deletion

# Create a new index for MiniLM embeddings (dimension 384)
if not pc.has_index(index_name):
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


# Wait until index is ready
while not pc.describe_index(index_name).status["ready"]:
    print("Waiting for index to be ready...")
    time.sleep(1)

index = pc.Index(index_name)

# -------------------- Embeddings --------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(file_text)

# Wrap each chunk as a Document with metadata
chunk_documents = [
    Document(page_content=chunk, metadata={"source": file_name, "chunk_id": str(i+1)})
    for i, chunk in enumerate(chunks)
]

# Generate unique ids for each chunk
chunk_ids = [str(uuid.uuid4()) for _ in chunk_documents]

# -------------------- Upload to Pinecone --------------------
vector_store.add_documents(documents=chunk_documents, ids=chunk_ids)

print(f"Successfully added {len(chunk_documents)} chunks from '{file_name}' to Pinecone index '{index_name}'!")