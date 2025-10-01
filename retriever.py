# retriever.py

import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("pinecone_api_key"))

# set the pinecone index

index_name = os.environ.get("Index_name")
index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# -------------------- Create Retriever --------------------
retriever = vector_store.as_retriever(
    search_type="similarity",  # can also try "mmr" for diverse results
    search_kwargs={"k": 1}     # number of top results to return
)

query = "what is machine learning?"
results = retriever.get_relevant_documents(query)

print("RESULTS:\n")
for res in results:
    print(f"* {res.page_content} [source: {res.metadata.get('source')}, chunk: {res.metadata.get('chunk_id')}]")



