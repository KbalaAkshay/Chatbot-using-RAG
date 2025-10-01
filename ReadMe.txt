# 📄 Chatbot with PDF/TXT Ingestion and Pinecone Retrieval

This project is a **Streamlit-based chatbot** that allows users to upload PDF or TXT files and ask questions about their content. It uses **LangChain**, **HuggingFace MiniLM embeddings**, and **Pinecone** to ingest documents, generate embeddings, and retrieve relevant information.

---

## **Project Overview**

1. **File Upload & Text Extraction**
   - Users can upload **PDF or TXT files**.
   - PDF text is extracted using `PyPDF2`.
   - TXT files are decoded as plain text.
   - Uploaded file content is displayed in the app for confirmation.

2. **Document Chunking**
   - Text is split into smaller chunks using **RecursiveCharacterTextSplitter**.
   - Each chunk has a **fixed size (500 characters)** with a **50-character overlap** to preserve context.

3. **Embeddings & Vector Store**
   - Chunks are converted to vector embeddings using **HuggingFace MiniLM model**.
   - Chunks are stored in a **Pinecone vector database** for fast similarity-based retrieval.
   - Pinecone index creation is performed **only once per session**.

4. **Retriever**
   - A **similarity-based retriever** fetches top relevant chunks from Pinecone.
   - User queries are matched against document embeddings to find the most relevant information.

5. **Chat Interface**
   - Built with **Streamlit** using `st.chat_message`.
   - User input triggers retrieval from Pinecone.
   - Retrieved chunks are displayed as a **contextual response**.
   - Chat history is preserved using **Streamlit session state**.

6. **Optimizations**
   - Prevents recreating the Pinecone index on every user interaction.
   - Overlapping chunks ensure meaningful context in retrieval.
   - Maintains chat session state for continuous conversation flow.

---

## **Technologies & Tools**
- **Frontend**: Streamlit  
- **PDF Parsing**: PyPDF2  
- **Embeddings**: HuggingFace MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)  
- **Vector Database**: Pinecone  
- **LangChain**: Document processing and retrieval  
- **Environment Variables**: `python-dotenv`

---

## **Setup Instructions**

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo-folder>
