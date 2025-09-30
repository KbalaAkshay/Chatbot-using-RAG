import streamlit as st
import PyPDF2

st.set_page_config(page_title="ChatBot")

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chatbot")

# File uploader (PDF or TXT)
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])

# Extract text from file
file_text = ""
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            file_text += page.extract_text()
    elif uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")

    st.success("✅ File uploaded successfully!")

# Chat UI
st.subheader("💬 Chat")

user_input = st.text_input("You:", "")

print(user_input)

if st.button("Send") and user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Simple dummy response (replace with RAG later)
    if file_text:
        response = f"I read your file. You asked: '{user_input}'.\n(Sample answer — integrate LLM later)"
    else:
        response = "Please upload a file first."

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
