"""
sprint.py

Extension for Chatbot-using-RAG:
- Retrieves relevant chunks from the vector DB (via a LangChain-style retriever),
- Combines those chunks with the user's query into a clear prompt,
- Sends the prompt to OpenAI (ChatCompletion) to generate a consolidated answer,
- Returns the LLM answer plus provenance (sources / chunk ids).

Usage (example, inside your Streamlit app.py):
    from sprint import answer_with_rag

    answer, metadata = answer_with_rag(user_query, retriever=st.session_state.retriever)
    # `answer` is the consolidated text from OpenAI
    # `metadata` contains the list of docs used and their source info

Requirements:
- pip install openai
- Set OPENAI_API_KEY in your environment (or .env)
- retriever should implement get_relevant_documents(query) and return Documents
  with .page_content and .metadata (like LangChain Documents used in this repo).
"""

import os
import time
import openai
from typing import List, Dict, Tuple, Optional

# Configure OpenAI key from env
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Default model — change to "gpt-4" if you have access and want higher quality
DEFAULT_MODEL = "gpt-3.5-turbo"


def _build_prompt(user_query: str, docs: List[object], max_chars_per_doc: int = 2000) -> List[Dict]:
    """
    Build a Chat API messages list with a system message and a user message that
    contains the retrieved contexts and the user question.

    - docs: list of Document-like objects with .page_content and .metadata (source, chunk_id)
    - max_chars_per_doc: truncate long chunks to avoid hitting token limits
    Returns messages suitable for openai.ChatCompletion.create(..., messages=messages)
    """
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant that composes concise, factual answers "
            "based ONLY on the provided context documents. If the answer is not "
            "contained in the context, say you don't know or that the information is not available. "
            "Cite which source(s) you used in a short 'Sources:' section at the end, "
            "including the filename and chunk id when available."
        ),
    }

    # Build the context block enumerating documents
    context_blocks = []
    for i, d in enumerate(docs, start=1):
        text = getattr(d, "page_content", "") or str(d)
        if max_chars_per_doc and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "...(truncated)"
        md = getattr(d, "metadata", {}) or {}
        src = md.get("source", "unknown_source")
        chunk_id = md.get("chunk_id", md.get("id", f"chunk_{i}"))
        header = f"[{i}] source: {src} | chunk: {chunk_id}"
        context_blocks.append(f"{header}\n{text}")

    context_str = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."

    user_content = (
        "Context documents (do not hallucinate beyond these):\n\n"
        f"{context_str}\n\n"
        "User question:\n"
        f"{user_query}\n\n"
        "Please produce a concise, consolidated answer that directly uses the context above. "
        "If multiple documents are used, combine their information and indicate which "
        "documents supported your statements. Finally, show a short 'Sources:' list."
    )

    user_message = {"role": "user", "content": user_content}

    return [system_message, user_message]


def answer_with_rag(
    user_query: str,
    retriever,
    openai_model: str = DEFAULT_MODEL,
    k_override: Optional[int] = None,
    max_chars_per_doc: int = 2000,
    temperature: float = 0.0,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> Tuple[str, Dict]:
    """
    Main helper: get relevant docs from retriever, call OpenAI, and return answer + metadata.

    Parameters:
    - user_query: the user's question string
    - retriever: a LangChain-style retriever object. It should have a method:
         get_relevant_documents(query) -> List[Document]
      If your retriever was configured at creation with k, it will return that many results.
      You can also override k by setting retriever.search_kwargs before calling if your retriever exposes it.
      (This function attempts to respect a provided k_override by setting search_kwargs if possible.)
    - openai_model: model name for OpenAI (default gpt-3.5-turbo)
    - k_override: integer to request top-k docs (optional)
    - max_chars_per_doc: truncate each doc to this many characters in the prompt
    - temperature: LLM temperature (0 for deterministic)
    - max_retries: number of times to retry on transient OpenAI errors
    - retry_delay: seconds to wait between retries

    Returns:
    - answer (str): consolidated LLM output
    - metadata (dict): includes 'docs_used' list and 'llm_raw' response object (minimalized)
    """

    if k_override is not None:
        # Try to set search kwargs on retriever if it's mutable (LangChain retriever often has .search_kwargs)
        try:
            setattr(retriever, "search_kwargs", {"k": k_override})
        except Exception:
            # If not possible, we'll just rely on retriever's default behavior
            pass

    # Retrieve relevant docs
    try:
        docs = retriever.get_relevant_documents(user_query)
    except Exception as e:
        raise RuntimeError(f"Failed to get documents from retriever: {e}")

    # Build prompt messages
    messages = _build_prompt(user_query, docs, max_chars_per_doc=max_chars_per_doc)

    # Call OpenAI ChatCompletion with retries
    attempt = 0
    while True:
        try:
            resp = openai.ChatCompletion.create(
                model=openai_model,
                messages=messages,
                temperature=temperature,
                # You can tune max_tokens as needed; leaving implicit lets OpenAI choose defaults.
            )
            break
        except openai.error.OpenAIError as e:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"OpenAI request failed after {max_retries} retries: {e}")
            time.sleep(retry_delay * attempt)

    # Extract assistant message
    try:
        assistant_text = resp["choices"][0]["message"]["content"].strip()
    except Exception:
        assistant_text = "<no content returned>"

    # Build metadata about docs used
    docs_used = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        docs_used.append(
            {
                "source": md.get("source", "unknown"),
                "chunk_id": md.get("chunk_id", md.get("id", None)),
                "preview": (getattr(d, "page_content", "") or "")[:400].replace("\n", " "),
            }
        )

    metadata = {
        "docs_used": docs_used,
        "llm_raw": {
            "model": openai_model,
            "id": resp.get("id"),
            "created": resp.get("created"),
        },
    }

    return assistant_text, metadata


# Optional helper for Streamlit usage
def streamlit_handle_query(user_query: str, retriever, **kwargs) -> Tuple[str, Dict]:
    """
    Convenience wrapper for Streamlit apps: call answer_with_rag and format a small
    human-friendly output. This function intentionally keeps no Streamlit imports so
    it can be tested outside of Streamlit; in the app, wrap calls in st.spinner / try/except.
    """
    return answer_with_rag(user_query, retriever, **kwargs)


# Simple CLI test (not executed when imported)
if __name__ == "__main__":
    print("sprint.py - quick test stub")
    print("This file is intended to be imported by your app. To test, supply a retriever object.")
